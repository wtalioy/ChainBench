"""Core preservation analysis over matched parent-child audio pairs."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

import numpy as np
import soundfile as sf

from chainbench.lib.config import resolve_path

from .backends import _call_backend
from .schema import DEFAULT_SPLITS, RESULT_FIELDNAMES

T = TypeVar("T")


@dataclass(frozen=True)
class AudioSummary:
    duration_sec: float
    rms_dbfs: float
    peak_abs: float
    sample_rate: int
    channels: int


@dataclass
class PreparedRow:
    row: Mapping[str, str]
    result: dict[str, Any]
    language: str
    reference_text: str
    parent_path: Path | None
    child_path: Path | None

    @property
    def is_ready(self) -> bool:
        return self.result.get("status") == "ok" and self.parent_path is not None and self.child_path is not None


def resolve_split(row: Mapping[str, str]) -> str:
    return str(row.get("split_standard", "") or row.get("split", "")).strip()


def _round_float(value: float | None, digits: int = 6) -> float | str:
    if value is None:
        return ""
    return round(float(value), digits)


def _base_result(row: Mapping[str, str]) -> dict[str, Any]:
    reference_text = str(row.get("transcript", "") or row.get("raw_transcript", ""))
    result = {field: "" for field in RESULT_FIELDNAMES}
    result.update(
        {
            "sample_id": str(row.get("sample_id", "")),
            "parent_id": str(row.get("parent_id", "")),
            "split": resolve_split(row),
            "language": str(row.get("language", "")),
            "label": str(row.get("label", "")),
            "chain_family": str(row.get("chain_family", "")),
            "generator_family": str(row.get("generator_family", "")),
            "generator_name": str(row.get("generator_name", "")),
            "speaker_id": str(row.get("speaker_id", "")),
            "source_speaker_id": str(row.get("source_speaker_id", "")),
            "clean_parent_path": str(row.get("clean_parent_path", "")),
            "file_name": str(row.get("file_name", "")),
            "status": "ok",
            "error": "",
            "reference_text": reference_text,
        }
    )
    result["reference_text_normalized"] = normalize_reference_text(reference_text, str(result["language"]))
    return result


def _failed_prepared_row(
    row: Mapping[str, str],
    result: dict[str, Any],
    *,
    language: str,
    reference_text: str,
    status: str,
    error: str,
) -> PreparedRow:
    result["status"] = status
    result["error"] = error
    return PreparedRow(
        row=row,
        result=result,
        language=language,
        reference_text=reference_text,
        parent_path=None,
        child_path=None,
    )


def _apply_audio_metrics(result: dict[str, Any], parent_audio: AudioSummary, child_audio: AudioSummary) -> None:
    result["parent_duration_sec"] = _round_float(parent_audio.duration_sec, 6)
    result["child_duration_sec"] = _round_float(child_audio.duration_sec, 6)
    result["duration_delta_sec"] = _round_float(child_audio.duration_sec - parent_audio.duration_sec, 6)
    result["duration_ratio_to_parent"] = _round_float(child_audio.duration_sec / parent_audio.duration_sec, 6)
    result["parent_rms_dbfs"] = _round_float(parent_audio.rms_dbfs, 6)
    result["child_rms_dbfs"] = _round_float(child_audio.rms_dbfs, 6)
    result["rms_dbfs_delta"] = _round_float(child_audio.rms_dbfs - parent_audio.rms_dbfs, 6)
    result["rms_dbfs_delta_abs"] = _round_float(abs(child_audio.rms_dbfs - parent_audio.rms_dbfs), 6)
    result["parent_peak_abs"] = _round_float(parent_audio.peak_abs, 6)
    result["child_peak_abs"] = _round_float(child_audio.peak_abs, 6)


def _apply_transcript_metrics(
    result: dict[str, Any],
    *,
    reference_text: str,
    parent_transcript: str,
    child_transcript: str,
    language: str,
) -> None:
    result["parent_transcript"] = parent_transcript
    result["child_transcript"] = child_transcript
    result["parent_transcript_normalized"] = normalize_reference_text(parent_transcript, language)
    result["child_transcript_normalized"] = normalize_reference_text(child_transcript, language)
    parent_wer = compute_wer(reference_text, parent_transcript, language)
    child_wer = compute_wer(reference_text, child_transcript, language)
    parent_cer = compute_cer(reference_text, parent_transcript, language)
    child_cer = compute_cer(reference_text, child_transcript, language)
    result["parent_wer"] = _round_float(parent_wer, 6)
    result["child_wer"] = _round_float(child_wer, 6)
    result["wer_drift"] = (
        _round_float(child_wer - parent_wer, 6) if parent_wer is not None and child_wer is not None else ""
    )
    result["parent_cer"] = _round_float(parent_cer, 6)
    result["child_cer"] = _round_float(child_cer, 6)
    result["cer_drift"] = (
        _round_float(child_cer - parent_cer, 6) if parent_cer is not None and child_cer is not None else ""
    )


def normalize_reference_text(text: str, language: str = "") -> str:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return ""
    if language == "zh":
        keep_chars = [char for char in normalized if ("\u4e00" <= char <= "\u9fff") or char.isalnum()]
        return "".join(keep_chars)
    lowered = normalized.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def transcript_word_tokens(text: str, language: str = "") -> list[str]:
    normalized = normalize_reference_text(text, language)
    if not normalized:
        return []
    if language == "zh" and " " not in normalized:
        return list(normalized)
    return normalized.split()


def transcript_char_tokens(text: str, language: str = "") -> list[str]:
    normalized = normalize_reference_text(text, language)
    compact = normalized.replace(" ", "")
    return list(compact)


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[Any], hypothesis: Sequence[Any]) -> float | None:
    if not reference:
        return None if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / float(len(reference))


def compute_wer(reference_text: str, hypothesis_text: str, language: str = "") -> float | None:
    return error_rate(
        transcript_word_tokens(reference_text, language),
        transcript_word_tokens(hypothesis_text, language),
    )


def compute_cer(reference_text: str, hypothesis_text: str, language: str = "") -> float | None:
    return error_rate(
        transcript_char_tokens(reference_text, language),
        transcript_char_tokens(hypothesis_text, language),
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_array = np.asarray(list(left), dtype=np.float32)
    right_array = np.asarray(list(right), dtype=np.float32)
    if left_array.size == 0 or right_array.size == 0 or left_array.shape != right_array.shape:
        return None
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(np.dot(left_array, right_array) / (left_norm * right_norm))


@lru_cache(maxsize=32768)
def inspect_audio_summary(path_str: str) -> AudioSummary | None:
    path = Path(path_str)
    if not path.exists():
        return None
    peak_abs = 0.0
    sum_squares = 0.0
    sample_count = 0
    sample_rate = 0
    channels = 0
    try:
        with sf.SoundFile(path, "r") as audio_file:
            sample_rate = int(audio_file.samplerate)
            channels = int(audio_file.channels)
            for block in audio_file.blocks(blocksize=65536, dtype="float32", always_2d=True):
                mono = np.mean(np.asarray(block, dtype=np.float32), axis=1)
                if mono.size == 0:
                    continue
                finite = mono[np.isfinite(mono)]
                if finite.size == 0:
                    continue
                peak_abs = max(peak_abs, float(np.max(np.abs(finite))))
                sum_squares += float(np.dot(finite, finite))
                sample_count += int(finite.size)
    except Exception:
        return None
    if sample_rate <= 0 or sample_count <= 0:
        return None
    rms = math.sqrt(max(sum_squares / sample_count, 1e-12))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
    duration_sec = sample_count / float(sample_rate)
    return AudioSummary(
        duration_sec=duration_sec,
        rms_dbfs=rms_dbfs,
        peak_abs=peak_abs,
        sample_rate=sample_rate,
        channels=channels,
    )


class PreservationAnalyzer:
    def __init__(
        self,
        workspace_root: Path,
        *,
        dataset_root: Path | None = None,
        asr_backend: Any = None,
        speaker_backend: Any = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.dataset_root = dataset_root.resolve() if dataset_root is not None else workspace_root
        self.asr_backend = asr_backend
        self.speaker_backend = speaker_backend
        self._transcript_cache: dict[tuple[str, str], str] = {}
        self._embedding_cache: dict[tuple[str, str], list[float]] = {}

    def analyze_rows(
        self,
        rows: Iterable[Mapping[str, str]],
        *,
        show_progress: bool = False,
        progress_desc: str = "Preservation",
    ) -> list[dict[str, Any]]:
        selected_rows = list(rows)
        prepared_rows = self._prepare_rows(
            selected_rows,
            show_progress=show_progress,
            progress_desc=f"{progress_desc} prep",
        )
        ready_rows = [prepared for prepared in prepared_rows if prepared.is_ready]
        if self.asr_backend is not None and ready_rows:
            self._apply_asr_stage(
                ready_rows,
                show_progress=show_progress,
                progress_desc=f"{progress_desc} ASR",
            )
        speaker_rows = [prepared for prepared in ready_rows if prepared.result.get("status") == "ok"]
        if self.speaker_backend is not None and speaker_rows:
            self._apply_speaker_stage(
                speaker_rows,
                show_progress=show_progress,
                progress_desc=f"{progress_desc} speaker",
            )
        return [prepared.result for prepared in prepared_rows]

    def analyze_row(self, row: Mapping[str, str]) -> dict[str, Any]:
        return self.analyze_rows([row], show_progress=False)[0]

    def _prepare_rows(
        self,
        rows: Sequence[Mapping[str, str]],
        *,
        show_progress: bool,
        progress_desc: str,
    ) -> list[PreparedRow]:
        progress = self._make_progress_bar(len(rows), progress_desc, show_progress, unit="row")
        prepared_rows: list[PreparedRow] = []
        for row in rows:
            prepared_rows.append(self._prepare_row(row))
            if progress is not None:
                progress.update(1)
        if progress is not None:
            progress.close()
        return prepared_rows

    def _prepare_row(self, row: Mapping[str, str]) -> PreparedRow:
        result = _base_result(row)
        language = str(result["language"])
        reference_text = str(result["reference_text"])

        child_path = self._resolve_child_file_name(result["file_name"]) if result["file_name"] else None
        if child_path is None or not child_path.exists():
            return _failed_prepared_row(
                row,
                result,
                language=language,
                reference_text=reference_text,
                status="missing_child_audio",
                error="delivered audio path is missing",
            )

        parent_path = (
            resolve_path(result["clean_parent_path"], self.workspace_root) if result["clean_parent_path"] else None
        )
        if parent_path is None or not parent_path.exists():
            return _failed_prepared_row(
                row,
                result,
                language=language,
                reference_text=reference_text,
                status="missing_clean_parent",
                error="clean parent path is missing",
            )

        parent_audio = inspect_audio_summary(str(parent_path))
        child_audio = inspect_audio_summary(str(child_path))
        if parent_audio is None or child_audio is None:
            return _failed_prepared_row(
                row,
                result,
                language=language,
                reference_text=reference_text,
                status="audio_read_error",
                error="unable to read parent or child audio",
            )

        _apply_audio_metrics(result, parent_audio, child_audio)
        return PreparedRow(
            row=row,
            result=result,
            language=language,
            reference_text=reference_text,
            parent_path=parent_path,
            child_path=child_path,
        )

    def _apply_asr_stage(
        self,
        prepared_rows: Sequence[PreparedRow],
        *,
        show_progress: bool,
        progress_desc: str,
    ) -> None:
        pending_by_language: dict[str, dict[tuple[str, str], tuple[Path, Mapping[str, str]]]] = defaultdict(dict)
        for prepared in prepared_rows:
            for path in (prepared.parent_path, prepared.child_path):
                if path is None:
                    continue
                key = (str(path), prepared.language)
                if key in self._transcript_cache:
                    continue
                pending_by_language[prepared.language].setdefault(key, (path, prepared.row))
        total_pending = sum(len(items) for items in pending_by_language.values())
        if total_pending:
            print(
                f"ASR stage: processing {total_pending:,} uncached audio files "
                f"across {len(pending_by_language):,} language bucket(s).",
                flush=True,
            )
        progress = self._make_progress_bar(total_pending, progress_desc, show_progress, unit="audio")
        transcript_errors: dict[tuple[str, str], str] = {}
        for language, language_items in sorted(pending_by_language.items()):
            items = [(key, path, row) for key, (path, row) in language_items.items()]
            self._populate_transcript_cache(items, language=language, progress=progress, errors=transcript_errors)
        if progress is not None:
            progress.close()

        for prepared in prepared_rows:
            if prepared.result.get("status") != "ok":
                continue
            parent_key = (str(prepared.parent_path), prepared.language)
            child_key = (str(prepared.child_path), prepared.language)
            parent_error = transcript_errors.get(parent_key)
            child_error = transcript_errors.get(child_key)
            if parent_error or child_error:
                prepared.result["status"] = "asr_error"
                prepared.result["error"] = parent_error or child_error or "missing ASR transcript"
                continue
            parent_transcript = self._transcript_cache.get(parent_key)
            child_transcript = self._transcript_cache.get(child_key)
            if parent_transcript is None or child_transcript is None:
                prepared.result["status"] = "asr_error"
                prepared.result["error"] = "missing ASR transcript"
                continue
            _apply_transcript_metrics(
                prepared.result,
                reference_text=prepared.reference_text,
                parent_transcript=parent_transcript,
                child_transcript=child_transcript,
                language=prepared.language,
            )

    def _apply_speaker_stage(
        self,
        prepared_rows: Sequence[PreparedRow],
        *,
        show_progress: bool,
        progress_desc: str,
    ) -> None:
        pending_items: dict[tuple[str, str], tuple[Path, str, Mapping[str, str]]] = {}
        for prepared in prepared_rows:
            for path in (prepared.parent_path, prepared.child_path):
                if path is None:
                    continue
                key = (str(path), prepared.language)
                if key in self._embedding_cache:
                    continue
                pending_items.setdefault(key, (path, prepared.language, prepared.row))
        total_pending = len(pending_items)
        if total_pending:
            print(f"Speaker stage: processing {total_pending:,} uncached audio files.", flush=True)
        progress = self._make_progress_bar(total_pending, progress_desc, show_progress, unit="audio")
        embedding_errors: dict[tuple[str, str], str] = {}
        self._populate_embedding_cache(list(pending_items.items()), progress=progress, errors=embedding_errors)
        if progress is not None:
            progress.close()

        for prepared in prepared_rows:
            if prepared.result.get("status") != "ok":
                continue
            parent_key = (str(prepared.parent_path), prepared.language)
            child_key = (str(prepared.child_path), prepared.language)
            parent_error = embedding_errors.get(parent_key)
            child_error = embedding_errors.get(child_key)
            if parent_error or child_error:
                prepared.result["status"] = "speaker_error"
                prepared.result["error"] = parent_error or child_error or "missing speaker embedding"
                continue
            parent_embedding = self._embedding_cache.get(parent_key)
            child_embedding = self._embedding_cache.get(child_key)
            if parent_embedding is None or child_embedding is None:
                prepared.result["status"] = "speaker_error"
                prepared.result["error"] = "missing speaker embedding"
                continue
            prepared.result["speaker_similarity"] = _round_float(
                cosine_similarity(parent_embedding, child_embedding),
                6,
            )

    def _populate_transcript_cache(
        self,
        items: Sequence[tuple[tuple[str, str], Path, Mapping[str, str]]],
        *,
        language: str,
        progress,
        errors: dict[tuple[str, str], str],
    ) -> None:
        batch_method = getattr(self.asr_backend, "transcribe_many", None)
        batch_size = self._get_backend_batch_size(self.asr_backend)
        if callable(batch_method):
            for chunk in _chunked(list(items), batch_size):
                keys = [key for key, _path, _row in chunk]
                paths = [path for _key, path, _row in chunk]
                rows = [row for _key, _path, row in chunk]
                try:
                    transcripts = list(batch_method(paths, language=language, rows=rows))
                    if len(transcripts) != len(chunk):
                        raise RuntimeError("ASR batch output length mismatch")
                    for key, transcript in zip(keys, transcripts):
                        self._transcript_cache[key] = str(transcript or "").strip()
                except Exception:
                    for key, path, row in chunk:
                        self._transcribe_single(key, path, language=language, row=row, errors=errors)
                if progress is not None:
                    progress.update(len(chunk))
            return
        for key, path, row in items:
            self._transcribe_single(key, path, language=language, row=row, errors=errors)
            if progress is not None:
                progress.update(1)

    def _populate_embedding_cache(
        self,
        items: Sequence[tuple[tuple[str, str], tuple[Path, str, Mapping[str, str]]]],
        *,
        progress,
        errors: dict[tuple[str, str], str],
    ) -> None:
        batch_method = getattr(self.speaker_backend, "embed_many", None)
        batch_size = self._get_backend_batch_size(self.speaker_backend)
        if callable(batch_method):
            for chunk in _chunked(list(items), batch_size):
                keys = [key for key, _payload in chunk]
                paths = [payload[0] for _key, payload in chunk]
                languages = [payload[1] for _key, payload in chunk]
                rows = [payload[2] for _key, payload in chunk]
                try:
                    embeddings = list(batch_method(paths, language=languages[0] if languages else "", rows=rows))
                    if len(embeddings) != len(chunk):
                        raise RuntimeError("speaker batch output length mismatch")
                    for key, embedding in zip(keys, embeddings):
                        self._embedding_cache[key] = list(np.asarray(embedding, dtype=np.float32).reshape(-1))
                except Exception:
                    for key, (path, language, row) in chunk:
                        self._embed_single(key, path, language=language, row=row, errors=errors)
                if progress is not None:
                    progress.update(len(chunk))
            return
        for key, (path, language, row) in items:
            self._embed_single(key, path, language=language, row=row, errors=errors)
            if progress is not None:
                progress.update(1)

    def _transcribe_single(
        self,
        key: tuple[str, str],
        path: Path,
        *,
        language: str,
        row: Mapping[str, str],
        errors: dict[tuple[str, str], str],
    ) -> None:
        try:
            self._transcript_cache[key] = self._transcribe(path, language=language, row=row)
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"

    def _embed_single(
        self,
        key: tuple[str, str],
        path: Path,
        *,
        language: str,
        row: Mapping[str, str],
        errors: dict[tuple[str, str], str],
    ) -> None:
        try:
            self._embedding_cache[key] = self._embed(path, language=language, row=row)
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"

    def _transcribe(self, path: Path, *, language: str, row: Mapping[str, str]) -> str:
        key = (str(path), language)
        if key not in self._transcript_cache:
            transcript = _call_backend(
                self.asr_backend,
                "transcribe",
                path,
                language=language,
                row=row,
            )
            self._transcript_cache[key] = str(transcript or "").strip()
        return self._transcript_cache[key]

    def _embed(self, path: Path, *, language: str, row: Mapping[str, str]) -> list[float]:
        key = (str(path), language)
        if key not in self._embedding_cache:
            embedding = _call_backend(
                self.speaker_backend,
                "embed",
                path,
                language=language,
                row=row,
            )
            self._embedding_cache[key] = list(np.asarray(embedding, dtype=np.float32).reshape(-1))
        return self._embedding_cache[key]

    def _resolve_child_file_name(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        dataset_candidate = self.dataset_root / path
        if dataset_candidate.exists():
            return dataset_candidate
        return resolve_path(path_value, self.workspace_root)

    @staticmethod
    def _make_progress_bar(total: int, desc: str, enabled: bool, *, unit: str) -> Any:
        if not enabled or total <= 0:
            return None
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return None
        return tqdm(total=total, desc=desc, unit=unit)

    @staticmethod
    def _get_backend_batch_size(backend: Any) -> int:
        value = getattr(backend, "batch_size", getattr(backend, "_batch_size", 1))
        try:
            return max(int(value), 1)
        except (TypeError, ValueError):
            return 1


def _chunked(items: Sequence[T], chunk_size: int) -> list[Sequence[T]]:
    if chunk_size <= 0:
        chunk_size = 1
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


__all__ = [
    "AudioSummary",
    "DEFAULT_SPLITS",
    "PreservationAnalyzer",
    "RESULT_FIELDNAMES",
    "compute_cer",
    "compute_wer",
    "cosine_similarity",
    "edit_distance",
    "error_rate",
    "inspect_audio_summary",
    "normalize_reference_text",
    "resolve_split",
    "transcript_char_tokens",
    "transcript_word_tokens",
]
