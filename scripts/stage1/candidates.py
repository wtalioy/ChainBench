"""Candidate loading from AISHELL-3 and Common Voice."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class Candidate:
    source_speaker_id: str
    utterance_id: str
    transcript: str
    raw_transcript: str
    source_audio_path: str
    source_split: str
    source_corpus: str
    language: str
    license_tag: str
    speaker_meta: dict[str, str]
    extra_meta: dict[str, str]


@dataclass
class ProbedCandidate:
    candidate: Candidate
    duration: float
    sample_rate: int
    channels: int
    codec_name: str


@dataclass
class AcceptedSample:
    candidate: Candidate
    duration: float
    sample_rate: int
    channels: int
    codec_name: str
    mean_volume_db: float
    max_volume_db: float
    silence_duration_sec: float
    speech_ratio: float


def normalize_english_transcript(text: str) -> str | None:
    tokens = EN_TOKEN_RE.findall(text.strip())
    if len(tokens) < 4:
        return None
    if tokens and all(t.isdigit() for t in tokens):
        return None
    return " ".join(tokens)


def normalize_aishell_transcript(text: str) -> str | None:
    tokens = text.strip().split()
    chars = "".join(tokens[::2]).strip()
    if len(ZH_CHAR_RE.findall(chars)) < 4:
        return None
    if chars.isdigit():
        return None
    return chars


def load_aishell_speaker_meta(dataset_root: Path) -> dict[str, dict[str, str]]:
    speaker_meta: dict[str, dict[str, str]] = {}
    meta_path = dataset_root / "spk-info.txt"
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            speaker_meta[parts[0]] = {
                "age_group": parts[1],
                "gender": parts[2],
                "accent": parts[3],
            }
    return speaker_meta


def load_aishell_candidates(
    lang_cfg: dict[str, Any],
    counters: dict[str, Any],
) -> dict[str, list[Candidate]]:
    dataset_root = Path(lang_cfg["dataset_root"])
    speaker_meta = load_aishell_speaker_meta(dataset_root)
    speaker_to_candidates: dict[str, list[Candidate]] = defaultdict(list)
    lines_seen = 0

    for source_split in ("train", "test"):
        transcript_path = dataset_root / source_split / "content.txt"
        wav_root = dataset_root / source_split / "wav"
        with transcript_path.open("r", encoding="utf-8") as f:
            for line in f:
                lines_seen += 1
                line = line.rstrip("\n")
                if not line:
                    continue
                utterance_id, raw_transcript = line.split("\t", 1)
                transcript = normalize_aishell_transcript(raw_transcript)
                if transcript is None:
                    counters["text"]["rejected_short_or_numeric"] += 1
                    continue
                source_speaker_id = utterance_id[:7]
                audio_path = wav_root / source_speaker_id / utterance_id
                if not audio_path.exists():
                    counters["text"]["missing_audio_file"] += 1
                    continue
                speaker_to_candidates[source_speaker_id].append(
                    Candidate(
                        source_speaker_id=source_speaker_id,
                        utterance_id=Path(utterance_id).stem,
                        transcript=transcript,
                        raw_transcript=raw_transcript,
                        source_audio_path=str(audio_path),
                        source_split=source_split,
                        source_corpus=lang_cfg["source_corpus"],
                        language="zh",
                        license_tag=lang_cfg["license_tag"],
                        speaker_meta=speaker_meta.get(source_speaker_id, {}),
                        extra_meta={},
                    )
                )
                counters["text"]["accepted"] += 1
    return speaker_to_candidates


def load_common_voice_candidates(
    lang_cfg: dict[str, Any],
    counters: dict[str, Any],
) -> dict[str, list[Candidate]]:
    import csv

    dataset_root = Path(lang_cfg["dataset_root"])
    tsv_path = dataset_root / "validated.tsv"
    clips_root = dataset_root / "clips"
    speaker_to_candidates: dict[str, list[Candidate]] = defaultdict(list)
    rows_seen = 0

    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows_seen += 1
            transcript = normalize_english_transcript(row.get("sentence", ""))
            if transcript is None:
                counters["text"]["rejected_short_or_numeric"] += 1
                continue
            source_speaker_id = row["client_id"]
            audio_path = clips_root / row["path"]
            if not audio_path.exists():
                counters["text"]["missing_audio_file"] += 1
                continue
            speaker_to_candidates[source_speaker_id].append(
                Candidate(
                    source_speaker_id=source_speaker_id,
                    utterance_id=Path(row["path"]).stem,
                    transcript=transcript,
                    raw_transcript=row.get("sentence", ""),
                    source_audio_path=str(audio_path),
                    source_split="validated",
                    source_corpus=lang_cfg["source_corpus"],
                    language="en",
                    license_tag=lang_cfg["license_tag"],
                    speaker_meta={
                        "age": row.get("age", "") or "",
                        "gender": row.get("gender", "") or "",
                        "accents": row.get("accents", "") or "",
                        "variant": row.get("variant", "") or "",
                    },
                    extra_meta={
                        "sentence_id": row.get("sentence_id", "") or "",
                        "locale": row.get("locale", "") or "",
                        "sentence_domain": row.get("sentence_domain", "") or "",
                    },
                )
            )
            counters["text"]["accepted"] += 1
    return speaker_to_candidates
