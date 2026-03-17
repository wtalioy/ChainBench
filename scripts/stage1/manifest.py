"""Symlinks and manifest row building for stage1."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lib.config import relative_to_workspace
from lib.io import write_csv

from .candidates import AcceptedSample


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if os.path.realpath(dst) == os.path.realpath(src):
            return
        dst.unlink()
    elif dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def sample_to_manifest_row(
    sample: AcceptedSample,
    speaker_id: str,
    split: str,
    seed: int,
    workspace_root: Path,
    raw_root: Path,
) -> dict[str, Any]:
    src = Path(sample.candidate.source_audio_path)
    dst = raw_root / sample.candidate.language / split / speaker_id / f"{sample.candidate.utterance_id}{src.suffix}"
    sample_id = f"{speaker_id}_{sample.candidate.utterance_id}"
    return {
        "sample_id": sample_id,
        "split": split,
        "language": sample.candidate.language,
        "source_corpus": sample.candidate.source_corpus,
        "speaker_id": speaker_id,
        "source_speaker_id": sample.candidate.source_speaker_id,
        "utterance_id": sample.candidate.utterance_id,
        "transcript": sample.candidate.transcript,
        "raw_transcript": sample.candidate.raw_transcript,
        "label": "bona_fide",
        "generator_family": "none",
        "generator_name": "none",
        "chain_family": "source_clean",
        "operator_seq": "[]",
        "operator_params": "{}",
        "seed": seed,
        "duration_sec": f"{sample.duration:.3f}",
        "sample_rate": sample.sample_rate,
        "channels": sample.channels,
        "codec_name": sample.codec_name,
        "mean_volume_db": f"{sample.mean_volume_db:.2f}",
        "max_volume_db": f"{sample.max_volume_db:.2f}",
        "silence_duration_sec": f"{sample.silence_duration_sec:.3f}",
        "effective_speech_ratio": f"{sample.speech_ratio:.3f}",
        "source_split": sample.candidate.source_split,
        "source_audio_path": relative_to_workspace(src, workspace_root),
        "stage1_audio_path": relative_to_workspace(dst, workspace_root),
        "license_tag": sample.candidate.license_tag,
        "speaker_gender": sample.candidate.speaker_meta.get("gender", ""),
        "speaker_age": sample.candidate.speaker_meta.get("age", sample.candidate.speaker_meta.get("age_group", "")),
        "speaker_accent": sample.candidate.speaker_meta.get("accent", sample.candidate.speaker_meta.get("accents", "")),
        "speaker_variant": sample.candidate.speaker_meta.get("variant", ""),
        "sentence_id": sample.candidate.extra_meta.get("sentence_id", ""),
        "locale": sample.candidate.extra_meta.get("locale", ""),
        "sentence_domain": sample.candidate.extra_meta.get("sentence_domain", ""),
    }
