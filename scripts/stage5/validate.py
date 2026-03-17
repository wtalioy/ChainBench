"""Audio validation and dataset summary helpers for stage5."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from lib.audio import ffprobe_audio
from lib.config import resolve_path


@dataclass
class ValidationResult:
    ok: bool
    status: str
    input_row: dict[str, str]
    error: str | None = None


def validate_probe(probe: dict[str, Any], validation_cfg: dict[str, Any]) -> str | None:
    if probe["duration"] < float(validation_cfg["min_duration_sec"]):
        return f"duration too short: {probe['duration']:.3f}s"
    if probe["duration"] > float(validation_cfg["max_duration_sec"]):
        return f"duration too long: {probe['duration']:.3f}s"
    if probe["sample_rate"] != int(validation_cfg["sample_rate"]):
        return f"unexpected sample rate: {probe['sample_rate']}"
    if probe["channels"] != int(validation_cfg["channels"]):
        return f"unexpected channels: {probe['channels']}"
    if probe["codec_name"] != str(validation_cfg["codec_name"]):
        return f"unexpected codec name: {probe['codec_name']}"
    if probe.get("size", 0) <= 0:
        return "empty output file"
    return None


def inspect_audio_samples(path: Path, clip_threshold: float) -> dict[str, Any]:
    audio, _sample_rate = sf.read(path, dtype="float32", always_2d=False)
    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return {
            "peak_abs": 0.0,
            "clipped_fraction": 0.0,
            "contains_nan": False,
            "contains_inf": False,
            "num_frames": 0,
        }

    contains_nan = bool(np.isnan(samples).any())
    contains_inf = bool(np.isinf(samples).any())
    finite_mask = np.isfinite(samples)
    finite_samples = samples[finite_mask]
    if finite_samples.size == 0:
        peak_abs = float("inf")
        clipped_fraction = 1.0
    else:
        peak_abs = float(np.max(np.abs(finite_samples)))
        clipped_fraction = float(np.mean(np.abs(finite_samples) >= clip_threshold))
    num_frames = int(samples.shape[0]) if samples.ndim > 0 else 0
    return {
        "peak_abs": peak_abs,
        "clipped_fraction": clipped_fraction,
        "contains_nan": contains_nan,
        "contains_inf": contains_inf,
        "num_frames": num_frames,
    }


def probe_parent_duration(
    row: dict[str, str],
    workspace_root: Path,
    timeout_sec: int,
) -> float | None:
    parent_path_value = row.get("clean_parent_path", "")
    if not parent_path_value:
        return None
    parent_path = resolve_path(parent_path_value, workspace_root)
    if not parent_path.exists():
        return None
    parent_probe = ffprobe_audio(parent_path, timeout_sec=timeout_sec)
    if parent_probe is None:
        return None
    return float(parent_probe["duration"])


def validate_single_row(
    row: dict[str, str],
    config: dict[str, Any],
    workspace_root: Path,
) -> ValidationResult:
    ffprobe_timeout_sec = int(config.get("ffprobe_sec", config.get("timeouts", {}).get("ffprobe_sec", 30)))
    validation_cfg = config["validation"]
    path = resolve_path(row["audio_path"], workspace_root)
    if not path.exists():
        return ValidationResult(
            ok=False,
            status="missing_audio",
            input_row=row,
            error=f"missing delivered audio: {path}",
        )

    probe = ffprobe_audio(path, timeout_sec=ffprobe_timeout_sec)
    if probe is None:
        return ValidationResult(
            ok=False,
            status="ffprobe_failed",
            input_row=row,
            error=f"ffprobe failed for {path}",
        )

    probe_error = validate_probe(probe, validation_cfg)
    if probe_error is not None:
        return ValidationResult(
            ok=False,
            status="probe_validation_failed",
            input_row=row,
            error=probe_error,
        )

    try:
        sample_stats = inspect_audio_samples(
            path,
            clip_threshold=float(validation_cfg["clip_threshold"]),
        )
    except Exception as exc:
        return ValidationResult(
            ok=False,
            status="audio_read_failed",
            input_row=row,
            error=f"{type(exc).__name__}: {exc}",
        )

    if sample_stats["contains_nan"] and not bool(validation_cfg.get("allow_nan", False)):
        return ValidationResult(
            ok=False,
            status="nonfinite_audio",
            input_row=row,
            error="audio contains NaN samples",
        )
    if sample_stats["contains_inf"] and not bool(validation_cfg.get("allow_inf", False)):
        return ValidationResult(
            ok=False,
            status="nonfinite_audio",
            input_row=row,
            error="audio contains infinite samples",
        )
    if sample_stats["peak_abs"] > float(validation_cfg["max_abs_peak"]):
        return ValidationResult(
            ok=False,
            status="peak_too_high",
            input_row=row,
            error=f"peak_abs too high: {sample_stats['peak_abs']:.6f}",
        )
    if sample_stats["clipped_fraction"] > float(validation_cfg["max_clipped_fraction"]):
        return ValidationResult(
            ok=False,
            status="clipped_audio",
            input_row=row,
            error=f"clipped_fraction too high: {sample_stats['clipped_fraction']:.6f}",
        )

    parent_duration_sec = probe_parent_duration(row, workspace_root, ffprobe_timeout_sec)
    if parent_duration_sec and parent_duration_sec > 0:
        duration_ratio_to_parent = float(probe["duration"]) / parent_duration_sec
        min_ratio = float(validation_cfg["min_duration_ratio_to_parent"])
        max_ratio = float(validation_cfg["max_duration_ratio_to_parent"])
        if duration_ratio_to_parent < min_ratio:
            return ValidationResult(
                ok=False,
                status="truncated_audio",
                input_row=row,
                error=(
                    "duration ratio to clean parent too small: "
                    f"{duration_ratio_to_parent:.3f} < {min_ratio:.3f}"
                ),
            )
        if duration_ratio_to_parent > max_ratio:
            return ValidationResult(
                ok=False,
                status="overlong_audio",
                input_row=row,
                error=(
                    "duration ratio to clean parent too large: "
                    f"{duration_ratio_to_parent:.3f} > {max_ratio:.3f}"
                ),
            )

    return ValidationResult(
        ok=True,
        status="validated",
        input_row=row,
    )


def summarize_validation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_language: dict[str, Counter] = defaultdict(Counter)
    by_label: Counter = Counter()
    by_chain: dict[str, Counter] = defaultdict(Counter)
    by_generator_family: dict[str, Counter] = defaultdict(Counter)
    speaker_sets: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        language = row["language"]
        split = row["split"]
        label = row["label"]
        chain_family = row["chain_family"]
        generator_family = row["generator_family"]
        by_language[language][split] += 1
        by_label[label] += 1
        by_chain[chain_family][label] += 1
        by_generator_family[generator_family][language] += 1
        speaker_sets[language].add(row["speaker_id"])

    return {
        "labels": dict(by_label),
        "languages": {
            language: {
                "selected_samples": sum(counter.values()),
                "selected_speakers": len(speaker_sets[language]),
                "split_sample_counts": dict(counter),
            }
            for language, counter in by_language.items()
        },
        "chain_families": {key: dict(counter) for key, counter in by_chain.items()},
        "generator_families": {
            key: dict(counter) for key, counter in by_generator_family.items()
        },
    }
