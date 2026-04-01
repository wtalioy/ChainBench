"""FFmpeg filter chain and single-row rendering for stage2."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chainbench.lib.audio import ffprobe_audio
from chainbench.lib.config import relative_to_workspace, resolve_path
from chainbench.lib.proc import run_command

from .validation import validate_output


@dataclass
class RenderResult:
    ok: bool
    status: str
    input_row: dict[str, str]
    output_relpath: str | None = None
    output_duration_sec: float | None = None
    output_sample_rate: int | None = None
    output_channels: int | None = None
    output_codec_name: str | None = None
    output_sample_fmt: str | None = None
    output_size_bytes: int | None = None
    error: str | None = None
    skipped: bool = False


def build_filter_chain(config: dict[str, Any]) -> str:
    filters: list[str] = []
    trim_cfg = config["trim"]
    if trim_cfg.get("enabled", True):
        threshold_db = float(trim_cfg["threshold_db"])
        start_duration = float(trim_cfg["start_duration_sec"])
        stop_duration = float(trim_cfg["stop_duration_sec"])
        filters.append(
            "silenceremove="
            f"start_periods=1:start_duration={start_duration}:start_threshold={threshold_db}dB:"
            f"stop_periods=-1:stop_duration={stop_duration}:stop_threshold={threshold_db}dB"
        )
    loudnorm_cfg = config["loudnorm"]
    if loudnorm_cfg.get("enabled", True):
        filters.append(
            "loudnorm="
            f"I={float(loudnorm_cfg['integrated_lufs'])}:"
            f"LRA={float(loudnorm_cfg['lra'])}:"
            f"TP={float(loudnorm_cfg['true_peak_db'])}"
        )
    return ",".join(filters)


def render_single_row(
    row: dict[str, str],
    config: dict[str, Any],
    workspace_root: Path,
    output_audio_root: Path,
    filter_chain: str,
) -> RenderResult:
    timeout_cfg = config.get("timeouts", {})
    ffmpeg_timeout_sec = int(timeout_cfg.get("ffmpeg_sec", 120))
    ffprobe_timeout_sec = int(timeout_cfg.get("ffprobe_sec", 30))

    input_path = resolve_path(row["stage1_audio_path"], workspace_root)
    if not input_path.exists():
        return RenderResult(
            ok=False,
            status="missing_input",
            input_row=row,
            error=f"missing input file: {input_path}",
        )

    extension = str(config["audio_output"]["extension"])
    output_path = output_audio_root / row["language"] / row["split"] / row["speaker_id"] / f"{row['sample_id']}{extension}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overwrite = bool(config.get("overwrite", False))
    if output_path.exists() and not overwrite:
        probe = ffprobe_audio(output_path, timeout_sec=ffprobe_timeout_sec)
        if probe is not None:
            validation_error = validate_output(probe, config)
            if validation_error is None:
                return RenderResult(
                    ok=True,
                    status="skipped_existing",
                    input_row=row,
                    output_relpath=relative_to_workspace(output_path, workspace_root),
                    output_duration_sec=probe["duration"],
                    output_sample_rate=probe["sample_rate"],
                    output_channels=probe["channels"],
                    output_codec_name=probe["codec_name"],
                    output_sample_fmt=probe.get("sample_fmt", ""),
                    output_size_bytes=probe.get("size"),
                    skipped=True,
                )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map_metadata",
        "-1",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(config["audio_output"]["channels"]),
        "-ar",
        str(config["audio_output"]["sample_rate"]),
    ]
    if filter_chain:
        command.extend(["-af", filter_chain])
    command.extend(
        [
            "-c:a",
            str(config["audio_output"]["codec_name"]),
            str(output_path),
        ]
    )

    result = run_command(command, timeout_sec=ffmpeg_timeout_sec)
    if result.returncode != 0:
        status = "ffmpeg_timeout" if result.returncode == 124 else "ffmpeg_failed"
        return RenderResult(
            ok=False,
            status=status,
            input_row=row,
            error=result.stderr.strip() or "ffmpeg failed with unknown error",
        )

    probe = ffprobe_audio(output_path, timeout_sec=ffprobe_timeout_sec)
    if probe is None:
        return RenderResult(
            ok=False,
            status="ffprobe_failed",
            input_row=row,
            error=f"ffprobe failed for output: {output_path}",
        )

    validation_error = validate_output(probe, config)
    if validation_error is not None:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
        return RenderResult(
            ok=False,
            status="validation_failed",
            input_row=row,
            error=validation_error,
        )

    return RenderResult(
        ok=True,
        status="rendered",
        input_row=row,
        output_relpath=relative_to_workspace(output_path, workspace_root),
        output_duration_sec=probe["duration"],
        output_sample_rate=probe["sample_rate"],
        output_channels=probe["channels"],
        output_codec_name=probe["codec_name"],
        output_sample_fmt=probe.get("sample_fmt", ""),
        output_size_bytes=probe.get("size"),
    )


def make_stage2_row(result: RenderResult, preprocess_desc: dict[str, Any]) -> dict[str, Any]:
    row = dict(result.input_row)
    row["parent_id"] = row["sample_id"]
    row["clean_parent_path"] = result.output_relpath or ""
    row["audio_path"] = result.output_relpath or ""
    row["preprocess_stage"] = "stage2_clean_master"
    row["preprocess_steps"] = json.dumps(preprocess_desc["steps"])
    row["preprocess_params"] = json.dumps(preprocess_desc["params"], sort_keys=True)
    row["source_duration_sec"] = row["duration_sec"]
    row["source_sample_rate"] = row["sample_rate"]
    row["source_channels"] = row["channels"]
    row["source_codec_name"] = row["codec_name"]
    row["duration_sec"] = f"{float(result.output_duration_sec or 0.0):.3f}"
    row["sample_rate"] = str(result.output_sample_rate or "")
    row["channels"] = str(result.output_channels or "")
    row["codec_name"] = result.output_codec_name or ""
    row["sample_fmt"] = result.output_sample_fmt or ""
    row["output_size_bytes"] = str(result.output_size_bytes or "")
    row["chain_family"] = "clean_parent"
    row["operator_seq"] = "[]"
    row["operator_params"] = "{}"
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts_by_lang: dict[str, Counter] = defaultdict(Counter)
    speaker_counts_by_lang: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        language = row["language"]
        split = row["split"]
        split_counts_by_lang[language][split] += 1
        speaker_counts_by_lang[language].add(row["speaker_id"])
    return {
        language: {
            "selected_samples": sum(counter.values()),
            "selected_speakers": len(speaker_counts_by_lang[language]),
            "split_sample_counts": dict(counter),
        }
        for language, counter in split_counts_by_lang.items()
    }
