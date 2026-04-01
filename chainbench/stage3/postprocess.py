"""Postprocess and validate spoof audio for stage3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chainbench.lib.proc import run_command


def build_postprocess_filter_chain(config: dict[str, Any]) -> str:
    post_cfg = config.get("postprocess", {})
    filters: list[str] = []
    trim_cfg = post_cfg.get("trim", {})
    if trim_cfg.get("enabled", False):
        filters.append(
            "silenceremove="
            f"start_periods=1:start_duration={float(trim_cfg['start_duration_sec'])}:start_threshold={float(trim_cfg['threshold_db'])}dB:"
            f"stop_periods=-1:stop_duration={float(trim_cfg['stop_duration_sec'])}:stop_threshold={float(trim_cfg['threshold_db'])}dB"
        )
    loudnorm_cfg = post_cfg.get("loudnorm", {})
    if loudnorm_cfg.get("enabled", False):
        filters.append(
            "loudnorm="
            f"I={float(loudnorm_cfg['integrated_lufs'])}:"
            f"LRA={float(loudnorm_cfg['lra'])}:"
            f"TP={float(loudnorm_cfg['true_peak_db'])}"
        )
    return ",".join(filters)


def postprocess_audio(raw_path: Path, final_path: Path, config: dict[str, Any]) -> str | None:
    post_cfg = config.get("postprocess", {})
    if not post_cfg.get("enabled", False):
        if raw_path.resolve() != final_path.resolve():
            if final_path.exists() and final_path.stat().st_size > 0:
                return None
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_bytes(raw_path.read_bytes())
        return None

    if final_path.exists() and final_path.stat().st_size > 0:
        return None

    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp.wav")
    audio_cfg = post_cfg["audio_output"]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_path),
        "-map_metadata",
        "-1",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(audio_cfg["channels"]),
        "-ar",
        str(audio_cfg["sample_rate"]),
    ]
    filter_chain = build_postprocess_filter_chain(config)
    if filter_chain:
        command.extend(["-af", filter_chain])
    command.extend(["-c:a", str(audio_cfg["codec_name"]), str(tmp_path)])
    timeout_cfg = config.get("timeouts", {})
    ffmpeg_timeout_sec = int(timeout_cfg.get("ffmpeg_sec", 120))
    result = run_command(command, timeout_sec=ffmpeg_timeout_sec)
    if result.returncode != 0:
        if result.returncode == 124:
            return result.stderr.strip() or f"ffmpeg postprocess TIMEOUT after {ffmpeg_timeout_sec}s"
        return result.stderr.strip() or "ffmpeg postprocess failed"
    tmp_path.replace(final_path)
    return None


def validate_spoof_output(
    probe: dict[str, Any],
    source_duration_sec: float,
    config: dict[str, Any],
) -> str | None:
    validation = config["validation"]
    if probe.get("size", 0) <= 0:
        return "empty output file"
    if probe["duration"] < float(validation["min_duration_sec"]):
        return f"duration too short: {probe['duration']:.3f}s"
    if probe["duration"] > float(validation["max_duration_sec"]):
        return f"duration too long: {probe['duration']:.3f}s"
    duration_ratio = probe["duration"] / max(source_duration_sec, 1e-6)
    if duration_ratio < float(validation["min_duration_ratio"]):
        return f"duration ratio too small: {duration_ratio:.3f}"
    if duration_ratio > float(validation["max_duration_ratio"]):
        return f"duration ratio too large: {duration_ratio:.3f}"

    post_cfg = config.get("postprocess", {})
    if post_cfg.get("enabled", False):
        audio_cfg = post_cfg["audio_output"]
        if probe["sample_rate"] != int(audio_cfg["sample_rate"]):
            return f"unexpected sample_rate={probe['sample_rate']}"
        if probe["channels"] != int(audio_cfg["channels"]):
            return f"unexpected channels={probe['channels']}"
        if probe["codec_name"] != str(audio_cfg["codec_name"]):
            return f"unexpected codec_name={probe['codec_name']}"
        if not str(probe.get("sample_fmt", "")).startswith(str(audio_cfg["sample_fmt"])):
            return f"unexpected sample_fmt={probe.get('sample_fmt')}"
    return None
