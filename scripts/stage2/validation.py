"""Output validation for stage2 rendered audio."""

from __future__ import annotations

from typing import Any


def validate_output(probe: dict[str, Any], config: dict[str, Any]) -> str | None:
    audio_cfg = config["audio_output"]
    validation_cfg = config["validation"]

    if probe["sample_rate"] != int(audio_cfg["sample_rate"]):
        return f"unexpected sample_rate={probe['sample_rate']}"
    if probe["channels"] != int(audio_cfg["channels"]):
        return f"unexpected channels={probe['channels']}"
    if probe["codec_name"] != str(audio_cfg["codec_name"]):
        return f"unexpected codec_name={probe['codec_name']}"
    sample_fmt = str(probe.get("sample_fmt", ""))
    if not sample_fmt.startswith(str(audio_cfg["sample_fmt"])):
        return f"unexpected sample_fmt={sample_fmt}"
    if probe["duration"] < float(validation_cfg["min_duration_sec"]):
        return f"duration too short: {probe['duration']:.3f}s"
    if probe["duration"] > float(validation_cfg["max_duration_sec"]):
        return f"duration too long: {probe['duration']:.3f}s"
    if probe.get("size", 0) <= 0:
        return "empty output file"
    return None
