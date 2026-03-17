"""Audio probing and metadata (ffprobe)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .proc import run_command


def ffprobe_audio(path: Path | str, timeout_sec: int = 30) -> dict[str, Any] | None:
    """Return duration, size, sample_rate, channels, codec_name, sample_fmt."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,codec_name,sample_fmt",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command, timeout_sec=timeout_sec)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        fmt = payload["format"]
        return {
            "duration": float(fmt["duration"]),
            "size": int(fmt.get("size", 0)),
            "sample_rate": int(stream["sample_rate"]),
            "channels": int(stream["channels"]),
            "codec_name": str(stream.get("codec_name", "")),
            "sample_fmt": str(stream.get("sample_fmt", "")),
        }
    except (KeyError, ValueError, IndexError, json.JSONDecodeError):
        return None
