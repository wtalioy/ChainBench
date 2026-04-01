"""Shared summary helpers for stage and eval entrypoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_json


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def print_json(payload: Any) -> None:
    print(json_dumps(payload))


def write_timestamped_json(path: Path, payload: dict[str, Any]) -> None:
    payload["generated_at_utc"] = utc_now_iso()
    write_json(path, payload)
