"""Config and path helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_str: str, workspace_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else workspace_root / path


def relative_to_workspace(path: Path, workspace_root: Path) -> str:
    resolved = path.resolve()
    workspace = workspace_root.resolve()
    try:
        return str(resolved.relative_to(workspace)).replace(os.sep, "/")
    except ValueError:
        return str(resolved).replace(os.sep, "/")
