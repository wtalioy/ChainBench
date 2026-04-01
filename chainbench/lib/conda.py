"""Shared conda environment helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .proc import run_command


def normalize_conda_env_ref(env_ref: str) -> str:
    env_ref = str(env_ref).strip()
    if not env_ref:
        raise ValueError("conda_env must be a non-empty string")
    path = Path(env_ref).expanduser()
    if path.is_absolute():
        return str(path)
    return env_ref


def is_conda_prefix_path(env_ref: str) -> bool:
    return Path(normalize_conda_env_ref(env_ref)).is_absolute()


def list_conda_env_names() -> set[str]:
    result = run_command(["conda", "env", "list", "--json"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query conda envs: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    return {Path(path).name for path in payload.get("envs", [])}


def require_conda_envs(env_refs: list[str] | set[str], *, label: str) -> None:
    requested_names: set[str] = set()
    requested_paths: set[str] = set()
    for env_ref in env_refs:
        normalized = normalize_conda_env_ref(env_ref)
        if Path(normalized).is_absolute():
            requested_paths.add(normalized)
        else:
            requested_names.add(normalized)

    missing = sorted(requested_names - list_conda_env_names())
    missing.extend(sorted(path for path in requested_paths if not Path(path).exists()))
    if missing:
        raise RuntimeError(f"Missing {label}: " + ", ".join(missing))


def conda_run_python_command(env_ref: str) -> list[str]:
    normalized = normalize_conda_env_ref(env_ref)
    if Path(normalized).is_absolute():
        return ["conda", "run", "--no-capture-output", "-p", normalized, "python", "-u"]
    return ["conda", "run", "--no-capture-output", "-n", normalized, "python", "-u"]
