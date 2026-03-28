"""Lightweight runtime resource snapshots for long-running scripts."""

from __future__ import annotations

import os
import resource
import subprocess
from typing import Any, Iterable


def _bytes_to_gib(value: int) -> float:
    return round(float(value) / (1024**3), 3)


def current_rss_bytes() -> int | None:
    status_path = "/proc/self/status"
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def max_rss_bytes() -> int | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (AttributeError, OSError):
        return None
    if usage.ru_maxrss <= 0:
        return None
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    scale = 1024 if os.name == "posix" and os.uname().sysname != "Darwin" else 1
    return int(usage.ru_maxrss) * scale


def _normalize_gpu_indices(devices: Iterable[str] | None = None) -> list[int]:
    tokens: list[str] = []
    if devices:
        tokens.extend(str(device).strip() for device in devices if str(device).strip())
    elif os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        tokens.extend(part.strip() for part in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if part.strip())

    normalized: list[int] = []
    for token in tokens:
        if token.startswith("cuda:"):
            token = token.split(":", 1)[1].strip()
        if token.isdigit():
            normalized.append(int(token))
    return sorted(set(normalized))


def gpu_memory_snapshot(devices: Iterable[str] | None = None) -> list[dict[str, int]]:
    gpu_indices = _normalize_gpu_indices(devices)
    if not gpu_indices:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []

    snapshots: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4 or not parts[0].isdigit():
            continue
        index = int(parts[0])
        if index not in gpu_indices:
            continue
        used = int(parts[1]) if parts[1].isdigit() else -1
        total = int(parts[2]) if parts[2].isdigit() else -1
        util = int(parts[3]) if parts[3].isdigit() else -1
        snapshots.append(
            {
                "index": index,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "utilization_gpu_pct": util,
            }
        )
    return sorted(snapshots, key=lambda item: item["index"])


def runtime_snapshot(label: str, *, devices: Iterable[str] | None = None) -> dict[str, Any]:
    rss_bytes = current_rss_bytes()
    max_bytes = max_rss_bytes()
    payload: dict[str, Any] = {
        "label": label,
        "pid": os.getpid(),
    }
    if rss_bytes is not None:
        payload["rss_gib"] = _bytes_to_gib(rss_bytes)
    if max_bytes is not None:
        payload["max_rss_gib"] = _bytes_to_gib(max_bytes)
    gpu = gpu_memory_snapshot(devices)
    if gpu:
        payload["gpus"] = gpu
    return payload

