"""Shared configuration loader for ChainBench baseline evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from lib.config import load_json, resolve_path
from .tasks import TASK_IDS

BASELINE_IDS = ("aasist", "sls_df", "safeear", "nes2net")
DEFAULT_TRAIN_CONFIG: dict[str, Any] = {
    "enabled": True,
    "devices": ["cuda:0"],
    "seed": 1234,
    "epochs": 100,
    "batch_size": 16,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "learning_rate": 1e-6,
    "weight_decay": 1e-4,
}
DEFAULT_EVAL_CONFIG: dict[str, Any] = {
    "enabled": True,
    "devices": ["cuda:0"],
    "batch_size": 8,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
}
DEFAULT_BASELINE_CONFIG: dict[str, Any] = {
    "repo_path": "",
    "conda_prefix": "",
    "env": {},
    "assets": {},
    "adapter": {},
    "train": DEFAULT_TRAIN_CONFIG,
    "eval": DEFAULT_EVAL_CONFIG,
}


def _merge_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_phase_config(phase_cfg: dict[str, Any], *, baseline_name: str, phase_name: str) -> dict[str, Any]:
    if "device" in phase_cfg:
        raise ValueError(
            f"Baseline {baseline_name!r} phase {phase_name!r} uses deprecated field 'device'; "
            "configure a 'devices' list instead"
        )
    raw_devices = phase_cfg.get("devices")
    if not isinstance(raw_devices, (list, tuple)):
        raise ValueError(
            f"Baseline {baseline_name!r} phase {phase_name!r} must set 'devices' to a non-empty list of strings"
        )
    devices = [str(device).strip() for device in raw_devices if str(device).strip()]
    if not devices:
        raise ValueError(f"Baseline {baseline_name!r} phase {phase_name!r} must configure at least one device")
    normalized = deepcopy(phase_cfg)
    normalized["devices"] = devices
    return normalized


def _normalize_baseline(name: str, raw: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    cfg = _merge_dict(DEFAULT_BASELINE_CONFIG, raw)
    if not cfg["repo_path"]:
        raise ValueError(f"Missing repo_path for baseline {name!r}")
    if not cfg["conda_prefix"]:
        raise ValueError(f"Missing conda_prefix for baseline {name!r}")

    cfg["id"] = name
    cfg["repo_path"] = str(resolve_path(cfg["repo_path"], workspace_root))
    cfg["conda_prefix"] = str(resolve_path(cfg["conda_prefix"], workspace_root))
    cfg["assets"] = {
        key: str(resolve_path(value, workspace_root)) if isinstance(value, str) and value else value
        for key, value in cfg.get("assets", {}).items()
    }
    cfg["train"] = _normalize_phase_config(
        _merge_dict(DEFAULT_TRAIN_CONFIG, cfg.get("train", {})),
        baseline_name=name,
        phase_name="train",
    )
    cfg["eval"] = _normalize_phase_config(
        _merge_dict(DEFAULT_EVAL_CONFIG, cfg.get("eval", {})),
        baseline_name=name,
        phase_name="eval",
    )
    return cfg


def load_eval_config(
    config_path: Path,
    workspace_root: Path,
    tasks_override: list[str] | None = None,
    baselines_override: list[str] | None = None,
) -> dict[str, Any]:
    raw = load_json(config_path)
    tasks = tasks_override or raw.get("tasks", list(TASK_IDS))
    baselines_requested = baselines_override or list(raw.get("baselines", {}).keys())

    for task_id in tasks:
        if task_id not in TASK_IDS:
            raise ValueError(f"Unknown task id: {task_id}")

    baselines_raw = raw.get("baselines")
    if not isinstance(baselines_raw, dict) or not baselines_raw:
        raise ValueError("Config must define a non-empty 'baselines' mapping")

    baselines: dict[str, dict[str, Any]] = {}
    for baseline_name in baselines_requested:
        if baseline_name not in BASELINE_IDS:
            raise ValueError(f"Unknown baseline id: {baseline_name}")
        if baseline_name not in baselines_raw:
            raise ValueError(f"Baseline {baseline_name!r} not present in config")
        baselines[baseline_name] = _normalize_baseline(baseline_name, baselines_raw[baseline_name], workspace_root)

    output_root = resolve_path(raw.get("output_root", "outputs/eval"), workspace_root)
    dataset_root = resolve_path(raw.get("dataset_root", "data/ChainBench"), workspace_root)
    metadata_path = resolve_path(raw["metadata_path"], workspace_root)
    sample_ratio = raw.get("sample_ratio")
    if sample_ratio is not None:
        sample_ratio = float(sample_ratio)
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError("Config field 'sample_ratio' must be in the interval (0, 1]")

    return {
        "config_path": str(config_path),
        "metadata_path": str(metadata_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "sample_ratio": sample_ratio,
        "tasks": tasks,
        "baselines": baselines,
    }
