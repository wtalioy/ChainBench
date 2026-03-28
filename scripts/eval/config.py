"""Shared configuration loader for ChainBench baseline evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from lib.config import load_json, resolve_path
from .generalization import normalize_generalization_config
from .shared import SPLIT_NAMES, normalize_sample_ratio_value, normalize_split_sample_ratio
from .tasks import TASK_IDS

BASELINE_IDS = ("aasist", "aasist-l", "sls_df", "safeear", "nes2net")
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

def _normalize_sample_ratio_spec(
    raw_value: Any,
    *,
    context: str,
) -> float | dict[str, float]:
    if isinstance(raw_value, (int, float)):
        return normalize_sample_ratio_value(raw_value, context=context)
    if isinstance(raw_value, (list, tuple)):
        return normalize_split_sample_ratio(raw_value, context=context)
    if isinstance(raw_value, dict):
        return normalize_split_sample_ratio(raw_value, context=context)
    raise ValueError(f"{context} must be either a float or a train/dev/test ratio triple")


def _normalize_sample_ratio_config(
    raw_sample_ratio: Any,
    *,
    raw_tasks: list[str],
    selected_tasks: list[str],
) -> float | dict[str, float | dict[str, float]] | None:
    if raw_sample_ratio is None:
        return None
    if isinstance(raw_sample_ratio, (int, float)):
        return normalize_sample_ratio_value(raw_sample_ratio, context="Config field 'sample_ratio'")
    if isinstance(raw_sample_ratio, list):
        if len(raw_sample_ratio) != len(raw_tasks):
            raise ValueError("Config field 'sample_ratio' list must match the configured tasks order and length")
        ratio_by_task: dict[str, float | dict[str, float]] = {
            task_id: _normalize_sample_ratio_spec(value, context=f"Config field 'sample_ratio' for task {task_id!r}")
            for task_id, value in zip(raw_tasks, raw_sample_ratio)
        }
        missing_tasks = [task_id for task_id in selected_tasks if task_id not in ratio_by_task]
        if missing_tasks:
            raise ValueError(
                "Config field 'sample_ratio' list is aligned to configured tasks and cannot serve selected tasks: "
                + ", ".join(missing_tasks)
            )
        return {task_id: ratio_by_task[task_id] for task_id in selected_tasks}
    if isinstance(raw_sample_ratio, dict):
        unknown_tasks = [task_id for task_id in raw_sample_ratio if task_id not in raw_tasks]
        if unknown_tasks:
            raise ValueError("Config field 'sample_ratio' contains unknown task ids: " + ", ".join(sorted(unknown_tasks)))
        missing_tasks = [task_id for task_id in selected_tasks if task_id not in raw_sample_ratio]
        if missing_tasks:
            raise ValueError("Config field 'sample_ratio' is missing selected tasks: " + ", ".join(missing_tasks))
        return {
            task_id: _normalize_sample_ratio_spec(
                raw_sample_ratio[task_id],
                context=f"Config field 'sample_ratio' for task {task_id!r}",
            )
            for task_id in selected_tasks
        }
    raise ValueError(
        "Config field 'sample_ratio' must be a float, a per-task list, or a per-task mapping"
    )


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
    # Some baseline wrappers still read the legacy singular field internally.
    normalized["device"] = devices[0]
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
    generalization_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw = load_json(config_path)
    raw_tasks = raw.get("tasks", list(TASK_IDS))
    tasks = tasks_override or raw_tasks
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
    sample_ratio = _normalize_sample_ratio_config(
        raw.get("sample_ratio"),
        raw_tasks=raw_tasks,
        selected_tasks=tasks,
    )
    raw_generalization = raw.get("generalization")
    if generalization_override is not None:
        merged_generalization = dict(raw_generalization) if isinstance(raw_generalization, dict) else {}
        merged_generalization.update(
            {key: value for key, value in generalization_override.items() if str(value).strip()}
        )
        raw_generalization = merged_generalization
    generalization = normalize_generalization_config(
        raw_generalization,
        selected_tasks=tasks,
    )

    return {
        "config_path": str(config_path),
        "metadata_path": str(metadata_path),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "sample_ratio": sample_ratio,
        "generalization": generalization,
        "tasks": tasks,
        "baselines": baselines,
    }
