"""Split and task sample-ratio normalization helpers for eval."""

from __future__ import annotations

from typing import Any

SPLIT_NAMES = ("train", "dev", "test")


def normalize_sample_ratio_value(value: Any, *, context: str) -> float:
    ratio = float(value)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"{context} must be in the interval (0, 1]")
    return ratio


def normalize_split_sample_ratio(
    value: Any,
    *,
    context: str,
) -> dict[str, float]:
    if isinstance(value, (int, float)):
        ratio = normalize_sample_ratio_value(value, context=context)
        return {split: ratio for split in SPLIT_NAMES}
    if isinstance(value, (list, tuple)):
        if len(value) != len(SPLIT_NAMES):
            raise ValueError(f"{context} must provide exactly three split ratios: train, dev, test")
        return {
            split: normalize_sample_ratio_value(split_value, context=f"{context} for split {split!r}")
            for split, split_value in zip(SPLIT_NAMES, value)
        }
    if isinstance(value, dict):
        unknown_splits = [split for split in value if split not in SPLIT_NAMES]
        if unknown_splits:
            raise ValueError(f"{context} contains unknown split keys: {', '.join(sorted(unknown_splits))}")
        missing_splits = [split for split in SPLIT_NAMES if split not in value]
        if missing_splits:
            raise ValueError(f"{context} must define ratios for train/dev/test; missing: {', '.join(missing_splits)}")
        return {
            split: normalize_sample_ratio_value(value[split], context=f"{context} for split {split!r}")
            for split in SPLIT_NAMES
        }
    raise ValueError(f"{context} must be either a float or a train/dev/test ratio triple")


def normalize_task_sample_ratio_mapping(
    task_ids: list[str],
    sample_ratio: Any,
    *,
    context: str = "sample_ratio",
) -> dict[str, float | dict[str, float]]:
    if sample_ratio is None:
        return {}
    if isinstance(sample_ratio, (int, float)):
        ratio = normalize_sample_ratio_value(sample_ratio, context=context)
        return {task_id: ratio for task_id in task_ids}
    if isinstance(sample_ratio, dict):
        unknown_tasks = [task_id for task_id in sample_ratio if task_id not in task_ids]
        if unknown_tasks:
            raise ValueError(f"{context} contains unknown task ids: {', '.join(sorted(unknown_tasks))}")
        return {
            str(task_id): normalize_split_sample_ratio(value, context=f"{context} for task {task_id!r}")
            for task_id, value in sample_ratio.items()
        }
    raise ValueError(f"{context} must be either a float or a per-task mapping")
