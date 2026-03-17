"""Task derivation for supported evaluation tasks from metadata."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

CHAIN_FAMILIES = ("direct", "platform_like", "telephony", "simreplay", "hybrid")
IN_CHAIN_TASK = "in_chain"
CROSS_CHAIN_TASK = "cross_chain"
UNSEEN_COMPOSITION_TASK = "unseen_composition"
UNSEEN_ORDER_TASK = "unseen_order"
COUNTERFACTUAL_TASK = "counterfactual_consistency"
TASK_IDS = (
    IN_CHAIN_TASK,
    CROSS_CHAIN_TASK,
    UNSEEN_COMPOSITION_TASK,
    UNSEEN_ORDER_TASK,
    COUNTERFACTUAL_TASK,
)


@dataclass
class TaskPack:
    """One evaluation task variant with train/dev/test rows."""

    task_id: str
    variant: str
    description: str
    train_rows: list[dict[str, Any]] = field(default_factory=list)
    dev_rows: list[dict[str, Any]] = field(default_factory=list)
    test_rows: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _stable_row_token(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    return json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _target_sample_size(size: int, ratio: float) -> int:
    if size <= 0 or ratio >= 1.0:
        return size
    return min(size, max(1, int(round(size * ratio))))


def _sample_rows(rows: list[dict[str, Any]], ratio: float, *, salt: str) -> list[dict[str, Any]]:
    if not rows or ratio >= 1.0:
        return rows
    target_size = _target_sample_size(len(rows), ratio)
    ranked = sorted(
        (
            hashlib.sha1(f"{salt}\0{_stable_row_token(row)}".encode("utf-8")).hexdigest(),
            index,
        )
        for index, row in enumerate(rows)
    )
    keep_indices = {index for _, index in ranked[:target_size]}
    return [row for index, row in enumerate(rows) if index in keep_indices]


def _sample_grouped_rows(
    rows: list[dict[str, Any]],
    ratio: float,
    *,
    salt: str,
    group_key_fn,
) -> list[dict[str, Any]]:
    if not rows or ratio >= 1.0:
        return rows
    group_keys = list(dict.fromkeys(group_key_fn(row) for row in rows))
    target_size = _target_sample_size(len(group_keys), ratio)
    ranked = sorted(
        (
            hashlib.sha1(f"{salt}\0{group_key}".encode("utf-8")).hexdigest(),
            group_key,
        )
        for group_key in group_keys
    )
    keep_group_keys = {group_key for _, group_key in ranked[:target_size]}
    return [row for row in rows if group_key_fn(row) in keep_group_keys]


def _counterfactual_meta(test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_rows:
        parent_id = str(row.get("parent_id", "")).strip()
        if parent_id:
            by_parent[parent_id].append(row)

    family_coverage_histogram: Counter[int] = Counter()
    paired_parent_count = 0
    for parent_rows in by_parent.values():
        families = {_chain_family(row) for row in parent_rows}
        if "direct" not in families or len(families) < 2:
            continue
        paired_parent_count += 1
        family_coverage_histogram[len(families)] += 1

    return {
        "paired_test_parent_count": paired_parent_count,
        "paired_test_family_coverage": {
            str(k): v for k, v in sorted(family_coverage_histogram.items())
        },
    }


def _composition_key(operator_seq_value: Any) -> str:
    try:
        seq = json.loads(operator_seq_value) if isinstance(operator_seq_value, str) else operator_seq_value
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(seq, list):
        return ""
    return json.dumps(sorted(str(x) for x in seq), ensure_ascii=False, separators=(",", ":"))


def _order_key(operator_seq_value: Any) -> str:
    try:
        seq = json.loads(operator_seq_value) if isinstance(operator_seq_value, str) else operator_seq_value
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(seq, list):
        return ""
    return json.dumps([str(x) for x in seq], ensure_ascii=False, separators=(",", ":"))


def _rows_for_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("split")) == split]


def _chain_family(row: dict[str, Any]) -> str:
    return str(row.get("chain_family", "")).strip()


def _supported_chain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _chain_family(row) in CHAIN_FAMILIES]


def _build_unseen_split_pack(
    rows: list[dict[str, Any]],
    *,
    task_id: str,
    variant: str,
    description: str,
    key_fn,
    meta_key: str,
) -> list[TaskPack]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = key_fn(row.get("operator_seq", "[]"))
        if key and key != "[]":
            by_key[key].append(row)

    seen_keys: set[str] = set()
    unseen_keys: set[str] = set()
    for key, grouped_rows in by_key.items():
        train_count = len(_rows_for_split(grouped_rows, "train"))
        test_count = len(_rows_for_split(grouped_rows, "test"))
        if test_count > 0 and train_count < 10:
            unseen_keys.add(key)
        elif train_count > 0:
            seen_keys.add(key)

    if not unseen_keys:
        return []

    train_rows = [row for row in _rows_for_split(rows, "train") if key_fn(row.get("operator_seq", "[]")) in seen_keys]
    dev_rows = [row for row in _rows_for_split(rows, "dev") if key_fn(row.get("operator_seq", "[]")) in seen_keys]
    test_rows = [row for row in _rows_for_split(rows, "test") if key_fn(row.get("operator_seq", "[]")) in unseen_keys]
    if not train_rows or not test_rows:
        return []

    return [
        TaskPack(
            task_id=task_id,
            variant=variant,
            description=description,
            train_rows=train_rows,
            dev_rows=dev_rows,
            test_rows=test_rows,
            meta={meta_key: list(unseen_keys)[:10]},
        )
    ]


def build_in_chain_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """One pack per chain family with train/dev/test from the same family."""
    packs: list[TaskPack] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("chain_family", ""))].append(row)

    for family in CHAIN_FAMILIES:
        fam_rows = by_family.get(family)
        if not fam_rows:
            continue
        train = _rows_for_split(fam_rows, "train")
        dev = _rows_for_split(fam_rows, "dev")
        test = _rows_for_split(fam_rows, "test")
        if not test and not dev:
            continue
        packs.append(
            TaskPack(
                task_id=IN_CHAIN_TASK,
                variant=family,
                description=f"In-chain detection: {family}",
                train_rows=train,
                dev_rows=dev,
                test_rows=test,
                meta={"chain_family": family},
            )
        )
    return packs


def build_cross_chain_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """Train on one chain family and test on another."""
    packs: list[TaskPack] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("chain_family", ""))].append(row)

    families = [f for f in CHAIN_FAMILIES if by_family.get(f)]
    for train_family in families:
        for test_family in families:
            if train_family == test_family:
                continue
            train_rows = _rows_for_split(by_family[train_family], "train")
            dev_rows = _rows_for_split(by_family[train_family], "dev")
            test_rows = _rows_for_split(by_family[test_family], "test")
            if not train_rows or not test_rows:
                continue
            variant = f"{train_family}_to_{test_family}"
            packs.append(
                TaskPack(
                    task_id=CROSS_CHAIN_TASK,
                    variant=variant,
                    description=f"Cross-chain: train {train_family} → test {test_family}",
                    train_rows=train_rows,
                    dev_rows=dev_rows,
                    test_rows=test_rows,
                    meta={"train_chain_family": train_family, "test_chain_family": test_family},
                )
            )
    return packs


def build_unseen_composition_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """Hold out some operator compositions for test."""
    return _build_unseen_split_pack(
        rows,
        task_id=UNSEEN_COMPOSITION_TASK,
        variant=UNSEEN_COMPOSITION_TASK,
        description="Unseen composition generalization",
        key_fn=_composition_key,
        meta_key="unseen_composition_keys",
    )


def build_unseen_order_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """Hold out some operator orders for test."""
    return _build_unseen_split_pack(
        rows,
        task_id=UNSEEN_ORDER_TASK,
        variant=UNSEEN_ORDER_TASK,
        description="Unseen order generalization",
        key_fn=_order_key,
        meta_key="unseen_order_keys",
    )


def build_counterfactual_packs(rows: list[dict[str, Any]]) -> list[TaskPack]:
    """Build one pack for matched-parent counterfactual evaluation."""
    usable_rows = _supported_chain_rows(rows)
    train_rows = _rows_for_split(usable_rows, "train")
    dev_rows = _rows_for_split(usable_rows, "dev")
    test_candidates = _rows_for_split(usable_rows, "test")

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in test_candidates:
        parent_id = str(row.get("parent_id", "")).strip()
        if parent_id:
            by_parent[parent_id].append(row)

    test_rows: list[dict[str, Any]] = []
    family_coverage_histogram: Counter[int] = Counter()
    for parent_rows in by_parent.values():
        families = {_chain_family(row) for row in parent_rows}
        if "direct" not in families or len(families) < 2:
            continue
        family_coverage_histogram[len(families)] += 1
        test_rows.extend(sorted(parent_rows, key=lambda row: (_chain_family(row), str(row.get("sample_id", "")))))

    if not train_rows or not test_rows:
        return []

    return [
        TaskPack(
            task_id=COUNTERFACTUAL_TASK,
            variant="matched_parents",
            description="Counterfactual consistency across matched chain families",
            train_rows=train_rows,
            dev_rows=dev_rows,
            test_rows=test_rows,
            meta={
                "reference_chain_family": "direct",
                "chain_families": list(CHAIN_FAMILIES),
                "paired_test_parent_count": sum(family_coverage_histogram.values()),
                "paired_test_family_coverage": {
                    str(k): v for k, v in sorted(family_coverage_histogram.items())
                },
            },
        )
    ]


TASK_BUILDERS = {
    IN_CHAIN_TASK: build_in_chain_packs,
    CROSS_CHAIN_TASK: build_cross_chain_packs,
    UNSEEN_COMPOSITION_TASK: build_unseen_composition_packs,
    UNSEEN_ORDER_TASK: build_unseen_order_packs,
    COUNTERFACTUAL_TASK: build_counterfactual_packs,
}


def _truncate_pack(
    pack: TaskPack,
    max_train: int,
    max_dev: int,
    max_test: int,
) -> TaskPack:
    """Truncate pack splits for smoke testing."""
    return TaskPack(
        task_id=pack.task_id,
        variant=pack.variant,
        description=pack.description,
        train_rows=pack.train_rows[:max_train],
        dev_rows=pack.dev_rows[:max_dev],
        test_rows=pack.test_rows[:max_test],
        meta={**pack.meta, "smoke_truncated": True},
    )


def _sample_pack(pack: TaskPack, sample_ratio: float) -> TaskPack:
    train_rows = _sample_rows(
        pack.train_rows,
        sample_ratio,
        salt=f"{pack.task_id}:{pack.variant}:train",
    )
    dev_rows = _sample_rows(
        pack.dev_rows,
        sample_ratio,
        salt=f"{pack.task_id}:{pack.variant}:dev",
    )
    if pack.task_id == COUNTERFACTUAL_TASK:
        test_rows = _sample_grouped_rows(
            pack.test_rows,
            sample_ratio,
            salt=f"{pack.task_id}:{pack.variant}:test",
            group_key_fn=lambda row: str(row.get("parent_id", "")).strip() or _stable_row_token(row),
        )
        meta = {**pack.meta, **_counterfactual_meta(test_rows), "sample_ratio": sample_ratio}
    else:
        test_rows = _sample_rows(
            pack.test_rows,
            sample_ratio,
            salt=f"{pack.task_id}:{pack.variant}:test",
        )
        meta = {**pack.meta, "sample_ratio": sample_ratio}
    return TaskPack(
        task_id=pack.task_id,
        variant=pack.variant,
        description=pack.description,
        train_rows=train_rows,
        dev_rows=dev_rows,
        test_rows=test_rows,
        meta=meta,
    )


def build_task_packs(
    rows: list[dict[str, Any]],
    task_ids: list[str],
    config: dict[str, Any] | None = None,
) -> list[TaskPack]:
    """Build all requested task packs from metadata rows."""
    config = config or {}
    sample_ratio = config.get("sample_ratio")
    smoke_limits = config.get("smoke_limits")  # (max_train, max_dev, max_test) or None
    packs = [pack for task_id in task_ids for pack in TASK_BUILDERS[task_id](rows)]

    if sample_ratio is not None:
        sample_ratio = float(sample_ratio)
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in the interval (0, 1]")
        packs = [_sample_pack(pack, sample_ratio) for pack in packs]

    if smoke_limits:
        max_train, max_dev, max_test = smoke_limits
        packs = [_truncate_pack(p, max_train, max_dev, max_test) for p in packs]

    return packs
