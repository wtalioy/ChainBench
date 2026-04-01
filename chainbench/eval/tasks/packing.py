"""Shared packing, annotation, and sampling helpers for task packs."""

from __future__ import annotations

from typing import Any

from chainbench.lib.chain_keys import operator_signature_sequence
from chainbench.lib.structural_metadata import (
    OPERATOR_MULTISET_FIELD,
    OPERATOR_SUBSTITUTION_GROUP_FIELD,
    ORDER_SWAP_GROUP_FIELD,
    PARAMETER_PERTURBATION_GROUP_FIELD,
    PATH_GROUP_FIELD,
    annotate_structural_group_fields,
)

from ..rows import (
    paired_test_id,
    rows_for_split,
    sample_rows,
    sample_rows_by_group,
    sample_units_within_primary_groups,
    stable_row_token,
)
from ..sample_ratio import normalize_split_sample_ratio
from .models import CHAIN_FAMILIES, IN_CHAIN_DETECTION_TASK, TaskPack


def chain_family(row: dict[str, Any]) -> str:
    return str(row.get("chain_family", "")).strip()


def supported_chain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if chain_family(row) in CHAIN_FAMILIES]


def ensure_annotations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    needs_annotations = any(
        OPERATOR_MULTISET_FIELD not in row
        or ORDER_SWAP_GROUP_FIELD not in row
        or PATH_GROUP_FIELD not in row
        or PARAMETER_PERTURBATION_GROUP_FIELD not in row
        or OPERATOR_SUBSTITUTION_GROUP_FIELD not in row
        for row in rows
    )
    return annotate_structural_group_fields(rows, copy_rows=False) if needs_annotations else rows


def intervention_train_dev_raw_test(
    rows: list[dict[str, Any]],
    group_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    usable_rows = supported_chain_rows(rows)
    train_rows = rows_for_split(usable_rows, "train")
    dev_rows = rows_for_split(usable_rows, "dev")
    raw_test_rows = [row for row in rows_for_split(usable_rows, "test") if str(row.get(group_field, "")).strip()]
    if not train_rows or not raw_test_rows:
        return None
    return train_rows, dev_rows, raw_test_rows


def intervention_train_dev_all_test(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    usable_rows = supported_chain_rows(rows)
    train_rows = rows_for_split(usable_rows, "train")
    dev_rows = rows_for_split(usable_rows, "dev")
    raw_test_rows = rows_for_split(usable_rows, "test")
    if not train_rows or not raw_test_rows:
        return None
    return train_rows, dev_rows, raw_test_rows


def order_swap_distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_sig = operator_signature_sequence(left)
    right_sig = operator_signature_sequence(right)
    if len(left_sig) != len(right_sig):
        return -1
    diff_indices = [index for index, (lhs, rhs) in enumerate(zip(left_sig, right_sig)) if lhs != rhs]
    if len(diff_indices) != 2:
        return -1
    return abs(diff_indices[1] - diff_indices[0])


def truncate_pack(
    pack: TaskPack,
    max_train: int,
    max_dev: int,
    max_test: int,
) -> TaskPack:
    return TaskPack(
        task_id=pack.task_id,
        variant=pack.variant,
        description=pack.description,
        train_rows=pack.train_rows[:max_train],
        dev_rows=pack.dev_rows[:max_dev],
        test_rows=pack.test_rows[:max_test],
        meta={**pack.meta, "smoke_truncated": True},
    )


def sample_pack(pack: TaskPack, sample_ratio: float | dict[str, float]) -> TaskPack:
    split_ratios = normalize_split_sample_ratio(sample_ratio, context=f"sample_ratio for task {pack.task_id!r}")
    shared_training_group = str(pack.meta.get("shared_training_group", "")).strip()
    train_dev_scope = shared_training_group or f"{pack.task_id}:{pack.variant}"
    train_rows = sample_rows(pack.train_rows, split_ratios["train"], salt=f"{train_dev_scope}:train")
    dev_rows = sample_rows(pack.dev_rows, split_ratios["dev"], salt=f"{train_dev_scope}:dev")

    if shared_training_group and any(paired_test_id(row) for row in pack.test_rows):
        test_rows = sample_units_within_primary_groups(
            pack.test_rows,
            split_ratios["test"],
            salt=f"{shared_training_group}:paired_test",
            primary_key_fn=chain_family,
            unit_key_fn=lambda row: paired_test_id(row) or stable_row_token(row),
        )
    elif pack.task_id == IN_CHAIN_DETECTION_TASK:
        test_rows = sample_rows_by_group(
            pack.test_rows,
            split_ratios["test"],
            salt=f"{pack.task_id}:{pack.variant}:test",
            group_key_fn=chain_family,
        )
    else:
        test_rows = sample_rows(
            pack.test_rows,
            split_ratios["test"],
            salt=f"{pack.task_id}:{pack.variant}:test",
        )
    return TaskPack(
        task_id=pack.task_id,
        variant=pack.variant,
        description=pack.description,
        train_rows=train_rows,
        dev_rows=dev_rows,
        test_rows=test_rows,
        meta={**pack.meta, "sample_ratio": sample_ratio},
    )
