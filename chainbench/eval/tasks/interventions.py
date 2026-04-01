"""Builders for paired intervention task packs."""

from __future__ import annotations

from chainbench.lib.structural_metadata import (
    OPERATOR_SUBSTITUTION_GROUP_FIELD,
    ORDER_SWAP_GROUP_FIELD,
    PARAMETER_PERTURBATION_GROUP_FIELD,
)
from ..rows import collect_paired_rows, group_rows_by_field, same_matched_binary_label
from .models import (
    INTERVENTION_ROBUSTNESS_TRAINING_GROUP,
    OPERATOR_SUBSTITUTION_TASK,
    ORDER_SWAP_TASK,
    PARAMETER_PERTURBATION_TASK,
    TaskPack,
)
from .packing import intervention_train_dev_raw_test, order_swap_distance


def build_label_matched_intervention_packs(
    rows: list[dict[str, object]],
    *,
    task_id: str,
    group_field: str,
    metric_profile: str,
    description: str,
) -> list[TaskPack]:
    splits = intervention_train_dev_raw_test(rows, group_field)
    if splits is None:
        return []
    train_rows, dev_rows, raw_test_rows = splits
    grouped_test_rows = group_rows_by_field(raw_test_rows, group_field)
    paired_test_rows = collect_paired_rows(
        grouped_test_rows,
        accept_pair=lambda _pid, left, right: same_matched_binary_label(left, right),
    )
    if not paired_test_rows:
        return []

    return [
        TaskPack(
            task_id=task_id,
            variant="default",
            description=description,
            train_rows=train_rows,
            dev_rows=dev_rows,
            test_rows=paired_test_rows,
            meta={
                "metric_profile": metric_profile,
                "shared_training_group": INTERVENTION_ROBUSTNESS_TRAINING_GROUP,
                "group_field": group_field,
                "pair_policy": "matched_label_pairs",
            },
        )
    ]


def build_operator_substitution_packs(rows: list[dict[str, object]]) -> list[TaskPack]:
    return build_label_matched_intervention_packs(
        rows,
        task_id=OPERATOR_SUBSTITUTION_TASK,
        group_field=OPERATOR_SUBSTITUTION_GROUP_FIELD,
        metric_profile="operator_substitution",
        description="Robustness to operator-name substitutions (multi-index allowed) under matched context",
    )


def build_parameter_perturbation_packs(rows: list[dict[str, object]]) -> list[TaskPack]:
    return build_label_matched_intervention_packs(
        rows,
        task_id=PARAMETER_PERTURBATION_TASK,
        group_field=PARAMETER_PERTURBATION_GROUP_FIELD,
        metric_profile="parameter_perturbation",
        description="Robustness to single-step operator parameter perturbations under matched context",
    )


def build_order_swap_pack(rows: list[dict[str, object]]) -> list[TaskPack]:
    splits = intervention_train_dev_raw_test(rows, ORDER_SWAP_GROUP_FIELD)
    if splits is None:
        return []
    train_rows, dev_rows, raw_test_rows = splits
    grouped_test_rows = group_rows_by_field(raw_test_rows, ORDER_SWAP_GROUP_FIELD)

    def accept_adjacent_swap(_pair_id: str, left: dict[str, object], right: dict[str, object]) -> bool:
        if not same_matched_binary_label(left, right):
            return False
        return order_swap_distance(left, right) == 1

    adjacent_pair_rows = collect_paired_rows(
        grouped_test_rows,
        accept_pair=accept_adjacent_swap,
        row_extras=lambda _pid, _l, _r: {"__order_swap_distance": "1"},
    )
    if not adjacent_pair_rows:
        return []
    return [
        TaskPack(
            task_id=ORDER_SWAP_TASK,
            variant="default",
            description="Robustness to adjacent minimal order swaps",
            train_rows=train_rows,
            dev_rows=dev_rows,
            test_rows=adjacent_pair_rows,
            meta={
                "metric_profile": "order_swap",
                "shared_training_group": ORDER_SWAP_TASK,
                "group_field": ORDER_SWAP_GROUP_FIELD,
                "pair_policy": "same_label_only",
                "order_swap_scope": "adjacent",
            },
        )
    ]
