"""Builders for delivery robustness task packs."""

from __future__ import annotations

from chainbench.lib.structural_metadata import lineage_bucket_key

from ..rows import bucket_rows, normalize_binary_label
from .models import DELIVERY_ROBUSTNESS_TASK, TaskPack
from .packing import intervention_train_dev_all_test


def build_delivery_robustness_pack(rows: list[dict[str, object]]) -> list[TaskPack]:
    splits = intervention_train_dev_all_test(rows)
    if splits is None:
        return []
    train_rows, dev_rows, raw_test_rows = splits
    lineage_groups = bucket_rows(raw_test_rows, lambda row: lineage_bucket_key(row, normalize_binary_label))
    selected_test_rows: list[dict[str, object]] = []
    for lineage_key, lineage_rows in lineage_groups.items():
        if lineage_key and len(lineage_rows) >= 2:
            selected_test_rows.extend(lineage_rows)
    if not selected_test_rows:
        return []

    return [
        TaskPack(
            task_id=DELIVERY_ROBUSTNESS_TASK,
            variant="default",
            description=(
                "Robustness to realistic delivery-chain edits within each "
                "(parent_id, chain_family, label) lineage graph, measured by shortest-path radius "
                "over observed atomic edits, Robust@k, and AURC-chain."
            ),
            train_rows=train_rows,
            dev_rows=dev_rows,
            test_rows=selected_test_rows,
            meta={
                "metric_profile": "delivery_robustness",
                "shared_training_group": DELIVERY_ROBUSTNESS_TASK,
                "lineage_group": "(parent_id, chain_family, label)",
                "reference_chain": "shortest_operator_signature",
            },
        )
    ]
