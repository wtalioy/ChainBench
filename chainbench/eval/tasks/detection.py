"""Builders for in-chain detection task packs."""

from __future__ import annotations

from typing import Any

from ..holdout_protocols import (
    LEAVE_ONE_TEMPLATE_OUT_PROTOCOL,
    build_template_holdout_detection_folds,
)
from ..rows import bucket_rows, rows_for_split
from .models import CHAIN_FAMILIES, IN_CHAIN_DETECTION_TASK, TaskPack
from .packing import chain_family, supported_chain_rows


def build_in_chain_detection_packs(
    rows: list[dict[str, Any]],
    generalization: dict[str, Any] | None = None,
) -> list[TaskPack]:
    if generalization and str(generalization.get("protocol", "")).strip() == LEAVE_ONE_TEMPLATE_OUT_PROTOCOL:
        return build_template_holdout_in_chain_detection_packs(rows, generalization=generalization)

    usable_rows = supported_chain_rows(rows)
    packs: list[TaskPack] = []
    by_family = bucket_rows(usable_rows, chain_family)

    for family in CHAIN_FAMILIES:
        family_rows = by_family.get(family, [])
        train_rows = rows_for_split(family_rows, "train")
        dev_rows = rows_for_split(family_rows, "dev")
        test_rows = rows_for_split(family_rows, "test")
        if not train_rows or not test_rows:
            continue
        packs.append(
            TaskPack(
                task_id=IN_CHAIN_DETECTION_TASK,
                variant=family,
                description=f"In-chain detection on {family}",
                train_rows=train_rows,
                dev_rows=dev_rows,
                test_rows=test_rows,
                meta={
                    "metric_profile": "binary",
                    "reference_group": "same_chain_family",
                    "chain_family": family,
                    "family_balanced_macro_group": "in_chain_detection",
                },
            )
        )
    return packs


def build_template_holdout_in_chain_detection_packs(
    rows: list[dict[str, Any]],
    *,
    generalization: dict[str, Any],
) -> list[TaskPack]:
    packs: list[TaskPack] = []
    for fold in build_template_holdout_detection_folds(supported_chain_rows(rows)):
        packs.append(
            TaskPack(
                task_id=IN_CHAIN_DETECTION_TASK,
                variant=fold.variant,
                description=(
                    "Leave-one-template-out in-chain detection on "
                    f"{fold.chain_family}, evaluated on held-out template {fold.template_id}"
                ),
                train_rows=fold.train_rows,
                dev_rows=fold.dev_rows,
                test_rows=fold.test_rows,
                meta={
                    "metric_profile": "binary",
                    "reference_group": "held_out_template",
                    "chain_family": fold.chain_family,
                    "held_out_chain_family": fold.chain_family,
                    "held_out_template_id": fold.template_id,
                    "generalization_protocol": str(generalization.get("protocol", "")).strip(),
                    "generalization_scope": str(generalization.get("scope", "")).strip(),
                    "fold_train_samples": len(fold.train_rows),
                    "fold_dev_samples": len(fold.dev_rows),
                    "fold_test_samples": len(fold.test_rows),
                    "fold_dropped_rows": fold.dropped_rows,
                    "family_balanced_macro_group": "in_chain_detection_template_holdout",
                },
            )
        )
    return packs
