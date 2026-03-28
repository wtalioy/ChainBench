"""Public task-pack construction entry point."""

from __future__ import annotations

from typing import Any

from ..shared import normalize_task_sample_ratio_mapping
from .common import ensure_annotations, sample_pack, truncate_pack
from .delivery import build_delivery_robustness_pack
from .detection import build_in_chain_detection_packs
from .interventions import (
    build_operator_substitution_packs,
    build_order_swap_pack,
    build_parameter_perturbation_packs,
)
from .models import (
    DELIVERY_ROBUSTNESS_TASK,
    IN_CHAIN_DETECTION_TASK,
    OPERATOR_SUBSTITUTION_TASK,
    ORDER_SWAP_TASK,
    PARAMETER_PERTURBATION_TASK,
    TaskPack,
)

TASK_BUILDERS = {
    IN_CHAIN_DETECTION_TASK: build_in_chain_detection_packs,
    OPERATOR_SUBSTITUTION_TASK: build_operator_substitution_packs,
    PARAMETER_PERTURBATION_TASK: build_parameter_perturbation_packs,
    ORDER_SWAP_TASK: build_order_swap_pack,
    DELIVERY_ROBUSTNESS_TASK: build_delivery_robustness_pack,
}


def build_task_packs(
    rows: list[dict[str, Any]],
    task_ids: list[str],
    config: dict[str, Any] | None = None,
) -> list[TaskPack]:
    config = config or {}
    sample_ratios = normalize_task_sample_ratio_mapping(task_ids, config.get("sample_ratio"))
    smoke_limits = config.get("smoke_limits")
    generalization = config.get("generalization")
    annotated_rows = ensure_annotations(rows)
    packs: list[TaskPack] = []

    for task_id in task_ids:
        builder = TASK_BUILDERS[task_id]
        if task_id == IN_CHAIN_DETECTION_TASK:
            task_packs = builder(annotated_rows, generalization=generalization)
        else:
            task_packs = builder(annotated_rows)
        sample_ratio = sample_ratios.get(task_id)
        if sample_ratio is not None:
            task_packs = [sample_pack(pack, sample_ratio) for pack in task_packs]
        packs.extend(task_packs)

    if smoke_limits:
        max_train, max_dev, max_test = smoke_limits
        packs = [truncate_pack(pack, max_train, max_dev, max_test) for pack in packs]

    return packs
