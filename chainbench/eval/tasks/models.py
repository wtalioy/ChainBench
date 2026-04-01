"""Task models and public task identifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CHAIN_FAMILIES = ("direct", "platform_like", "telephony", "simreplay", "hybrid")

IN_CHAIN_DETECTION_TASK = "in_chain_detection"
OPERATOR_SUBSTITUTION_TASK = "operator_substitution"
PARAMETER_PERTURBATION_TASK = "parameter_perturbation"
ORDER_SWAP_TASK = "order_swap"
DELIVERY_ROBUSTNESS_TASK = "delivery_robustness"

INTERVENTION_ROBUSTNESS_TRAINING_GROUP = "intervention_robustness"

TASK_IDS = (
    IN_CHAIN_DETECTION_TASK,
    OPERATOR_SUBSTITUTION_TASK,
    PARAMETER_PERTURBATION_TASK,
    ORDER_SWAP_TASK,
    DELIVERY_ROBUSTNESS_TASK,
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
