"""Public per-run metric computation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..tasks.structural_groups import (
    OPERATOR_SUBSTITUTION_DETAIL_FIELD,
    OPERATOR_SUBSTITUTION_GROUP_FIELD,
    PARAMETER_PERTURBATION_AXIS_FIELD,
    PARAMETER_PERTURBATION_GROUP_FIELD,
)
from ..tasks.task_keys import parse_operator_seq

from ..tasks import TaskPack
from .binary import (
    compute_accuracy_from_binary_scores,
    compute_auc_from_binary_scores,
    compute_eer_from_binary_scores,
    compute_eer_from_labels,
    compute_f1_from_binary_scores,
)
from .core import (
    binary_label_counts,
    binary_scores,
    enrich_scores,
    load_scores_csv,
    validate_score_coverage,
)
from .delivery import compute_delivery_robustness_metrics
from .interventions import (
    compute_intervention_pair_robustness_metrics,
    compute_order_swap_metrics,
    decision_threshold_for_pairs,
)


def summarize_binary_subset(scores: list[dict[str, Any]]) -> dict[str, Any]:
    score_pairs = binary_scores(scores, "score", "label")
    return {
        "n_samples": len(scores),
        "label_counts": binary_label_counts(scores),
        "auc": compute_auc_from_binary_scores(score_pairs),
        "eer": compute_eer_from_binary_scores(score_pairs),
    }


def subgroup_metric_summaries(scores: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    groupers = {
        "chain_template_id": lambda row: str(row.get("chain_template_id", "")).strip() or "unknown",
        "operator_seq_length": lambda row: str(len(parse_operator_seq(row.get("operator_seq", "[]")))),
        "language": lambda row: str(row.get("language", "")).strip() or "unknown",
        "generator_family": lambda row: str(row.get("generator_family", "")).strip() or "unknown",
    }
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for group_name, group_fn in groupers.items():
        grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scores:
            grouped_rows[str(group_fn(row))].append(row)
        summaries[group_name] = {
            group_value: summarize_binary_subset(group_rows)
            for group_value, group_rows in sorted(grouped_rows.items())
        }
    return summaries


def compute_metrics_for_scores(
    scores_path: Path,
    label_map: dict[str, str] | None = None,
    task_pack: TaskPack | None = None,
) -> dict[str, Any]:
    scores = enrich_scores(
        load_scores_csv(scores_path),
        label_map=label_map,
        metadata_rows=task_pack.test_rows if task_pack is not None else None,
    )
    validate_score_coverage(scores, task_pack)
    score_pairs = binary_scores(scores, "score", "label")
    calibrated_threshold = 0.5
    if score_pairs:
        labels = np.array([label for _, label in score_pairs], dtype=np.int32)
        predictions = np.array([score for score, _ in score_pairs], dtype=np.float64)
        _, calibrated_threshold = compute_eer_from_labels(labels, predictions)
    metrics: dict[str, Any] = {
        "eer": compute_eer_from_binary_scores(score_pairs),
        "auc": compute_auc_from_binary_scores(score_pairs),
        "accuracy": compute_accuracy_from_binary_scores(score_pairs, calibrated_threshold),
        "f1": compute_f1_from_binary_scores(score_pairs, calibrated_threshold),
        "calibrated_threshold": calibrated_threshold,
        "n_samples": len(scores),
    }
    metric_profile = str(task_pack.meta.get("metric_profile", "binary")).strip() if task_pack is not None else "binary"
    pair_threshold = decision_threshold_for_pairs(score_pairs, calibrated_threshold)
    if metric_profile == "operator_substitution":
        metrics.update(
            compute_intervention_pair_robustness_metrics(
                scores,
                group_field=OPERATOR_SUBSTITUTION_GROUP_FIELD,
                breakdown_field=OPERATOR_SUBSTITUTION_DETAIL_FIELD,
                breakdown_key="by_substitution_pattern",
                threshold=pair_threshold,
                prefix="operator_substitution",
            )
        )
    elif metric_profile == "parameter_perturbation":
        metrics.update(
            compute_intervention_pair_robustness_metrics(
                scores,
                group_field=PARAMETER_PERTURBATION_GROUP_FIELD,
                breakdown_field=PARAMETER_PERTURBATION_AXIS_FIELD,
                breakdown_key="by_axis",
                threshold=pair_threshold,
                prefix="parameter_perturbation",
            )
        )
    elif metric_profile in ("order_swap", "minimal_order_swap"):
        metrics.update(compute_order_swap_metrics(scores, threshold=pair_threshold))
    elif metric_profile == "delivery_robustness":
        metrics.update(compute_delivery_robustness_metrics(scores, threshold=pair_threshold))
    if scores:
        metrics["subgroup_metrics"] = subgroup_metric_summaries(scores)
    return metrics
