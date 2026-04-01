"""Pair-based robustness metrics for intervention tasks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from chainbench.lib.structural_metadata import ORDER_SWAP_GROUP_FIELD

from ..rows import binary_label_as_int, group_rows_by_field
from .core import parse_score


def decision_threshold_for_pairs(
    score_pairs: list[tuple[float, int]],
    calibrated_threshold: float,
) -> float:
    if not score_pairs:
        return 0.5
    labels = [label for _, label in score_pairs]
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return calibrated_threshold


def zero_one_misclassification_risk(score: float, label: int, threshold: float) -> float:
    pred = 1 if score >= threshold else 0
    return 1.0 if pred != label else 0.0


def paired_prediction(value: float, threshold: float) -> int:
    return 1 if value >= threshold else 0


def summarize_paired_rows(rows_subset: list[dict[str, Any]]) -> dict[str, Any]:
    n_pairs = len(rows_subset)
    if not n_pairs:
        return {"n_pairs": 0, "pair_consistency_rate": 0.0, "pair_joint_accuracy": 0.0}
    consistency = sum(1 for row in rows_subset if row["consistent"]) / n_pairs
    pair_joint_accuracy = sum(1 for row in rows_subset if row["both_correct"]) / n_pairs
    return {
        "n_pairs": n_pairs,
        "pair_consistency_rate": consistency,
        "pair_joint_accuracy": pair_joint_accuracy,
    }


def intervention_metrics_keys(
    *,
    prefix: str,
    breakdown_key: str,
    order_swap_style: bool,
) -> dict[str, str]:
    if order_swap_style:
        return {
            "pair_count": "order_swap_pair_count",
            "pair_consistency_rate": "pair_consistency_rate",
            "pair_joint_accuracy": "pair_joint_accuracy",
            "pair_symmetric_misclassification_risk": "pair_symmetric_misclassification_risk",
            "mean_normalized_score_drift": "mean_normalized_score_drift",
            "by_label": "order_swap_robustness_by_label",
        }
    return {
        "pair_count": f"{prefix}_pair_count",
        "pair_consistency_rate": f"{prefix}_pair_consistency_rate",
        "pair_joint_accuracy": f"{prefix}_pair_joint_accuracy",
        "pair_symmetric_misclassification_risk": f"{prefix}_pair_symmetric_misclassification_risk",
        "mean_normalized_score_drift": f"{prefix}_mean_normalized_score_drift",
        "by_breakdown": f"{prefix}_pair_robustness_{breakdown_key}",
        "by_label": f"{prefix}_pair_robustness_by_label",
    }


def compute_intervention_pair_robustness_metrics(
    scores: list[dict[str, Any]],
    *,
    group_field: str,
    breakdown_field: str = "",
    breakdown_key: str = "by_axis",
    additional_breakdowns: list[tuple[str, str]] | None = None,
    threshold: float,
    prefix: str,
    order_swap_style: bool = False,
) -> dict[str, Any]:
    keys = intervention_metrics_keys(prefix=prefix, breakdown_key=breakdown_key, order_swap_style=order_swap_style)
    use_breakdown = bool(breakdown_field.strip()) and not order_swap_style
    extra_specs = list(additional_breakdowns or [])
    per_extra_breakdown: dict[str, defaultdict[str, list[dict[str, Any]]]] = {
        suffix: defaultdict(list) for _, suffix in extra_specs
    }

    grouped_rows = group_rows_by_field(scores, group_field)
    all_raw_scores = [parse_score(row.get("score")) for row in scores]
    score_iqr = float(np.percentile(all_raw_scores, 75) - np.percentile(all_raw_scores, 25)) if len(all_raw_scores) >= 2 else 0.0

    pair_count = 0
    consistent_count = 0
    both_correct_count = 0
    score_drifts: list[float] = []
    symmetric_misclass: list[float] = []
    per_breakdown_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    per_label_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for group_rows in grouped_rows.values():
        if len(group_rows) != 2:
            continue
        left, right = sorted(group_rows, key=lambda row: str(row.get("sample_id", "")))
        left_label = binary_label_as_int(left.get("label"))
        right_label = binary_label_as_int(right.get("label"))
        if left_label is None or right_label is None or left_label != right_label:
            continue

        pair_count += 1
        left_score = parse_score(left.get("score"))
        right_score = parse_score(right.get("score"))
        left_pred = paired_prediction(left_score, threshold)
        right_pred = paired_prediction(right_score, threshold)
        consistent = left_pred == right_pred
        both_correct = left_pred == left_label and right_pred == right_label
        if consistent:
            consistent_count += 1
        if both_correct:
            both_correct_count += 1
        drift = abs(left_score - right_score)
        score_drifts.append(drift)
        z_left = zero_one_misclassification_risk(left_score, left_label, threshold)
        z_right = zero_one_misclassification_risk(right_score, right_label, threshold)
        symmetric_misclass.append(0.5 * (z_left + z_right))

        label_key = "bonafide" if left_label == 1 else "spoof"
        row_summary = {
            "consistent": consistent,
            "both_correct": both_correct,
        }
        if use_breakdown:
            bucket = str(left.get(breakdown_field, "") or right.get(breakdown_field, "")).strip() or "unknown"
            per_breakdown_rows[bucket].append(row_summary)
        if extra_specs and not order_swap_style:
            for field, suffix in extra_specs:
                extra_bucket = str(left.get(field, "") or right.get(field, "")).strip() or "unknown"
                per_extra_breakdown[suffix][extra_bucket].append(row_summary)
        per_label_rows[label_key].append(row_summary)

    consistency_rate = (consistent_count / pair_count) if pair_count else 0.0
    joint_accuracy = (both_correct_count / pair_count) if pair_count else 0.0
    mean_score_drift = (sum(score_drifts) / len(score_drifts)) if score_drifts else 0.0
    denom = score_iqr if score_iqr > 1e-12 else 1.0
    mean_normalized_drift = mean_score_drift / denom

    out: dict[str, Any] = {
        keys["pair_count"]: pair_count,
        keys["pair_consistency_rate"]: consistency_rate,
        keys["pair_joint_accuracy"]: joint_accuracy,
        keys["pair_symmetric_misclassification_risk"]: (
            (sum(symmetric_misclass) / len(symmetric_misclass)) if symmetric_misclass else 0.0
        ),
        keys["mean_normalized_score_drift"]: mean_normalized_drift,
        keys["by_label"]: {
            label: summarize_paired_rows(rows_subset) for label, rows_subset in sorted(per_label_rows.items())
        },
    }
    if use_breakdown:
        out[keys["by_breakdown"]] = {
            key: summarize_paired_rows(rows_subset) for key, rows_subset in sorted(per_breakdown_rows.items())
        }
    if extra_specs and not order_swap_style:
        for _field, suffix in extra_specs:
            out[f"{prefix}_pair_robustness_{suffix}"] = {
                key: summarize_paired_rows(rows_subset) for key, rows_subset in sorted(per_extra_breakdown[suffix].items())
            }
    return out


def compute_order_swap_metrics(scores: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    return compute_intervention_pair_robustness_metrics(
        scores,
        group_field=ORDER_SWAP_GROUP_FIELD,
        threshold=threshold,
        prefix="order_swap",
        order_swap_style=True,
    )
