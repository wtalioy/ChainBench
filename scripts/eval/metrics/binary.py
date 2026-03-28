"""Binary classification metrics for eval score files."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_curve

from .core import binary_scores


def compute_eer_from_labels(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    """Compute EER and its threshold from binary labels and positive-class scores."""
    if labels.size == 0 or predictions.size == 0 or labels.size != predictions.size:
        return 1.0, 0.0
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 1.0, 0.0
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    fnr = 1.0 - tpr
    best_index = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[best_index] + fnr[best_index]) / 2.0)
    threshold = float(thresholds[best_index])
    return eer, threshold


def compute_eer_from_binary_scores(score_pairs: list[tuple[float, int]]) -> float:
    if not score_pairs:
        return 0.0
    labels = np.array([label for _, label in score_pairs], dtype=np.int32)
    predictions = np.array([score for score, _ in score_pairs], dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    eer, _ = compute_eer_from_labels(labels, predictions)
    return eer


def compute_eer(scores: list[dict[str, Any]], score_key: str = "score", label_key: str = "label") -> float:
    return compute_eer_from_binary_scores(binary_scores(scores, score_key, label_key))


def compute_auc_from_binary_scores(score_pairs: list[tuple[float, int]]) -> float:
    if not score_pairs:
        return 0.0
    ordered_scores = sorted(score_pairs, key=lambda item: item[0])
    n_pos = sum(1 for _, label in ordered_scores if label == 1)
    n_neg = len(ordered_scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank_sum = 0
    for index, (_, label) in enumerate(ordered_scores):
        if label == 1:
            rank_sum += index + 1
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return max(0.0, min(1.0, auc))


def compute_auc_simple(scores: list[dict[str, Any]], score_key: str = "score", label_key: str = "label") -> float:
    return compute_auc_from_binary_scores(binary_scores(scores, score_key, label_key))


def compute_accuracy_from_binary_scores(score_pairs: list[tuple[float, int]], threshold: float) -> float:
    if not score_pairs:
        return 0.0
    correct = 0
    for score, label in score_pairs:
        if (1 if score >= threshold else 0) == label:
            correct += 1
    return correct / len(score_pairs)


def compute_accuracy(
    scores: list[dict[str, Any]],
    score_key: str = "score",
    label_key: str = "label",
    threshold: float | None = None,
) -> float:
    effective_threshold = 0.5 if threshold is None else threshold
    return compute_accuracy_from_binary_scores(binary_scores(scores, score_key, label_key), effective_threshold)


def compute_f1_from_binary_scores(score_pairs: list[tuple[float, int]], threshold: float) -> float:
    if not score_pairs:
        return 0.0
    tp = fp = fn = 0
    for score, label in score_pairs:
        pred = 1 if score >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_f1(
    scores: list[dict[str, Any]],
    score_key: str = "score",
    label_key: str = "label",
    threshold: float = 0.5,
) -> float:
    return compute_f1_from_binary_scores(binary_scores(scores, score_key, label_key), threshold)
