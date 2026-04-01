"""Public metrics API for eval."""

from __future__ import annotations

from .binary import (
    compute_accuracy,
    compute_auc_simple,
    compute_eer,
    compute_eer_from_labels,
    compute_f1,
)
from .core import build_label_map, load_scores_csv
from .compute import compute_metrics_for_scores
from .reporting import aggregate_run_metrics

__all__ = [
    "aggregate_run_metrics",
    "build_label_map",
    "compute_accuracy",
    "compute_auc_simple",
    "compute_eer",
    "compute_eer_from_labels",
    "compute_f1",
    "compute_metrics_for_scores",
    "load_scores_csv",
]
