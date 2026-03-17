"""Metrics aggregation and snapshot writing for the evaluation pipeline."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from lib.io import write_json

from ..tasks import TaskPack
from .models import RunRecord
from .status import utc_now_iso


def task_label_maps(
    packs: list[TaskPack],
    *,
    build_label_map_fn: Callable[[list[dict[str, Any]]], dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (pack.task_id, pack.variant): build_label_map_fn(pack.test_rows)
        for pack in packs
    }


def task_pack_lookup(packs: list[TaskPack]) -> dict[tuple[str, str], TaskPack]:
    return {
        (pack.task_id, pack.variant): pack
        for pack in packs
    }


def compute_baseline_metrics(
    output_root: Path,
    packs: list[TaskPack],
    baseline_results: list[dict[str, Any]],
    *,
    aggregate_metrics_fn: Callable[..., list[dict[str, Any]]],
    build_label_map_fn: Callable[[list[dict[str, Any]]], dict[str, str]],
) -> list[dict[str, Any]]:
    return aggregate_metrics_fn(
        output_root,
        baseline_results,
        task_label_maps(packs, build_label_map_fn=build_label_map_fn),
        task_pack_lookup(packs),
    )


def write_metrics_files(output_root: Path, baseline_metrics: list[dict[str, Any]]) -> None:
    for run_meta in baseline_metrics:
        if run_meta.get("metrics"):
            run_dir = output_root / run_meta["task_id"] / run_meta["variant"] / run_meta["baseline"]
            write_json(run_dir / "metrics.json", run_meta["metrics"])


class ResultsSnapshotWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        packs: list[TaskPack],
        total_jobs: int,
        aggregate_metrics_fn: Callable[..., list[dict[str, Any]]],
        build_label_map_fn: Callable[[list[dict[str, Any]]], dict[str, str]],
        on_snapshot: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self.output_root = output_root
        self.packs = packs
        self.total_jobs = total_jobs
        self.aggregate_metrics_fn = aggregate_metrics_fn
        self.build_label_map_fn = build_label_map_fn
        self.on_snapshot = on_snapshot
        self.snapshot_path = output_root / "eval_results_live.json"
        self._records_by_index: dict[int, RunRecord] = {}
        self._lock = threading.Lock()

    def record(self, index: int, record: RunRecord) -> None:
        with self._lock:
            self._records_by_index[index] = record
            baseline_results = [
                current_record.as_dict()
                for _, current_record in sorted(self._records_by_index.items(), key=lambda item: item[0])
            ]
            baseline_metrics = compute_baseline_metrics(
                self.output_root,
                self.packs,
                baseline_results,
                aggregate_metrics_fn=self.aggregate_metrics_fn,
                build_label_map_fn=self.build_label_map_fn,
            )
            write_metrics_files(self.output_root, baseline_metrics)
            write_json(
                self.snapshot_path,
                {
                    "generated_at_utc": utc_now_iso(),
                    "total_jobs": self.total_jobs,
                    "completed_jobs": len(self._records_by_index),
                    "baseline_runs": baseline_results,
                    "baseline_metrics": baseline_metrics,
                },
            )
            if self.on_snapshot is not None:
                self.on_snapshot(baseline_results, baseline_metrics)
