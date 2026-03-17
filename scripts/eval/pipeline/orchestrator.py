"""Top-level orchestration for evaluation pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from lib.logging import get_logger

from ..baselines import BASELINE_MAP
from ..metrics import aggregate_run_metrics, build_label_map
from ..progress import create_progress
from ..tasks import TaskPack
from .models import RunRecord
from .scheduler import assign_jobs_to_devices, execute_assigned_jobs
from .snapshots import ResultsSnapshotWriter, compute_baseline_metrics, write_metrics_files
from .state import PipelineState
from .status import PipelineMonitor

LOGGER = get_logger("eval.pipeline")


def run_all_baselines(
    *,
    output_root: Path,
    dataset_root: Path,
    packs: list[TaskPack],
    baseline_names: list[str],
    baseline_configs: dict[str, dict[str, object]],
    eval_only: bool,
    train_only: bool,
    force_retrain: bool = False,
    on_snapshot: Callable[[list[dict[str, object]], list[dict[str, object]]], None] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str | None]:
    records: list[RunRecord] = []
    baseline_error = None
    pipeline_state = PipelineState(
        output_root=output_root,
        dataset_root=dataset_root,
        eval_only=eval_only,
        train_only=train_only,
        force_retrain=force_retrain,
        baseline_map=BASELINE_MAP,
    )
    try:
        assigned_jobs = assign_jobs_to_devices(
            packs,
            baseline_names,
            baseline_configs,
            eval_only=eval_only,
            train_only=train_only,
        )
        pipeline_state.progress = create_progress(total=len(assigned_jobs), desc="eval pipeline", unit="run")
        pipeline_state.monitor = PipelineMonitor(output_root, assigned_jobs, pipeline_state.progress)
        pipeline_state.results_snapshot_writer = ResultsSnapshotWriter(
            output_root=output_root,
            packs=packs,
            total_jobs=len(assigned_jobs),
            aggregate_metrics_fn=aggregate_run_metrics,
            build_label_map_fn=build_label_map,
            on_snapshot=on_snapshot,
        )
        indexed_records = execute_assigned_jobs(
            assigned_jobs,
            baseline_configs=baseline_configs,
            pipeline_state=pipeline_state,
        )
        records = [record for _, record in sorted(indexed_records, key=lambda item: item[0])]
    except Exception as exc:
        LOGGER.exception("baseline run failed: %s", exc)
        baseline_error = str(exc)
        if pipeline_state.monitor is not None:
            pipeline_state.monitor.fail_pipeline(str(exc))
    finally:
        if pipeline_state.progress is not None:
            pipeline_state.progress.close()

    baseline_results = [record.as_dict() for record in records]
    baseline_metrics = compute_baseline_metrics(
        output_root,
        packs,
        baseline_results,
        aggregate_metrics_fn=aggregate_run_metrics,
        build_label_map_fn=build_label_map,
    )
    write_metrics_files(output_root, baseline_metrics)
    return baseline_results, baseline_metrics, baseline_error
