"""Top-level orchestration for evaluation pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from lib.logging import get_logger

from ..baselines import BASELINE_MAP
from ..metrics import aggregate_run_metrics, build_label_map
from ..progress import create_progress
from ..tasks import TaskPack
from .models import RunRecord
from .observability import PipelineObserver
from .scheduler import assign_jobs_to_devices, execute_assigned_jobs
from .state import PipelineState
from ..metrics.reporting import compute_baseline_metrics, write_metrics_files

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
    baseline_map: dict[str, Any] | None = None,
    aggregate_metrics_fn: Callable[..., list[dict[str, Any]]] | None = None,
    build_label_map_fn: Callable[[list[dict[str, Any]]], dict[str, str]] | None = None,
    progress_factory: Callable[..., Any] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str | None]:
    aggregate_metrics_fn = aggregate_metrics_fn or aggregate_run_metrics
    build_label_map_fn = build_label_map_fn or build_label_map
    progress_factory = progress_factory or create_progress
    records: list[RunRecord] = []
    baseline_error = None
    pipeline_state = PipelineState(
        output_root=output_root,
        dataset_root=dataset_root,
        eval_only=eval_only,
        train_only=train_only,
        force_retrain=force_retrain,
        baseline_map=baseline_map or BASELINE_MAP,
    )
    try:
        assigned_jobs = assign_jobs_to_devices(
            packs,
            baseline_names,
            baseline_configs,
            eval_only=eval_only,
            train_only=train_only,
        )
        pipeline_state.progress = progress_factory(total=len(assigned_jobs), desc="eval pipeline", unit="run")
        pipeline_state.observer = PipelineObserver(
            output_root=output_root,
            jobs=assigned_jobs,
            packs=packs,
            progress=pipeline_state.progress,
            aggregate_metrics_fn=aggregate_metrics_fn,
            build_label_map_fn=build_label_map_fn,
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
        if pipeline_state.observer is not None:
            pipeline_state.observer.fail_pipeline(str(exc))
    finally:
        if pipeline_state.progress is not None:
            pipeline_state.progress.close()

    baseline_results = [record.as_dict() for record in records]
    baseline_metrics = compute_baseline_metrics(
        output_root,
        packs,
        baseline_results,
        aggregate_metrics_fn=aggregate_metrics_fn,
        build_label_map_fn=build_label_map_fn,
    )
    write_metrics_files(output_root, baseline_metrics)
    return baseline_results, baseline_metrics, baseline_error
