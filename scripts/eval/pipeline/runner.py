"""Single-job execution helpers for the evaluation pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from lib.logging import get_logger

from ..tasks import TaskPack
from .checkpoints import (
    checkpoint_for_run,
    clear_stale_run_artifacts,
    shared_training_key,
    write_checkpoint_manifest,
    write_checkpoint_skip_note,
    write_shared_checkpoint_note,
)
from .models import AssignedJob, RunRecord, TrainingState
from .state import PipelineState

LOGGER = get_logger("eval.pipeline")


def job_config_with_device(
    baseline_cfg: dict[str, Any],
    *,
    execution_device: str,
    eval_only: bool,
    train_only: bool,
) -> dict[str, Any]:
    job_cfg = deepcopy(baseline_cfg)
    if baseline_cfg["train"]["enabled"] and not eval_only:
        # Train-mode runtimes perform train+eval in one process, so keep the
        # whole run pinned to a single configured execution slot.
        job_cfg["train"]["device"] = execution_device
        job_cfg["eval"]["device"] = execution_device
    elif baseline_cfg["eval"]["enabled"] and not train_only:
        job_cfg["eval"]["device"] = execution_device
    else:
        job_cfg["train"]["device"] = execution_device
    return job_cfg


def build_run_record(
    *,
    pack: TaskPack,
    baseline_name: str,
    execution_device: str,
    training_state: TrainingState,
    ok: bool,
    returncode: int,
    scores_path: Path | None = None,
    model_path: Path | None = None,
) -> RunRecord:
    resolved_model_path = model_path if model_path is not None else training_state.checkpoint
    return RunRecord(
        task_id=pack.task_id,
        variant=pack.variant,
        baseline=baseline_name,
        device=execution_device,
        train_status=training_state.train_status,
        train_returncode=training_state.train_returncode,
        ok=ok,
        returncode=returncode,
        model_path=str(resolved_model_path) if resolved_model_path else None,
        scores_path=str(scores_path) if scores_path else None,
    )


def load_training_state(
    *,
    run_dir: Path,
    runner: Any,
    pack: TaskPack,
    baseline_name: str,
    baseline_cfg: dict[str, Any],
    pipeline_state: PipelineState,
) -> tuple[str | None, TrainingState]:
    shared_key = shared_training_key(pack, baseline_name, baseline_cfg)
    if pipeline_state.force_retrain and baseline_cfg["train"]["enabled"] and not pipeline_state.eval_only:
        LOGGER.info("force retrain enabled; clearing cached checkpoint artifacts for %s", run_dir)
        clear_stale_run_artifacts(run_dir, runner)
        return shared_key, TrainingState()

    checkpoint = checkpoint_for_run(run_dir, runner, shared_key)
    training_state = TrainingState(checkpoint=checkpoint)
    if checkpoint is None and shared_key and not pipeline_state.force_retrain:
        reused = pipeline_state.reuse_shared_checkpoint(shared_key, run_dir)
        if reused is not None:
            source_checkpoint, local_checkpoint = reused
            training_state.checkpoint = local_checkpoint
            training_state.train_status = "reused_shared_checkpoint"
            write_checkpoint_manifest(run_dir, shared_key)
            write_shared_checkpoint_note(run_dir, source_checkpoint)
    return shared_key, training_state


def run_training_phase(
    *,
    pack: TaskPack,
    baseline_name: str,
    baseline_cfg: dict[str, Any],
    execution_device: str,
    runner: Any,
    prepared_view: Any,
    run_dir: Path,
    shared_key: str | None,
    training_state: TrainingState,
    pipeline_state: PipelineState,
    job: AssignedJob | None,
) -> RunRecord | None:
    if not (baseline_cfg["train"]["enabled"] and not pipeline_state.eval_only and training_state.checkpoint is None):
        return None

    pipeline_state.start_phase(job, phase="train", log_path=run_dir / "train.log")
    if shared_key:
        # Record expected inputs before launch so interrupted runs remain attributable.
        write_checkpoint_manifest(run_dir, shared_key)

    train_result = runner.train(prepared_view, run_dir)
    training_state.train_status = "trained"
    training_state.train_returncode = train_result.returncode
    training_state.train_scores_path = train_result.scores_path
    if not train_result.ok:
        return build_run_record(
            pack=pack,
            baseline_name=baseline_name,
            execution_device=execution_device,
            training_state=training_state,
            ok=False,
            returncode=training_state.train_returncode,
        )

    training_state.checkpoint = train_result.model_path or runner.find_checkpoint(run_dir)
    if training_state.checkpoint is not None and shared_key:
        write_checkpoint_manifest(run_dir, shared_key)
        pipeline_state.remember_checkpoint(shared_key, training_state.checkpoint)
    return None


def apply_non_training_status(
    *,
    run_dir: Path,
    job_cfg: dict[str, Any],
    shared_key: str | None,
    training_state: TrainingState,
    pipeline_state: PipelineState,
) -> None:
    if pipeline_state.eval_only:
        training_state.train_status = "eval_only"
        return
    if pipeline_state.train_only:
        training_state.train_status = "train_only"
        return
    if training_state.checkpoint is None:
        return
    if training_state.train_status != "reused_shared_checkpoint":
        training_state.train_status = "skipped_existing_checkpoint"
        write_checkpoint_skip_note(run_dir, training_state.checkpoint, job_cfg["train"]["epochs"])
    if shared_key:
        write_checkpoint_manifest(run_dir, shared_key)
        pipeline_state.remember_checkpoint(shared_key, training_state.checkpoint)


def run_baseline(
    pack: TaskPack,
    baseline_name: str,
    baseline_cfg: dict[str, Any],
    *,
    execution_device: str,
    pipeline_state: PipelineState,
    job: AssignedJob | None = None,
) -> RunRecord:
    run_dir = pipeline_state.output_root / pack.task_id / pack.variant / baseline_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pipeline_state.start_phase(job, phase="prepare_view")
    job_cfg = job_config_with_device(
        baseline_cfg,
        execution_device=execution_device,
        eval_only=pipeline_state.eval_only,
        train_only=pipeline_state.train_only,
    )
    runner = pipeline_state.baseline_map[baseline_name](job_cfg)
    prepared_view = pipeline_state.prepare_view(runner, pack, run_dir)
    shared_key, training_state = load_training_state(
        run_dir=run_dir,
        runner=runner,
        pack=pack,
        baseline_name=baseline_name,
        baseline_cfg=baseline_cfg,
        pipeline_state=pipeline_state,
    )
    train_failure = run_training_phase(
        pack=pack,
        baseline_name=baseline_name,
        baseline_cfg=baseline_cfg,
        execution_device=execution_device,
        runner=runner,
        prepared_view=prepared_view,
        run_dir=run_dir,
        shared_key=shared_key,
        training_state=training_state,
        pipeline_state=pipeline_state,
        job=job,
    )
    if train_failure is not None:
        return pipeline_state.finish_record(job, train_failure)

    if training_state.train_status != "trained":
        apply_non_training_status(
            run_dir=run_dir,
            job_cfg=job_cfg,
            shared_key=shared_key,
            training_state=training_state,
            pipeline_state=pipeline_state,
        )

    if pipeline_state.train_only or not baseline_cfg["eval"]["enabled"]:
        return pipeline_state.finish_record(
            job,
            build_run_record(
                pack=pack,
                baseline_name=baseline_name,
                execution_device=execution_device,
                training_state=training_state,
                ok=True,
                returncode=0,
            ),
        )

    if training_state.train_scores_path is not None:
        LOGGER.info("reusing training-time scores for %s on %s/%s", baseline_name, pack.task_id, pack.variant)
        return pipeline_state.finish_record(
            job,
            build_run_record(
                pack=pack,
                baseline_name=baseline_name,
                execution_device=execution_device,
                training_state=training_state,
                ok=True,
                returncode=0,
                scores_path=training_state.train_scores_path,
            ),
        )

    pipeline_state.start_phase(job, phase="eval", log_path=run_dir / "eval.log")
    eval_result = runner.evaluate(prepared_view, run_dir, training_state.checkpoint)
    return pipeline_state.finish_record(
        job,
        build_run_record(
            pack=pack,
            baseline_name=baseline_name,
            execution_device=execution_device,
            training_state=training_state,
            ok=eval_result.ok,
            returncode=eval_result.returncode,
            scores_path=eval_result.scores_path,
        ),
    )
