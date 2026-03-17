"""Shared mutable state for pipeline execution."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tasks import TaskPack
from .checkpoints import materialize_shared_checkpoint
from .models import AssignedJob, RunRecord


def view_lock_for_key(
    view_lock_key: str,
    view_locks: dict[str, threading.Lock],
    view_locks_lock: threading.Lock,
) -> threading.Lock:
    with view_locks_lock:
        return view_locks.setdefault(view_lock_key, threading.Lock())


@dataclass
class PipelineState:
    output_root: Path
    dataset_root: Path
    eval_only: bool
    train_only: bool
    force_retrain: bool
    baseline_map: dict[str, Any]
    progress: Any | None = None
    monitor: Any | None = None
    results_snapshot_writer: Any | None = None
    training_cache: dict[str, Path] = field(default_factory=dict)
    cache_lock: threading.Lock = field(default_factory=threading.Lock)
    view_locks: dict[str, threading.Lock] = field(default_factory=dict)
    view_locks_lock: threading.Lock = field(default_factory=threading.Lock)

    def start_phase(
        self,
        job: AssignedJob | None,
        *,
        phase: str,
        log_path: Path | None = None,
        note: str | None = None,
    ) -> None:
        if self.monitor is not None and job is not None:
            self.monitor.start_phase(job, phase=phase, log_path=log_path, note=note)

    def finish_record(self, job: AssignedJob | None, record: RunRecord) -> RunRecord:
        if self.results_snapshot_writer is not None and job is not None:
            self.results_snapshot_writer.record(job.index, record)
        if self.monitor is not None and job is not None:
            self.monitor.finish(job, record)
            return record
        if self.progress is not None:
            self.progress.update(1)
        return record

    def prepare_view(self, runner: Any, pack: TaskPack, run_dir: Path) -> Any:
        view_lock_key = str((run_dir.parent / "_views").resolve())
        with view_lock_for_key(view_lock_key, self.view_locks, self.view_locks_lock):
            return runner.prepare_view(pack, run_dir, self.dataset_root)

    def reuse_shared_checkpoint(self, shared_training_key_value: str, run_dir: Path) -> tuple[Path, Path] | None:
        with self.cache_lock:
            cached_checkpoint = self.training_cache.get(shared_training_key_value)
        if cached_checkpoint is None or not cached_checkpoint.exists():
            return None
        return cached_checkpoint, materialize_shared_checkpoint(run_dir, cached_checkpoint)

    def remember_checkpoint(self, shared_training_key_value: str | None, checkpoint: Path | None) -> None:
        if self.force_retrain or shared_training_key_value is None or checkpoint is None:
            return
        with self.cache_lock:
            self.training_cache.setdefault(shared_training_key_value, checkpoint.resolve())
