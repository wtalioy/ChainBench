"""Live status and snapshot handling for the evaluation pipeline."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from chainbench.lib.io import write_json
from chainbench.lib.logging import get_logger
from chainbench.lib.summary import utc_now_iso

from ..metrics.reporting import compute_baseline_metrics, write_metrics_files
from ..tasks import TaskPack
from .models import AssignedJob, RunRecord

LOGGER = get_logger("eval.pipeline")


class PipelineMonitor:
    def __init__(self, output_root: Path, jobs: list[AssignedJob], progress: Any | None) -> None:
        self.output_root = output_root
        self.progress = progress
        self.status_path = output_root / "eval_live_status.json"
        self.total_jobs = len(jobs)
        self.completed_jobs = 0
        self._lock = threading.Lock()
        self._started_at_monotonic: dict[int, float] = {}
        self._jobs: dict[int, dict[str, Any]] = {}
        timestamp = utc_now_iso()
        for job in jobs:
            run_dir = output_root / job.pack.task_id / job.pack.variant / job.baseline_name
            self._jobs[job.index] = {
                "job_id": self.job_id(job),
                "index": job.index,
                "task_id": job.pack.task_id,
                "variant": job.pack.variant,
                "baseline": job.baseline_name,
                "device": job.execution_device,
                "status": "queued",
                "phase": "queued",
                "run_dir": self._display_path(run_dir),
                "train_log_path": self._display_path(run_dir / "train.log"),
                "eval_log_path": self._display_path(run_dir / "eval.log"),
                "current_log_path": None,
                "started_at_utc": None,
                "updated_at_utc": timestamp,
                "finished_at_utc": None,
                "note": None,
            }
        with self._lock:
            self._write_snapshot_locked()
            self._update_progress_locked()

    def job_id(self, job: AssignedJob) -> str:
        return f"{job.index + 1}/{self.total_jobs}"

    def job_label(self, job: AssignedJob) -> str:
        return f"{job.baseline_name} {job.pack.task_id}/{job.pack.variant}"

    def _display_path(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.relative_to(self.output_root))
        except ValueError:
            return str(path)

    def _running_summary_locked(self) -> str:
        running = [entry for entry in self._jobs.values() if entry["status"] == "running"]
        if not running:
            return "idle"
        parts = [
            f'{entry["device"]}:{entry["baseline"]}/{entry["task_id"]}/{entry["variant"]}:{entry["phase"]}'
            for entry in sorted(running, key=lambda item: (item["device"], item["index"]))[:3]
        ]
        if len(running) > 3:
            parts.append(f"+{len(running) - 3} more")
        return " | ".join(parts)

    def _update_progress_locked(self) -> None:
        if self.progress is None or not hasattr(self.progress, "set_postfix_str"):
            return
        postfix = self._running_summary_locked()
        try:
            self.progress.set_postfix_str(postfix, refresh=False)
        except TypeError:
            self.progress.set_postfix_str(postfix)

    def _snapshot_payload_locked(self) -> dict[str, Any]:
        jobs = [self._jobs[index].copy() for index in sorted(self._jobs)]
        running_jobs = sum(1 for job in jobs if job["status"] == "running")
        queued_jobs = sum(1 for job in jobs if job["status"] == "queued")
        failed_jobs = sum(1 for job in jobs if job["status"] == "failed")
        return {
            "generated_at_utc": utc_now_iso(),
            "status_path": self._display_path(self.status_path),
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "running_jobs": running_jobs,
            "queued_jobs": queued_jobs,
            "failed_jobs": failed_jobs,
            "active_summary": self._running_summary_locked(),
            "jobs": jobs,
        }

    def _write_snapshot_locked(self) -> None:
        write_json(self.status_path, self._snapshot_payload_locked())

    def start_phase(self, job: AssignedJob, *, phase: str, log_path: Path | None = None, note: str | None = None) -> None:
        with self._lock:
            entry = self._jobs[job.index]
            timestamp = utc_now_iso()
            if entry["started_at_utc"] is None:
                entry["started_at_utc"] = timestamp
                self._started_at_monotonic[job.index] = time.monotonic()
            entry["status"] = "running"
            entry["phase"] = phase
            entry["updated_at_utc"] = timestamp
            entry["finished_at_utc"] = None
            entry["note"] = note
            if log_path is not None:
                entry["current_log_path"] = self._display_path(log_path)
            self._update_progress_locked()
            self._write_snapshot_locked()
        if log_path is not None:
            LOGGER.info(
                "job %s start %s: %s on %s (log: %s)",
                self.job_id(job),
                phase,
                self.job_label(job),
                job.execution_device,
                self._display_path(log_path),
            )
        else:
            LOGGER.info("job %s start %s: %s on %s", self.job_id(job), phase, self.job_label(job), job.execution_device)

    def finish(self, job: AssignedJob, record: RunRecord) -> None:
        elapsed_seconds = None
        with self._lock:
            entry = self._jobs[job.index]
            already_done = entry["status"] in {"finished", "failed"}
            entry["status"] = "finished" if record.ok else "failed"
            entry["phase"] = "done"
            entry["updated_at_utc"] = utc_now_iso()
            entry["finished_at_utc"] = entry["updated_at_utc"]
            entry["current_log_path"] = None
            entry["note"] = None
            entry["ok"] = record.ok
            entry["train_status"] = record.train_status
            entry["train_returncode"] = record.train_returncode
            entry["returncode"] = record.returncode
            entry["model_path"] = record.model_path
            entry["scores_path"] = record.scores_path
            started_at = self._started_at_monotonic.get(job.index)
            if started_at is not None:
                elapsed_seconds = max(0.0, time.monotonic() - started_at)
                entry["elapsed_seconds"] = round(elapsed_seconds, 3)
            if not already_done:
                self.completed_jobs += 1
                if self.progress is not None:
                    self.progress.update(1)
            self._update_progress_locked()
            self._write_snapshot_locked()
        LOGGER.info(
            "job %s done: %s on %s status=%s train=%s rc=%d elapsed=%.1fs",
            self.job_id(job),
            self.job_label(job),
            job.execution_device,
            "ok" if record.ok else "failed",
            record.train_status,
            record.returncode,
            elapsed_seconds or 0.0,
        )

    def fail_pipeline(self, message: str) -> None:
        with self._lock:
            timestamp = utc_now_iso()
            for entry in self._jobs.values():
                if entry["status"] in {"queued", "running"}:
                    entry["status"] = "failed"
                    entry["phase"] = "aborted"
                    entry["updated_at_utc"] = timestamp
                    entry["finished_at_utc"] = timestamp
                    entry["note"] = message
            self.completed_jobs = sum(1 for entry in self._jobs.values() if entry["status"] in {"finished", "failed"})
            self._update_progress_locked()
            self._write_snapshot_locked()
        LOGGER.error("evaluation pipeline failed: %s", message)


class LiveResultsWriter:
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


class PipelineObserver:
    def __init__(
        self,
        *,
        output_root: Path,
        jobs: list[AssignedJob],
        packs: list[TaskPack],
        progress: Any | None,
        aggregate_metrics_fn: Callable[..., list[dict[str, Any]]],
        build_label_map_fn: Callable[[list[dict[str, Any]]], dict[str, str]],
        on_snapshot: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    ) -> None:
        self.monitor = PipelineMonitor(output_root, jobs, progress)
        self.live_results = LiveResultsWriter(
            output_root=output_root,
            packs=packs,
            total_jobs=len(jobs),
            aggregate_metrics_fn=aggregate_metrics_fn,
            build_label_map_fn=build_label_map_fn,
            on_snapshot=on_snapshot,
        )

    @property
    def status_path(self) -> Path:
        return self.monitor.status_path

    def start_phase(self, job: AssignedJob, *, phase: str, log_path: Path | None = None, note: str | None = None) -> None:
        self.monitor.start_phase(job, phase=phase, log_path=log_path, note=note)

    def finish(self, job: AssignedJob, record: RunRecord) -> None:
        self.live_results.record(job.index, record)
        self.monitor.finish(job, record)

    def fail_pipeline(self, message: str) -> None:
        self.monitor.fail_pipeline(message)
