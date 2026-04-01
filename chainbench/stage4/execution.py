"""Concurrent rendering helpers for Stage 4."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from chainbench.lib.execution import run_bounded_tasks

from .render import build_manifest_row, render_single_job


def _render_failure(job: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "sample_id": job["sample_id"],
        "parent_id": job["parent_id"],
        "chain_family": job["family_name"],
        "chain_template_id": job["template_id"],
        "error": error,
    }


def render_stage4_jobs(
    jobs: Iterable[dict[str, Any]],
    total: int,
    *,
    config: dict[str, Any],
    workspace_root: Path,
    output_root: Path,
    workers: int,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counts: Counter = Counter()
    run_bounded_tasks(
        jobs,
        total,
        workers=workers,
        desc="stage4 render",
        unit="job",
        submit_fn=lambda executor, job: executor.submit(
            render_single_job,
            job,
            config,
            workspace_root,
            output_root,
        ),
        on_result=lambda result: (
            manifest_rows.append(build_manifest_row(result["job"], result, workspace_root))
            if result["status"] in {"ok", "skipped_existing"}
            else failures.append(_render_failure(result["job"], result["error"]))
        ),
        counts=counts,
        log_every=log_every,
        progress_postfix=lambda _completed: {
            "ok": counts["ok"],
            "skipped": counts["skipped_existing"],
            "failed": counts["failed"],
        },
        status_fn=lambda result: str(result["status"]),
    )
    return manifest_rows, failures, counts
