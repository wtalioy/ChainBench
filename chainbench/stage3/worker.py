"""Internal Stage-3 worker for sequential per-generator job execution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from chainbench.lib.cli import add_log_level_argument
from chainbench.lib.config import load_json
from chainbench.lib.io import load_jsonl
from chainbench.lib.logging import format_elapsed, get_logger, progress_bar, setup_logging

from .runner import AdapterRunner, RUNNER_REGISTRY

INTERNAL_WORKER_FLAG = "--internal-worker"

LOGGER = get_logger("stage3-worker")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Adapter key to use.")
    parser.add_argument("--repo-path", required=True, help="Generator repo root.")
    parser.add_argument("--config-path", required=True, help="Adapter config JSON.")
    parser.add_argument("--jobs-path", required=True, help="JSONL jobs file.")
    parser.add_argument("--results-path", required=True, help="JSONL results file.")
    add_log_level_argument(parser)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Emit aggregate progress every N jobs. Default auto-selects by batch size.",
    )
    return parser.parse_args(argv)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_worker(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)

    repo_path = Path(args.repo_path).resolve()
    config = load_json(Path(args.config_path))
    jobs = load_jsonl(Path(args.jobs_path))
    results_path = Path(args.results_path)
    if results_path.exists():
        results_path.unlink()

    runner_cls: type[AdapterRunner] | None = RUNNER_REGISTRY.get(args.adapter)
    if runner_cls is None:
        raise RuntimeError(f"Unsupported adapter: {args.adapter}")

    LOGGER.info("boot adapter=%s repo=%s", args.adapter, repo_path)
    startup_started_at = time.monotonic()
    runner = runner_cls(repo_path=repo_path, config=config)
    runner.setup()
    total_jobs = len(jobs)
    progress_every = (
        args.progress_every
        if args.progress_every > 0
        else (1 if total_jobs <= 10 else 5 if total_jobs <= 50 else 25)
    )
    ok_count = 0
    skipped_count = 0
    failed_count = 0
    started_at = time.monotonic()
    LOGGER.info(
        "ready in %s | jobs=%d",
        format_elapsed(time.monotonic() - startup_started_at),
        total_jobs,
    )

    for idx, job in enumerate(jobs, start=1):
        job_started_at = time.monotonic()
        LOGGER.info(
            "%s %d/%d starting sample=%s speaker=%s split=%s",
            progress_bar(idx - 1, total_jobs),
            idx,
            total_jobs,
            job["sample_id"],
            job["speaker_id"],
            job["split"],
        )
        output_path = Path(job["output_path"])
        if output_path.exists():
            skipped_count += 1
            append_jsonl(
                results_path,
                {
                    "job_id": job["job_id"],
                    "status": "skipped_existing",
                    "output_path": job["output_path"],
                    "sample_id": job["sample_id"],
                    "parent_id": job["parent_id"],
                },
            )
            LOGGER.info(
                "%s %d/%d skip  | %s | sample=%s | ok=%d skip=%d fail=%d",
                progress_bar(idx, total_jobs),
                idx,
                total_jobs,
                format_elapsed(time.monotonic() - job_started_at),
                job["sample_id"],
                ok_count,
                skipped_count,
                failed_count,
            )
            continue

        try:
            meta = runner.run_job(job)
            ok_count += 1
            append_jsonl(
                results_path,
                {
                    "job_id": job["job_id"],
                    "status": "ok",
                    "output_path": job["output_path"],
                    "sample_id": job["sample_id"],
                    "parent_id": job["parent_id"],
                    "meta": meta,
                },
            )
            LOGGER.success(
                "%s %d/%d done  | %s | sample=%s | ok=%d skip=%d fail=%d",
                progress_bar(idx, total_jobs),
                idx,
                total_jobs,
                format_elapsed(time.monotonic() - job_started_at),
                job["sample_id"],
                ok_count,
                skipped_count,
                failed_count,
            )
        except Exception as exc:
            failed_count += 1
            append_jsonl(
                results_path,
                {
                    "job_id": job["job_id"],
                    "status": "failed",
                    "output_path": job["output_path"],
                    "sample_id": job["sample_id"],
                    "parent_id": job["parent_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            LOGGER.error(
                "%s %d/%d fail  | %s | sample=%s | %s | ok=%d skip=%d fail=%d",
                progress_bar(idx, total_jobs),
                idx,
                total_jobs,
                format_elapsed(time.monotonic() - job_started_at),
                job["sample_id"],
                f"{type(exc).__name__}: {exc}",
                ok_count,
                skipped_count,
                failed_count,
            )

        if idx <= 3 or idx % progress_every == 0 or idx == total_jobs:
            LOGGER.info(
                "%s %d/%d pulse | %s | ok=%d skip=%d fail=%d",
                progress_bar(idx, total_jobs),
                idx,
                total_jobs,
                format_elapsed(time.monotonic() - started_at),
                ok_count,
                skipped_count,
                failed_count,
            )

    LOGGER.success(
        "complete | adapter=%s | jobs=%d | elapsed=%s | ok=%d skip=%d fail=%d",
        args.adapter,
        total_jobs,
        format_elapsed(time.monotonic() - started_at),
        ok_count,
        skipped_count,
        failed_count,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_worker(parse_args(argv))
