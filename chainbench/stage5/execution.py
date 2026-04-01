"""Execution helpers for Stage 5 validation and export."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from chainbench.lib.execution import run_bounded_tasks

from .export import export_single_audio
from .validate import validate_single_row


def make_failure_record(row: dict[str, Any], status: str, error: str | None) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "parent_id": row["parent_id"],
        "split": row["split"],
        "language": row["language"],
        "speaker_id": row["speaker_id"],
        "status": status,
        "error": error or "",
        "audio_path": row["audio_path"],
    }


def validate_dataset_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    workspace_root: Path,
    workers: int,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    prepared_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counts = Counter()
    run_bounded_tasks(
        rows,
        len(rows),
        workers=workers,
        desc="stage5 validate",
        unit="file",
        submit_fn=lambda executor, row: executor.submit(validate_single_row, row, config, workspace_root),
        on_result=lambda result: (
            prepared_rows.append(dict(result.input_row))
            if result.ok
            else failures.append(make_failure_record(result.input_row, result.status, result.error))
        ),
        counts=counts,
        log_every=log_every,
        progress_postfix=lambda completed: {
            "kept": len(prepared_rows),
            "failed": completed - len(prepared_rows),
        },
    )
    return prepared_rows, failures, counts


def export_dataset_audio(
    rows: list[dict[str, Any]],
    workspace_root: Path,
    dataset_root: Path,
    workers: int,
    overwrite: bool,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    exported_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counts = Counter()
    run_bounded_tasks(
        rows,
        len(rows),
        workers=workers,
        desc="stage5 export",
        unit="audio",
        submit_fn=lambda executor, row: executor.submit(
            export_single_audio,
            row,
            workspace_root,
            dataset_root,
            overwrite,
        ),
        on_result=lambda result: (
            exported_rows.append(result.output_row)
            if result.ok and result.output_row is not None
            else failures.append(make_failure_record(result.input_row, result.status, result.error))
        ),
        counts=counts,
        log_every=log_every,
        progress_postfix=lambda completed: {
            "copied": counts["exported_audio"],
            "skipped": counts["skipped_existing_audio"],
            "failed": completed - (counts["exported_audio"] + counts["skipped_existing_audio"]),
        },
    )
    return exported_rows, failures, counts


def sort_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (row["language"], row["split"], row["speaker_id"], row["sample_id"]))
    return rows
