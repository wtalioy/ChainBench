"""Internal row-cache and resume-state helpers for preservation analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from chainbench.lib.io import write_json
from chainbench.lib.summary import utc_now_iso

from .schema import RESULT_FIELDNAMES
from .selection import result_row_key
from .summary import PreservationSummaryAccumulator


def state_root(workspace_root: Path) -> Path:
    path = workspace_root / "results" / ".preservation_state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_rows_path(workspace_root: Path, *, num_shards: int, shard_index: int) -> Path:
    root = state_root(workspace_root)
    if num_shards <= 1:
        return root / "preservation_rows.csv"
    return root / f"preservation_rows.shard{shard_index}of{num_shards}.csv"


def resume_state_path(output_rows_path: Path) -> Path:
    return output_rows_path.with_name(f"{output_rows_path.name}.progress.json")


def analysis_fingerprint(
    *,
    metadata_path: Path,
    requested_splits: list[str],
    args,
    num_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    return {
        "metadata_path": str(metadata_path),
        "requested_splits": list(requested_splits),
        "limit": int(getattr(args, "limit", 0)),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "asr_backend": str(getattr(args, "asr_backend", "")),
        "speaker_backend": str(getattr(args, "speaker_backend", "")),
    }


def load_resume_rows(
    output_rows_path: Path,
) -> tuple[set[str], PreservationSummaryAccumulator, int]:
    completed_keys: set[str] = set()
    summary_accumulator = PreservationSummaryAccumulator()
    completed_rows = 0
    if not output_rows_path.exists():
        return completed_keys, summary_accumulator, completed_rows
    with output_rows_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = dict(row)
            key = result_row_key(normalized)
            if key in completed_keys:
                continue
            completed_keys.add(key)
            summary_accumulator.update(normalized)
            completed_rows += 1
    return completed_keys, summary_accumulator, completed_rows


def load_resume_manifest(path: Path, *, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    prior = state.get("fingerprint", {})
    if prior != fingerprint:
        raise ValueError(
            "Resume state does not match the current analysis arguments; "
            "remove the old preservation state files or rerun with --no-resume."
        )
    return state


def write_resume_manifest(
    path: Path,
    *,
    fingerprint: dict[str, Any],
    completed_rows: int,
    scanned_rows: int,
    selected_rows: int,
    status: str,
) -> None:
    write_json(
        path,
        {
            "updated_at_utc": utc_now_iso(),
            "status": status,
            "completed_result_rows": completed_rows,
            "scanned_rows": scanned_rows,
            "selected_rows": selected_rows,
            "fingerprint": fingerprint,
        },
    )


def open_result_rows_writer(
    path: Path | None,
    *,
    append: bool = False,
) -> tuple[Any | None, csv.DictWriter | None]:
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    mode = "a" if append and file_exists else "w"
    handle = path.open(mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
    if mode == "w" or not file_exists or path.stat().st_size == 0:
        writer.writeheader()
    return handle, writer


def write_result_rows(writer: csv.DictWriter | None, rows: list[dict[str, Any]]) -> None:
    if writer is None:
        return
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDNAMES})
