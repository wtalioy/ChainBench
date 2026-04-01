"""Row selection and sharding helpers for preservation analysis."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from chainbench.lib.config import resolve_path


def infer_dataset_root(dataset_root_arg: str, metadata_path: Path, workspace_root: Path) -> Path:
    if dataset_root_arg:
        return resolve_path(dataset_root_arg, workspace_root)
    if metadata_path.parent.name == "manifest":
        return metadata_path.parent.parent
    return metadata_path.parent


def row_in_shard(row: Mapping[str, str], *, num_shards: int, shard_index: int) -> bool:
    if num_shards <= 1:
        return True
    parent_id = str(row.get("parent_id", "")).strip()
    sample_id = str(row.get("sample_id", "")).strip()
    shard_key = parent_id or sample_id
    digest = hashlib.md5(shard_key.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards == shard_index


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")


def row_matches_selection(
    row: Mapping[str, str],
    *,
    split_set: set[str],
    num_shards: int,
    shard_index: int,
) -> bool:
    split = str(row.get("split_standard", "") or row.get("split", "")).strip()
    if split_set and split not in split_set:
        return False
    return row_in_shard(row, num_shards=num_shards, shard_index=shard_index)


def result_row_key(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    file_name = str(row.get("file_name", "")).strip()
    if file_name:
        return file_name
    return json.dumps(
        {
            "parent_id": str(row.get("parent_id", "")).strip(),
            "clean_parent_path": str(row.get("clean_parent_path", "")).strip(),
            "file_name": file_name,
            "language": str(row.get("language", "")).strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def iter_selected_rows(
    metadata_path: Path,
    requested_splits: list[str],
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> Iterator[dict[str, str]]:
    split_set = set(requested_splits)
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row_matches_selection(
                row,
                split_set=split_set,
                num_shards=num_shards,
                shard_index=shard_index,
            ):
                continue
            yield dict(row)


def load_selected_rows(
    metadata_path: Path,
    requested_splits: list[str],
    limit: int,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> tuple[list[dict[str, str]], int]:
    selected: list[dict[str, str]] = []
    scanned_rows = 0
    split_set = set(requested_splits)
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scanned_rows += 1
            if not row_matches_selection(
                row,
                split_set=split_set,
                num_shards=num_shards,
                shard_index=shard_index,
            ):
                continue
            selected.append(dict(row))
            if limit > 0 and len(selected) >= limit:
                break
    return selected, scanned_rows
