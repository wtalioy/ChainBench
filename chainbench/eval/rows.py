"""Row-level helpers for eval metadata and score dictionaries."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

# Use ``typing.Dict`` for the alias: baseline conda envs may run Python 3.7, where
# ``dict[str, Any]`` is evaluated at import time and is not subscriptable.
RowDict = Dict[str, Any]

def stable_row_token(row: RowDict) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    return json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def paired_test_id(row: RowDict) -> str:
    return str(row.get("__paired_test_id", "")).strip()


def standard_split_value(row: RowDict) -> str:
    return str(row.get("split_standard", row.get("split", ""))).strip()


def rows_for_split(rows: list[RowDict], split: str) -> list[RowDict]:
    return [row for row in rows if standard_split_value(row) == split]


def normalize_binary_label(label: Any) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in {"bonafide", "bona_fide", "1"}:
        return "bonafide"
    if normalized in {"spoof", "0"}:
        return "spoof"
    return ""


def binary_label_as_int(label: Any) -> Optional[int]:
    """Map coarse labels to 1 = bonafide, 0 = spoof; unknown/empty → ``None``."""
    slug = normalize_binary_label(label)
    if slug == "bonafide":
        return 1
    if slug == "spoof":
        return 0
    return None


def same_matched_binary_label(left: RowDict, right: RowDict) -> bool:
    left_label = normalize_binary_label(left.get("label"))
    right_label = normalize_binary_label(right.get("label"))
    return bool(left_label and left_label == right_label)


def group_rows_by_field(rows: list[RowDict], field: str) -> dict[str, list[RowDict]]:
    """Bucket rows by stripped string value of ``field``; omit rows with empty keys."""
    grouped: dict[str, list[RowDict]] = defaultdict(list)
    for row in rows:
        key = str(row.get(field, "")).strip()
        if key:
            grouped[key].append(row)
    return grouped


def bucket_rows(rows: list[RowDict], key_fn: Callable[[RowDict], Any]) -> dict[str, list[RowDict]]:
    """Group rows by ``str(key_fn(row))`` (including empty string buckets)."""
    out: dict[str, list[RowDict]] = defaultdict(list)
    for row in rows:
        out[str(key_fn(row))].append(row)
    return out


def target_sample_size(size: int, ratio: float) -> int:
    if size <= 0 or ratio >= 1.0:
        return size
    return min(size, max(1, int(round(size * ratio))))


def sample_exact_rows(rows: list[RowDict], target_size: int, *, salt: str) -> list[RowDict]:
    if not rows or target_size >= len(rows):
        return rows
    if target_size <= 0:
        return []
    ranked = sorted(
        (
            hashlib.sha1(f"{salt}\0{stable_row_token(row)}".encode("utf-8")).hexdigest(),
            index,
        )
        for index, row in enumerate(rows)
    )
    keep_indices = {index for _, index in ranked[:target_size]}
    return [row for index, row in enumerate(rows) if index in keep_indices]


def sample_rows(rows: list[RowDict], ratio: float, *, salt: str) -> list[RowDict]:
    if not rows or ratio >= 1.0:
        return rows
    return sample_exact_rows(rows, target_sample_size(len(rows), ratio), salt=salt)


def sample_rows_by_group(
    rows: list[RowDict],
    ratio: float,
    *,
    salt: str,
    group_key_fn: Callable[[RowDict], Any],
) -> list[RowDict]:
    if not rows or ratio >= 1.0:
        return rows
    rows_by_group = bucket_rows(rows, group_key_fn)
    sampled_rows: list[RowDict] = []
    for group_key in sorted(rows_by_group):
        group_rows = rows_by_group[group_key]
        sampled_rows.extend(
            sample_exact_rows(
                group_rows,
                target_sample_size(len(group_rows), ratio),
                salt=f"{salt}:{group_key}",
            )
        )
    return sampled_rows


def sample_units_within_primary_groups(
    rows: list[RowDict],
    ratio: float,
    *,
    salt: str,
    primary_key_fn: Callable[[RowDict], Any],
    unit_key_fn: Callable[[RowDict], str],
) -> list[RowDict]:
    """Sample whole unit buckets (e.g. paired test rows) within each primary group (e.g. chain family)."""
    if not rows or ratio >= 1.0:
        return rows
    nested: dict[str, dict[str, list[RowDict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        nested[str(primary_key_fn(row))][unit_key_fn(row)].append(row)

    sampled_rows: list[RowDict] = []
    for primary in sorted(nested):
        unit_map = nested[primary]
        unit_ids = list(unit_map.keys())
        target_size = target_sample_size(len(unit_ids), ratio)
        ranked = sorted(
            (
                hashlib.sha1(f"{salt}:{primary}\0{unit_id}".encode("utf-8")).hexdigest(),
                unit_id,
            )
            for unit_id in unit_ids
        )
        keep_ids = {uid for _, uid in ranked[:target_size]}
        for unit_id in unit_ids:
            if unit_id in keep_ids:
                sampled_rows.extend(unit_map[unit_id])
    return sampled_rows


def collect_paired_rows(
    grouped_rows: dict[str, list[RowDict]],
    *,
    accept_pair: Callable[[str, RowDict, RowDict], bool],
    pair_id_field: str = "__paired_test_id",
    row_extras: Optional[Callable[[str, RowDict, RowDict], RowDict]] = None,
) -> list[RowDict]:
    """Keep groups of exactly two rows that pass ``accept_pair``; attach ``pair_id_field`` (+ optional extras)."""
    out: list[RowDict] = []
    for pair_id, pair_rows in grouped_rows.items():
        if len(pair_rows) != 2:
            continue
        left, right = pair_rows[0], pair_rows[1]
        if not accept_pair(pair_id, left, right):
            continue
        extras = row_extras(pair_id, left, right) if row_extras else {}
        out.extend({**row, pair_id_field: pair_id, **extras} for row in pair_rows)
    return out
