"""Low-level score loading and enrichment helpers."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..rows import binary_label_as_int
from ..tasks import TaskPack


def build_label_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        (row.get("sample_id") or "").strip(): (row.get("label") or "").strip()
        for row in rows
        if (row.get("sample_id") or "").strip()
    }


def load_scores_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def binary_scores(scores: list[dict[str, Any]], score_key: str, label_key: str) -> list[tuple[float, int]]:
    result: list[tuple[float, int]] = []
    for row in scores:
        label = binary_label_as_int(row.get(label_key))
        if label is not None:
            result.append((parse_score(row.get(score_key)), label))
    return result


def binary_label_counts(scores: list[dict[str, Any]], label_key: str = "label") -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in scores:
        label = binary_label_as_int(row.get(label_key))
        if label == 1:
            counts["bona_fide"] += 1
        elif label == 0:
            counts["spoof"] += 1
        else:
            counts["unknown"] += 1
    return dict(counts)


def apply_label_map(scores: list[dict[str, Any]], label_map: dict[str, str] | None) -> list[dict[str, Any]]:
    if not label_map:
        return scores
    enriched: list[dict[str, Any]] = []
    for row in scores:
        sample_id = (row.get("sample_id") or "").strip()
        if sample_id and not (row.get("label") or "").strip() and sample_id in label_map:
            enriched.append({**row, "label": label_map[sample_id]})
        else:
            enriched.append(row)
    return enriched


def enrich_scores(
    scores: list[dict[str, Any]],
    *,
    label_map: dict[str, str] | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    enriched = apply_label_map(scores, label_map)
    if not metadata_rows:
        return enriched

    metadata_by_sample = {
        (row.get("sample_id") or "").strip(): row
        for row in metadata_rows
        if (row.get("sample_id") or "").strip()
    }
    merged_rows: list[dict[str, Any]] = []
    for row in enriched:
        sample_id = (row.get("sample_id") or "").strip()
        metadata = metadata_by_sample.get(sample_id)
        if not metadata:
            merged_rows.append(row)
            continue
        merged = {**metadata, **row}
        if not (merged.get("label") or "").strip():
            merged["label"] = metadata.get("label", "")
        merged_rows.append(merged)
    return merged_rows


def validate_score_coverage(scores: list[dict[str, Any]], task_pack: TaskPack | None) -> None:
    if task_pack is None:
        return
    expected_ids = [
        (row.get("sample_id") or "").strip()
        for row in task_pack.test_rows
        if (row.get("sample_id") or "").strip()
    ]
    if not expected_ids:
        return
    seen_counts: dict[str, int] = defaultdict(int)
    unexpected_ids: set[str] = set()
    for row in scores:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        seen_counts[sample_id] += 1
        if sample_id not in expected_ids:
            unexpected_ids.add(sample_id)
    duplicate_ids = sorted(sample_id for sample_id, count in seen_counts.items() if count > 1)
    missing_ids = sorted(sample_id for sample_id in expected_ids if seen_counts.get(sample_id, 0) == 0)
    if duplicate_ids or missing_ids or unexpected_ids:
        raise ValueError(
            "scores.csv coverage mismatch: "
            f"missing={len(missing_ids)} duplicate={len(duplicate_ids)} unexpected={len(unexpected_ids)}"
        )
