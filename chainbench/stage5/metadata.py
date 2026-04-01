"""Metadata annotation and validation helpers for Stage 5 outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from chainbench.lib.structural_metadata import annotate_structural_group_fields


def annotate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return annotate_structural_group_fields(rows)


def check_speaker_disjoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    speakers_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        speakers_by_split[str(row["split"])].add(str(row["speaker_id"]))
    overlaps: dict[str, int] = {}
    split_names = sorted(speakers_by_split)
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            overlap = speakers_by_split[left] & speakers_by_split[right]
            overlaps[f"{left}__{right}"] = len(overlap)
    return {
        "splits": {split: len(speakers) for split, speakers in speakers_by_split.items()},
        "pairwise_overlap_counts": overlaps,
        "speaker_disjoint": all(count == 0 for count in overlaps.values()),
    }


def summarize_parent_coverage(rows: list[dict[str, Any]], required_families: list[str]) -> dict[str, Any]:
    families_by_parent: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        families_by_parent[str(row["parent_id"])].add(str(row["chain_family"]))
    coverage_counts = Counter()
    required = set(required_families)
    for families in families_by_parent.values():
        missing = required - families
        coverage_counts["parents_total"] += 1
        if not missing:
            coverage_counts["parents_with_required_families"] += 1
        else:
            coverage_counts["parents_missing_required_families"] += 1
    return dict(coverage_counts)
