"""Summary builders for preservation analysis outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from .schema import SUMMARY_FIELDNAMES, SUMMARY_NUMERIC_FIELDS


def _round_float(value: float | None, digits: int = 6) -> float | str:
    if value is None:
        return ""
    return round(float(value), digits)


def _safe_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class _GroupSummaryAccumulator:
    def __init__(self) -> None:
        self.rows = 0
        self.parents: set[str] = set()
        self.status_counts: Counter[str] = Counter()
        self.numeric_sums: dict[str, float] = defaultdict(float)
        self.numeric_counts: dict[str, int] = defaultdict(int)

    def update(self, row: Mapping[str, Any]) -> None:
        self.rows += 1
        parent_id = str(row.get("parent_id", "")).strip()
        if parent_id:
            self.parents.add(parent_id)
        self.status_counts[str(row.get("status", ""))] += 1
        for field in SUMMARY_NUMERIC_FIELDS:
            parsed = _safe_float(row.get(field, ""))
            if parsed is None:
                continue
            self.numeric_sums[field] += parsed
            self.numeric_counts[field] += 1

    def as_row(self, group_key: str, *, chain_family: str = "", language: str = "") -> dict[str, Any]:
        def averaged(field: str) -> float | None:
            count = self.numeric_counts.get(field, 0)
            if count <= 0:
                return None
            return self.numeric_sums[field] / count

        return {
            "group_key": group_key,
            "chain_family": chain_family,
            "language": language,
            "rows": self.rows,
            "parents": len(self.parents),
            "status_ok": int(self.status_counts.get("ok", 0)),
            "asr_rows": self.numeric_counts.get("child_wer", 0),
            "speaker_rows": self.numeric_counts.get("speaker_similarity", 0),
            "audio_rows": self.numeric_counts.get("duration_ratio_to_parent", 0),
            "parent_wer_mean": _round_float(averaged("parent_wer"), 6),
            "child_wer_mean": _round_float(averaged("child_wer"), 6),
            "wer_drift_mean": _round_float(averaged("wer_drift"), 6),
            "parent_cer_mean": _round_float(averaged("parent_cer"), 6),
            "child_cer_mean": _round_float(averaged("child_cer"), 6),
            "cer_drift_mean": _round_float(averaged("cer_drift"), 6),
            "speaker_similarity_mean": _round_float(averaged("speaker_similarity"), 6),
            "duration_ratio_to_parent_mean": _round_float(averaged("duration_ratio_to_parent"), 6),
            "duration_delta_sec_mean": _round_float(averaged("duration_delta_sec"), 6),
            "rms_dbfs_delta_mean": _round_float(averaged("rms_dbfs_delta"), 6),
            "rms_dbfs_delta_abs_mean": _round_float(averaged("rms_dbfs_delta_abs"), 6),
        }


class PreservationSummaryAccumulator:
    def __init__(self) -> None:
        self._overall = _GroupSummaryAccumulator()
        self._by_family: dict[str, _GroupSummaryAccumulator] = defaultdict(_GroupSummaryAccumulator)
        self._by_family_language: dict[tuple[str, str], _GroupSummaryAccumulator] = defaultdict(_GroupSummaryAccumulator)
        self._status_counts: Counter[str] = Counter()
        self._split_counts: Counter[str] = Counter()

    def update(self, row: Mapping[str, Any]) -> None:
        family = str(row.get("chain_family", ""))
        language = str(row.get("language", ""))
        split = str(row.get("split", ""))
        status = str(row.get("status", ""))

        self._overall.update(row)
        self._by_family[family].update(row)
        self._by_family_language[(family, language)].update(row)
        self._status_counts[status] += 1
        self._split_counts[split] += 1

    def update_many(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            self.update(row)

    def build_tables(self) -> dict[str, Any]:
        return {
            "overall": self._overall.as_row("overall"),
            "by_family": [
                accumulator.as_row(family, chain_family=family)
                for family, accumulator in sorted(self._by_family.items())
            ],
            "by_family_language": [
                accumulator.as_row(f"{family}::{language}", chain_family=family, language=language)
                for (family, language), accumulator in sorted(self._by_family_language.items())
            ],
            "status_counts": dict(self._status_counts),
            "split_counts": dict(self._split_counts),
        }


def build_summary_tables(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    accumulator = PreservationSummaryAccumulator()
    accumulator.update_many(results)
    return accumulator.build_tables()


def build_summary_payload_from_tables(
    tables: Mapping[str, Any],
    *,
    metadata_path: str,
    requested_splits: list[str],
    asr_backend: str,
    speaker_backend: str,
) -> dict[str, Any]:
    total_rows = int(tables.get("overall", {}).get("rows", 0))
    overall = dict(tables.get("overall", {}))
    return {
        "metadata_path": metadata_path,
        "requested_splits": requested_splits,
        "backends": {
            "asr": asr_backend,
            "speaker": speaker_backend,
        },
        "coverage": {
            "rows": total_rows,
            "status_ok": overall["status_ok"],
            "ok_fraction": _round_float((overall["status_ok"] / total_rows) if total_rows else None, 6),
            "asr_rows": overall["asr_rows"],
            "speaker_rows": overall["speaker_rows"],
            "audio_rows": overall["audio_rows"],
            "status_counts": dict(tables.get("status_counts", {})),
        },
        "overall": overall,
        "by_family": list(tables.get("by_family", [])),
        "by_family_language": list(tables.get("by_family_language", [])),
        "split_counts": dict(tables.get("split_counts", {})),
    }


def build_summary_payload(
    results: list[Mapping[str, Any]],
    *,
    metadata_path: str,
    requested_splits: list[str],
    asr_backend: str,
    speaker_backend: str,
) -> dict[str, Any]:
    return build_summary_payload_from_tables(
        build_summary_tables(results),
        metadata_path=metadata_path,
        requested_splits=requested_splits,
        asr_backend=asr_backend,
        speaker_backend=speaker_backend,
    )


__all__ = [
    "PreservationSummaryAccumulator",
    "SUMMARY_FIELDNAMES",
    "build_summary_payload",
    "build_summary_payload_from_tables",
    "build_summary_tables",
]
