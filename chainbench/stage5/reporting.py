"""Reporting and metadata export helpers for Stage 5."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from chainbench.lib.config import relative_to_workspace
from chainbench.lib.io import write_csv
from chainbench.lib.structural_metadata import (
    OPERATOR_MULTISET_FIELD,
    OPERATOR_SUBSTITUTION_DETAIL_FIELD,
    OPERATOR_SUBSTITUTION_GROUP_FIELD,
    ORDER_SWAP_GROUP_FIELD,
    PARAMETER_PERTURBATION_AXIS_FIELD,
    PARAMETER_PERTURBATION_GROUP_FIELD,
    PATH_ENDPOINT_FIELD,
    PATH_GROUP_FIELD,
    PATH_STEP_FIELD,
)
from chainbench.lib.summary import utc_now_iso

DEFAULT_COVERAGE_CHAIN_FAMILIES = ["direct", "platform_like", "telephony", "simreplay", "hybrid"]
STANDARD_SPLIT_EXPORT_FIELD = "split_standard"
RELEASE_METADATA_FIELD_ORDER = [
    "sample_id",
    "parent_id",
    "file_name",
    "clean_parent_path",
    "trace_path",
    "label",
    STANDARD_SPLIT_EXPORT_FIELD,
    OPERATOR_SUBSTITUTION_GROUP_FIELD,
    OPERATOR_SUBSTITUTION_DETAIL_FIELD,
    PARAMETER_PERTURBATION_GROUP_FIELD,
    PARAMETER_PERTURBATION_AXIS_FIELD,
    ORDER_SWAP_GROUP_FIELD,
    PATH_GROUP_FIELD,
    "language",
    "speaker_id",
    "source_speaker_id",
    "utterance_id",
    "transcript",
    "raw_transcript",
    "source_corpus",
    "license_tag",
    "generator_family",
    "generator_name",
    "chain_family",
    "chain_template_id",
    "chain_variant_index",
    "operator_seq",
    OPERATOR_MULTISET_FIELD,
    PATH_ENDPOINT_FIELD,
    PATH_STEP_FIELD,
    "operator_params",
    "codec",
    "bitrate",
    "packet_loss",
    "bandwidth_mode",
    "snr",
    "rt60",
    "rir_backend",
    "room_dim",
    "distance",
    "seed",
    "duration_sec",
    "sample_rate",
    "channels",
    "codec_name",
    "sample_fmt",
]


def write_stats_tables(
    rows: list[dict[str, Any]],
    stats_root: Path,
    workspace_root: Path,
) -> dict[str, str]:
    label_language_split_rows: list[dict[str, Any]] = []
    chain_family_rows: list[dict[str, Any]] = []
    generator_family_rows: list[dict[str, Any]] = []

    counts_by_label_lang_split: dict[tuple[str, str, str], int] = Counter()
    counts_by_chain_family_label: dict[tuple[str, str], int] = Counter()
    counts_by_generator_family_label: dict[tuple[str, str], int] = Counter()

    for row in rows:
        counts_by_label_lang_split[(row["label"], row["language"], row["split"])] += 1
        counts_by_chain_family_label[(row["chain_family"], row["label"])] += 1
        counts_by_generator_family_label[(row["generator_family"], row["label"])] += 1

    for (label, language, split), count in sorted(counts_by_label_lang_split.items()):
        label_language_split_rows.append(
            {
                "label": label,
                "language": language,
                "split": split,
                "samples": str(count),
            }
        )
    for (chain_family, label), count in sorted(counts_by_chain_family_label.items()):
        chain_family_rows.append(
            {
                "chain_family": chain_family,
                "label": label,
                "samples": str(count),
            }
        )
    for (generator_family, label), count in sorted(counts_by_generator_family_label.items()):
        generator_family_rows.append(
            {
                "generator_family": generator_family,
                "label": label,
                "samples": str(count),
            }
        )

    write_csv(stats_root / "stats_label_language_split.csv", label_language_split_rows)
    write_csv(stats_root / "stats_chain_family_label.csv", chain_family_rows)
    write_csv(stats_root / "stats_generator_family_label.csv", generator_family_rows)
    return {
        "label_language_split": relative_to_workspace(stats_root / "stats_label_language_split.csv", workspace_root),
        "chain_family_label": relative_to_workspace(stats_root / "stats_chain_family_label.csv", workspace_root),
        "generator_family_label": relative_to_workspace(stats_root / "stats_generator_family_label.csv", workspace_root),
    }


def summarize_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_id_counts = Counter(row["sample_id"] for row in rows)
    audio_path_counts = Counter(row["audio_path"] for row in rows)
    duplicate_sample_ids = {key: count for key, count in sample_id_counts.items() if count > 1}
    duplicate_audio_paths = {key: count for key, count in audio_path_counts.items() if count > 1}
    return {
        "duplicate_sample_id_count": len(duplicate_sample_ids),
        "duplicate_audio_path_count": len(duplicate_audio_paths),
    }


def build_release_metadata_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_rows: list[dict[str, Any]] = []
    for row in rows:
        metadata_row = {
            "sample_id": row.get("sample_id", ""),
            "parent_id": row.get("parent_id", ""),
            "file_name": row.get("audio_path", ""),
            "clean_parent_path": row.get("clean_parent_path", ""),
            "trace_path": row.get("trace_path", ""),
            "label": row.get("label", ""),
            STANDARD_SPLIT_EXPORT_FIELD: row.get("split", ""),
            OPERATOR_SUBSTITUTION_GROUP_FIELD: row.get(OPERATOR_SUBSTITUTION_GROUP_FIELD, ""),
            OPERATOR_SUBSTITUTION_DETAIL_FIELD: row.get(OPERATOR_SUBSTITUTION_DETAIL_FIELD, ""),
            PARAMETER_PERTURBATION_GROUP_FIELD: row.get(PARAMETER_PERTURBATION_GROUP_FIELD, ""),
            PARAMETER_PERTURBATION_AXIS_FIELD: row.get(PARAMETER_PERTURBATION_AXIS_FIELD, ""),
            ORDER_SWAP_GROUP_FIELD: row.get(ORDER_SWAP_GROUP_FIELD, ""),
            PATH_GROUP_FIELD: row.get(PATH_GROUP_FIELD, ""),
            "language": row.get("language", ""),
            "speaker_id": row.get("speaker_id", ""),
            "source_speaker_id": row.get("source_speaker_id", ""),
            "utterance_id": row.get("utterance_id", ""),
            "transcript": row.get("transcript", ""),
            "raw_transcript": row.get("raw_transcript", ""),
            "source_corpus": row.get("source_corpus", ""),
            "license_tag": row.get("license_tag", ""),
            "generator_family": row.get("generator_family", ""),
            "generator_name": row.get("generator_name", ""),
            "chain_family": row.get("chain_family", ""),
            "chain_template_id": row.get("chain_template_id", ""),
            "chain_variant_index": row.get("chain_variant_index", ""),
            "operator_seq": row.get("operator_seq", ""),
            OPERATOR_MULTISET_FIELD: row.get(OPERATOR_MULTISET_FIELD, ""),
            PATH_ENDPOINT_FIELD: row.get(PATH_ENDPOINT_FIELD, ""),
            PATH_STEP_FIELD: row.get(PATH_STEP_FIELD, ""),
            "operator_params": row.get("operator_params", ""),
            "codec": row.get("codec", ""),
            "bitrate": row.get("bitrate", ""),
            "packet_loss": row.get("packet_loss", ""),
            "bandwidth_mode": row.get("bandwidth_mode", ""),
            "snr": row.get("snr", ""),
            "rt60": row.get("rt60", ""),
            "rir_backend": row.get("rir_backend", ""),
            "room_dim": row.get("room_dim", ""),
            "distance": row.get("distance", ""),
            "seed": row.get("seed", ""),
            "duration_sec": row.get("duration_sec", ""),
            "sample_rate": row.get("sample_rate", ""),
            "channels": row.get("channels", ""),
            "codec_name": row.get("codec_name", ""),
            "sample_fmt": row.get("sample_fmt", ""),
        }
        metadata_rows.append({field: metadata_row.get(field, "") for field in RELEASE_METADATA_FIELD_ORDER})
    return metadata_rows


def build_split_release_metadata_rows(metadata_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for row in metadata_rows:
        split_name = str(row.get(STANDARD_SPLIT_EXPORT_FIELD, "")).strip()
        if not split_name:
            continue
        split_prefix = f"{split_name}/"
        file_name = str(row.get("file_name", "")).strip()
        if not file_name.startswith(split_prefix):
            raise ValueError(f"Expected file_name to start with {split_prefix!r}, got {file_name!r}")
        split_row = dict(row)
        split_row["file_name"] = file_name[len(split_prefix) :]
        split_rows.setdefault(split_name, []).append(split_row)
    return split_rows


def write_split_metadata_files(
    dataset_root: Path,
    metadata_rows: list[dict[str, Any]],
    workspace_root: Path,
) -> dict[str, str]:
    split_metadata_rows = build_split_release_metadata_rows(metadata_rows)
    split_metadata_paths: dict[str, str] = {}
    for split_name, rows in split_metadata_rows.items():
        split_metadata_path = dataset_root / split_name / "metadata.csv"
        write_csv(split_metadata_path, rows, fieldnames=RELEASE_METADATA_FIELD_ORDER)
        split_metadata_paths[split_name] = relative_to_workspace(split_metadata_path, workspace_root)
    return split_metadata_paths


def resolve_coverage_families(config: dict[str, Any]) -> list[str]:
    return list(config.get("coverage_chain_families", DEFAULT_COVERAGE_CHAIN_FAMILIES))


def build_stage5_summary(
    *,
    config: dict[str, Any],
    config_path: Path,
    input_manifest_path: Path,
    output_root: Path,
    metadata_path: Path,
    failures_path: Path,
    workspace_root: Path,
    validation_enabled: bool,
    input_rows: int,
    annotated_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    counts: Counter,
    split_metadata_paths: dict[str, str],
    duplicate_checks: dict[str, Any],
    speaker_disjoint_check: dict[str, Any],
    counterfactual_parent_coverage: dict[str, Any],
    validation_stats: dict[str, Any],
    stats_tables: dict[str, str],
) -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now_iso(),
        "dataset_name": str(config.get("dataset_name", "ChainBench")),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "input_manifest": relative_to_workspace(input_manifest_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "validation_enabled": validation_enabled,
        "input_rows": input_rows,
        "dataset_rows": len(annotated_rows),
        "failed_rows": len(failures),
        "metadata_path": relative_to_workspace(metadata_path, workspace_root),
        "split_metadata_paths": split_metadata_paths,
        "failures_path": relative_to_workspace(failures_path, workspace_root),
        "status_counts": dict(counts),
        "speaker_disjoint_check": speaker_disjoint_check,
        "counterfactual_parent_coverage": counterfactual_parent_coverage,
        "duplicate_checks": duplicate_checks,
        "stats": validation_stats,
        "stats_tables": stats_tables,
    }
