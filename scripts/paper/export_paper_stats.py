"""Compute publication-ready dataset statistics from release metadata."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from eval.tasks.task_keys import parse_operator_seq
from lib.chain_families import CHAIN_FAMILY_MACRO_NAMES

GROUP_FIELDS = (
    "operator_substitution_group_id",
    "parameter_perturbation_group_id",
    "order_swap_group_id",
    "path_group_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        default="data/ChainBench/manifest/metadata.csv",
        help="Path to the packaged dataset metadata CSV.",
    )
    parser.add_argument(
        "--output-json",
        default="stat.json",
        help="Optional path to write the computed statistics as JSON.",
    )
    parser.add_argument(
        "--output-tex",
        default="",
        help="Optional path to write LaTeX macros for the main statistics.",
    )
    parser.add_argument(
        "--dataset-name",
        default="ChainSpeech",
        help="Paper-facing dataset name used in the LaTeX output.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def _dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def compute_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = Counter()
    languages = Counter()
    splits = Counter()
    chain_families = Counter()
    chain_family_label = Counter()
    generator_families = Counter()
    generator_names = Counter()
    chain_lengths = Counter()
    operator_counts = Counter()

    unique_parents: set[str] = set()
    unique_templates: set[tuple[str, str]] = set()
    unique_orders: set[str] = set()
    unique_group_ids: dict[str, set[str]] = {field: set() for field in GROUP_FIELDS}
    unique_speakers: set[tuple[str, str]] = set()
    unique_source_speakers: set[tuple[str, str]] = set()
    split_speakers: dict[str, set[str]] = defaultdict(set)
    language_parents: dict[str, set[str]] = defaultdict(set)
    durations: list[float] = []

    label_language_split = Counter()

    for row in rows:
        label = str(row.get("label", "")).strip()
        language = str(row.get("language", "")).strip()
        split = str(row.get("split_standard", "")).strip()
        chain_family = str(row.get("chain_family", "")).strip()
        chain_template_id = str(row.get("chain_template_id", "")).strip()
        generator_family = str(row.get("generator_family", "")).strip() or "none"
        generator_name = str(row.get("generator_name", "")).strip() or "none"
        parent_id = str(row.get("parent_id", "")).strip()
        speaker_id = str(row.get("speaker_id", "")).strip()
        source_speaker_id = str(row.get("source_speaker_id", "")).strip()
        labels[label] += 1
        languages[language] += 1
        splits[split] += 1
        chain_families[chain_family] += 1
        chain_family_label[(chain_family, label)] += 1
        generator_families[generator_family] += 1
        generator_names[generator_name] += 1
        label_language_split[(label, language, split)] += 1

        unique_parents.add(parent_id)
        unique_templates.add((chain_family, chain_template_id))
        for group_field in GROUP_FIELDS:
            group_value = str(row.get(group_field, "")).strip()
            if group_value:
                unique_group_ids[group_field].add(group_value)
        if speaker_id:
            unique_speakers.add((language, speaker_id))
            split_speakers[split].add(f"{language}::{speaker_id}")
        if source_speaker_id:
            unique_source_speakers.add((language, source_speaker_id))
        if parent_id:
            language_parents[language].add(parent_id)

        operator_seq = parse_operator_seq(str(row.get("operator_seq", "")))
        chain_lengths[len(operator_seq)] += 1
        unique_orders.add(json.dumps(operator_seq or [], separators=(",", ":")))
        for operator_name in operator_seq:
            operator_counts[operator_name] += 1

        duration = _safe_float(str(row.get("duration_sec", "")))
        if duration is not None:
            durations.append(duration)

    chain_family_table: dict[str, dict[str, int]] = {}
    for family in sorted(chain_families):
        family_counts = {
            "total": int(chain_families[family]),
        }
        for label in sorted(labels):
            family_counts[label] = int(chain_family_label[(family, label)])
        chain_family_table[family] = family_counts

    label_language_split_table: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (label, language, split), count in sorted(label_language_split.items()):
        label_language_split_table[label][language][split] = int(count)

    return {
        "totals": {
            "samples": len(rows),
            "parents": len(unique_parents),
            "speakers": len(unique_speakers),
            "source_speakers": len(unique_source_speakers),
            "languages": len({key for key, value in languages.items() if value > 0}),
            "chain_families": len({key for key, value in chain_families.items() if value > 0}),
            "chain_templates": len(unique_templates),
            "operator_orders": len(unique_orders),
            "operator_substitution_groups": len(unique_group_ids["operator_substitution_group_id"]),
            "parameter_perturbation_groups": len(unique_group_ids["parameter_perturbation_group_id"]),
            "order_swap_groups": len(unique_group_ids["order_swap_group_id"]),
            "path_groups": len(unique_group_ids["path_group_id"]),
            "generator_families": len(
                {
                    key
                    for key, value in generator_families.items()
                    if value > 0 and key != "none"
                }
            ),
            "generator_models": len({key for key, value in generator_names.items() if value > 0 and key != "none"}),
        },
        "labels": _dict(labels),
        "languages": {
            language: {
                "samples": int(count),
                "parents": len(language_parents.get(language, set())),
            }
            for language, count in languages.items()
        },
        "splits": {
            split: {
                "samples": int(count),
                "speakers": len(split_speakers.get(split, set())),
            }
            for split, count in splits.items()
        },
        "chain_families": chain_family_table,
        "generator_families": _dict(generator_families),
        "generator_models": _dict(generator_names),
        "label_language_split": label_language_split_table,
        "chain_lengths": {str(length): int(count) for length, count in sorted(chain_lengths.items())},
        "operator_counts": _dict(operator_counts),
        "duration_seconds": _duration_summary(durations),
    }


def _format_int(value: int) -> str:
    return f"{int(value):,}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_tex(path: Path, dataset_name: str, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = stats["totals"]
    splits = stats["splits"]
    labels = stats["labels"]
    languages = stats["languages"]
    chain_families = stats["chain_families"]
    duration = stats["duration_seconds"]

    lines = [
        "% Auto-generated by scripts/paper/export_paper_stats.py",
        f"\\providecommand{{\\ChainDatasetName}}{{{dataset_name}}}",
        f"\\providecommand{{\\ChainDatasetTotalSamples}}{{{_format_int(totals['samples'])}}}",
        f"\\providecommand{{\\ChainDatasetTotalParents}}{{{_format_int(totals['parents'])}}}",
        f"\\providecommand{{\\ChainDatasetTotalSpeakers}}{{{_format_int(totals['speakers'])}}}",
        f"\\providecommand{{\\ChainDatasetChainFamilies}}{{{_format_int(totals['chain_families'])}}}",
        f"\\providecommand{{\\ChainDatasetChainTemplates}}{{{_format_int(totals['chain_templates'])}}}",
        f"\\providecommand{{\\ChainDatasetOperatorOrders}}{{{_format_int(totals['operator_orders'])}}}",
        f"\\providecommand{{\\ChainDatasetGeneratorFamilies}}{{{_format_int(totals['generator_families'])}}}",
        f"\\providecommand{{\\ChainDatasetGeneratorModels}}{{{_format_int(totals['generator_models'])}}}",
        f"\\providecommand{{\\ChainDatasetBonaFide}}{{{_format_int(labels.get('bona_fide', 0))}}}",
        f"\\providecommand{{\\ChainDatasetSpoof}}{{{_format_int(labels.get('spoof', 0))}}}",
        f"\\providecommand{{\\ChainDatasetTrain}}{{{_format_int(splits.get('train', {}).get('samples', 0))}}}",
        f"\\providecommand{{\\ChainDatasetDev}}{{{_format_int(splits.get('dev', {}).get('samples', 0))}}}",
        f"\\providecommand{{\\ChainDatasetTest}}{{{_format_int(splits.get('test', {}).get('samples', 0))}}}",
        f"\\providecommand{{\\ChainDatasetTrainSpeakers}}{{{_format_int(splits.get('train', {}).get('speakers', 0))}}}",
        f"\\providecommand{{\\ChainDatasetDevSpeakers}}{{{_format_int(splits.get('dev', {}).get('speakers', 0))}}}",
        f"\\providecommand{{\\ChainDatasetTestSpeakers}}{{{_format_int(splits.get('test', {}).get('speakers', 0))}}}",
        f"\\providecommand{{\\ChainDatasetEnglishSamples}}{{{_format_int(languages.get('en', {}).get('samples', 0))}}}",
        f"\\providecommand{{\\ChainDatasetChineseSamples}}{{{_format_int(languages.get('zh', {}).get('samples', 0))}}}",
        f"\\providecommand{{\\ChainDatasetEnglishParents}}{{{_format_int(languages.get('en', {}).get('parents', 0))}}}",
        f"\\providecommand{{\\ChainDatasetChineseParents}}{{{_format_int(languages.get('zh', {}).get('parents', 0))}}}",
        f"\\providecommand{{\\ChainDatasetDurationMean}}{{{_format_float(duration.get('mean', 0.0))}}}",
        f"\\providecommand{{\\ChainDatasetDurationMedian}}{{{_format_float(duration.get('median', 0.0))}}}",
        f"\\providecommand{{\\ChainDatasetDurationMin}}{{{_format_float(duration.get('min', 0.0))}}}",
        f"\\providecommand{{\\ChainDatasetDurationMax}}{{{_format_float(duration.get('max', 0.0))}}}",
    ]

    for family, macro_suffix in CHAIN_FAMILY_MACRO_NAMES.items():
        family_stats = chain_families.get(family, {})
        lines.extend(
            [
                f"\\providecommand{{\\ChainDataset{macro_suffix}Total}}{{{_format_int(family_stats.get('total', 0))}}}",
                f"\\providecommand{{\\ChainDataset{macro_suffix}BonaFide}}{{{_format_int(family_stats.get('bona_fide', 0))}}}",
                f"\\providecommand{{\\ChainDataset{macro_suffix}Spoof}}{{{_format_int(family_stats.get('spoof', 0))}}}",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    rows = load_rows(metadata_path)
    stats = compute_stats(rows)

    payload = {
        "dataset_name": args.dataset_name,
        "metadata_path": str(metadata_path),
        "stats": stats,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.output_json:
        write_json(Path(args.output_json), payload)
    if args.output_tex:
        write_tex(Path(args.output_tex), args.dataset_name, stats)


if __name__ == "__main__":
    main()
