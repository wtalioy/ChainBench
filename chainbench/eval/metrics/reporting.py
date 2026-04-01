"""Pipeline-facing metric aggregation and artifact helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from chainbench.lib.config import load_json
from chainbench.lib.io import write_json

from ..tasks import TaskPack
from .core import build_label_map
from .compute import compute_metrics_for_scores


def task_label_maps(
    packs: list[TaskPack],
    *,
    build_label_map_fn: Callable[[list[dict[str, Any]]], dict[str, str]] = build_label_map,
) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (pack.task_id, pack.variant): build_label_map_fn(pack.test_rows)
        for pack in packs
    }


def task_pack_lookup(packs: list[TaskPack]) -> dict[tuple[str, str], TaskPack]:
    return {
        (pack.task_id, pack.variant): pack
        for pack in packs
    }


def compute_baseline_metrics(
    output_root: Path,
    packs: list[TaskPack],
    baseline_results: list[dict[str, Any]],
    *,
    aggregate_metrics_fn: Callable[..., list[dict[str, Any]]],
    build_label_map_fn: Callable[[list[dict[str, Any]]], dict[str, str]] = build_label_map,
) -> list[dict[str, Any]]:
    return aggregate_metrics_fn(
        output_root,
        baseline_results,
        task_label_maps(packs, build_label_map_fn=build_label_map_fn),
        task_pack_lookup(packs),
    )


def write_metrics_files(output_root: Path, baseline_metrics: list[dict[str, Any]]) -> None:
    for run_meta in baseline_metrics:
        if run_meta.get("metrics"):
            run_dir = output_root / run_meta["task_id"] / run_meta["variant"] / run_meta["baseline"]
            write_json(run_dir / "metrics.json", run_meta["metrics"])


def aggregate_run_metrics(
    output_root: Path,
    baseline_results: list[dict[str, Any]],
    label_maps: dict[tuple[str, str], dict[str, str]] | None = None,
    task_packs: dict[tuple[str, str], TaskPack] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in baseline_results:
        scores_path = run.get("scores_path")
        if not scores_path:
            out.append({**run, "metrics": None})
            continue
        path = Path(scores_path)
        if not path.is_absolute():
            path = output_root / path
        if not path.exists():
            out.append({**run, "metrics": None})
            continue
        metrics_path = output_root / run["task_id"] / run["variant"] / run["baseline"] / "metrics.json"
        if metrics_path.exists() and metrics_path.stat().st_mtime >= path.stat().st_mtime:
            out.append({**run, "metrics": load_json(metrics_path)})
            continue
        task_key = (run["task_id"], run["variant"])
        label_map = (label_maps or {}).get(task_key)
        task_pack = (task_packs or {}).get(task_key)
        out.append(
            {
                **run,
                "metrics": compute_metrics_for_scores(path, label_map=label_map, task_pack=task_pack),
            }
        )

    macro_entries: list[dict[str, Any]] = []
    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in out:
        if run.get("task_id") == "in_chain_detection" and isinstance(run.get("metrics"), dict):
            by_baseline[str(run.get("baseline", ""))].append(run)
    for baseline, runs in sorted(by_baseline.items()):
        eers = [float(run["metrics"]["eer"]) for run in runs if run.get("metrics") is not None]
        aucs = [float(run["metrics"]["auc"]) for run in runs if run.get("metrics") is not None]
        if not eers or not aucs:
            continue
        macro_entries.append(
            {
                "task_id": "in_chain_detection",
                "variant": "macro_average",
                "baseline": baseline,
                "scores_path": "",
                "metrics": {
                    "eer": sum(eers) / len(eers),
                    "auc": sum(aucs) / len(aucs),
                    "n_families": len(eers),
                    "macro_source_variants": sorted(str(run.get("variant", "")) for run in runs),
                },
            }
        )
    out.extend(macro_entries)
    return out
