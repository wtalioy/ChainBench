"""Top-level orchestration for eval CLI runs."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

from chainbench.lib import runtime_snapshot
from chainbench.lib.conda import require_conda_envs
from chainbench.lib.config import default_workspace_root, relative_to_workspace, resolve_path
from chainbench.lib.io import load_csv_rows
from chainbench.lib.logging import get_logger
from chainbench.lib.summary import utc_now_iso, write_timestamped_json

from .config import load_eval_config
from .holdout_protocols import LEAVE_ONE_TEMPLATE_OUT_PROTOCOL, PER_FAMILY_SCOPE
from .pipeline.run import run_all_baselines
from .tasks import TaskPack, build_task_packs

LOGGER = get_logger("eval")


def apply_smoke_mode(rows: list[dict[str, Any]], config: dict[str, Any], smoke_limit: int = 500) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        split = str(row.get("split_standard", "")).strip() or "train"
        by_split.setdefault(split, []).append(row)

    per_split = max(1, smoke_limit // 3)
    smoke_rows = [row for split in ("train", "dev", "test") for row in by_split.get(split, [])[:per_split]]
    for baseline_cfg in config["baselines"].values():
        baseline_cfg["train"]["epochs"] = min(baseline_cfg["train"]["epochs"], 2)
    config["smoke_limits"] = (100, 50, 50)
    return smoke_rows


def validate_args(args: argparse.Namespace) -> str | None:
    if args.eval_only and args.train_only:
        return "--eval-only and --train-only cannot be used together"
    if args.eval_only and args.force_retrain:
        return "--force-retrain cannot be used with --eval-only"
    if args.sample_ratio is not None and not 0.0 < args.sample_ratio <= 1.0:
        return "--sample-ratio must be in the interval (0, 1]"
    return None


def generalization_override_from_args(args: argparse.Namespace) -> dict[str, str] | None:
    if not args.template_holdout:
        return None
    return {
        "protocol": LEAVE_ONE_TEMPLATE_OUT_PROTOCOL,
        "scope": PER_FAMILY_SCOPE,
    }


def load_config_from_args(args: argparse.Namespace, workspace_root: Path) -> tuple[Path, dict[str, Any] | None]:
    config_path = resolve_path(args.config, workspace_root)
    if not config_path.exists():
        LOGGER.error("Config not found: %s", config_path)
        return config_path, None

    try:
        config = load_eval_config(
            config_path,
            workspace_root,
            tasks_override=args.tasks,
            baselines_override=args.baselines,
            generalization_override=generalization_override_from_args(args),
            output_root_override=args.output_root,
        )
    except ValueError as exc:
        LOGGER.error(str(exc))
        return config_path, None

    if args.sample_ratio is not None:
        config["sample_ratio"] = args.sample_ratio
    return config_path, config


def build_task_pack_config(config: dict[str, Any]) -> dict[str, Any]:
    pack_config: dict[str, Any] = {}
    if config.get("sample_ratio") is not None:
        pack_config["sample_ratio"] = config["sample_ratio"]
    if config.get("generalization") is not None:
        pack_config["generalization"] = config["generalization"]
    if "smoke_limits" in config:
        pack_config["smoke_limits"] = config["smoke_limits"]
    return pack_config


def build_summary(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    config_path: Path,
    metadata_path: Path,
    output_root: Path,
    workspace_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now_iso(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "metadata_path": relative_to_workspace(metadata_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "eval_only": args.eval_only,
        "train_only": args.train_only,
        "dry_run": args.dry_run,
        "smoke": args.smoke,
        "sample_ratio": config.get("sample_ratio"),
        "generalization": config.get("generalization"),
        "force_retrain": args.force_retrain,
        "tasks": config["tasks"],
        "baselines": list(config["baselines"].keys()),
        "metadata_rows": len(rows),
        "baseline_runs": [],
        "baseline_metrics": [],
    }


def write_summary_snapshot(path: Path, summary: dict[str, Any]) -> None:
    write_timestamped_json(path, summary)


def configured_devices(config: dict[str, Any]) -> list[str]:
    devices: set[str] = set()
    for baseline_cfg in config.get("baselines", {}).values():
        for phase_name in ("train", "eval"):
            phase_cfg = baseline_cfg.get(phase_name, {})
            for device in phase_cfg.get("devices", []):
                device_text = str(device).strip()
                if device_text:
                    devices.add(device_text)
    return sorted(devices)


def preflight_baseline_envs(config: dict[str, Any], *, skip_env_check: bool) -> None:
    if skip_env_check:
        return
    require_conda_envs(
        [baseline_cfg["conda_env"] for baseline_cfg in config.get("baselines", {}).values()],
        label="baseline conda envs",
    )


def log_runtime_snapshot(label: str, *, devices: list[str]) -> None:
    LOGGER.info("runtime snapshot: %s", json.dumps(runtime_snapshot(label, devices=devices), sort_keys=True))


def build_task_packs_with_logging(config: dict[str, Any], rows: list[dict[str, Any]]) -> list[TaskPack]:
    packs = build_task_packs(rows, config["tasks"], build_task_pack_config(config))
    LOGGER.info("effective sample_ratio=%s", config.get("sample_ratio"))
    for pack in packs:
        LOGGER.info(
            "task pack %s/%s: train=%d dev=%d test=%d",
            pack.task_id,
            pack.variant,
            len(pack.train_rows),
            len(pack.dev_rows),
            len(pack.test_rows),
        )
    return packs


def attach_task_pack_counts(summary: dict[str, Any], packs: list[TaskPack]) -> None:
    task_packs: list[dict[str, Any]] = []
    totals = {"train": 0, "dev": 0, "test": 0}
    for pack in packs:
        train_count = len(pack.train_rows)
        dev_count = len(pack.dev_rows)
        test_count = len(pack.test_rows)
        totals["train"] += train_count
        totals["dev"] += dev_count
        totals["test"] += test_count
        pack_summary = {
            "task_id": pack.task_id,
            "variant": pack.variant,
            "train_samples": train_count,
            "dev_samples": dev_count,
            "test_samples": test_count,
        }
        generalization_meta = {
            "protocol": str(pack.meta.get("generalization_protocol", "")).strip(),
            "scope": str(pack.meta.get("generalization_scope", "")).strip(),
            "held_out_chain_family": str(pack.meta.get("held_out_chain_family", "")).strip(),
            "held_out_template_id": str(pack.meta.get("held_out_template_id", "")).strip(),
        }
        generalization_meta = {key: value for key, value in generalization_meta.items() if value}
        if generalization_meta:
            pack_summary["generalization"] = generalization_meta
        task_packs.append(pack_summary)
    summary["task_packs"] = task_packs
    summary["task_pack_totals"] = {
        **totals,
        "all_splits": totals["train"] + totals["dev"] + totals["test"],
    }


def finalize_task_pack_summary(summary: dict[str, Any], packs: list[TaskPack]) -> None:
    summary["task_packs_built"] = len(packs)
    if packs:
        summary["status"] = "ready"
        return
    summary["status"] = "no_packs"
    summary["message"] = "No task packs produced (e.g. no data for selected tasks)."


def persist_final_summary(output_root: Path, summary: dict[str, Any], start_time: float) -> Path:
    summary_path = output_root / f"eval_summary_{start_time}.json"
    write_summary_snapshot(summary_path, summary)
    return summary_path


def run_eval_from_args(args: argparse.Namespace, *, workspace_root: Path | None = None) -> int:
    start_time = time.time()
    workspace_root = workspace_root or default_workspace_root()
    error = validate_args(args)
    if error is not None:
        LOGGER.error(error)
        return 1

    config_path, config = load_config_from_args(args, workspace_root)
    if config is None:
        return 1

    preflight_baseline_envs(config, skip_env_check=args.dry_run)

    metadata_path = Path(config["metadata_path"])
    output_root = Path(config["output_root"])

    if not metadata_path.exists():
        LOGGER.error("Metadata not found: %s", metadata_path)
        return 1

    execution_devices = configured_devices(config)
    log_runtime_snapshot("before_metadata_load", devices=execution_devices)
    rows = load_csv_rows(metadata_path)
    log_runtime_snapshot("after_metadata_load", devices=execution_devices)
    LOGGER.info("loaded %d rows from %s", len(rows), metadata_path)

    if args.smoke:
        rows = apply_smoke_mode(rows, config)
        LOGGER.info("smoke mode: sampled %d rows across splits (train/dev/test)", len(rows))

    dataset_root = Path(config["dataset_root"])
    baselines = list(config["baselines"].keys())
    summary = build_summary(
        args=args,
        config=config,
        config_path=config_path,
        metadata_path=metadata_path,
        output_root=output_root,
        workspace_root=workspace_root,
        rows=rows,
    )
    packs = build_task_packs_with_logging(config, rows)
    log_runtime_snapshot("after_task_pack_build", devices=execution_devices)
    rows = []
    gc.collect()
    log_runtime_snapshot("after_metadata_release", devices=execution_devices)
    finalize_task_pack_summary(summary, packs)
    attach_task_pack_counts(summary, packs)

    if args.dry_run:
        summary["message"] = "Dry run only: task packs built and baseline execution skipped."
        LOGGER.info("dry run requested: skipping baseline execution")
        LOGGER.info("summary:\n%s", json.dumps(summary, ensure_ascii=False, indent=2))
        LOGGER.info("time taken: %s", time.time() - start_time)
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    summary_latest_path = output_root / "eval_summary_latest.json"
    write_summary_snapshot(summary_latest_path, summary)

    if summary["baselines"] and packs:

        def persist_baseline_snapshot(
            baseline_results: list[dict[str, Any]],
            baseline_metrics: list[dict[str, Any]],
        ) -> None:
            summary["baseline_runs"] = baseline_results
            summary["baseline_metrics"] = baseline_metrics
            write_summary_snapshot(summary_latest_path, summary)
            log_runtime_snapshot(
                f"baseline_snapshot_completed_{len(baseline_results)}",
                devices=execution_devices,
            )

        baseline_results, baseline_metrics, baseline_error = run_all_baselines(
            output_root=output_root,
            dataset_root=dataset_root,
            packs=packs,
            baseline_names=baselines,
            baseline_configs=config["baselines"],
            eval_only=args.eval_only,
            train_only=args.train_only,
            force_retrain=args.force_retrain,
            on_snapshot=persist_baseline_snapshot,
        )
        if baseline_error:
            summary["baseline_error"] = baseline_error
        summary["baseline_runs"] = baseline_results
        summary["baseline_metrics"] = baseline_metrics
    else:
        summary["baseline_runs"] = []

    write_summary_snapshot(summary_latest_path, summary)
    summary_path = persist_final_summary(output_root, summary, start_time)
    log_runtime_snapshot("after_eval_pipeline", devices=execution_devices)
    LOGGER.info("updated live summary %s", summary_latest_path)
    LOGGER.info("wrote %s", summary_path)
    LOGGER.info("summary:\n%s", json.dumps(summary, ensure_ascii=False, indent=2))
    LOGGER.info("time taken: %s", time.time() - start_time)
    return 0
