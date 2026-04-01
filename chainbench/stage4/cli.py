"""Stage 4 CLI: delivery-chain rendering."""

from __future__ import annotations

import argparse
import sys

from chainbench.lib.cli import (
    add_language_filter_argument,
    add_limit_argument,
    add_log_level_argument,
    load_rows_with_filters,
    resolve_config_argument,
    resolve_worker_count,
)
from chainbench.lib.config import default_workspace_root, load_json, relative_to_workspace, resolve_path
from chainbench.lib.io import write_csv, write_json
from chainbench.lib.logging import get_logger, setup_logging
from chainbench.lib.summary import print_json, utc_now_iso

from .chains import count_jobs, iter_sample_jobs
from .execution import render_stage4_jobs
from .render import summarize_manifest

LOGGER = get_logger("stage4")
CHAIN_FAMILY_CHOICES = ("direct", "platform_like", "telephony", "simreplay", "hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage4.json",
        help="Path to the Stage-4 config JSON.",
    )
    add_log_level_argument(parser)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=CHAIN_FAMILY_CHOICES,
        help="Restrict rendering to selected chain families.",
    )
    add_language_filter_argument(parser, help_text="Restrict processing to one or more languages.")
    add_limit_argument(parser, help_text="Optionally process only the first N clean parents.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Override worker count from config.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Sample/render plan but do not actually create delivered children.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Update progress display every N finished jobs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    workspace_root = default_workspace_root()
    config_path = resolve_config_argument(args.config, workspace_root)
    config = load_json(config_path)
    output_root = resolve_path(config["output_root"], workspace_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_manifest_path = resolve_path(config["input_manifest"], workspace_root)
    rows = load_rows_with_filters(
        input_manifest_path,
        logger=LOGGER,
        row_label="clean parents",
        empty_error="No clean-parent rows selected for Stage-4 processing",
        languages=args.language,
        limit=args.limit,
    )

    selected_families = args.families or [
        name for name, family_cfg in config["families"].items() if family_cfg.get("enabled", False)
    ]
    LOGGER.info("selected chain families: %s", ", ".join(selected_families))

    jobs_per_family = count_jobs(rows, config, selected_families)
    jobs_total = sum(jobs_per_family.values())
    jobs_dir = output_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    plan_path = jobs_dir / "stage4_job_plan.json"
    write_json(
        plan_path,
        {
            "generated_at_utc": utc_now_iso(),
            "config_path": relative_to_workspace(config_path, workspace_root),
            "input_manifest": relative_to_workspace(input_manifest_path, workspace_root),
            "input_clean_parents": len(rows),
            "selected_families": selected_families,
            "jobs_total": jobs_total,
            "jobs_per_family": dict(jobs_per_family),
        },
    )
    if args.plan_only:
        LOGGER.info("plan-only mode: wrote job plan to %s", plan_path)
        return 0

    workers = resolve_worker_count(args.workers, config)
    LOGGER.info("using %d worker(s)", workers)
    manifest_rows, failures, counts = render_stage4_jobs(
        iter_sample_jobs(rows, config, selected_families, workspace_root),
        jobs_total,
        config=config,
        workspace_root=workspace_root,
        output_root=output_root,
        workers=workers,
        log_every=args.log_every,
    )
    if not manifest_rows:
        raise RuntimeError("Stage-4 produced zero delivered samples")

    manifest_rows.sort(
        key=lambda row: (
            row["chain_family"],
            row["language"],
            row["split"],
            row["speaker_id"],
            row["sample_id"],
        )
    )
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    write_csv(manifest_root / "delivered_manifest.csv", manifest_rows)
    for family_name in selected_families:
        subset = [row for row in manifest_rows if row["chain_family"] == family_name]
        if subset:
            write_csv(manifest_root / f"delivered_manifest_{family_name}.csv", subset)
    for language in sorted({row["language"] for row in manifest_rows}):
        subset = [row for row in manifest_rows if row["language"] == language]
        write_csv(manifest_root / f"delivered_manifest_{language}.csv", subset)

    failures_path = manifest_root / "stage4_failures.json"
    write_json(failures_path, failures)
    summary = {
        "generated_at_utc": utc_now_iso(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "input_manifest": relative_to_workspace(input_manifest_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "input_clean_parents": len(rows),
        "jobs_total": jobs_total,
        "delivered_samples": len(manifest_rows),
        "failed_jobs": len(failures),
        "status_counts": dict(counts),
        "stats": summarize_manifest(manifest_rows),
    }
    summary_path = manifest_root / "stage4_summary.json"
    write_json(summary_path, summary)

    if failures and not bool(config.get("allow_partial_failures", True)):
        raise RuntimeError(f"Stage-4 had {len(failures)} failures and allow_partial_failures=false")

    LOGGER.info(
        "Stage-4 finished: delivered_samples=%d, failed_jobs=%d, manifests=%s",
        len(manifest_rows),
        len(failures),
        manifest_root,
    )
    print_json(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
