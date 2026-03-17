"""Stage 3 CLI: spoof clean-parent generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from tqdm.auto import tqdm

from lib.logging import get_logger, setup_logging, format_elapsed
from lib.config import load_json, relative_to_workspace, resolve_path
from lib.io import load_csv_rows, write_csv

from .collect import collect_spoof_rows, extract_traceback_or_tail, summarize_spoof_rows
from .jobs import assign_generators, enrich_jobs, get_active_generators, preflight_generators
from .runners import materialize_generator_jobs, run_generator_batch


LOGGER = get_logger("stage3-main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage3_spoof_generation.json",
        help="Path to the Stage-3 config JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--runner-log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level passed to generator batch runners.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Emit Stage-3 progress every N validated spoof samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally process only the first N real clean parents.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write job plans/manifests but do not launch generators.",
    )
    parser.add_argument(
        "--only-generator",
        action="append",
        help="Restrict processing to one or more configured generator keys.",
    )
    parser.add_argument(
        "--language",
        action="append",
        choices=("zh", "en"),
        help="Restrict processing to one or more languages.",
    )
    parser.add_argument(
        "--generators-per-parent",
        type=int,
        default=0,
        help="Override the configured number of assigned generators per parent.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    stage3_started_at = time.monotonic()

    workspace_root = Path.cwd()
    config_path = resolve_path(args.config, workspace_root)
    config = load_json(config_path)
    output_root = resolve_path(config["output_root"], workspace_root)
    output_root.mkdir(parents=True, exist_ok=True)

    stage2_manifest_path = resolve_path(config["stage2_manifest"], workspace_root)
    rows = load_csv_rows(stage2_manifest_path)
    LOGGER.info("loaded %d clean parents from %s", len(rows), stage2_manifest_path)
    if args.language:
        rows = [r for r in rows if r["language"] in set(args.language)]
        LOGGER.info("language filter -> %d rows", len(rows))
    if args.limit > 0:
        rows = rows[: args.limit]
        LOGGER.info("limit -> %d rows", len(rows))
    if not rows:
        raise RuntimeError("No Stage-2 rows selected for Stage-3 processing")

    generator_cfgs = get_active_generators(config, args.only_generator)
    LOGGER.info("generators=%s", ", ".join(sorted(generator_cfgs)))
    preflight_generators(generator_cfgs, workspace_root, args.plan_only)
    if args.plan_only:
        LOGGER.success("preflight ok | checked repos")
    else:
        LOGGER.success("preflight ok | checked repos + envs")

    generators_per_parent = (
        int(args.generators_per_parent)
        if args.generators_per_parent > 0
        else int(config["generators_per_parent"])
    )
    if generators_per_parent > len(generator_cfgs):
        raise RuntimeError(
            f"Requested generators_per_parent={generators_per_parent}, but only {len(generator_cfgs)} active generators selected"
        )

    assignments = assign_generators(rows, generator_cfgs, generators_per_parent, int(config["seed"]))
    LOGGER.info(
        "planned jobs=%d from parents=%d with generators_per_parent=%d",
        len(assignments),
        len(rows),
        generators_per_parent,
    )
    jobs_by_generator = enrich_jobs(assignments, generator_cfgs, config, workspace_root, output_root)

    assignment_summary = Counter(item["generator_key"] for item in assignments)
    (output_root / "jobs").mkdir(parents=True, exist_ok=True)
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    materialize_generator_jobs(jobs_by_generator, generator_cfgs, output_root)
    plan_path = output_root / "jobs" / "stage3_job_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": relative_to_workspace(config_path, workspace_root),
                "stage2_manifest": relative_to_workspace(stage2_manifest_path, workspace_root),
                "input_clean_parents": len(rows),
                "jobs_total": len(assignments),
                "generators_per_parent": generators_per_parent,
                "jobs_per_generator": dict(assignment_summary),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.plan_only:
        LOGGER.success("plan-only | wrote job plan to %s", plan_path)
        return 0

    futures = []
    results_meta = []
    max_workers = max(1, int(config.get("workers", 1)))
    generation_started_at = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for generator_key, jobs in jobs_by_generator.items():
            LOGGER.info("%s > queued jobs=%d", generator_key, len(jobs))
            futures.append(
                executor.submit(
                    run_generator_batch,
                    generator_key,
                    generator_cfgs[generator_key],
                    jobs,
                    output_root,
                    workspace_root,
                    args.runner_log_level,
                    LOGGER,
                )
            )
        with tqdm(total=len(futures), desc="stage3 generate", unit="gen", dynamic_ncols=True) as progress:
            for future in as_completed(futures):
                result = future.result()
                results_meta.append(result)
                progress.update(1)
                progress.set_postfix(
                    finished=f"{len(results_meta)}/{len(futures)}",
                    elapsed=format_elapsed(time.monotonic() - generation_started_at),
                )
                LOGGER.success(
                    "%s > runner done | rc=%d | finished=%d/%d | elapsed=%s | log=%s",
                    result["generator_key"],
                    result["returncode"],
                    len(results_meta),
                    len(futures),
                    format_elapsed(time.monotonic() - generation_started_at),
                    result["log_path"],
                )

    failed_runners = [r for r in results_meta if r["returncode"] != 0]
    if failed_runners:
        for item in failed_runners:
            LOGGER.error(
                "%s > runner failed | rc=%d | log=%s\n%s",
                item["generator_key"],
                item["returncode"],
                item["log_path"],
                extract_traceback_or_tail(item["log_path"]),
            )
        raise RuntimeError(
            f"Stage-3 generation failed for {len(failed_runners)} runner(s); see logged traceback snippets above"
        )

    spoof_rows, failures, stats_by_generator = collect_spoof_rows(
        jobs_by_generator,
        generator_cfgs,
        config,
        output_root,
        workspace_root,
        args.log_every,
    )
    spoof_rows.sort(key=lambda r: (r["language"], r["split"], r["speaker_id"], r["sample_id"]))

    if not spoof_rows:
        raise RuntimeError("Stage-3 produced zero valid spoof clean parents")

    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    write_csv(manifest_root / "spoof_clean_manifest.csv", spoof_rows)
    for language in sorted({r["language"] for r in spoof_rows}):
        subset = [r for r in spoof_rows if r["language"] == language]
        write_csv(manifest_root / f"spoof_clean_manifest_{language}.csv", subset)

    all_rows = []
    all_fieldnames = []
    fieldname_set = set()
    for collection in (rows, spoof_rows):
        for row in collection:
            for key in row.keys():
                if key not in fieldname_set:
                    fieldname_set.add(key)
                    all_fieldnames.append(key)
    all_rows.extend(rows)
    all_rows.extend(spoof_rows)
    write_csv(manifest_root / "clean_parent_manifest_all.csv", all_rows, fieldnames=all_fieldnames)

    failures_path = manifest_root / "stage3_failures.json"
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "stage2_manifest": relative_to_workspace(stage2_manifest_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "input_clean_parents": len(rows),
        "jobs_total": len(assignments),
        "generators_per_parent": generators_per_parent,
        "valid_spoof_clean_parents": len(spoof_rows),
        "failed_jobs": len(failures),
        "jobs_per_generator": dict(assignment_summary),
        "runner_status": {
            item["generator_key"]: {
                "returncode": item["returncode"],
                "log_path": relative_to_workspace(item["log_path"], workspace_root),
            }
            for item in results_meta
        },
        "generator_result_counts": {
            gk: dict(counter) for gk, counter in stats_by_generator.items()
        },
        "spoof_stats": summarize_spoof_rows(spoof_rows),
    }
    summary_path = manifest_root / "stage3_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures and not bool(config.get("allow_partial_failures", True)):
        raise RuntimeError(f"Stage-3 had {len(failures)} failures and allow_partial_failures=false")

    LOGGER.success(
        "finished | elapsed=%s | valid=%d | fail=%d | manifests=%s",
        format_elapsed(time.monotonic() - stage3_started_at),
        len(spoof_rows),
        len(failures),
        manifest_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
