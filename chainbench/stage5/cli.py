"""Stage 5 CLI: validation plus lean dataset metadata export."""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from chainbench.lib.cli import (
    add_language_filter_argument,
    add_limit_argument,
    add_log_level_argument,
    load_rows_with_filters,
    resolve_config_argument,
    resolve_worker_count,
)
from chainbench.lib.config import default_workspace_root, load_json, resolve_path
from chainbench.lib.io import write_csv, write_json
from chainbench.lib.logging import get_logger, setup_logging
from chainbench.lib.summary import print_json

from .execution import export_dataset_audio, sort_dataset_rows, validate_dataset_rows
from .metadata import annotate_rows, check_speaker_disjoint, summarize_parent_coverage
from .reporting import (
    RELEASE_METADATA_FIELD_ORDER,
    build_release_metadata_rows,
    build_stage5_summary,
    resolve_coverage_families,
    summarize_duplicates,
    write_split_metadata_files,
    write_stats_tables,
)
from .validate import summarize_validation_rows

LOGGER = get_logger("stage5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage5.json",
        help="Path to the Stage-5 JSON config file.",
    )
    add_log_level_argument(parser)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Update the progress display every N validated files.",
    )
    add_limit_argument(parser, help_text="Optionally validate/package only the first N rows after filtering.")
    add_language_filter_argument(parser, help_text="Restrict processing to one or more languages.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Override worker count from config.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Stage-5 validation and directly construct the final dataset package.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    workspace_root = default_workspace_root()
    config_path = resolve_config_argument(args.config, workspace_root)
    config = load_json(config_path)
    input_manifest_path = resolve_path(config["input_manifest"], workspace_root)
    output_root = resolve_path(config["output_root"], workspace_root)
    manifest_root = output_root / "manifest"
    manifest_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows_with_filters(
        input_manifest_path,
        logger=LOGGER,
        row_label="Stage-4 rows",
        empty_error="No Stage-4 rows selected for Stage-5 processing",
        languages=args.language,
        limit=args.limit,
    )

    workers = resolve_worker_count(args.workers, config, fallback=1)
    if args.skip_validation:
        LOGGER.info("skip-validation mode: constructing dataset directly from input manifest")
        prepared_rows = [dict(row) for row in rows]
        failures: list[dict[str, str]] = []
        counts = Counter({"packaged_without_validation": len(prepared_rows)})
    else:
        prepared_rows, failures, counts = validate_dataset_rows(
            rows,
            config,
            workspace_root,
            workers,
            args.log_every,
        )
    if not prepared_rows:
        raise RuntimeError("Stage-5 produced zero rows for dataset construction")

    LOGGER.info("copying final audio into dataset structure ...")
    exported_rows, export_failures, export_counts = export_dataset_audio(
        prepared_rows,
        workspace_root,
        output_root,
        workers,
        overwrite=bool(config.get("overwrite_audio", False)),
        log_every=args.log_every,
    )
    failures.extend(export_failures)
    counts.update(export_counts)
    if not exported_rows:
        raise RuntimeError("Stage-5 produced zero exported dataset rows")

    annotated_rows = sort_dataset_rows(annotate_rows(exported_rows))
    LOGGER.info("writing lean dataset metadata ...")
    metadata_rows = build_release_metadata_rows(annotated_rows)
    metadata_path = output_root / "metadata.csv"
    write_csv(metadata_path, metadata_rows, fieldnames=RELEASE_METADATA_FIELD_ORDER)
    split_metadata_paths = write_split_metadata_files(output_root, metadata_rows, workspace_root)

    failures_path = manifest_root / "stage5_failures.json"
    write_json(failures_path, failures)
    LOGGER.info("writing statistics tables ...")
    stats_tables = write_stats_tables(annotated_rows, manifest_root, workspace_root)

    coverage_families = resolve_coverage_families(config)
    summary = build_stage5_summary(
        config=config,
        config_path=config_path,
        input_manifest_path=input_manifest_path,
        output_root=output_root,
        metadata_path=metadata_path,
        failures_path=failures_path,
        workspace_root=workspace_root,
        validation_enabled=not args.skip_validation,
        input_rows=len(rows),
        annotated_rows=annotated_rows,
        failures=failures,
        counts=counts,
        split_metadata_paths=split_metadata_paths,
        duplicate_checks=summarize_duplicates(annotated_rows),
        speaker_disjoint_check=check_speaker_disjoint(annotated_rows),
        counterfactual_parent_coverage=summarize_parent_coverage(annotated_rows, coverage_families),
        validation_stats=summarize_validation_rows(annotated_rows),
        stats_tables=stats_tables,
    )
    summary_path = manifest_root / "dataset_summary.json"
    write_json(summary_path, summary)

    if failures and not bool(config.get("allow_partial_failures", True)):
        raise RuntimeError(f"Stage-5 had {len(failures)} failures and allow_partial_failures=false")

    LOGGER.info(
        "Stage-5 finished: dataset_rows=%d, failed=%d, manifest_root=%s",
        len(annotated_rows),
        len(failures),
        manifest_root,
    )
    print_json(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
