"""Stage 2 CLI: clean master preparation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from lib.logging import get_logger, setup_logging
from lib.config import load_json, relative_to_workspace, resolve_path
from lib.io import load_csv_rows, write_csv

from .render import build_filter_chain, make_stage2_row, render_single_row, summarize_rows


LOGGER = get_logger("stage2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/stage2_clean_master_preparation.json",
        help="Path to the Stage-2 JSON config file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Emit a progress update every N files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally process only the first N rows after filtering.",
    )
    parser.add_argument(
        "--language",
        action="append",
        choices=("zh", "en"),
        help="Restrict processing to one or more languages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    workspace_root = Path.cwd()
    config_path = resolve_path(args.config, workspace_root)
    config = load_json(config_path)
    stage1_manifest_path = resolve_path(config["stage1_manifest"], workspace_root)
    output_root = resolve_path(config["output_root"], workspace_root)
    output_audio_root = output_root / "audio"
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    output_audio_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("loading Stage-1 manifest from %s", stage1_manifest_path)
    rows = load_csv_rows(stage1_manifest_path)
    LOGGER.info("loaded %d Stage-1 rows", len(rows))

    if args.language:
        rows = [r for r in rows if r["language"] in set(args.language)]
        LOGGER.info("after language filter: %d rows", len(rows))
    if args.limit > 0:
        rows = rows[: args.limit]
        LOGGER.info("after --limit: %d rows", len(rows))
    if not rows:
        raise RuntimeError("No Stage-1 rows selected for Stage-2 processing")

    filter_chain = build_filter_chain(config)
    LOGGER.info("using ffmpeg filter chain: %s", filter_chain if filter_chain else "<none>")
    LOGGER.info(
        "timeouts: ffmpeg=%ss, ffprobe=%ss",
        int(config.get("timeouts", {}).get("ffmpeg_sec", 120)),
        int(config.get("timeouts", {}).get("ffprobe_sec", 30)),
    )

    preprocess_desc = {
        "steps": [
            "trim_silence",
            "downmix_to_mono",
            "resample_16khz",
            "loudness_normalization",
            "encode_pcm_s16le",
        ],
        "params": {
            "trim": config["trim"],
            "loudnorm": config["loudnorm"],
            "audio_output": config["audio_output"],
            "validation": config["validation"],
        },
    }

    success_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    counters = Counter()
    workers = int(config["workers"])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                render_single_row,
                row,
                config,
                workspace_root,
                output_audio_root,
                filter_chain,
            ): row
            for row in rows
        }
        total = len(future_map)
        with tqdm(total=total, desc="stage2 render", unit="file", dynamic_ncols=True) as progress:
            for idx, future in enumerate(as_completed(future_map), start=1):
                result = future.result()
                counters[result.status] += 1
                if result.ok:
                    success_rows.append(make_stage2_row(result, preprocess_desc))
                else:
                    failures.append(
                        {
                            "sample_id": result.input_row["sample_id"],
                            "language": result.input_row["language"],
                            "split": result.input_row["split"],
                            "speaker_id": result.input_row["speaker_id"],
                            "status": result.status,
                            "error": result.error or "",
                        }
                    )
                progress.update(1)
                if idx <= 5 or idx % args.log_every == 0 or idx == total:
                    progress.set_postfix(
                        rendered=counters["rendered"],
                        skipped=counters["skipped_existing"],
                        failed=idx - (counters["rendered"] + counters["skipped_existing"]),
                    )

    LOGGER.info("Sorting success rows ...")
    success_rows.sort(key=lambda r: (r["language"], r["split"], r["speaker_id"], r["utterance_id"]))
    if not success_rows:
        raise RuntimeError("Stage-2 produced zero valid clean masters")

    write_csv(manifest_root / "clean_parent_manifest.csv", success_rows)
    for language in sorted({r["language"] for r in success_rows}):
        subset = [r for r in success_rows if r["language"] == language]
        write_csv(manifest_root / f"clean_parent_manifest_{language}.csv", subset)

    with (manifest_root / "stage2_failures.json").open("w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": relative_to_workspace(config_path, workspace_root),
        "stage1_manifest": relative_to_workspace(stage1_manifest_path, workspace_root),
        "output_root": relative_to_workspace(output_root, workspace_root),
        "total_input_rows": len(rows),
        "successful_rows": len(success_rows),
        "failed_rows": len(failures),
        "status_counts": dict(counters),
        "languages": summarize_rows(success_rows),
    }
    with (manifest_root / "stage2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if failures and not bool(config.get("allow_partial_failures", True)):
        raise RuntimeError(f"Stage-2 had {len(failures)} failures and allow_partial_failures=false")

    LOGGER.info(
        "Stage-2 finished: success=%d, failed=%d, manifests written to %s",
        len(success_rows),
        len(failures),
        manifest_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
