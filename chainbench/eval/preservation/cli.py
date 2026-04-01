"""CLI for preservation analysis summary generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))

from chainbench.eval.preservation.execution import run_from_args, write_output_json

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--metadata",
        default="data/ChainBench/metadata.csv",
        help="Path to the packaged dataset metadata CSV.",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(REPO_ROOT),
        help="Workspace root used to resolve relative audio paths.",
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help=(
            "Optional packaged dataset root used to resolve child audio paths. "
            "Defaults to the metadata directory for root-level metadata exports."
        ),
    )
    parser.add_argument(
        "--split",
        action="append",
        dest="splits",
        help="Restrict analysis to one or more exported split names. Defaults to test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally analyze only the first N rows after split filtering.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parent-aware shards for multi-process / multi-GPU runs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index within --num-shards.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the row-wise progress bar.",
    )
    parser.add_argument(
        "--gpus",
        default="",
        help="Comma-separated GPU IDs for one-command multi-GPU execution, e.g. 0,1.",
    )
    parser.add_argument(
        "--asr-backend",
        default="transformers_asr",
        help=(
            "Optional ASR backend factory. Use module:function or the shortcut "
            "'identity_asr' for smoke tests."
        ),
    )
    parser.add_argument(
        "--asr-backend-kwargs",
        default='{"model_id":"openai/whisper-large-v3-turbo","device":0,"chunk_length_s":15.0,"batch_size":16,"generate_kwargs":{"task":"transcribe"},"model_kwargs":{"dtype":"float16"}}',
        help="Optional JSON object passed as kwargs to the ASR backend factory.",
    )
    parser.add_argument(
        "--speaker-backend",
        default="transformers_speaker",
        help=(
            "Optional speaker backend factory. Use module:function or the shortcut "
            "'speechbrain_speaker'."
        ),
    )
    parser.add_argument(
        "--speaker-backend-kwargs",
        default='{"model_id":"/remote-home/wangruiming/ChainBench/.cache_models/wavlm-base-sv","device":0,"model_kwargs":{"dtype":"float16"},"batch_size":16}',
        help="Optional JSON object passed as kwargs to the speaker backend factory.",
    )
    parser.add_argument(
        "--output-json",
        default="results/preservation_summary.json",
        help="Path to the summary JSON output.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Rows per analysis chunk when streaming through the internal row cache.",
    )
    parser.add_argument(
        "--in-memory-results",
        action="store_true",
        help="Keep full result rows in memory instead of streaming through the internal row cache.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore the internal row cache and recompute the analysis from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_from_args(args)
    write_output_json(args, summary)
    if args.output_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
