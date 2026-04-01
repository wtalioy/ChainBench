"""Command-line entrypoint for ChainBench evaluation."""

from __future__ import annotations

import argparse
import sys

from chainbench.lib.cli import add_log_level_argument
from chainbench.lib.config import default_workspace_root
from chainbench.lib.logging import setup_logging
from .runner import run_eval_from_args
from .config import BASELINE_IDS, TASK_IDS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChainBench evaluation pipeline.")
    parser.add_argument(
        "--config",
        default="config/eval.json",
        help="Path to evaluation JSON config.",
    )
    parser.add_argument(
        "--output-root",
        help="Override the config output_root for this run.",
    )
    add_log_level_argument(parser)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_IDS,
        help="Tasks to run (default: from config).",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=BASELINE_IDS,
        help="Baselines to run (default: from config).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run evaluation.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run training only and skip evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build task packs and report train/dev/test sample counts without running baselines.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: limit metadata rows and per-split sizes for fast validation.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        help="Deterministically subsample each task pack split by this ratio (0, 1].",
    )
    parser.add_argument(
        "--template-holdout",
        action="store_true",
        help="Override the config to run leave-one-template-out generalization for `in_chain_detection`.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore cached checkpoints and retrain selected baselines from scratch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    return run_eval_from_args(args, workspace_root=default_workspace_root())


if __name__ == "__main__":
    sys.exit(main())
