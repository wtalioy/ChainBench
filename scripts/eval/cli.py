"""Command-line entrypoint for ChainBench evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lib.logging import setup_logging
from .app import run_eval_from_args
from .config import BASELINE_IDS, TASK_IDS
from .generalization import GENERALIZATION_PROTOCOLS, GENERALIZATION_SCOPES

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChainBench evaluation pipeline.")
    parser.add_argument(
        "--config",
        default="config/eval_baselines.json",
        help="Path to evaluation JSON config.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
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
        "--generalization-protocol",
        choices=GENERALIZATION_PROTOCOLS,
        help="Override the config generalization protocol for eval-time held-out folds.",
    )
    parser.add_argument(
        "--generalization-scope",
        choices=GENERALIZATION_SCOPES,
        help="Override the config generalization scope.",
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
    return run_eval_from_args(args, workspace_root=Path.cwd())


if __name__ == "__main__":
    sys.exit(main())
