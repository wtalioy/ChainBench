#!/usr/bin/env python3
"""Entrypoint for Stage 3 spoof generation and internal batch runs."""

from __future__ import annotations

import argparse
import os
import sys

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--batch-runner",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args, remaining = parse_args(raw_argv)
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0], *remaining]
    try:
        if args.batch_runner:
            from stage3.runner import main as target_main
        else:
            from stage3.cli import main as target_main
        return target_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
