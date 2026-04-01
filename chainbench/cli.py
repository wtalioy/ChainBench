"""Unified CLI for ChainBench stage pipelines."""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    "fetch": ("Download and extract the ChainBench dataset release.", "chainbench.release.fetch:main"),
    "package": ("Package a local ChainBench dataset release.", "chainbench.release.package:main"),
    "stage1": ("Run Stage 1 source curation.", "chainbench.stage1.cli:main"),
    "stage2": ("Run Stage 2 clean-master preparation.", "chainbench.stage2.cli:main"),
    "stage3": ("Run Stage 3 spoof generation.", "chainbench.stage3.cli:main"),
    "stage4": ("Run Stage 4 delivery-chain rendering.", "chainbench.stage4.cli:main"),
    "stage5": ("Run Stage 5 validation and packaging.", "chainbench.stage5.cli:main"),
    "eval": ("Run the evaluation pipeline.", "chainbench.eval.cli:main"),
    "preservation": ("Run preservation analysis and write a JSON summary.", "chainbench.eval.preservation.cli:main"),
}


def _print_help() -> None:
    print("ChainBench CLI")
    print("")
    print("Usage:")
    print("  chainbench <command> [args...]")
    print("")
    print("Commands:")
    for name, (description, _) in COMMANDS.items():
        print(f"  {name:<11} {description}")
    print("")
    print("Examples:")
    print("  chainbench fetch --extract-parent data")
    print("  chainbench package --dataset-root data/ChainBench")
    print("  chainbench stage1 --config config/stage1.json")
    print("  chainbench eval --config config/eval.json --dry-run")
    print("  chainbench preservation --metadata data/ChainBench/metadata.csv")


def _run_command(command: str, argv: list[str]) -> int:
    _, handler_path = COMMANDS[command]
    module_name, attr_name = handler_path.split(":", 1)
    handler = getattr(importlib.import_module(module_name), attr_name)
    original_argv = sys.argv[:]
    sys.argv = [f"chainbench {command}", *argv]
    try:
        result = handler()
    finally:
        sys.argv = original_argv
    return 0 if result is None else int(result)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv or raw_argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    command, *command_argv = raw_argv
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run `chainbench --help` for available commands.", file=sys.stderr)
        return 2

    return _run_command(command, command_argv)


if __name__ == "__main__":
    raise SystemExit(main())

