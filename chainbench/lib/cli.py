"""Shared CLI helpers for ChainBench command modules."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .config import resolve_path
from .io import load_csv_rows

LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR")
LANGUAGE_CHOICES = ("zh", "en")


def add_log_level_argument(parser: argparse.ArgumentParser, *, default: str = "INFO") -> None:
    parser.add_argument(
        "--log-level",
        default=default,
        choices=LOG_LEVEL_CHOICES,
        help="Logging verbosity.",
    )


def add_language_filter_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--language",
        action="append",
        choices=LANGUAGE_CHOICES,
        help=help_text,
    )


def add_limit_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=help_text,
    )


def load_rows_with_filters(
    input_manifest_path: Path,
    *,
    logger: Any,
    row_label: str,
    empty_error: str,
    languages: list[str] | None = None,
    limit: int = 0,
) -> list[dict[str, str]]:
    logger.info("loading %s from %s", row_label, input_manifest_path)
    rows = load_csv_rows(input_manifest_path)
    logger.info("loaded %d %s", len(rows), row_label)
    if languages:
        allowed_languages = set(languages)
        rows = [row for row in rows if row["language"] in allowed_languages]
        logger.info("after language filter: %d rows", len(rows))
    if limit > 0:
        rows = rows[:limit]
        logger.info("after --limit: %d rows", len(rows))
    if not rows:
        raise RuntimeError(empty_error)
    return rows


def resolve_worker_count(
    args_workers: int,
    config: dict[str, Any],
    *,
    fallback: int | None = None,
    maximum_default: int = 8,
) -> int:
    if args_workers > 0:
        return args_workers
    config_workers = int(config.get("workers", 0))
    if config_workers > 0:
        return config_workers
    if fallback is not None:
        return max(1, fallback)
    cpu_count = os.cpu_count() or 4
    return max(1, min(maximum_default, cpu_count))


def resolve_config_argument(config_arg: str, workspace_root: Path) -> Path:
    return resolve_path(config_arg, workspace_root)
