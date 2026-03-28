"""Generator batch launching and job materialization for stage3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.io import write_jsonl
from lib.logging import clean_stream_line
from lib.proc import run_command_streaming


@dataclass(frozen=True)
class GeneratorBatchPaths:
    jobs_path: Path
    adapter_cfg_path: Path
    results_path: Path
    log_path: Path


def get_generator_batch_paths(output_root: Path, generator_key: str) -> GeneratorBatchPaths:
    return GeneratorBatchPaths(
        jobs_path=output_root / "jobs" / f"{generator_key}.jsonl",
        adapter_cfg_path=output_root / "jobs" / f"{generator_key}.adapter_config.json",
        results_path=output_root / "results" / f"{generator_key}.jsonl",
        log_path=output_root / "logs" / f"{generator_key}.log",
    )


def materialize_generator_jobs(
    jobs_by_generator: dict[str, list[dict[str, Any]]],
    generator_cfgs: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, GeneratorBatchPaths]:
    batch_paths: dict[str, GeneratorBatchPaths] = {}
    for generator_key, jobs in jobs_by_generator.items():
        paths = get_generator_batch_paths(output_root, generator_key)
        write_jsonl(paths.jobs_path, jobs)
        paths.adapter_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        paths.adapter_cfg_path.write_text(
            json.dumps(generator_cfgs[generator_key]["adapter_config"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        batch_paths[generator_key] = paths
    return batch_paths


def run_generator_batch(
    generator_key: str,
    generator: dict[str, Any],
    batch_paths: GeneratorBatchPaths,
    workspace_root: Path,
    runner_log_level: str,
    logger: Any,
) -> dict[str, Any]:
    runner_script = workspace_root / "scripts" / "run_stage3.py"
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        generator["conda_env"],
        "python",
        "-u",
        str(runner_script),
        "--batch-runner",
        "--adapter",
        generator["adapter"],
        "--repo-path",
        str((workspace_root / generator["repo_path"]).resolve()),
        "--config-path",
        str(batch_paths.adapter_cfg_path.resolve()),
        "--jobs-path",
        str(batch_paths.jobs_path.resolve()),
        "--results-path",
        str(batch_paths.results_path.resolve()),
        "--log-level",
        runner_log_level,
    ]
    log_prefix = generator_key

    def on_line(line: str) -> None:
        logger.info("%s > %s", log_prefix, clean_stream_line(line))

    returncode = run_command_streaming(command, cwd=workspace_root, log_path=batch_paths.log_path, on_line=on_line)
    return {
        "generator_key": generator_key,
        "returncode": returncode,
        "results_path": batch_paths.results_path,
        "log_path": batch_paths.log_path,
    }
