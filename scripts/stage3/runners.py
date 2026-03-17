"""Generator batch launching and job materialization for stage3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lib.io import write_jsonl
from lib.proc import run_command_streaming
from lib.logging import clean_stream_line


def materialize_generator_jobs(
    jobs_by_generator: dict[str, list[dict[str, Any]]],
    generator_cfgs: dict[str, dict[str, Any]],
    output_root: Path,
) -> None:
    for generator_key, jobs in jobs_by_generator.items():
        jobs_path = output_root / "jobs" / f"{generator_key}.jsonl"
        adapter_cfg_path = output_root / "jobs" / f"{generator_key}.adapter_config.json"
        write_jsonl(jobs_path, jobs)
        adapter_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_cfg_path.write_text(
            json.dumps(generator_cfgs[generator_key]["adapter_config"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def run_generator_batch(
    generator_key: str,
    generator: dict[str, Any],
    jobs: list[dict[str, Any]],
    output_root: Path,
    workspace_root: Path,
    runner_log_level: str,
    logger: Any,
) -> dict[str, Any]:
    jobs_path = output_root / "jobs" / f"{generator_key}.jsonl"
    results_path = output_root / "results" / f"{generator_key}.jsonl"
    adapter_cfg_path = output_root / "jobs" / f"{generator_key}.adapter_config.json"
    log_path = output_root / "logs" / f"{generator_key}.log"
    write_jsonl(jobs_path, jobs)
    adapter_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_cfg_path.write_text(
        json.dumps(generator["adapter_config"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    runner_script = workspace_root / "scripts" / "run_stage3_batch_runner.py"
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        generator["conda_env"],
        "python",
        "-u",
        str(runner_script),
        "--adapter",
        generator["adapter"],
        "--repo-path",
        str((workspace_root / generator["repo_path"]).resolve()),
        "--config-path",
        str(adapter_cfg_path.resolve()),
        "--jobs-path",
        str(jobs_path.resolve()),
        "--results-path",
        str(results_path.resolve()),
        "--log-level",
        runner_log_level,
    ]
    log_prefix = generator_key

    def on_line(line: str) -> None:
        logger.info("%s > %s", log_prefix, clean_stream_line(line))

    returncode = run_command_streaming(command, cwd=workspace_root, log_path=log_path, on_line=on_line)
    return {
        "generator_key": generator_key,
        "returncode": returncode,
        "results_path": results_path,
        "log_path": log_path,
    }
