"""Worker launching and job materialization for Stage 3."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm

from chainbench.lib.conda import conda_run_python_command
from chainbench.lib.io import write_jsonl
from chainbench.lib.logging import clean_stream_line, format_elapsed
from chainbench.lib.proc import run_command_streaming

from .collect import extract_traceback_or_tail
from .worker import INTERNAL_WORKER_FLAG


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
    command = conda_run_python_command(generator["conda_env"]) + [
        "-m",
        "chainbench.stage3",
        INTERNAL_WORKER_FLAG,
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


def run_generator_batches(
    jobs_by_generator: dict[str, list[dict[str, Any]]],
    generator_cfgs: dict[str, dict[str, Any]],
    batch_paths_by_generator: dict[str, GeneratorBatchPaths],
    *,
    workspace_root: Path,
    runner_log_level: str,
    workers: int,
    logger: Any,
) -> list[dict[str, Any]]:
    futures = []
    results_meta = []
    generation_started_at = time.monotonic()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for generator_key, jobs in jobs_by_generator.items():
            logger.info("%s > queued jobs=%d", generator_key, len(jobs))
            futures.append(
                executor.submit(
                    run_generator_batch,
                    generator_key,
                    generator_cfgs[generator_key],
                    batch_paths_by_generator[generator_key],
                    workspace_root,
                    runner_log_level,
                    logger,
                )
            )

        with tqdm(total=len(futures), desc="stage3 generate", unit="gen", dynamic_ncols=True) as progress:
            for future in as_completed(futures):
                result = future.result()
                results_meta.append(result)
                progress.update(1)
                progress.set_postfix(
                    finished=f"{len(results_meta)}/{len(futures)}",
                    elapsed=format_elapsed(time.monotonic() - generation_started_at),
                )
                logger.success(
                    "%s > runner done | rc=%d | finished=%d/%d | elapsed=%s | log=%s",
                    result["generator_key"],
                    result["returncode"],
                    len(results_meta),
                    len(futures),
                    format_elapsed(time.monotonic() - generation_started_at),
                    result["log_path"],
                )

    failed_runners = [result for result in results_meta if result["returncode"] != 0]
    if failed_runners:
        for item in failed_runners:
            logger.error(
                "%s > runner failed | rc=%d | log=%s\n%s",
                item["generator_key"],
                item["returncode"],
                item["log_path"],
                extract_traceback_or_tail(item["log_path"]),
            )
        raise RuntimeError(
            f"Stage-3 generation failed for {len(failed_runners)} runner(s); see logged traceback snippets above"
        )

    return results_meta
