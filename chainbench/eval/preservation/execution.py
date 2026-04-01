"""Workflow orchestration for preservation analysis runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chainbench.lib import runtime_snapshot
from chainbench.lib.config import resolve_path
from chainbench.lib.io import write_json

from .analysis import PreservationAnalyzer
from .backends import load_backend_from_cli
from .schema import DEFAULT_SPLITS
from .selection import (
    infer_dataset_root,
    load_selected_rows,
    result_row_key,
    row_matches_selection,
    validate_shard_args,
)
from .state import (
    analysis_fingerprint,
    load_resume_manifest,
    load_resume_rows,
    open_result_rows_writer,
    result_rows_path,
    resume_state_path,
    write_result_rows,
    write_resume_manifest,
)
from .summary import PreservationSummaryAccumulator, build_summary_payload, build_summary_payload_from_tables


@dataclass(frozen=True)
class RunContext:
    workspace_root: Path
    metadata_path: Path
    requested_splits: list[str]
    num_shards: int
    shard_index: int


def _stream_results_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "in_memory_results", False))


def _resume_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "resume", False))


def _parse_gpu_list(raw_value: str) -> list[str]:
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def _log_runtime_snapshot(label: str) -> None:
    print(f"runtime snapshot: {json.dumps(runtime_snapshot(label), sort_keys=True)}", flush=True)


def _resolve_run_context(args: argparse.Namespace) -> RunContext:
    workspace_root = Path(args.workspace_root).resolve()
    metadata_path = resolve_path(args.metadata, workspace_root)
    requested_splits = list(args.splits or DEFAULT_SPLITS)
    num_shards = max(int(getattr(args, "num_shards", 1)), 1)
    shard_index = int(getattr(args, "shard_index", 0))
    validate_shard_args(num_shards, shard_index)
    return RunContext(
        workspace_root=workspace_root,
        metadata_path=metadata_path,
        requested_splits=requested_splits,
        num_shards=num_shards,
        shard_index=shard_index,
    )


def _build_analyzer(args: argparse.Namespace, context: RunContext) -> PreservationAnalyzer:
    return PreservationAnalyzer(
        context.workspace_root,
        dataset_root=infer_dataset_root(
            getattr(args, "dataset_root", ""),
            context.metadata_path,
            context.workspace_root,
        ),
        asr_backend=load_backend_from_cli(args.asr_backend, args.asr_backend_kwargs),
        speaker_backend=load_backend_from_cli(args.speaker_backend, args.speaker_backend_kwargs),
    )


def _attach_shard_info(summary: dict[str, Any], *, num_shards: int, shard_index: int | str) -> dict[str, Any]:
    summary["shard"] = {
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    return summary


def _print_selection_stats(*, scanned_rows: int, selected_rows: int, context: RunContext) -> None:
    print(
        f"Scanned {scanned_rows:,} metadata rows; analyzing {selected_rows:,} rows "
        f"for splits={context.requested_splits}, shard={context.shard_index + 1}/{context.num_shards}.",
        flush=True,
    )


def run_in_memory_analysis(args: argparse.Namespace) -> dict[str, Any]:
    context = _resolve_run_context(args)
    selected_rows, scanned_rows = load_selected_rows(
        context.metadata_path,
        context.requested_splits,
        int(args.limit),
        num_shards=context.num_shards,
        shard_index=context.shard_index,
    )
    _print_selection_stats(scanned_rows=scanned_rows, selected_rows=len(selected_rows), context=context)

    results = _build_analyzer(args, context).analyze_rows(
        selected_rows,
        show_progress=not bool(getattr(args, "no_progress", False)),
        progress_desc="Preservation analysis",
    )
    return _attach_shard_info(
        build_summary_payload(
            results,
            metadata_path=str(context.metadata_path),
            requested_splits=context.requested_splits,
            asr_backend=args.asr_backend,
            speaker_backend=args.speaker_backend,
        ),
        num_shards=context.num_shards,
        shard_index=context.shard_index,
    )


def run_streaming_analysis(args: argparse.Namespace) -> dict[str, Any]:
    context = _resolve_run_context(args)
    analyzer = _build_analyzer(args, context)
    output_rows_path = result_rows_path(
        context.workspace_root,
        num_shards=context.num_shards,
        shard_index=context.shard_index,
    )
    fingerprint = analysis_fingerprint(
        metadata_path=context.metadata_path,
        requested_splits=context.requested_splits,
        args=args,
        num_shards=context.num_shards,
        shard_index=context.shard_index,
    )
    resume_manifest_path = resume_state_path(output_rows_path)

    completed_keys: set[str] = set()
    resumed_rows = 0
    if _resume_enabled(args):
        load_resume_manifest(resume_manifest_path, fingerprint=fingerprint)
        completed_keys, summary_accumulator, resumed_rows = load_resume_rows(output_rows_path)
    else:
        summary_accumulator = PreservationSummaryAccumulator()

    rows_handle, row_writer = open_result_rows_writer(output_rows_path, append=_resume_enabled(args))
    split_set = set(context.requested_splits)
    scanned_rows = 0
    selected_rows = resumed_rows
    chunk_index = 0
    pending_rows: list[dict[str, str]] = []
    limit = int(getattr(args, "limit", 0))

    def process_chunk(chunk_rows: list[dict[str, str]], current_chunk_index: int) -> None:
        chunk_results = analyzer.analyze_rows(
            chunk_rows,
            show_progress=not bool(getattr(args, "no_progress", False)),
            progress_desc=f"Preservation analysis chunk {current_chunk_index}",
        )
        summary_accumulator.update_many(chunk_results)
        write_result_rows(row_writer, chunk_results)
        completed_keys.update(result_row_key(row) for row in chunk_results)
        if _resume_enabled(args):
            write_resume_manifest(
                resume_manifest_path,
                fingerprint=fingerprint,
                completed_rows=len(completed_keys),
                scanned_rows=scanned_rows,
                selected_rows=selected_rows,
                status="running",
            )
        _log_runtime_snapshot(f"preservation_chunk_{current_chunk_index}")

    _log_runtime_snapshot("preservation_start")
    try:
        with context.metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scanned_rows += 1
                if not row_matches_selection(
                    row,
                    split_set=split_set,
                    num_shards=context.num_shards,
                    shard_index=context.shard_index,
                ):
                    continue
                row_key = result_row_key(row)
                if row_key in completed_keys:
                    continue
                if limit > 0 and selected_rows >= limit:
                    break
                pending_rows.append(dict(row))
                selected_rows += 1
                if len(pending_rows) >= max(int(getattr(args, "chunk_size", 2048)), 1):
                    chunk_index += 1
                    process_chunk(pending_rows, chunk_index)
                    pending_rows = []
        if pending_rows:
            chunk_index += 1
            process_chunk(pending_rows, chunk_index)
    finally:
        if rows_handle is not None:
            rows_handle.close()

    _print_selection_stats(scanned_rows=scanned_rows, selected_rows=selected_rows, context=context)
    if _resume_enabled(args):
        write_resume_manifest(
            resume_manifest_path,
            fingerprint=fingerprint,
            completed_rows=len(completed_keys),
            scanned_rows=scanned_rows,
            selected_rows=selected_rows,
            status="completed",
        )
    _log_runtime_snapshot("preservation_done")
    return _attach_shard_info(
        build_summary_payload_from_tables(
            summary_accumulator.build_tables(),
            metadata_path=str(context.metadata_path),
            requested_splits=context.requested_splits,
            asr_backend=args.asr_backend,
            speaker_backend=args.speaker_backend,
        ),
        num_shards=context.num_shards,
        shard_index=context.shard_index,
    )


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    if _stream_results_enabled(args):
        return run_streaming_analysis(args)
    return run_in_memory_analysis(args)


def merge_row_csvs(row_csvs: list[Path], *, args: argparse.Namespace, context: RunContext) -> dict[str, Any]:
    summary_accumulator = PreservationSummaryAccumulator()
    merged_row_count = 0
    for index, row_csv in enumerate(row_csvs, start=1):
        with row_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                summary_accumulator.update(dict(row))
                merged_row_count += 1
        _log_runtime_snapshot(f"merge_rows_csv_{index}")
    print(f"Merging {len(row_csvs):,} shard row cache file(s) into {merged_row_count:,} rows.", flush=True)
    return _attach_shard_info(
        build_summary_payload_from_tables(
            summary_accumulator.build_tables(),
            metadata_path=str(context.metadata_path),
            requested_splits=context.requested_splits,
            asr_backend=args.asr_backend,
            speaker_backend=args.speaker_backend,
        ),
        num_shards=len(row_csvs),
        shard_index="merged",
    )


def _build_shard_command(
    args: argparse.Namespace,
    *,
    shard_index: int,
    num_shards: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "chainbench.eval.preservation.cli",
        "--metadata",
        str(args.metadata),
        "--workspace-root",
        str(args.workspace_root),
        "--dataset-root",
        str(getattr(args, "dataset_root", "")),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
        "--output-json",
        "",
    ]
    if bool(getattr(args, "no_progress", False)):
        command.append("--no-progress")
    for split in list(args.splits or []):
        command.extend(["--split", str(split)])
    if int(getattr(args, "limit", 0)) > 0:
        command.extend(["--limit", str(args.limit)])
    if int(getattr(args, "chunk_size", 2048)) > 0:
        command.extend(["--chunk-size", str(args.chunk_size)])
    if bool(getattr(args, "in_memory_results", False)):
        command.append("--in-memory-results")
    if not bool(getattr(args, "resume", True)):
        command.append("--no-resume")
    if str(getattr(args, "asr_backend", "")):
        command.extend(["--asr-backend", str(args.asr_backend)])
    if str(getattr(args, "asr_backend_kwargs", "")):
        command.extend(["--asr-backend-kwargs", str(args.asr_backend_kwargs)])
    if str(getattr(args, "speaker_backend", "")):
        command.extend(["--speaker-backend", str(args.speaker_backend)])
    if str(getattr(args, "speaker_backend_kwargs", "")):
        command.extend(["--speaker-backend-kwargs", str(args.speaker_backend_kwargs)])
    return command


def run_multi_gpu(args: argparse.Namespace) -> dict[str, Any]:
    gpu_ids = _parse_gpu_list(getattr(args, "gpus", ""))
    if len(gpu_ids) <= 1:
        return run_analysis(args)

    base_context = _resolve_run_context(args)
    num_shards = len(gpu_ids)
    print(f"Launching {num_shards} preservation shard workers on GPUs {gpu_ids}.", flush=True)
    if not bool(getattr(args, "no_progress", False)):
        print("Child progress output may interleave across GPUs.", flush=True)

    processes: list[tuple[int, str, subprocess.Popen[str]]] = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        command = _build_shard_command(args, shard_index=shard_index, num_shards=num_shards)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        process = subprocess.Popen(command, cwd=str(base_context.workspace_root), env=env, text=True)
        processes.append((shard_index, gpu_id, process))

    failures: list[str] = []
    row_csvs: list[Path] = []
    for shard_index, gpu_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append(f"shard {shard_index} on GPU {gpu_id} exited with code {return_code}")
            continue
        row_csvs.append(
            result_rows_path(
                base_context.workspace_root,
                num_shards=num_shards,
                shard_index=shard_index,
            )
        )

    if failures:
        raise RuntimeError("; ".join(failures))

    print(f"Merging shard outputs from {len(row_csvs)} row cache file(s).", flush=True)
    return merge_row_csvs(row_csvs, args=args, context=base_context)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if len(_parse_gpu_list(getattr(args, "gpus", ""))) > 1:
        return run_multi_gpu(args)
    return run_analysis(args)


def write_output_json(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    if not getattr(args, "output_json", ""):
        return
    workspace_root = Path(args.workspace_root).resolve()
    write_json(resolve_path(args.output_json, workspace_root), summary)


__all__ = [
    "merge_row_csvs",
    "run_analysis",
    "run_from_args",
    "run_in_memory_analysis",
    "run_multi_gpu",
    "run_streaming_analysis",
    "write_output_json",
]
