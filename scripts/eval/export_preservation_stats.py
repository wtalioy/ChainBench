"""Compute counterfactual preservation-validation statistics from release metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from eval.preservation import (
    DEFAULT_SPLITS,
    PreservationAnalyzer,
    PreservationSummaryAccumulator,
    RESULT_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    build_summary_payload,
    build_summary_payload_from_tables,
    load_backend_from_cli,
    write_csv_with_headers,
    write_preservation_table_tex,
    write_preservation_tex,
)
from lib import runtime_snapshot
from lib.config import resolve_path
from lib.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--output-rows-csv",
        default="results/preservation_rows.csv",
        help="Path to the row-level preservation output CSV.",
    )
    parser.add_argument(
        "--output-family-csv",
        default="results/preservation_by_family.csv",
        help="Path to the family-level summary CSV.",
    )
    parser.add_argument(
        "--output-family-language-csv",
        default="results/preservation_by_family_language.csv",
        help="Path to the family-language summary CSV.",
    )
    parser.add_argument(
        "--output-json",
        default="results/preservation_summary.json",
        help="Path to the summary JSON output.",
    )
    parser.add_argument(
        "--output-tex",
        default="results/preservation_macros.tex",
        help="Optional path to write LaTeX macros.",
    )
    parser.add_argument(
        "--output-table-tex",
        default="results/preservation_table.tex",
        help="Optional path to write a LaTeX table snippet.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Rows per analysis chunk when streaming metadata/results to limit peak memory.",
    )
    parser.add_argument(
        "--in-memory-results",
        action="store_true",
        help="Keep full result rows in memory instead of streaming to disk/summary accumulators.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from an existing output rows CSV by skipping already-written result rows.",
    )
    parser.add_argument(
        "--merge-rows-csv",
        action="append",
        dest="merge_rows_csvs",
        default=[],
        help="Merge mode: one or more row-level preservation CSVs to combine into final outputs.",
    )
    return parser.parse_args()


def run_analysis(args: argparse.Namespace) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    if _stream_results_enabled(args):
        return _run_analysis_streaming(args)

    workspace_root = Path(args.workspace_root).resolve()
    metadata_path = resolve_path(args.metadata, workspace_root)
    dataset_root = _infer_dataset_root(getattr(args, "dataset_root", ""), metadata_path, workspace_root)
    requested_splits = list(args.splits or DEFAULT_SPLITS)
    num_shards = max(int(getattr(args, "num_shards", 1)), 1)
    shard_index = int(getattr(args, "shard_index", 0))
    _validate_shard_args(num_shards, shard_index)
    selected_rows, scanned_rows = _load_selected_rows(
        metadata_path,
        requested_splits,
        int(args.limit),
        num_shards=num_shards,
        shard_index=shard_index,
    )
    print(
        f"Scanned {scanned_rows:,} metadata rows; analyzing {len(selected_rows):,} rows "
        f"for splits={requested_splits}, shard={shard_index + 1}/{num_shards}.",
        flush=True,
    )

    analyzer = PreservationAnalyzer(
        workspace_root,
        dataset_root=dataset_root,
        asr_backend=load_backend_from_cli(args.asr_backend, args.asr_backend_kwargs),
        speaker_backend=load_backend_from_cli(args.speaker_backend, args.speaker_backend_kwargs),
    )
    results = analyzer.analyze_rows(
        selected_rows,
        show_progress=not bool(getattr(args, "no_progress", False)),
        progress_desc="Preservation analysis",
    )
    summary = build_summary_payload(
        results,
        metadata_path=str(metadata_path),
        requested_splits=requested_splits,
        asr_backend=args.asr_backend,
        speaker_backend=args.speaker_backend,
    )
    summary["shard"] = {
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    return results, summary


def _infer_dataset_root(dataset_root_arg: str, metadata_path: Path, workspace_root: Path) -> Path:
    if dataset_root_arg:
        return resolve_path(dataset_root_arg, workspace_root)
    if metadata_path.parent.name == "manifest":
        return metadata_path.parent.parent
    return metadata_path.parent


def _load_selected_rows(
    metadata_path: Path,
    requested_splits: list[str],
    limit: int,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> tuple[list[dict[str, str]], int]:
    selected: list[dict[str, str]] = []
    split_set = set(requested_splits)
    scanned_rows = 0
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scanned_rows += 1
            split = str(row.get("split_standard", "") or row.get("split", "")).strip()
            if split_set and split not in split_set:
                continue
            if not _row_in_shard(row, num_shards=num_shards, shard_index=shard_index):
                continue
            selected.append(dict(row))
            if limit > 0 and len(selected) >= limit:
                break
    return selected, scanned_rows


def _row_in_shard(row: dict[str, str], *, num_shards: int, shard_index: int) -> bool:
    if num_shards <= 1:
        return True
    parent_id = str(row.get("parent_id", "")).strip()
    sample_id = str(row.get("sample_id", "")).strip()
    shard_key = parent_id or sample_id
    digest = hashlib.md5(shard_key.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards == shard_index


def _validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")


def _stream_results_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "in_memory_results", False))


def _resume_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "resume", False))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_runtime_snapshot(label: str) -> None:
    print(f"runtime snapshot: {json.dumps(runtime_snapshot(label), sort_keys=True)}", flush=True)


def _resume_state_path(output_rows_path: Path) -> Path:
    return output_rows_path.with_name(f"{output_rows_path.name}.progress.json")


def _analysis_fingerprint(
    *,
    metadata_path: Path,
    requested_splits: list[str],
    args: argparse.Namespace,
    num_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    return {
        "metadata_path": str(metadata_path),
        "requested_splits": list(requested_splits),
        "limit": int(getattr(args, "limit", 0)),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "asr_backend": str(getattr(args, "asr_backend", "")),
        "speaker_backend": str(getattr(args, "speaker_backend", "")),
    }


def _result_row_key(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    file_name = str(row.get("file_name", "")).strip()
    if file_name:
        return file_name
    return json.dumps(
        {
            "parent_id": str(row.get("parent_id", "")).strip(),
            "clean_parent_path": str(row.get("clean_parent_path", "")).strip(),
            "file_name": file_name,
            "language": str(row.get("language", "")).strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_resume_rows(
    output_rows_path: Path,
    *,
    stream_results: bool,
) -> tuple[set[str], PreservationSummaryAccumulator, list[dict[str, Any]] | None, int]:
    completed_keys: set[str] = set()
    summary_accumulator = PreservationSummaryAccumulator()
    results: list[dict[str, Any]] | None = None if stream_results else []
    completed_rows = 0
    if not output_rows_path.exists():
        return completed_keys, summary_accumulator, results, completed_rows
    with output_rows_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = dict(row)
            key = _result_row_key(normalized)
            if key in completed_keys:
                continue
            completed_keys.add(key)
            summary_accumulator.update(normalized)
            if results is not None:
                results.append(normalized)
            completed_rows += 1
    return completed_keys, summary_accumulator, results, completed_rows


def _load_resume_manifest(path: Path, *, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    prior = state.get("fingerprint", {})
    if prior != fingerprint:
        raise ValueError(
            "Resume state does not match the current analysis arguments; "
            "remove the old output rows/progress files or rerun without --resume."
        )
    return state


def _write_resume_manifest(
    path: Path,
    *,
    fingerprint: dict[str, Any],
    completed_rows: int,
    scanned_rows: int,
    selected_rows: int,
    status: str,
) -> None:
    write_json(
        path,
        {
            "updated_at_utc": _utc_now_iso(),
            "status": status,
            "completed_result_rows": completed_rows,
            "scanned_rows": scanned_rows,
            "selected_rows": selected_rows,
            "fingerprint": fingerprint,
        },
    )


def _open_result_rows_writer(
    path: Path | None,
    *,
    append: bool = False,
) -> tuple[Any | None, csv.DictWriter | None]:
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    mode = "a" if append and file_exists else "w"
    handle = path.open(mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
    if mode == "w" or not file_exists or path.stat().st_size == 0:
        writer.writeheader()
    return handle, writer


def _write_result_rows(writer: csv.DictWriter | None, rows: list[dict[str, Any]]) -> None:
    if writer is None:
        return
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDNAMES})


def _run_analysis_streaming(args: argparse.Namespace) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    workspace_root = Path(args.workspace_root).resolve()
    metadata_path = resolve_path(args.metadata, workspace_root)
    dataset_root = _infer_dataset_root(getattr(args, "dataset_root", ""), metadata_path, workspace_root)
    requested_splits = list(args.splits or DEFAULT_SPLITS)
    num_shards = max(int(getattr(args, "num_shards", 1)), 1)
    shard_index = int(getattr(args, "shard_index", 0))
    chunk_size = max(int(getattr(args, "chunk_size", 2048)), 1)
    _validate_shard_args(num_shards, shard_index)

    analyzer = PreservationAnalyzer(
        workspace_root,
        dataset_root=dataset_root,
        asr_backend=load_backend_from_cli(args.asr_backend, args.asr_backend_kwargs),
        speaker_backend=load_backend_from_cli(args.speaker_backend, args.speaker_backend_kwargs),
    )
    stream_results = _stream_results_enabled(args)
    output_rows_path = resolve_path(args.output_rows_csv, workspace_root) if args.output_rows_csv else None
    if _resume_enabled(args) and output_rows_path is None:
        raise ValueError("--resume requires --output-rows-csv so completed rows can be recovered")
    fingerprint = _analysis_fingerprint(
        metadata_path=metadata_path,
        requested_splits=requested_splits,
        args=args,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    resume_manifest_path = _resume_state_path(output_rows_path) if output_rows_path is not None else None
    completed_keys: set[str] = set()
    resumed_rows = 0
    if _resume_enabled(args) and output_rows_path is not None:
        if resume_manifest_path is not None:
            _load_resume_manifest(resume_manifest_path, fingerprint=fingerprint)
        completed_keys, summary_accumulator, results, resumed_rows = _load_resume_rows(
            output_rows_path,
            stream_results=stream_results,
        )
    else:
        summary_accumulator = PreservationSummaryAccumulator()
        results = None if stream_results else []
    rows_handle, row_writer = _open_result_rows_writer(
        output_rows_path,
        append=_resume_enabled(args),
    )
    split_set = set(requested_splits)
    scanned_rows = 0
    selected_rows = resumed_rows
    chunk_index = 0
    pending_rows: list[dict[str, str]] = []

    def process_chunk(chunk_rows: list[dict[str, str]], current_chunk_index: int) -> None:
        chunk_results = analyzer.analyze_rows(
            chunk_rows,
            show_progress=not bool(getattr(args, "no_progress", False)),
            progress_desc=f"Preservation analysis chunk {current_chunk_index}",
        )
        summary_accumulator.update_many(chunk_results)
        _write_result_rows(row_writer, chunk_results)
        if results is not None:
            results.extend(chunk_results)
        completed_keys.update(_result_row_key(row) for row in chunk_results)
        if resume_manifest_path is not None:
            _write_resume_manifest(
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
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scanned_rows += 1
                split = str(row.get("split_standard", "") or row.get("split", "")).strip()
                if split_set and split not in split_set:
                    continue
                if not _row_in_shard(row, num_shards=num_shards, shard_index=shard_index):
                    continue
                if _result_row_key(row) in completed_keys:
                    continue
                pending_rows.append(dict(row))
                selected_rows += 1
                if len(pending_rows) >= chunk_size:
                    chunk_index += 1
                    process_chunk(pending_rows, chunk_index)
                    pending_rows = []
                if int(args.limit) > 0 and selected_rows >= int(args.limit):
                    break
        if pending_rows:
            chunk_index += 1
            process_chunk(pending_rows, chunk_index)
    finally:
        if rows_handle is not None:
            rows_handle.close()

    print(
        f"Scanned {scanned_rows:,} metadata rows; analyzing {selected_rows:,} rows "
        f"for splits={requested_splits}, shard={shard_index + 1}/{num_shards}.",
        flush=True,
    )
    summary = build_summary_payload_from_tables(
        summary_accumulator.build_tables(),
        metadata_path=str(metadata_path),
        requested_splits=requested_splits,
        asr_backend=args.asr_backend,
        speaker_backend=args.speaker_backend,
    )
    summary["shard"] = {
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    if resume_manifest_path is not None:
        _write_resume_manifest(
            resume_manifest_path,
            fingerprint=fingerprint,
            completed_rows=len(completed_keys),
            scanned_rows=scanned_rows,
            selected_rows=selected_rows,
            status="completed",
        )
    _log_runtime_snapshot("preservation_done")
    return results, summary


def _load_result_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_row_csvs(args: argparse.Namespace) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    workspace_root = Path(args.workspace_root).resolve()
    requested_splits = list(args.splits or DEFAULT_SPLITS)
    stream_results = _stream_results_enabled(args)
    merged_rows: list[dict[str, Any]] | None = [] if not stream_results else None
    summary_accumulator = PreservationSummaryAccumulator()
    merged_row_count = 0
    output_rows_path = resolve_path(args.output_rows_csv, workspace_root) if getattr(args, "output_rows_csv", "") else None
    rows_handle, row_writer = _open_result_rows_writer(output_rows_path)
    try:
        for index, row_csv in enumerate(args.merge_rows_csvs, start=1):
            with resolve_path(row_csv, workspace_root).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    normalized = dict(row)
                    merged_row_count += 1
                    summary_accumulator.update(normalized)
                    _write_result_rows(row_writer, [normalized])
                    if merged_rows is not None:
                        merged_rows.append(normalized)
            _log_runtime_snapshot(f"merge_rows_csv_{index}")
    finally:
        if rows_handle is not None:
            rows_handle.close()
    print(f"Merging {len(args.merge_rows_csvs):,} row CSV file(s) into {merged_row_count:,} rows.", flush=True)
    summary = build_summary_payload_from_tables(
        summary_accumulator.build_tables(),
        metadata_path=str(resolve_path(args.metadata, workspace_root)),
        requested_splits=requested_splits,
        asr_backend=args.asr_backend,
        speaker_backend=args.speaker_backend,
    )
    summary["shard"] = {
        "num_shards": len(args.merge_rows_csvs),
        "shard_index": "merged",
    }
    return merged_rows, summary


def _parse_gpu_list(raw_value: str) -> list[str]:
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def run_multi_gpu(args: argparse.Namespace) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    gpu_ids = _parse_gpu_list(getattr(args, "gpus", ""))
    if len(gpu_ids) <= 1:
        return run_analysis(args)

    workspace_root = Path(args.workspace_root).resolve()
    shard_dir = resolve_path("results/.preservation_shards", workspace_root)
    shard_dir.mkdir(parents=True, exist_ok=True)
    num_shards = len(gpu_ids)
    print(f"Launching {num_shards} preservation shard workers on GPUs {gpu_ids}.", flush=True)
    if not bool(getattr(args, "no_progress", False)):
        print("Child progress output may interleave across GPUs.", flush=True)

    processes: list[tuple[int, str, Path, subprocess.Popen[str]]] = []
    for shard_index, gpu_id in enumerate(gpu_ids):
        shard_rows_path = shard_dir / f"preservation_rows.shard{shard_index}of{num_shards}.csv"
        command = _build_shard_command(args, shard_index=shard_index, num_shards=num_shards, row_csv_path=shard_rows_path)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        process = subprocess.Popen(command, cwd=str(workspace_root), env=env, text=True)
        processes.append((shard_index, gpu_id, shard_rows_path, process))

    failures: list[str] = []
    row_csvs: list[str] = []
    for shard_index, gpu_id, shard_rows_path, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append(f"shard {shard_index} on GPU {gpu_id} exited with code {return_code}")
        else:
            row_csvs.append(str(shard_rows_path))

    if failures:
        raise RuntimeError("; ".join(failures))

    merge_args = argparse.Namespace(**vars(args))
    merge_args.merge_rows_csvs = row_csvs
    merge_args.gpus = ""
    print(f"Merging shard outputs from {len(row_csvs)} row CSV file(s).", flush=True)
    return merge_row_csvs(merge_args)


def _build_shard_command(
    args: argparse.Namespace,
    *,
    shard_index: int,
    num_shards: int,
    row_csv_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
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
        "--output-rows-csv",
        str(row_csv_path),
        "--output-family-csv",
        "",
        "--output-family-language-csv",
        "",
        "--output-json",
        "",
        "--output-tex",
        "",
        "--output-table-tex",
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
    if bool(getattr(args, "resume", False)):
        command.append("--resume")
    if str(getattr(args, "asr_backend", "")):
        command.extend(["--asr-backend", str(args.asr_backend)])
    if str(getattr(args, "asr_backend_kwargs", "")):
        command.extend(["--asr-backend-kwargs", str(args.asr_backend_kwargs)])
    if str(getattr(args, "speaker_backend", "")):
        command.extend(["--speaker-backend", str(args.speaker_backend)])
    if str(getattr(args, "speaker_backend_kwargs", "")):
        command.extend(["--speaker-backend-kwargs", str(args.speaker_backend_kwargs)])
    return command


def write_outputs(
    args: argparse.Namespace,
    *,
    results: list[dict[str, Any]] | None,
    summary: dict[str, Any],
) -> None:
    workspace_root = Path(args.workspace_root).resolve()
    if args.output_rows_csv and results is not None:
        write_csv_with_headers(resolve_path(args.output_rows_csv, workspace_root), RESULT_FIELDNAMES, results)
    if args.output_family_csv:
        write_csv_with_headers(
            resolve_path(args.output_family_csv, workspace_root),
            SUMMARY_FIELDNAMES,
            summary.get("by_family", []),
        )
    if args.output_family_language_csv:
        write_csv_with_headers(
            resolve_path(args.output_family_language_csv, workspace_root),
            SUMMARY_FIELDNAMES,
            summary.get("by_family_language", []),
        )
    if args.output_json:
        write_json(resolve_path(args.output_json, workspace_root), summary)
    if args.output_tex:
        write_preservation_tex(resolve_path(args.output_tex, workspace_root), summary)
    if args.output_table_tex:
        write_preservation_table_tex(resolve_path(args.output_table_tex, workspace_root), summary)


def main() -> None:
    args = parse_args()
    if args.merge_rows_csvs:
        results, summary = merge_row_csvs(args)
    elif len(_parse_gpu_list(getattr(args, "gpus", ""))) > 1:
        results, summary = run_multi_gpu(args)
    else:
        results, summary = run_analysis(args)
    write_outputs(args, results=results, summary=summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
