"""Collect and validate spoof results for stage3."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from chainbench.lib.logging import format_elapsed
from chainbench.lib.audio import ffprobe_audio
from chainbench.lib.config import relative_to_workspace
from chainbench.lib.io import load_jsonl

from .postprocess import postprocess_audio, validate_spoof_output


def extract_traceback_or_tail(log_path: Path, max_lines: int = 40) -> str:
    if not log_path.exists():
        return "<log file missing>"
    with log_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    if not lines:
        return "<log file empty>"
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].startswith("Traceback (most recent call last):"):
            return "\n".join(lines[idx : idx + max_lines])
    return "\n".join(lines[-max_lines:])


def _postprocess_one(
    args: tuple[str, str, dict[str, Any], Path, Path, dict[str, Any]],
) -> tuple[str, str, str | None]:
    job_id, generator_key, _job, raw_path, final_path, config = args
    err = postprocess_audio(raw_path, final_path, config)
    return (job_id, generator_key, err)


def collect_spoof_rows(
    jobs_by_generator: dict[str, list[dict[str, Any]]],
    generator_cfgs: dict[str, dict[str, Any]],
    config: dict[str, Any],
    output_root: Path,
    workspace_root: Path,
    log_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Counter]]:
    spoof_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stats_by_generator: dict[str, Counter] = defaultdict(Counter)
    processed = 0
    total_results = sum(len(jobs) for jobs in jobs_by_generator.values())
    started_at = time.monotonic()
    timeout_cfg = config.get("timeouts", {})
    ffprobe_timeout_sec = int(timeout_cfg.get("ffprobe_sec", 30))

    need_postprocess: list[tuple[str, str, dict[str, Any], Path, Path, dict[str, Any]]] = []
    for generator_key, jobs in jobs_by_generator.items():
        job_map = {job["job_id"]: job for job in jobs}
        results_path = output_root / "results" / f"{generator_key}.jsonl"
        results = load_jsonl(results_path)
        for result in results:
            if result["status"] == "failed":
                continue
            job = job_map[result["job_id"]]
            raw_path = Path(job["raw_output_path"])
            final_path = Path(job["final_output_path"])
            if not (final_path.exists() and final_path.stat().st_size > 0):
                need_postprocess.append((job["job_id"], generator_key, job, raw_path, final_path, config))

    postprocess_results: dict[tuple[str, str], str | None] = {}
    if need_postprocess:
        postprocess_workers = max(1, int(config.get("postprocess_workers", 8)))
        n_postprocess = len(need_postprocess)
        with ThreadPoolExecutor(max_workers=postprocess_workers) as executor:
            with tqdm(total=n_postprocess, desc="postprocess", unit="audio", dynamic_ncols=True) as progress:
                for job_id, generator_key, err in executor.map(_postprocess_one, need_postprocess):
                    postprocess_results[(job_id, generator_key)] = err
                    progress.update(1)

    with tqdm(total=total_results, desc="stage3 validate", unit="job", dynamic_ncols=True) as validate_progress:
        for generator_key, jobs in jobs_by_generator.items():
            job_map = {job["job_id"]: job for job in jobs}
            results_path = output_root / "results" / f"{generator_key}.jsonl"
            results = load_jsonl(results_path)
            stats = stats_by_generator[generator_key]
            for result in results:
                processed += 1
                job = job_map[result["job_id"]]
                stats[result["status"]] += 1
                if result["status"] == "failed":
                    failures.append(
                        {
                            "job_id": job["job_id"],
                            "generator_key": generator_key,
                            "stage": "generation",
                            "error": result.get("error", ""),
                            "source_parent_id": job["source_parent_id"],
                            "prompt_parent_id": job["prompt_parent_id"],
                        }
                    )
                    validate_progress.update(1)
                    if processed <= 5 or processed % log_every == 0 or processed == total_results:
                        validate_progress.set_postfix(
                            generator=generator_key,
                            ok=len(spoof_rows),
                            fail=len(failures),
                            elapsed=format_elapsed(time.monotonic() - started_at),
                        )
                    continue

                raw_path = Path(job["raw_output_path"])
                final_path = Path(job["final_output_path"])
                key = (job["job_id"], generator_key)
                post_error = (
                    postprocess_results.get(key)
                    if key in postprocess_results
                    else postprocess_audio(raw_path, final_path, config)
                )
                if post_error is not None:
                    stats["postprocess_failed"] += 1
                    failures.append(
                        {
                            "job_id": job["job_id"],
                            "generator_key": generator_key,
                            "stage": "postprocess",
                            "error": post_error,
                            "source_parent_id": job["source_parent_id"],
                        }
                    )
                    validate_progress.update(1)
                    if processed <= 5 or processed % log_every == 0 or processed == total_results:
                        validate_progress.set_postfix(
                            generator=generator_key,
                            ok=len(spoof_rows),
                            fail=len(failures),
                            elapsed=format_elapsed(time.monotonic() - started_at),
                        )
                    continue

                probe = ffprobe_audio(final_path, timeout_sec=ffprobe_timeout_sec)
                if probe is None:
                    stats["ffprobe_failed"] += 1
                    failures.append(
                        {
                            "job_id": job["job_id"],
                            "generator_key": generator_key,
                            "stage": "ffprobe",
                            "error": f"ffprobe failed for {final_path}",
                            "source_parent_id": job["source_parent_id"],
                        }
                    )
                    validate_progress.update(1)
                    if processed <= 5 or processed % log_every == 0 or processed == total_results:
                        validate_progress.set_postfix(
                            generator=generator_key,
                            ok=len(spoof_rows),
                            fail=len(failures),
                            elapsed=format_elapsed(time.monotonic() - started_at),
                        )
                    continue

                validation_error = validate_spoof_output(
                    probe, float(job["source_duration_sec"]), config
                )
                if validation_error is not None:
                    stats["validation_failed"] += 1
                    failures.append(
                        {
                            "job_id": job["job_id"],
                            "generator_key": generator_key,
                            "stage": "validation",
                            "error": validation_error,
                            "source_parent_id": job["source_parent_id"],
                        }
                    )
                    validate_progress.update(1)
                    if processed <= 5 or processed % log_every == 0 or processed == total_results:
                        validate_progress.set_postfix(
                            generator=generator_key,
                            ok=len(spoof_rows),
                            fail=len(failures),
                            elapsed=format_elapsed(time.monotonic() - started_at),
                        )
                    continue

                generator = generator_cfgs[generator_key]
                spoof_rows.append(
                    {
                        "sample_id": job["sample_id"],
                        "parent_id": job["parent_id"],
                        "source_parent_id": job["source_parent_id"],
                        "prompt_parent_id": job["prompt_parent_id"],
                        "split": job["split"],
                        "language": job["language"],
                        "source_corpus": job["source_corpus"],
                        "speaker_id": job["speaker_id"],
                        "source_speaker_id": job["source_speaker_id"],
                        "utterance_id": job["utterance_id"],
                        "transcript": job["transcript"],
                        "raw_transcript": job.get("raw_transcript", ""),
                        "label": "spoof",
                        "generator_family": generator["generator_family"],
                        "generator_name": generator["generator_name"],
                        "chain_family": "clean_parent",
                        "operator_seq": "[]",
                        "operator_params": "{}",
                        "seed": str(job["seed"]),
                        "duration_sec": f"{probe['duration']:.3f}",
                        "sample_rate": str(probe["sample_rate"]),
                        "channels": str(probe["channels"]),
                        "codec_name": probe["codec_name"],
                        "sample_fmt": probe.get("sample_fmt", ""),
                        "source_duration_sec": str(job["source_duration_sec"]),
                        "source_sample_rate": job["source_sample_rate"],
                        "source_channels": job.get("source_channels", ""),
                        "source_codec_name": job.get("source_codec_name", ""),
                        "clean_parent_path": job["final_output_relpath"],
                        "audio_path": job["final_output_relpath"],
                        "source_clean_parent_path": job.get("source_clean_parent_path", ""),
                        "raw_generator_output_path": job["raw_output_relpath"],
                        "prompt_audio_path": relative_to_workspace(Path(job["prompt_audio_path"]), workspace_root),
                        "output_size_bytes": str(probe.get("size", 0)),
                        "license_tag": job.get("license_tag", ""),
                        "speaker_gender": job.get("speaker_gender", ""),
                        "speaker_age": job.get("speaker_age", ""),
                        "speaker_accent": job.get("speaker_accent", ""),
                        "speaker_variant": job.get("speaker_variant", ""),
                        "sentence_id": job.get("sentence_id", ""),
                        "locale": job.get("locale", ""),
                        "sentence_domain": job.get("sentence_domain", ""),
                        "assignment_idx": str(job["assignment_idx"]),
                        "generator_key": generator_key,
                        "stage3_status": result["status"],
                    }
                )
                stats["validated_ok"] += 1
                validate_progress.update(1)
                if processed <= 5 or processed % log_every == 0 or processed == total_results:
                    validate_progress.set_postfix(
                        generator=generator_key,
                        ok=len(spoof_rows),
                        fail=len(failures),
                        elapsed=format_elapsed(time.monotonic() - started_at),
                    )

            missing = len(jobs) - len(results)
            if missing > 0:
                stats["missing_results"] += missing
                for job in jobs:
                    if job["job_id"] not in {r["job_id"] for r in results}:
                        failures.append(
                            {
                                "job_id": job["job_id"],
                                "generator_key": generator_key,
                                "stage": "results",
                                "error": "missing result entry",
                                "source_parent_id": job["source_parent_id"],
                            }
                        )

    return spoof_rows, failures, stats_by_generator


def summarize_spoof_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_language: dict[str, Counter] = defaultdict(Counter)
    by_generator: dict[str, Counter] = defaultdict(Counter)
    speakers_by_language: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_language[row["language"]][row["split"]] += 1
        by_generator[row["generator_key"]][row["language"]] += 1
        speakers_by_language[row["language"]].add(row["speaker_id"])
    return {
        "languages": {
            language: {
                "selected_samples": sum(counter.values()),
                "selected_speakers": len(speakers_by_language[language]),
                "split_sample_counts": dict(counter),
            }
            for language, counter in by_language.items()
        },
        "generators": {key: dict(counter) for key, counter in by_generator.items()},
    }
