"""Job assignment and enrichment for stage3."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lib.config import resolve_path, relative_to_workspace


def choose_prompt_reference(
    row: dict[str, str],
    speaker_rows: list[dict[str, str]],
    generator_key: str,
    seed: int,
) -> dict[str, str]:
    alternatives = [c for c in speaker_rows if c["parent_id"] != row["parent_id"]]
    if not alternatives:
        return row
    alternatives.sort(key=lambda item: item["parent_id"])
    idx = random.Random(f"{seed}:{generator_key}:{row['parent_id']}").randrange(len(alternatives))
    return alternatives[idx]


def get_active_generators(config: dict[str, Any], only_keys: list[str] | None) -> dict[str, dict[str, Any]]:
    selected = {}
    only = set(only_keys or [])
    for key, value in config["generators"].items():
        if not value.get("enabled", False):
            continue
        if only and key not in only:
            continue
        selected[key] = value
    if not selected:
        raise RuntimeError("No active generators selected")
    return selected


def get_conda_env_names() -> set[str]:
    from lib.proc import run_command
    result = run_command(["conda", "env", "list", "--json"])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to query conda envs: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    return {Path(p).name for p in payload.get("envs", [])}


def preflight_generators(
    generator_cfgs: dict[str, dict[str, Any]],
    workspace_root: Path,
    plan_only: bool,
) -> None:
    missing_envs = []
    missing_repos = []
    for key, generator in generator_cfgs.items():
        repo_path = workspace_root / generator["repo_path"]
        if not repo_path.exists():
            missing_repos.append(f"{key}:{repo_path}")
    if missing_repos:
        raise RuntimeError("Missing generator repo paths: " + ", ".join(missing_repos))
    if plan_only:
        return
    env_names = get_conda_env_names()
    for key, generator in generator_cfgs.items():
        env_name = generator["conda_env"]
        if env_name not in env_names:
            missing_envs.append(f"{key}:{env_name}")
    if missing_envs:
        raise RuntimeError("Missing generator conda envs: " + ", ".join(missing_envs))


def assign_generators(
    rows: list[dict[str, str]],
    generator_cfgs: dict[str, dict[str, Any]],
    generators_per_parent: int,
    seed: int,
) -> list[dict[str, Any]]:
    counts = Counter()
    tiebreak = {key: idx for idx, key in enumerate(sorted(generator_cfgs))}
    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)
    jobs: list[dict[str, Any]] = []
    rows_by_speaker: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_speaker[(row["language"], row["speaker_id"])].append(row)

    for row in shuffled_rows:
        supported = [
            key
            for key, gen_cfg in generator_cfgs.items()
            if row["language"] in gen_cfg.get("supported_languages", [])
        ]
        if len(supported) < generators_per_parent:
            raise RuntimeError(
                f"Parent {row['parent_id']} language={row['language']} has only {len(supported)} compatible generators"
            )
        supported.sort(key=lambda k: (counts[k], tiebreak[k], k))
        chosen = supported[:generators_per_parent]
        for assignment_idx, generator_key in enumerate(chosen, start=1):
            counts[generator_key] += 1
            prompt_row = choose_prompt_reference(
                row,
                rows_by_speaker[(row["language"], row["speaker_id"])],
                generator_key,
                seed,
            )
            spoof_parent_id = f"{row['parent_id']}__{generator_key}"
            jobs.append(
                {
                    "source_row": row,
                    "generator_key": generator_key,
                    "assignment_idx": assignment_idx,
                    "prompt_row": prompt_row,
                    "spoof_parent_id": spoof_parent_id,
                }
            )
    return jobs


def enrich_jobs(
    assignments: list[dict[str, Any]],
    generator_cfgs: dict[str, dict[str, Any]],
    config: dict[str, Any],
    workspace_root: Path,
    output_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    jobs_by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_root = output_root / "audio_raw"
    final_root = output_root / "audio"
    seed = int(config["seed"])

    for item in assignments:
        row = item["source_row"]
        generator_key = item["generator_key"]
        prompt_row = item["prompt_row"]
        generator = generator_cfgs[generator_key]
        job_id = item["spoof_parent_id"]
        raw_path = raw_root / generator_key / row["language"] / row["split"] / row["speaker_id"] / f"{job_id}.wav"
        final_path = final_root / row["language"] / row["split"] / row["speaker_id"] / f"{job_id}.wav"
        jobs_by_generator[generator_key].append(
            {
                "source_row": row,
                "prompt_row": prompt_row,
                "job_id": job_id,
                "sample_id": job_id,
                "parent_id": job_id,
                "source_parent_id": row["parent_id"],
                "prompt_parent_id": prompt_row["parent_id"],
                "assignment_idx": item["assignment_idx"],
                "generator_key": generator_key,
                "generator_name": generator["generator_name"],
                "generator_family": generator["generator_family"],
                "language": row["language"],
                "split": row["split"],
                "source_corpus": row["source_corpus"],
                "speaker_id": row["speaker_id"],
                "source_speaker_id": row["source_speaker_id"],
                "utterance_id": row["utterance_id"],
                "text": row["transcript"],
                "transcript": row["transcript"],
                "raw_transcript": row.get("raw_transcript", ""),
                "prompt_text": prompt_row["transcript"],
                "prompt_audio_path": str(resolve_path(prompt_row["clean_parent_path"], workspace_root)),
                "source_audio_path": str(resolve_path(row["clean_parent_path"], workspace_root)),
                "source_duration_sec": float(row["duration_sec"]),
                "source_sample_rate": row["sample_rate"],
                "source_channels": row.get("channels", ""),
                "source_codec_name": row.get("codec_name", ""),
                "raw_output_path": str(raw_path.resolve()),
                "output_path": str(raw_path.resolve()),
                "final_output_path": str(final_path.resolve()),
                "raw_output_relpath": relative_to_workspace(raw_path, workspace_root),
                "final_output_relpath": relative_to_workspace(final_path, workspace_root),
                "source_clean_parent_path": row.get("clean_parent_path", ""),
                "license_tag": row.get("license_tag", ""),
                "speaker_gender": row.get("speaker_gender", ""),
                "speaker_age": row.get("speaker_age", ""),
                "speaker_accent": row.get("speaker_accent", ""),
                "speaker_variant": row.get("speaker_variant", ""),
                "sentence_id": row.get("sentence_id", ""),
                "locale": row.get("locale", ""),
                "sentence_domain": row.get("sentence_domain", ""),
                "seed": seed,
            }
        )
    return jobs_by_generator
