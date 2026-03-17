"""Chain template sampling and job creation for stage4."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from lib.config import resolve_path


def sample_spec(
    spec: Any,
    rng: random.Random,
    pools: dict[str, Any],
    context: dict[str, Any],
) -> Any:
    if isinstance(spec, dict):
        if "$pool" in spec:
            return rng.choice(pools[spec["$pool"]])
        if "$choice" in spec:
            return sample_spec(rng.choice(spec["$choice"]), rng, pools, context)
        if "$previous" in spec:
            return context.get(spec["$previous"])
        return {key: sample_spec(value, rng, pools, context) for key, value in spec.items()}
    if isinstance(spec, list):
        return [sample_spec(item, rng, pools, context) for item in spec]
    return spec


def concretize_template(
    family_name: str,
    template: dict[str, Any],
    pools: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    context: dict[str, Any] = {}
    operators: list[dict[str, Any]] = []
    for operator in template["operators"]:
        sampled = sample_spec(operator, rng, pools, context)
        if sampled["op"] in {"codec", "reencode"}:
            mode = sampled.get("mode")
            if sampled["op"] == "reencode":
                if mode == "same":
                    sampled["codec"] = context.get("last_codec", sampled.get("default_codec", "aac"))
                elif mode == "cross":
                    previous = context.get("last_codec", sampled.get("default_codec", "aac"))
                    candidates = [c for c in ["aac", "opus"] if c != previous]
                    sampled["codec"] = sampled.get("codec") if sampled.get("codec") in candidates else rng.choice(candidates)
                elif not sampled.get("codec"):
                    sampled["codec"] = sampled.get("default_codec", "aac")
            context["last_codec"] = sampled["codec"]
            if sampled.get("bitrate"):
                context["last_bitrate"] = sampled["bitrate"]
        if sampled["op"] == "bandlimit":
            context["bandwidth_mode"] = sampled["mode"]
        operators.append(sampled)
    return operators


def sample_jobs(
    rows: list[dict[str, str]],
    config: dict[str, Any],
    selected_families: list[str],
    workspace_root: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    family_cfgs = config["families"]
    seed = int(config["seed"])
    for row in rows:
        parent_id = row["parent_id"]
        for family_name in selected_families:
            family_cfg = family_cfgs[family_name]
            rng = random.Random(f"{seed}:{parent_id}:{family_name}")
            template = rng.choice(family_cfg["templates"])
            operators = concretize_template(family_name, template, config["parameter_pools"], rng)
            sample_id = f"{parent_id}__{family_name}__{template['template_id']}"
            jobs.append(
                {
                    "job_id": sample_id,
                    "sample_id": sample_id,
                    "parent_id": parent_id,
                    "family_name": family_name,
                    "template_id": template["template_id"],
                    "operators": operators,
                    "source_row": row,
                    "source_audio_path_abs": str(resolve_path(row["audio_path"], workspace_root)),
                }
            )
    return jobs
