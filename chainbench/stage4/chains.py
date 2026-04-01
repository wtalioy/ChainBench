"""Chain template sampling and job creation for stage4."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from chainbench.lib.config import resolve_path


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
            codec_name = str(sampled["codec"]).lower()
            if codec_name == "gsm":
                sampled["encode_sample_rate"] = 8000
                sampled["decode_sample_rate"] = int(sampled.get("decode_sample_rate", 8000) or 8000)
            context["last_codec"] = sampled["codec"]
            if sampled.get("bitrate"):
                context["last_bitrate"] = sampled["bitrate"]
        if sampled["op"] == "bandlimit":
            context["bandwidth_mode"] = sampled["mode"]
        operators.append(sampled)
    return operators


def _family_variants_per_parent(family_cfg: dict[str, Any], config: dict[str, Any]) -> int:
    return max(1, int(family_cfg.get("variants_per_parent", config.get("variants_per_parent", 1))))


def _paired_group_probability(family_cfg: dict[str, Any]) -> float:
    probability = float(family_cfg.get("paired_group_probability", 0.0))
    return min(1.0, max(0.0, probability))


def _template_by_id(templates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(template["template_id"]): template for template in templates}


def _template_operator_names(template: dict[str, Any]) -> list[str]:
    return [str(operator["op"]) for operator in template["operators"]]


def _reorder_sampled_operators(
    sampled_operators: list[dict[str, Any]],
    target_template: dict[str, Any],
) -> list[dict[str, Any]]:
    source_by_op: dict[str, dict[str, Any]] = {}
    for operator in sampled_operators:
        op_name = str(operator.get("op", "")).strip()
        if not op_name:
            raise ValueError("sampled operator is missing 'op'")
        if op_name in source_by_op:
            raise ValueError(f"paired template reordering does not support duplicate operator names: {op_name}")
        source_by_op[op_name] = dict(operator)

    reordered: list[dict[str, Any]] = []
    for operator in target_template["operators"]:
        op_name = str(operator["op"])
        if op_name not in source_by_op:
            raise ValueError(f"paired template target op {op_name!r} missing from sampled operators")
        reordered.append(dict(source_by_op[op_name]))
    return reordered


def _paired_template_jobs(
    *,
    row: dict[str, str],
    family_name: str,
    family_cfg: dict[str, Any],
    config: dict[str, Any],
    workspace_root: Path,
    seed: int,
    template_id_group: list[str],
    start_variant_index: int,
) -> list[dict[str, Any]]:
    templates = _template_by_id(list(family_cfg["templates"]))
    group_templates = [templates[template_id] for template_id in template_id_group]
    if len(group_templates) < 2:
        return []
    canonical_template = group_templates[0]
    canonical_ops = _template_operator_names(canonical_template)
    canonical_op_set = sorted(canonical_ops)
    for template in group_templates[1:]:
        target_ops = _template_operator_names(template)
        if sorted(target_ops) != canonical_op_set:
            raise ValueError(
                f"paired templates must share the same operator multiset: "
                f"{canonical_template['template_id']} vs {template['template_id']}"
            )

    rng = random.Random(f"{seed}:{row['parent_id']}:{family_name}:paired:{','.join(template_id_group)}")
    sampled_base = concretize_template(family_name, canonical_template, config["parameter_pools"], rng)
    jobs: list[dict[str, Any]] = []
    for offset, template in enumerate(group_templates):
        operators = (
            [dict(operator) for operator in sampled_base]
            if offset == 0
            else _reorder_sampled_operators(sampled_base, template)
        )
        variant_index = start_variant_index + offset
        variant_suffix = f"__v{variant_index + 1:02d}"
        sample_id = f"{row['parent_id']}__{family_name}__{template['template_id']}{variant_suffix}"
        jobs.append(
            {
                "job_id": sample_id,
                "sample_id": sample_id,
                "parent_id": row["parent_id"],
                "family_name": family_name,
                "template_id": template["template_id"],
                "variant_index": variant_index,
                "operators": operators,
                "source_row": row,
                "source_audio_path_abs": str(resolve_path(row["audio_path"], workspace_root)),
            }
        )
    return jobs


def count_jobs(
    rows: Iterable[dict[str, str]],
    config: dict[str, Any],
    selected_families: list[str],
) -> Counter:
    family_counts: Counter = Counter()
    family_cfgs = config["families"]
    for _row in rows:
        for family_name in selected_families:
            family_cfg = family_cfgs[family_name]
            templates = family_cfg["templates"]
            if not templates:
                continue
            family_counts[family_name] += _family_variants_per_parent(family_cfg, config)
    return family_counts


def iter_sample_jobs(
    rows: list[dict[str, str]],
    config: dict[str, Any],
    selected_families: list[str],
    workspace_root: Path,
) -> Iterator[dict[str, Any]]:
    family_cfgs = config["families"]
    seed = int(config["seed"])
    for row in rows:
        parent_id = row["parent_id"]
        for family_name in selected_families:
            family_cfg = family_cfgs[family_name]
            templates = list(family_cfg["templates"])
            if not templates:
                continue
            variants_per_parent = _family_variants_per_parent(family_cfg, config)
            family_rng = random.Random(f"{seed}:{parent_id}:{family_name}:template_order")
            template_indices = list(range(len(templates)))
            family_rng.shuffle(template_indices)

            emitted_jobs: list[dict[str, Any]] = []
            used_template_ids: set[str] = set()
            paired_template_groups = [
                [str(template_id) for template_id in group.get("template_ids", [])]
                for group in family_cfg.get("paired_template_groups", [])
                if len(group.get("template_ids", [])) >= 2
            ]
            pair_probability = _paired_group_probability(family_cfg)
            if paired_template_groups and variants_per_parent >= 2 and family_rng.random() < pair_probability:
                eligible_groups = [
                    group
                    for group in paired_template_groups
                    if len(group) <= variants_per_parent
                ]
                if eligible_groups:
                    selected_group = family_rng.choice(eligible_groups)
                    emitted_jobs.extend(
                        _paired_template_jobs(
                            row=row,
                            family_name=family_name,
                            family_cfg=family_cfg,
                            config=config,
                            workspace_root=workspace_root,
                            seed=seed,
                            template_id_group=selected_group,
                            start_variant_index=0,
                        )
                    )
                    used_template_ids.update(selected_group)

            variant_index = len(emitted_jobs)
            for template_index in template_indices:
                if variant_index >= variants_per_parent:
                    break
                template = templates[template_index]
                if str(template["template_id"]) in used_template_ids:
                    continue
                rng = random.Random(f"{seed}:{parent_id}:{family_name}:{variant_index}")
                operators = concretize_template(family_name, template, config["parameter_pools"], rng)
                variant_suffix = f"__v{variant_index + 1:02d}" if variants_per_parent > 1 else ""
                sample_id = f"{parent_id}__{family_name}__{template['template_id']}{variant_suffix}"
                emitted_jobs.append(
                    {
                        "job_id": sample_id,
                        "sample_id": sample_id,
                        "parent_id": parent_id,
                        "family_name": family_name,
                        "template_id": template["template_id"],
                        "variant_index": variant_index,
                        "operators": operators,
                        "source_row": row,
                        "source_audio_path_abs": str(resolve_path(row["audio_path"], workspace_root)),
                    }
                )
                variant_index += 1

            for job in emitted_jobs:
                yield job


def sample_jobs(
    rows: list[dict[str, str]],
    config: dict[str, Any],
    selected_families: list[str],
    workspace_root: Path,
) -> list[dict[str, Any]]:
    return list(iter_sample_jobs(rows, config, selected_families, workspace_root))
