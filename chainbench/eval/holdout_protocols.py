"""Holdout protocol helpers for evaluation-time generalization experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .rows import sample_exact_rows, stable_row_token, standard_split_value, target_sample_size

IN_CHAIN_DETECTION_TASK_ID = "in_chain_detection"
LEAVE_ONE_TEMPLATE_OUT_PROTOCOL = "leave_one_template_out"
PER_FAMILY_SCOPE = "per_family"
GENERALIZATION_PROTOCOLS = (LEAVE_ONE_TEMPLATE_OUT_PROTOCOL,)
GENERALIZATION_SCOPES = (PER_FAMILY_SCOPE,)
SUPPORTED_TASKS_BY_PROTOCOL = {
    LEAVE_ONE_TEMPLATE_OUT_PROTOCOL: (IN_CHAIN_DETECTION_TASK_ID,),
}


@dataclass(frozen=True)
class TemplateHoldoutFold:
    chain_family: str
    template_id: str
    variant: str
    train_rows: list[dict[str, Any]]
    dev_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    dropped_rows: int


def normalize_generalization_config(raw: Any, *, selected_tasks: list[str]) -> dict[str, str] | None:
    if raw in (None, ""):
        return None
    if not isinstance(raw, dict):
        raise ValueError("Config field 'generalization' must be a mapping when provided")

    protocol = str(raw.get("protocol", "")).strip()
    if protocol not in GENERALIZATION_PROTOCOLS:
        raise ValueError(
            "Config field 'generalization.protocol' must be one of: "
            + ", ".join(sorted(GENERALIZATION_PROTOCOLS))
        )

    scope = str(raw.get("scope", PER_FAMILY_SCOPE)).strip() or PER_FAMILY_SCOPE
    if scope not in GENERALIZATION_SCOPES:
        raise ValueError(
            "Config field 'generalization.scope' must be one of: " + ", ".join(sorted(GENERALIZATION_SCOPES))
        )

    supported_tasks = set(SUPPORTED_TASKS_BY_PROTOCOL[protocol])
    unsupported_tasks = [task_id for task_id in selected_tasks if task_id not in supported_tasks]
    if unsupported_tasks:
        raise ValueError(
            f"Generalization protocol {protocol!r} currently supports only tasks: "
            + ", ".join(sorted(supported_tasks))
        )

    return {
        "protocol": protocol,
        "scope": scope,
    }


def build_template_holdout_detection_folds(rows: list[dict[str, Any]]) -> list[TemplateHoldoutFold]:
    by_family_template: dict[tuple[str, str], list[dict[str, Any]]] = {}
    family_rows: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        family = _chain_family(row)
        template_id = _template_id(row)
        if not family or not template_id:
            continue
        row_copy = dict(row)
        by_family_template.setdefault((family, template_id), []).append(row_copy)
        family_rows.setdefault(family, []).append(row_copy)

    folds: list[TemplateHoldoutFold] = []
    for family, template_id in sorted(by_family_template):
        train_rows: list[dict[str, Any]] = []
        dev_rows: list[dict[str, Any]] = []
        test_rows: list[dict[str, Any]] = []
        dropped_rows = 0

        for row in family_rows.get(family, []):
            split = standard_split_value(row)
            row_template_id = _template_id(row)
            if not split:
                dropped_rows += 1
                continue
            if row_template_id == template_id:
                if split == "test":
                    test_rows.append(_with_split(row, "test"))
                else:
                    dropped_rows += 1
                continue
            if split == "train":
                train_rows.append(_with_split(row, "train"))
            elif split == "dev":
                dev_rows.append(_with_split(row, "dev"))
            else:
                dropped_rows += 1

        if not train_rows or not test_rows:
            continue
        if not dev_rows:
            train_rows, dev_rows = _fallback_dev_split(train_rows, family=family, template_id=template_id)
        if not dev_rows:
            continue

        folds.append(
            TemplateHoldoutFold(
                chain_family=family,
                template_id=template_id,
                variant=template_holdout_variant(family, template_id),
                train_rows=train_rows,
                dev_rows=dev_rows,
                test_rows=test_rows,
                dropped_rows=dropped_rows,
            )
        )
    return folds


def template_holdout_variant(chain_family: str, template_id: str) -> str:
    raw_variant = f"{chain_family}__template__{template_id}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_variant).strip("._-") or "template_holdout"


def _chain_family(row: dict[str, Any]) -> str:
    return str(row.get("chain_family", "")).strip()


def _template_id(row: dict[str, Any]) -> str:
    return str(row.get("chain_template_id", "")).strip()


def _with_split(row: dict[str, Any], split: str) -> dict[str, Any]:
    out = dict(row)
    out["split_standard"] = split
    return out


def _fallback_dev_split(
    train_rows: list[dict[str, Any]],
    *,
    family: str,
    template_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(train_rows) < 2:
        return train_rows, []
    dev_target = min(len(train_rows) - 1, target_sample_size(len(train_rows), 0.1))
    if dev_target <= 0:
        return train_rows, []
    fallback_dev_rows = sample_exact_rows(
        train_rows,
        dev_target,
        salt=f"template_holdout_dev:{family}:{template_id}",
    )
    dev_tokens = {stable_row_token(row) for row in fallback_dev_rows}
    next_train_rows = [row for row in train_rows if stable_row_token(row) not in dev_tokens]
    next_dev_rows = [_with_split(row, "dev") for row in fallback_dev_rows]
    return next_train_rows, next_dev_rows
