"""Derived group IDs for structure-first evaluation tasks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from .task_keys import operator_multiset_key, operator_signature_sequence, path_endpoint_key

OPERATOR_SUBSTITUTION_GROUP_FIELD = "operator_substitution_group_id"
OPERATOR_SUBSTITUTION_DETAIL_FIELD = "operator_substitution_detail"
PARAMETER_PERTURBATION_GROUP_FIELD = "parameter_perturbation_group_id"
PARAMETER_PERTURBATION_AXIS_FIELD = "parameter_perturbation_axis"
ORDER_SWAP_GROUP_FIELD = "order_swap_group_id"
PATH_GROUP_FIELD = "path_group_id"
OPERATOR_MULTISET_FIELD = "operator_multiset_key"
PATH_ENDPOINT_FIELD = "path_endpoint_key"
PATH_STEP_FIELD = "path_step_index"


def _stable_row_token(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    return json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("parent_id", "")).strip(),
        str(row.get("chain_family", "")).strip(),
        str(row.get("label", "")).strip(),
    )


def _pair_id(prefix: str, left: dict[str, Any], right: dict[str, Any]) -> str:
    left_id, right_id = sorted((_stable_row_token(left), _stable_row_token(right)))
    return hashlib.sha1(f"{prefix}\0{left_id}\0{right_id}".encode("utf-8")).hexdigest()


def _same_except_one(left: list[str], right: list[str]) -> tuple[bool, str]:
    if len(left) != len(right) or not left:
        return False, ""
    diffs = [index for index, (lval, rval) in enumerate(zip(left, right)) if lval != rval]
    if len(diffs) != 1:
        return False, ""
    diff_index = diffs[0]
    left_name = left[diff_index].split("[", 1)[0]
    right_name = right[diff_index].split("[", 1)[0]
    axis = "parameter" if left_name == right_name else "operator"
    return True, f"{axis}@{diff_index}"


def _operator_substitution_match(left: list[str], right: list[str]) -> tuple[bool, str]:
    """Same-length chains differ only where the operator *name* at that index changes.

    Allows multiple differing indices (not restricted to a single-index edit).
    Any differing index must have different operator names; same-name parameter-only
    differences are excluded.
    """
    if len(left) != len(right) or not left:
        return False, ""
    diff_indices = [index for index, (lval, rval) in enumerate(zip(left, right)) if lval != rval]
    if not diff_indices:
        return False, ""
    for index in diff_indices:
        left_name = left[index].split("[", 1)[0].strip()
        right_name = right[index].split("[", 1)[0].strip()
        if left_name == right_name:
            return False, ""
    detail = "indices=" + ",".join(str(index) for index in diff_indices)
    return True, detail


def _is_minimal_swap(left: list[str], right: list[str]) -> bool:
    if len(left) != len(right) or len(left) < 2 or sorted(left) != sorted(right):
        return False
    diffs = [index for index, (lval, rval) in enumerate(zip(left, right)) if lval != rval]
    if len(diffs) != 2:
        return False
    i, j = diffs
    swapped = list(left)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    return swapped == right


def _is_immediate_prefix(prefix: list[str], extended: list[str]) -> bool:
    return len(extended) == len(prefix) + 1 and extended[: len(prefix)] == prefix


def annotate_structural_group_fields(
    rows: list[dict[str, Any]],
    *,
    copy_rows: bool = True,
) -> list[dict[str, Any]]:
    annotated = [dict(row) for row in rows] if copy_rows else rows
    rows_by_token = {_stable_row_token(row): row for row in annotated}

    for row in annotated:
        row[OPERATOR_MULTISET_FIELD] = operator_multiset_key(row)
        row[PATH_ENDPOINT_FIELD] = path_endpoint_key(row)
        row[OPERATOR_SUBSTITUTION_GROUP_FIELD] = str(row.get(OPERATOR_SUBSTITUTION_GROUP_FIELD, "")).strip()
        row[OPERATOR_SUBSTITUTION_DETAIL_FIELD] = str(row.get(OPERATOR_SUBSTITUTION_DETAIL_FIELD, "")).strip()
        row[PARAMETER_PERTURBATION_GROUP_FIELD] = str(row.get(PARAMETER_PERTURBATION_GROUP_FIELD, "")).strip()
        row[PARAMETER_PERTURBATION_AXIS_FIELD] = str(row.get(PARAMETER_PERTURBATION_AXIS_FIELD, "")).strip()
        row[ORDER_SWAP_GROUP_FIELD] = str(row.get(ORDER_SWAP_GROUP_FIELD, "")).strip()
        row[PATH_GROUP_FIELD] = str(row.get(PATH_GROUP_FIELD, "")).strip()
        row[PATH_STEP_FIELD] = str(row.get(PATH_STEP_FIELD, "")).strip()

    grouped_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in annotated:
        if all(_group_key(row)):
            grouped_rows[_group_key(row)].append(row)

    for group_rows in grouped_rows.values():
        signatures_by_token = {
            _stable_row_token(row): operator_signature_sequence(row)
            for row in group_rows
        }

        used_intervention_rows: set[str] = set()
        parameter_candidates: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
        for left_index, left in enumerate(group_rows):
            left_id = _stable_row_token(left)
            left_sig = signatures_by_token[left_id]
            for right in group_rows[left_index + 1 :]:
                right_id = _stable_row_token(right)
                right_sig = signatures_by_token[right_id]
                matches, axis = _same_except_one(left_sig, right_sig)
                if not matches or not str(axis).startswith("parameter@"):
                    continue
                pair_id = _pair_id("parameter_perturbation", left, right)
                parameter_candidates.append((pair_id, left, right, axis))
        for pair_id, left, right, axis in sorted(parameter_candidates, key=lambda item: item[0]):
            left_id = _stable_row_token(left)
            right_id = _stable_row_token(right)
            if left_id in used_intervention_rows or right_id in used_intervention_rows:
                continue
            rows_by_token[left_id][PARAMETER_PERTURBATION_GROUP_FIELD] = pair_id
            rows_by_token[right_id][PARAMETER_PERTURBATION_GROUP_FIELD] = pair_id
            rows_by_token[left_id][PARAMETER_PERTURBATION_AXIS_FIELD] = axis
            rows_by_token[right_id][PARAMETER_PERTURBATION_AXIS_FIELD] = axis
            used_intervention_rows.update({left_id, right_id})

        operator_candidates: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
        for left_index, left in enumerate(group_rows):
            left_id = _stable_row_token(left)
            left_sig = signatures_by_token[left_id]
            for right in group_rows[left_index + 1 :]:
                right_id = _stable_row_token(right)
                right_sig = signatures_by_token[right_id]
                matches, detail = _operator_substitution_match(left_sig, right_sig)
                if not matches:
                    continue
                pair_id = _pair_id("operator_substitution", left, right)
                operator_candidates.append((pair_id, left, right, detail))
        for pair_id, left, right, detail in sorted(operator_candidates, key=lambda item: item[0]):
            left_id = _stable_row_token(left)
            right_id = _stable_row_token(right)
            if left_id in used_intervention_rows or right_id in used_intervention_rows:
                continue
            rows_by_token[left_id][OPERATOR_SUBSTITUTION_GROUP_FIELD] = pair_id
            rows_by_token[right_id][OPERATOR_SUBSTITUTION_GROUP_FIELD] = pair_id
            rows_by_token[left_id][OPERATOR_SUBSTITUTION_DETAIL_FIELD] = detail
            rows_by_token[right_id][OPERATOR_SUBSTITUTION_DETAIL_FIELD] = detail
            used_intervention_rows.update({left_id, right_id})

        used_order_swaps: set[str] = set()
        order_swap_candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for left_index, left in enumerate(group_rows):
            left_id = _stable_row_token(left)
            left_sig = signatures_by_token[left_id]
            for right in group_rows[left_index + 1 :]:
                right_id = _stable_row_token(right)
                right_sig = signatures_by_token[right_id]
                if not _is_minimal_swap(left_sig, right_sig):
                    continue
                pair_id = _pair_id("order_swap", left, right)
                order_swap_candidates.append((pair_id, left, right))
        for pair_id, left, right in sorted(order_swap_candidates, key=lambda item: item[0]):
            left_id = _stable_row_token(left)
            right_id = _stable_row_token(right)
            if left_id in used_order_swaps or right_id in used_order_swaps:
                continue
            rows_by_token[left_id][ORDER_SWAP_GROUP_FIELD] = pair_id
            rows_by_token[right_id][ORDER_SWAP_GROUP_FIELD] = pair_id
            used_order_swaps.update({left_id, right_id})

        used_path_rows: set[str] = set()
        path_candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for prefix in group_rows:
            prefix_id = _stable_row_token(prefix)
            prefix_sig = signatures_by_token[prefix_id]
            if not prefix_sig:
                continue
            for extended in group_rows:
                extended_id = _stable_row_token(extended)
                if prefix_id == extended_id:
                    continue
                extended_sig = signatures_by_token[extended_id]
                if not _is_immediate_prefix(prefix_sig, extended_sig):
                    continue
                pair_id = _pair_id("path", prefix, extended)
                path_candidates.append((pair_id, prefix, extended))
        for pair_id, prefix, extended in sorted(path_candidates, key=lambda item: item[0]):
            prefix_id = _stable_row_token(prefix)
            extended_id = _stable_row_token(extended)
            if prefix_id in used_path_rows or extended_id in used_path_rows:
                continue
            rows_by_token[prefix_id][PATH_GROUP_FIELD] = pair_id
            rows_by_token[extended_id][PATH_GROUP_FIELD] = pair_id
            step_index = str(len(signatures_by_token[extended_id]))
            rows_by_token[prefix_id][PATH_STEP_FIELD] = step_index
            rows_by_token[extended_id][PATH_STEP_FIELD] = step_index
            used_path_rows.update({prefix_id, extended_id})

    return annotated
