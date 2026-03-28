"""Delivery-lineage robustness metrics."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from ..tasks.task_keys import operator_signature_sequence

from ..row_utils import binary_label_as_int, bucket_rows, normalize_binary_label, stable_row_token
from .core import parse_score
from .interventions import paired_prediction


def delivery_lineage_key(row: dict[str, Any]) -> str:
    label = normalize_binary_label(row.get("label"))
    if not label:
        return ""
    return "|".join(
        [
            str(row.get("parent_id", "")).strip(),
            str(row.get("chain_family", "")).strip(),
            label,
        ]
    )


def operator_signature_tokens(row: dict[str, Any]) -> list[str]:
    return [str(token) for token in operator_signature_sequence(row)]


def operator_name(token: str) -> str:
    return str(token).split("[", 1)[0].strip()


def single_difference_index(tokens_a: list[str], tokens_b: list[str]) -> int:
    if len(tokens_a) != len(tokens_b):
        return -1
    diffs = [index for index, (left, right) in enumerate(zip(tokens_a, tokens_b)) if left != right]
    if len(diffs) != 1:
        return -1
    return diffs[0]


def is_single_parameter_perturbation(tokens_a: list[str], tokens_b: list[str]) -> bool:
    diff_index = single_difference_index(tokens_a, tokens_b)
    if diff_index < 0:
        return False
    left_name = operator_name(tokens_a[diff_index])
    right_name = operator_name(tokens_b[diff_index])
    return bool(left_name and left_name == right_name)


def is_single_operator_substitution(tokens_a: list[str], tokens_b: list[str]) -> bool:
    diff_index = single_difference_index(tokens_a, tokens_b)
    if diff_index < 0:
        return False
    left_name = operator_name(tokens_a[diff_index])
    right_name = operator_name(tokens_b[diff_index])
    return bool(left_name and right_name and left_name != right_name)


def is_adjacent_order_swap(tokens_a: list[str], tokens_b: list[str]) -> bool:
    if len(tokens_a) != len(tokens_b) or len(tokens_a) < 2:
        return False
    diffs = [index for index, (left, right) in enumerate(zip(tokens_a, tokens_b)) if left != right]
    if len(diffs) != 2:
        return False
    left_index, right_index = diffs
    if right_index != left_index + 1:
        return False
    swapped = list(tokens_a)
    swapped[left_index], swapped[right_index] = swapped[right_index], swapped[left_index]
    return swapped == tokens_b


def is_single_insertion_or_deletion(tokens_a: list[str], tokens_b: list[str]) -> bool:
    if abs(len(tokens_a) - len(tokens_b)) != 1:
        return False
    shorter, longer = (tokens_a, tokens_b) if len(tokens_a) < len(tokens_b) else (tokens_b, tokens_a)
    mismatch_seen = False
    short_index = 0
    long_index = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        if mismatch_seen:
            return False
        mismatch_seen = True
        long_index += 1
    return True


def is_atomic_delivery_edit(tokens_a: list[str], tokens_b: list[str]) -> bool:
    if tokens_a == tokens_b:
        return False
    if is_single_parameter_perturbation(tokens_a, tokens_b):
        return True
    if is_single_operator_substitution(tokens_a, tokens_b):
        return True
    if is_adjacent_order_swap(tokens_a, tokens_b):
        return True
    if is_single_insertion_or_deletion(tokens_a, tokens_b):
        return True
    return False


def signature_key(tokens: list[str]) -> tuple[str, ...]:
    return tuple(tokens)


def delivery_graph_distances(lineage_rows: list[dict[str, Any]], reference_signature: tuple[str, ...]) -> dict[tuple[str, ...], int]:
    signatures = sorted({signature_key(operator_signature_tokens(row)) for row in lineage_rows})
    if reference_signature not in signatures:
        signatures.append(reference_signature)
        signatures.sort()

    adjacency: dict[tuple[str, ...], set[tuple[str, ...]]] = {signature: set() for signature in signatures}
    for index, left in enumerate(signatures):
        left_tokens = list(left)
        for right in signatures[index + 1 :]:
            if not is_atomic_delivery_edit(left_tokens, list(right)):
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)
    distances: dict[tuple[str, ...], int] = {reference_signature: 0}
    queue: deque[tuple[str, ...]] = deque([reference_signature])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current]):
            if neighbor in distances:
                continue
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    return distances


def compute_delivery_robustness_metrics(scores: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    lineages = bucket_rows(scores, delivery_lineage_key)
    lineage_records: list[dict[str, Any]] = []
    max_distance = 0
    for lineage_key, lineage_rows in lineages.items():
        if not lineage_key or len(lineage_rows) < 2:
            continue
        ordered = sorted(
            lineage_rows,
            key=lambda row: (len(operator_signature_tokens(row)), stable_row_token(row)),
        )
        reference_row = ordered[0]
        reference_signature = signature_key(operator_signature_tokens(reference_row))
        graph_distances = delivery_graph_distances(lineage_rows, reference_signature)
        failure_distances: list[int] = []
        by_distance: dict[int, list[bool]] = defaultdict(list)
        for row in lineage_rows:
            label = binary_label_as_int(row.get("label"))
            if label is None:
                continue
            score = parse_score(row.get("score"))
            pred = paired_prediction(score, threshold)
            correct = pred == label
            distance = graph_distances.get(signature_key(operator_signature_tokens(row)))
            if distance is None:
                continue
            max_distance = max(max_distance, distance)
            by_distance[distance].append(correct)
            if not correct:
                failure_distances.append(distance)
        if not by_distance:
            continue
        lineage_records.append(
            {
                "label": normalize_binary_label(reference_row.get("label")),
                "radius": min(failure_distances) if failure_distances else None,
                "by_distance": by_distance,
            }
        )
    if not lineage_records:
        return {
            "delivery_robustness_n_lineages": 0,
            "delivery_robustness_mean_radius": 0.0,
            "delivery_robustness_mean_radius_capped": 0.0,
            "delivery_robustness_aurc_chain": 0.0,
            "delivery_robustness_robust_at_k": {},
        }

    cap_radius = max_distance + 1
    robust_at_k: dict[str, float] = {}
    for budget in range(max_distance + 1):
        robust_count = 0
        for record in lineage_records:
            robust = True
            for distance, flags in record["by_distance"].items():
                if distance <= budget and not all(flags):
                    robust = False
                    break
            if robust:
                robust_count += 1
        robust_at_k[str(budget)] = robust_count / len(lineage_records)

    observed_radii = [record["radius"] for record in lineage_records if record["radius"] is not None]
    mean_radius = sum(observed_radii) / max(1, len(observed_radii))
    mean_radius_capped = sum((record["radius"] if record["radius"] is not None else cap_radius) for record in lineage_records) / len(
        lineage_records
    )
    by_label: dict[str, list[float]] = defaultdict(list)
    for record in lineage_records:
        by_label[record["label"] or "unknown"].append(record["radius"] if record["radius"] is not None else cap_radius)
    return {
        "delivery_robustness_n_lineages": len(lineage_records),
        "delivery_robustness_mean_radius": mean_radius,
        "delivery_robustness_mean_radius_capped": mean_radius_capped,
        "delivery_robustness_aurc_chain": sum(robust_at_k.values()) / max(1, len(robust_at_k)),
        "delivery_robustness_robust_at_k": robust_at_k,
        "delivery_robustness_mean_radius_by_label": {
            label: sum(values) / len(values) for label, values in sorted(by_label.items())
        },
    }
