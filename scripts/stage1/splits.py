"""Train/dev/test split assignment."""

from __future__ import annotations

import math
import random
from typing import Any


def compute_split_counts(total: int, split_cfg: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in split_cfg.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    priorities = sorted(raw.items(), key=lambda item: item[1] - math.floor(item[1]), reverse=True)
    for name, _ in priorities[:remainder]:
        counts[name] += 1
    return counts


def assign_splits(
    selected_speakers: list[dict[str, Any]],
    split_cfg: dict[str, float],
    rng: random.Random,
) -> dict[str, str]:
    split_counts = compute_split_counts(len(selected_speakers), split_cfg)
    shuffled = list(selected_speakers)
    rng.shuffle(shuffled)
    split_map: dict[str, str] = {}
    cursor = 0
    for split_name in ("train", "dev", "test"):
        count = split_counts.get(split_name, 0)
        for speaker_bundle in shuffled[cursor : cursor + count]:
            split_map[speaker_bundle["speaker_id"]] = split_name
        cursor += count
    return split_map
