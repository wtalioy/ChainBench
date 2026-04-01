"""Shared dataclasses for pipeline internals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..tasks import TaskPack


@dataclass
class RunRecord:
    task_id: str
    variant: str
    baseline: str
    device: str
    train_status: str
    train_returncode: int
    ok: bool
    returncode: int
    model_path: str | None
    scores_path: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssignedJob:
    index: int
    pack: TaskPack
    baseline_name: str
    execution_device: str


@dataclass
class TrainingState:
    checkpoint: Path | None = None
    train_status: str = "skipped"
    train_returncode: int = 0
    train_scores_path: Path | None = None
