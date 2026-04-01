"""Small tqdm helpers."""

from __future__ import annotations

from typing import Any, Iterable, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")

def create_progress(
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
    leave: bool = True,
    position: int | None = None,
) -> Any:
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        position=position,
        dynamic_ncols=True,
    )


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
    leave: bool = True,
    position: int | None = None,
) -> Iterable[T]:
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        position=position,
        dynamic_ncols=True,
    )
