"""Shared concurrent execution helpers for stage pipelines."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Iterable, TypeVar

from tqdm.auto import tqdm

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


def _default_status(result: Any) -> str:
    return str(getattr(result, "status", ""))


def run_bounded_tasks(
    items: Iterable[ItemT],
    total: int,
    *,
    workers: int,
    desc: str,
    unit: str,
    submit_fn: Callable[[ThreadPoolExecutor, ItemT], Future],
    on_result: Callable[[ResultT], None],
    counts: Counter,
    log_every: int,
    progress_postfix: Callable[[int], dict[str, Any]],
    status_fn: Callable[[ResultT], str] | None = None,
) -> None:
    in_flight_limit = max(1, workers * 2)
    item_iter = iter(items)
    pending: set[Future] = set()
    result_status = status_fn or _default_status
    progress_interval = max(1, log_every)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for _ in range(min(in_flight_limit, total)):
            item = next(item_iter, None)
            if item is None:
                break
            pending.add(submit_fn(executor, item))

        with tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True) as progress:
            completed = 0
            while pending:
                done, still_pending = wait(pending, return_when=FIRST_COMPLETED)
                pending = set(still_pending)
                for future in done:
                    completed += 1
                    result = future.result()
                    status = result_status(result)
                    if status:
                        counts[status] += 1
                    on_result(result)
                    progress.update(1)
                    if completed <= 5 or completed % progress_interval == 0 or completed == total:
                        progress.set_postfix(**progress_postfix(completed))
                while len(pending) < in_flight_limit:
                    item = next(item_iter, None)
                    if item is None:
                        break
                    pending.add(submit_fn(executor, item))
            if total > 0 and (completed <= 5 or completed % progress_interval != 0):
                progress.set_postfix(**progress_postfix(completed))
