from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, Optional, TypeVar


ItemT = TypeVar("ItemT")
PreparedT = TypeVar("PreparedT")
ResultT = TypeVar("ResultT")


def _release_future_result(
    future: Future[PreparedT],
    release_prepared: Optional[Callable[[PreparedT], None]],
) -> None:
    if release_prepared is None:
        return
    if not future.cancel():
        release_prepared(future.result())


def ordered_prefetch_map(
    items: Iterable[ItemT],
    *,
    prepare: Callable[[ItemT], PreparedT],
    consume: Callable[[ItemT, PreparedT], ResultT],
    release_prepared: Optional[Callable[[PreparedT], None]] = None,
) -> Iterator[ResultT]:
    """Run one-item lookahead preparation while consuming the current item.

    The first item is prepared before any result is produced. After that, the
    next item's prepare work is submitted before the current item is consumed,
    so independent producer work can overlap with current-item consumer work.
    Results are yielded in input order.
    """
    iterator = iter(items)
    try:
        current_item = next(iterator)
    except StopIteration:
        return

    pending_next: Future[PreparedT] | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pd-prefetch") as executor:
        current_future = executor.submit(prepare, current_item)
        while True:
            current_prepared = current_future.result()
            try:
                next_item = next(iterator)
            except StopIteration:
                pending_next = None
            else:
                pending_next = executor.submit(prepare, next_item)

            try:
                yield consume(current_item, current_prepared)
            except BaseException:
                if pending_next is not None:
                    _release_future_result(pending_next, release_prepared)
                raise

            if pending_next is None:
                break
            current_item = next_item
            current_future = pending_next
