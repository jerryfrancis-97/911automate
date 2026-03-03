"""Async utilities: run blocking sync code in a shared ThreadPoolExecutor."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

_thread_pool: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    """Return the shared ThreadPoolExecutor, creating it lazily."""
    global _thread_pool
    if _thread_pool is None:
        max_workers = min(32, (os.cpu_count() or 4) + 4)
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _thread_pool


async def run_in_thread(
    fn: Callable[..., _T],
    *args: object,
    **kwargs: object,
) -> _T:
    """Run a blocking callable in the shared thread pool. Returns the result."""
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    return await loop.run_in_executor(
        executor,
        lambda: fn(*args, **kwargs),
    )
