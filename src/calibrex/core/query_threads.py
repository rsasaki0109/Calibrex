"""How many threads the nearest-neighbour queries of the LiDAR estimators use.

This is an execution setting, not a method option: a KD-tree query answers each point on
its own, so the thread count changes how long a query takes and nothing else. It is
therefore kept out of the estimator options, the artifacts and the cache keys.

The setting is process-wide. ``calibrex check``, ``estimate`` and ``drift`` set it from
``--threads``; a library caller sets it with :func:`query_threads` or the
``CALIBREX_QUERY_THREADS`` environment variable. Unset, queries run on one thread, as
they always did.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

ENV = "CALIBREX_QUERY_THREADS"
_threads: int | None = None


def available_threads() -> int:
    """The CPUs this process may run on."""

    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except AttributeError:  # not on Linux
        return max(os.cpu_count() or 1, 1)


def _from_environment() -> int:
    text = os.environ.get(ENV, "").strip()
    if not text:
        return 1
    try:
        count = int(text)
    except ValueError:
        raise ValueError(f"{ENV} must be an integer, got {text!r}") from None
    return available_threads() if count <= 0 else count


def query_workers(configured: int = 1) -> int:
    """``workers=`` for a KD-tree query.

    An estimator option that asks for more than one worker is kept as given; the default
    of one follows the process setting.
    """

    if configured != 1:
        return configured
    return _threads if _threads is not None else _from_environment()


@contextmanager
def query_threads(count: int | None) -> Iterator[None]:
    """Run the enclosed queries on ``count`` threads (0 or less: every available CPU).

    ``None`` leaves the setting as it is.
    """

    global _threads
    previous = _threads
    if count is not None:
        _threads = available_threads() if count <= 0 else count
    try:
        yield
    finally:
        _threads = previous
