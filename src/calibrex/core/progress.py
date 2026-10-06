"""Opt-in progress hooks for long calibration passes (ROS-independent, dependency-free).

Estimators and readers call :func:`emit_stage` / :func:`emit_tick` at natural points
(a new pass starts, one more scan was processed).  Without an installed sink they do
nothing, and they never touch the data being processed, so results are bit-identical
with progress on or off.  A caller (the ``calibrex check`` runner) installs a sink for
the duration of one pair with :func:`progress_sink`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProgressEvent:
    """One progress report: a new ``stage`` or a ``tick`` of the current stage."""

    kind: Literal["stage", "tick"]
    text: str = ""
    """Stage label (``stage`` events)."""
    done: int | None = None
    """Items processed so far in the current stage (``tick`` events)."""
    total: int | None = None
    """Items expected in the current stage, when known."""


ProgressSink = Callable[[ProgressEvent], None]

_SINK: ContextVar[ProgressSink | None] = ContextVar("calibrex_progress_sink", default=None)


@contextmanager
def progress_sink(sink: ProgressSink | None) -> Iterator[None]:
    """Route :func:`emit_stage` / :func:`emit_tick` to ``sink`` inside the block."""

    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)


def emit_stage(text: str) -> None:
    """Announce that a new stage (pass, solver, cache reuse) started."""

    sink = _SINK.get()
    if sink is not None:
        sink(ProgressEvent("stage", text=text))


def emit_tick(done: int, total: int | None = None) -> None:
    """Report ``done`` items processed in the current stage (``total`` when known)."""

    sink = _SINK.get()
    if sink is not None:
        sink(ProgressEvent("tick", done=done, total=total))
