"""Structured, dependency-free progress for ``calibrex check``.

The runner reports to a :class:`CheckProgress`.  :class:`LegacyProgress` adapts a
plain ``Callable[[str], None]`` (the original API), :class:`PlainProgress` writes one
line per event, and :class:`TtyProgress` keeps a single updating status line on a
terminal.  Progress is written to a stream (stderr in the CLI) and never changes the
data being processed.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable, Mapping
from typing import TextIO

from calibrex.core.progress import ProgressEvent

Clock = Callable[[], float]

CLEAR_LINE = "\r\x1b[K"
_COLORS = {
    "pass": "\x1b[32m",
    "warn": "\x1b[33m",
    "fail": "\x1b[31m",
    "inconclusive": "\x1b[2m",
    "skipped": "\x1b[2m",
}
_RESET = "\x1b[0m"


def use_color(stream: TextIO, environ: Mapping[str, str] | None = None) -> bool:
    """Colour only on a TTY and when ``NO_COLOR`` is unset or empty (no-color.org)."""

    env = os.environ if environ is None else environ
    isatty = getattr(stream, "isatty", None)
    if isatty is None or not isatty():
        return False
    return not env.get("NO_COLOR") and env.get("TERM") != "dumb"


def paint_verdict(text: str, verdict: str | None = None) -> str:
    """Wrap ``text`` in the colour of ``verdict`` (default: its first word)."""

    key = verdict if verdict is not None else (text.split() or [""])[0]
    code = _COLORS.get(key)
    return f"{code}{text}{_RESET}" if code else text


def format_duration(seconds: float) -> str:
    """``45s``, ``3m05s``, ``1h02m``."""

    total = max(round(seconds), 0)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def eta_seconds(elapsed_s: float, done: int | None, total: int | None) -> float | None:
    """Remaining time extrapolated from the rate so far; ``None`` when not known."""

    if not done or not total or done <= 0 or total < done or elapsed_s <= 0.0:
        return None
    return elapsed_s * (total - done) / done


class CheckProgress:
    """Receiver of the runner's progress; every method is a no-op by default."""

    def run_started(self, total_pairs: int) -> None: ...

    def pair_started(self, index: int, total: int, label: str) -> None: ...

    def set_scan_total(self, total: int | None) -> None:
        """Items (bag messages) the current pair reads per pass, when known."""

    def stage(self, text: str) -> None: ...

    def tick(self, done: int, total: int | None = None) -> None: ...

    def pair_skipped(self, label: str, code: str, reason: str) -> None: ...

    def pair_failed(self, label: str, error: str) -> None: ...

    def pair_finished(
        self, label: str, status: str, runtime_s: float, from_cache: bool | None
    ) -> None: ...

    def run_finished(self) -> None: ...

    def __call__(self, message: str) -> None:
        """Free-text stage message, as the estimators send them."""

        self.stage(message)

    def on_event(self, event: ProgressEvent) -> None:
        """Sink for :func:`calibrex.core.progress.progress_sink`."""

        if event.kind == "stage":
            self.stage(event.text)
        elif event.done is not None:
            self.tick(event.done, event.total)


class LegacyProgress(CheckProgress):
    """Adapter for a ``Callable[[str], None]``: the original one-line messages."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit

    def pair_started(self, index: int, total: int, label: str) -> None:
        self._emit(f"check: running {label}")

    def stage(self, text: str) -> None:
        self._emit(text)

    def pair_skipped(self, label: str, code: str, reason: str) -> None:
        self._emit(f"check: {label} skipped ({code}): {reason}")

    def pair_failed(self, label: str, error: str) -> None:
        self._emit(f"check: {label} failed: {error}")

    def pair_finished(
        self, label: str, status: str, runtime_s: float, from_cache: bool | None
    ) -> None:
        self._emit(f"check: {label} -> {status} ({runtime_s:.0f} s)")


def as_progress(progress: Callable[[str], None] | CheckProgress | None) -> CheckProgress:
    """A :class:`CheckProgress` for whatever the caller passed."""

    if progress is None:
        return CheckProgress()
    if isinstance(progress, CheckProgress):
        return progress
    return LegacyProgress(progress)


class _Timeline(CheckProgress):
    """State shared by the stream renderers: pair, stage and scan counters."""

    def __init__(self, stream: TextIO, clock: Clock = time.monotonic) -> None:
        self.stream = stream
        self.clock = clock
        self.lock = threading.RLock()
        self.total_pairs = 0
        self.index = 0
        self.label = ""
        # A message before the first pair (``estimate`` announces its plan) must not
        # measure its elapsed time from the clock's origin (system boot).
        self.pair_started_at = clock()
        self.stage_text = ""
        self.stage_started_at = self.pair_started_at
        self.scan_total: int | None = None
        self.done: int | None = None
        self.total: int | None = None

    def run_started(self, total_pairs: int) -> None:
        self.total_pairs = total_pairs

    def pair_started(self, index: int, total: int, label: str) -> None:
        with self.lock:
            self.index, self.total_pairs, self.label = index, total, label
            self.pair_started_at = self.stage_started_at = self.clock()
            self.stage_text, self.done, self.total = "starting", None, None
            self.scan_total = None

    def set_scan_total(self, total: int | None) -> None:
        self.scan_total = total

    def stage(self, text: str) -> None:
        with self.lock:
            self.stage_text = text
            self.stage_started_at = self.clock()
            self.done = self.total = None

    def tick(self, done: int, total: int | None = None) -> None:
        with self.lock:
            if self.done is not None and done < self.done:  # a new pass over the data
                self.stage_started_at = self.clock()
            self.done = done
            self.total = total if total is not None else self.scan_total

    def header(self) -> str:
        return f"pair {self.index}/{self.total_pairs}"

    def bits(self) -> str:
        """``done/total scans (pct), elapsed, eta`` for the current stage."""

        now = self.clock()
        parts: list[str] = []
        if self.done is not None:
            if self.total:
                pct = min(100.0, 100.0 * self.done / self.total)
                parts.append(f"{self.done}/{self.total} msgs ({pct:.0f}%)")
            else:
                parts.append(f"{self.done} msgs")
        parts.append(f"{format_duration(now - self.pair_started_at)} elapsed")
        eta = eta_seconds(now - self.stage_started_at, self.done, self.total)
        if eta is not None:
            parts.append(f"eta {format_duration(eta)}")
        return ", ".join(parts)

    def detail(self) -> str:
        return f"{self.stage_text} [{self.bits()}]"

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    @staticmethod
    def _cache_suffix(from_cache: bool | None) -> str:
        return " [cache hit]" if from_cache else ""


class PlainProgress(_Timeline):
    """One line per event; scan counts at most every ``tick_interval_s`` seconds."""

    def __init__(
        self, stream: TextIO, clock: Clock = time.monotonic, tick_interval_s: float = 30.0
    ) -> None:
        super().__init__(stream, clock)
        self.tick_interval_s = tick_interval_s
        self._last_tick_line = float("-inf")

    def run_started(self, total_pairs: int) -> None:
        super().run_started(total_pairs)
        noun = "pair" if total_pairs == 1 else "pairs"
        self._write(f"check: {total_pairs} {noun} to run\n")

    def pair_started(self, index: int, total: int, label: str) -> None:
        super().pair_started(index, total, label)
        self._last_tick_line = float("-inf")
        self._write(f"check: running {label} [pair {index}/{total}]\n")

    def stage(self, text: str) -> None:
        super().stage(text)
        self._last_tick_line = self.clock()
        elapsed = format_duration(self.clock() - self.pair_started_at)
        self._write(f"check:   {text} [{elapsed} elapsed]\n")

    def tick(self, done: int, total: int | None = None) -> None:
        super().tick(done, total)
        now = self.clock()
        if now - self._last_tick_line >= self.tick_interval_s:
            self._last_tick_line = now
            self._write(f"check:   {self.detail()}\n")

    def pair_skipped(self, label: str, code: str, reason: str) -> None:
        self._write(f"check: {label} skipped ({code}): {reason}\n")

    def pair_failed(self, label: str, error: str) -> None:
        self._write(f"check: {label} failed: {error}\n")

    def pair_finished(
        self, label: str, status: str, runtime_s: float, from_cache: bool | None
    ) -> None:
        self._write(
            f"check: {label} -> {status} ({runtime_s:.0f} s){self._cache_suffix(from_cache)}"
            f" [pair {self.index}/{self.total_pairs} done]\n"
        )


class TtyProgress(_Timeline):
    """A single status line rewritten in place; permanent lines for finished pairs."""

    def __init__(
        self,
        stream: TextIO,
        clock: Clock = time.monotonic,
        refresh_s: float = 0.2,
        heartbeat_s: float = 1.0,
        color: bool = False,
    ) -> None:
        super().__init__(stream, clock)
        self.refresh_s = refresh_s
        self.heartbeat_s = heartbeat_s
        self.color = color
        self._last_paint = float("-inf")
        self._drawn = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _paint(self, force: bool = False) -> None:
        with self.lock:
            now = self.clock()
            if not force and now - self._last_paint < self.refresh_s:
                return
            self._last_paint = now
            # counts first, so a truncated line keeps the numbers
            name = self.label.split(" ")[0]
            stage = self.stage_text.removeprefix(f"{name}: ")
            line = f"check: {self.header()} [{self.bits()}] {name}: {stage}"
            width = _terminal_width(self.stream)
            if width and len(line) >= width:
                line = line[: width - 2] + ".."
            self._write(CLEAR_LINE + line)
            self._drawn = True

    def _permanent(self, text: str) -> None:
        with self.lock:
            self._write((CLEAR_LINE if self._drawn else "") + text + "\n")
            self._drawn = False

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            if self.label and self._drawn:
                self._paint(force=True)

    def run_started(self, total_pairs: int) -> None:
        super().run_started(total_pairs)
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def pair_started(self, index: int, total: int, label: str) -> None:
        super().pair_started(index, total, label)
        self._paint(force=True)

    def stage(self, text: str) -> None:
        super().stage(text)
        self._paint(force=True)

    def tick(self, done: int, total: int | None = None) -> None:
        super().tick(done, total)
        self._paint()

    def pair_skipped(self, label: str, code: str, reason: str) -> None:
        self._permanent(f"check: {label} skipped ({code}): {reason}")

    def pair_failed(self, label: str, error: str) -> None:
        self._permanent(f"check: {label} failed: {error}")

    def pair_finished(
        self, label: str, status: str, runtime_s: float, from_cache: bool | None
    ) -> None:
        shown = paint_verdict(status) if self.color else status
        self._permanent(
            f"check: {label} -> {shown} ({runtime_s:.0f} s){self._cache_suffix(from_cache)}"
            f" [pair {self.index}/{self.total_pairs} done]"
        )
        self.label = ""

    def run_finished(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self.lock:
            if self._drawn:
                self._write(CLEAR_LINE)
                self._drawn = False


def _terminal_width(stream: TextIO) -> int:
    try:
        return os.get_terminal_size(stream.fileno()).columns
    except (OSError, ValueError, AttributeError):
        return 0


def make_progress(
    stream: TextIO | None = None,
    *,
    quiet: bool = False,
    plain: bool = False,
    environ: Mapping[str, str] | None = None,
) -> CheckProgress:
    """Pick the renderer: none when ``quiet``, an updating line on a TTY, else line-per-event."""

    if quiet:
        return CheckProgress()
    out = stream if stream is not None else sys.stderr
    env = os.environ if environ is None else environ
    isatty = getattr(out, "isatty", None)
    interactive = bool(isatty and isatty()) and env.get("TERM") != "dumb"
    if interactive and not plain:
        return TtyProgress(out, color=use_color(out, env))
    return PlainProgress(out)
