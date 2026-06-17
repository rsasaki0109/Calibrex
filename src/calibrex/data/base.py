"""Dataset reader abstractions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TimestampedRecord:
    """A normalized dataset record."""

    stream: str
    timestamp_ns: int
    payload_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamSummary:
    """A discovered dataset stream."""

    name: str
    kind: str
    message_count: int | None = None
    topic: str | None = None
    sensor: str | None = None


class DatasetReader(Protocol):
    """Common reader protocol for adapter implementations."""

    def streams(self) -> list[StreamSummary]:
        """Return available streams."""

    def records(self, stream: str) -> Iterable[TimestampedRecord]:
        """Yield normalized records for a stream."""
