"""Timestamp normalization and stream synchronization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class TimestampNormalizer:
    """Normalize supported timestamp bases into integer nanoseconds."""

    time_base: str = "sensor_time_ns"

    def to_nanoseconds(self, value: int | float) -> int:
        """Convert a timestamp into nanoseconds."""

        if self.time_base in {"sensor_time_ns", "unix_time_ns", "nanoseconds"}:
            return int(value)
        if self.time_base in {"sensor_time_us", "microseconds"}:
            return int(value) * 1_000
        if self.time_base in {"sensor_time_ms", "milliseconds"}:
            return int(value) * 1_000_000
        if self.time_base in {"sensor_time_sec", "seconds"}:
            return floor(float(value) * 1_000_000_000)
        msg = f"unsupported time base: {self.time_base}"
        raise ValueError(msg)


def apply_time_offset_ns(sensor_timestamp_ns: int, offset_seconds: float) -> int:
    """Apply slac time offset convention.

    slac defines `sensor_time + dt_sensor = reference_time`.
    """

    return sensor_timestamp_ns + round(offset_seconds * 1_000_000_000)


def nearest_timestamp_pairs(
    reference_timestamps_ns: list[int],
    candidate_timestamps_ns: list[int],
    tolerance_ns: int,
) -> list[tuple[int, int]]:
    """Pair each reference timestamp with the nearest candidate within tolerance."""

    if tolerance_ns < 0:
        msg = "tolerance_ns must be non-negative"
        raise ValueError(msg)
    candidates = sorted(candidate_timestamps_ns)
    pairs: list[tuple[int, int]] = []
    cursor = 0
    for reference in sorted(reference_timestamps_ns):
        while cursor + 1 < len(candidates) and abs(candidates[cursor + 1] - reference) <= abs(
            candidates[cursor] - reference
        ):
            cursor += 1
        if candidates and abs(candidates[cursor] - reference) <= tolerance_ns:
            pairs.append((reference, candidates[cursor]))
    return pairs
