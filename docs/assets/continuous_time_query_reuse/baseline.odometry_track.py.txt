"""Odometry pose track with SE(3) interpolation for motion compensation."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Literal

from calibrex.core.geometry import SE3, interpolate_se3


@dataclass(frozen=True)
class OdometryPoseSample:
    """One timestamped ``T_world_base`` pose sample."""

    timestamp_ns: int
    pose: SE3


OdometryBurstPolicy = Literal["preserve", "keep_first", "keep_last"]


class OdometryTrack:
    """Time-ordered odometry poses with SE(3) interpolation.

    Translation is linearly interpolated; rotation uses quaternion slerp on the
    shortest arc (antipodal ``q`` / ``-q`` pairs are handled). Queries outside
    the covered time range clamp to the nearest pose; ``clamp_count`` records how
    many queries were clamped.
    """

    def __init__(
        self,
        samples: list[OdometryPoseSample],
        *,
        burst_policy: OdometryBurstPolicy = "preserve",
        min_interval_s: float = 0.001,
    ) -> None:
        if burst_policy not in {"preserve", "keep_first", "keep_last"}:
            raise ValueError(f"unsupported odometry burst policy: {burst_policy!r}")
        if min_interval_s <= 0.0:
            raise ValueError("odometry burst min_interval_s must be positive")
        ordered = sorted(samples, key=lambda sample: sample.timestamp_ns)
        self.raw_message_count = len(ordered)
        self.burst_policy = burst_policy
        self.burst_min_interval_s = float(min_interval_s)
        if burst_policy != "preserve":
            ordered, removed_count = _reduce_timestamp_bursts(
                ordered,
                burst_policy=burst_policy,
                min_interval_ns=round(min_interval_s * 1_000_000_000),
            )
        else:
            removed_count = 0
        self.burst_removed_count = removed_count
        self.samples: tuple[OdometryPoseSample, ...] = tuple(ordered)
        self._timestamps_ns: tuple[int, ...] = tuple(
            sample.timestamp_ns for sample in self.samples
        )
        self.clamp_count = 0
        self.interpolation_count = 0
        self.max_extrapolation_s = 0.0

    @property
    def message_count(self) -> int:
        return len(self.samples)

    @property
    def first_timestamp_ns(self) -> int | None:
        return self.samples[0].timestamp_ns if self.samples else None

    @property
    def last_timestamp_ns(self) -> int | None:
        return self.samples[-1].timestamp_ns if self.samples else None

    def interpolate(self, timestamp_ns: int) -> tuple[SE3, bool, float]:
        """Return ``(T_world_base, was_clamped, extrapolation_s)`` at ``timestamp_ns``."""

        self.interpolation_count += 1
        if not self.samples:
            return SE3.identity(), False, 0.0
        first_ns = self._timestamps_ns[0]
        last_ns = self._timestamps_ns[-1]
        if timestamp_ns <= first_ns:
            extrapolation_s = max(0.0, (first_ns - timestamp_ns) / 1_000_000_000)
            if timestamp_ns < first_ns:
                self.clamp_count += 1
            self._record_extrapolation(extrapolation_s)
            return self.samples[0].pose, timestamp_ns < first_ns, extrapolation_s
        if timestamp_ns >= last_ns:
            extrapolation_s = max(0.0, (timestamp_ns - last_ns) / 1_000_000_000)
            if timestamp_ns > last_ns:
                self.clamp_count += 1
            self._record_extrapolation(extrapolation_s)
            return self.samples[-1].pose, timestamp_ns > last_ns, extrapolation_s

        right_idx = bisect.bisect_right(self._timestamps_ns, timestamp_ns)
        left = self.samples[right_idx - 1]
        right = self.samples[right_idx]
        span_ns = right.timestamp_ns - left.timestamp_ns
        if span_ns <= 0:
            self._record_extrapolation(0.0)
            return left.pose, False, 0.0
        alpha = (timestamp_ns - left.timestamp_ns) / span_ns
        self._record_extrapolation(0.0)
        return interpolate_se3(left.pose, right.pose, alpha), False, 0.0

    def _record_extrapolation(self, extrapolation_s: float) -> None:
        if extrapolation_s > self.max_extrapolation_s:
            self.max_extrapolation_s = extrapolation_s


def _reduce_timestamp_bursts(
    samples: list[OdometryPoseSample],
    *,
    burst_policy: Literal["keep_first", "keep_last"],
    min_interval_ns: int,
) -> tuple[list[OdometryPoseSample], int]:
    """Reduce near-coincident pose samples without changing retained timestamps."""

    if not samples:
        return [], 0
    retained: list[OdometryPoseSample] = [samples[0]]
    removed_count = 0
    for sample in samples[1:]:
        delta_ns = sample.timestamp_ns - retained[-1].timestamp_ns
        if delta_ns < min_interval_ns:
            removed_count += 1
            if burst_policy == "keep_last":
                retained[-1] = sample
            continue
        retained.append(sample)
    return retained, removed_count
