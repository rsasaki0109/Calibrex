"""Odometry pose track with SE(3) interpolation for motion compensation."""

from __future__ import annotations

from dataclasses import dataclass

from calibrex.core.geometry import SE3, interpolate_se3


@dataclass(frozen=True)
class OdometryPoseSample:
    """One timestamped ``T_world_base`` pose sample."""

    timestamp_ns: int
    pose: SE3


class OdometryTrack:
    """Time-ordered odometry poses with SE(3) interpolation.

    Translation is linearly interpolated; rotation uses quaternion slerp on the
    shortest arc (antipodal ``q`` / ``-q`` pairs are handled). Queries outside
    the covered time range clamp to the nearest pose; ``clamp_count`` records how
    many queries were clamped.
    """

    def __init__(self, samples: list[OdometryPoseSample]) -> None:
        ordered = sorted(samples, key=lambda sample: sample.timestamp_ns)
        self.samples: tuple[OdometryPoseSample, ...] = tuple(ordered)
        self.clamp_count = 0

    @property
    def message_count(self) -> int:
        return len(self.samples)

    @property
    def first_timestamp_ns(self) -> int | None:
        return self.samples[0].timestamp_ns if self.samples else None

    @property
    def last_timestamp_ns(self) -> int | None:
        return self.samples[-1].timestamp_ns if self.samples else None

    def interpolate(self, timestamp_ns: int) -> tuple[SE3, bool]:
        """Return ``(T_world_base, was_clamped)`` at ``timestamp_ns``."""

        if not self.samples:
            return SE3.identity(), False
        if timestamp_ns <= self.samples[0].timestamp_ns:
            if timestamp_ns < self.samples[0].timestamp_ns:
                self.clamp_count += 1
            return self.samples[0].pose, timestamp_ns < self.samples[0].timestamp_ns
        if timestamp_ns >= self.samples[-1].timestamp_ns:
            if timestamp_ns > self.samples[-1].timestamp_ns:
                self.clamp_count += 1
            return self.samples[-1].pose, timestamp_ns > self.samples[-1].timestamp_ns

        for left, right in zip(self.samples, self.samples[1:], strict=False):
            if left.timestamp_ns <= timestamp_ns <= right.timestamp_ns:
                span_ns = right.timestamp_ns - left.timestamp_ns
                if span_ns <= 0:
                    return left.pose, False
                alpha = (timestamp_ns - left.timestamp_ns) / span_ns
                return interpolate_se3(left.pose, right.pose, alpha), False
        return self.samples[-1].pose, True
