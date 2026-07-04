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
        first_ns = self.samples[0].timestamp_ns
        last_ns = self.samples[-1].timestamp_ns
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

        for left, right in zip(self.samples, self.samples[1:], strict=False):
            if left.timestamp_ns <= timestamp_ns <= right.timestamp_ns:
                span_ns = right.timestamp_ns - left.timestamp_ns
                if span_ns <= 0:
                    self._record_extrapolation(0.0)
                    return left.pose, False, 0.0
                alpha = (timestamp_ns - left.timestamp_ns) / span_ns
                self._record_extrapolation(0.0)
                return interpolate_se3(left.pose, right.pose, alpha), False, 0.0
        self._record_extrapolation(0.0)
        return self.samples[-1].pose, True, 0.0

    def _record_extrapolation(self, extrapolation_s: float) -> None:
        if extrapolation_s > self.max_extrapolation_s:
            self.max_extrapolation_s = extrapolation_s
