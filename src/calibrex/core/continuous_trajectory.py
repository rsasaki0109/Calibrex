"""ROS-independent continuous body-pose trajectory interpolation."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from itertools import pairwise

from calibrex.core.geometry import SE3, Vector3, interpolate_se3


@dataclass(frozen=True)
class ContinuousTrajectoryPose:
    """One timestamped ``T_world_body`` trajectory knot."""

    timestamp_sec: float
    transform_world_body: SE3

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_sec):
            raise ValueError("trajectory timestamp must be finite")


@dataclass(frozen=True)
class PiecewiseSE3Trajectory:
    """Interpolate a bounded sequence of ``T_world_body`` poses.

    Translation is linear and rotation uses shortest-arc quaternion slerp.
    Evaluation outside the supplied interval is rejected rather than clamped,
    so missing temporal support cannot silently bias calibration.
    """

    poses: tuple[ContinuousTrajectoryPose, ...]
    _timestamps: tuple[float, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if len(self.poses) < 2:
            raise ValueError("continuous trajectory requires at least two poses")
        timestamps = tuple(item.timestamp_sec for item in self.poses)
        if any(
            right <= left
            for left, right in pairwise(timestamps)
        ):
            raise ValueError(
                "continuous trajectory timestamps must be strictly increasing"
            )
        object.__setattr__(self, "_timestamps", timestamps)

    @property
    def minimum_time_sec(self) -> float:
        """Return the first supported timestamp."""

        return self.poses[0].timestamp_sec

    @property
    def maximum_time_sec(self) -> float:
        """Return the last supported timestamp."""

        return self.poses[-1].timestamp_sec

    def pose_at(self, timestamp_sec: float) -> SE3:
        """Return interpolated ``T_world_body`` at an in-range timestamp."""

        if not math.isfinite(timestamp_sec):
            raise ValueError("trajectory query timestamp must be finite")
        if (
            timestamp_sec < self.minimum_time_sec
            or timestamp_sec > self.maximum_time_sec
        ):
            raise ValueError(
                "trajectory query is outside the supplied temporal support"
            )
        right_index = bisect.bisect_left(self._timestamps, timestamp_sec)
        if right_index == 0:
            return self.poses[0].transform_world_body
        if right_index == len(self.poses):
            return self.poses[-1].transform_world_body
        right = self.poses[right_index]
        if right.timestamp_sec == timestamp_sec:
            return right.transform_world_body
        left = self.poses[right_index - 1]
        alpha = (timestamp_sec - left.timestamp_sec) / (
            right.timestamp_sec - left.timestamp_sec
        )
        return interpolate_se3(
            left.transform_world_body,
            right.transform_world_body,
            alpha,
        )

    def transform_static_point_between_body_frames(
        self,
        point_body_source: Vector3,
        *,
        source_time_sec: float,
        target_time_sec: float,
    ) -> Vector3:
        """Express a static world point from source body frame at target time."""

        transform_world_source = self.pose_at(source_time_sec)
        transform_world_target = self.pose_at(target_time_sec)
        transform_target_source = transform_world_target.inverse().compose(
            transform_world_source
        )
        return transform_target_source.transform_point(point_body_source)
