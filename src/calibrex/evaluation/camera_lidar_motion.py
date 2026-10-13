"""Move LiDAR points to the image time with the rig's measured motion.

A LiDAR point and the image it is compared with are not measured at the same time:
an unsynchronized camera fires up to half a frame away from the scan, and a spinning
LiDAR measures each point at its own time during the sweep. On a moving rig the
LiDAR has moved and turned in between, so its points no longer line up with the
image even under the right extrinsic. Over a short interval the rig's velocity is
nearly constant, so a point measured ``dt`` seconds after the image is moved to
where the LiDAR saw it from at the image time::

    p_image_time = R(omega * dt) p + v * dt

with ``omega`` the angular rate and ``v`` the velocity of the LiDAR origin, both in
the LiDAR frame. They come from a velocity topic of the same bag: an odometry or
twist topic (rotation and translation) or an IMU (gyro; rotation only), expressed
in the LiDAR frame through the static transform between the two frames.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.camera_lidar_edge import CameraLidarEdgeMotion
from calibrex.data import ros_cdr
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.data.rosbag2 import iter_messages

FloatArray: TypeAlias = NDArray[np.float64]
MotionKind = Literal["twist", "gyro"]

ODOMETRY_TYPE = "nav_msgs/msg/Odometry"
TWIST_TYPES = frozenset(
    {"geometry_msgs/msg/TwistStamped", "geometry_msgs/msg/TwistWithCovarianceStamped"}
)
IMU_TYPE = "sensor_msgs/msg/Imu"


@dataclass(frozen=True)
class LidarMotion:
    """The rig's velocity in the LiDAR frame, sampled over time."""

    source_topic: str
    kind: MotionKind
    times_s: FloatArray
    """Increasing sample times (seconds, the bag's clock)."""
    angular_rad_s: FloatArray
    """``(N, 3)`` angular rate in the LiDAR frame."""
    linear_m_s: FloatArray | None
    """``(N, 3)`` velocity of the LiDAR origin in the LiDAR frame; ``None`` for a gyro."""
    max_sample_gap_s: float = 0.25
    """A time further than this from every sample has no motion."""

    def __post_init__(self) -> None:
        count = self.times_s.shape[0]
        if count == 0 or self.angular_rad_s.shape != (count, 3):
            raise ValueError("motion needs at least one sample and (N, 3) angular rates")
        if self.linear_m_s is not None and self.linear_m_s.shape != (count, 3):
            raise ValueError("linear velocity must be (N, 3) like the angular rate")
        if np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("motion sample times must increase")

    def velocity_at(self, time_s: float) -> tuple[FloatArray, FloatArray | None] | None:
        """Interpolated ``(omega, v)`` at ``time_s``; ``None`` without a sample nearby."""

        index = int(np.searchsorted(self.times_s, time_s))
        neighbours = self.times_s[max(index - 1, 0) : index + 1]
        if float(np.min(np.abs(neighbours - time_s))) > self.max_sample_gap_s:
            return None

        def interpolate(values: FloatArray) -> FloatArray:
            return np.asarray(
                [np.interp(time_s, self.times_s, values[:, axis]) for axis in range(3)]
            )

        linear = interpolate(self.linear_m_s) if self.linear_m_s is not None else None
        return interpolate(self.angular_rad_s), linear


def lidar_motion_from_body(
    source_topic: str,
    kind: MotionKind,
    times_s: Sequence[float],
    angular_body: Sequence[Sequence[float]],
    linear_body: Sequence[Sequence[float]] | None,
    t_body_lidar: FloatArray,
) -> LidarMotion:
    """Express a body's angular rate and velocity in the LiDAR frame.

    ``t_body_lidar`` is the LiDAR's pose in the body frame the samples are given in.
    The LiDAR origin's velocity is the body velocity plus ``omega x r``.
    """

    order = np.argsort(np.asarray(times_s, dtype=np.float64), kind="stable")
    times = np.asarray(times_s, dtype=np.float64)[order]
    keep = np.concatenate([[True], np.diff(times) > 0.0])
    rotation = np.asarray(t_body_lidar, dtype=np.float64)[:3, :3]
    lever = np.asarray(t_body_lidar, dtype=np.float64)[:3, 3]
    omega = np.asarray(angular_body, dtype=np.float64).reshape(-1, 3)[order][keep]
    linear: FloatArray | None = None
    if linear_body is not None:
        velocity = np.asarray(linear_body, dtype=np.float64).reshape(-1, 3)[order][keep]
        linear = (velocity + np.cross(omega, lever)) @ rotation
    return LidarMotion(
        source_topic=source_topic,
        kind=kind,
        times_s=times[keep],
        angular_rad_s=omega @ rotation,
        linear_m_s=linear,
    )


def read_lidar_motion(
    bag: str | Path,
    topic: str,
    message_type: str,
    t_body_lidar: FloatArray,
) -> LidarMotion | None:
    """Read an odometry, twist or IMU topic as the LiDAR's motion (``None`` if empty).

    Odometry twists are in the ``child_frame_id`` and twist messages in their header
    frame (REP 105); ``t_body_lidar`` must be the LiDAR's pose in that frame.
    """

    times: list[float] = []
    angular: list[tuple[float, float, float]] = []
    linear: list[tuple[float, float, float]] = []
    kind: MotionKind = "gyro" if message_type == IMU_TYPE else "twist"
    for _connection, timestamp_ns, payload in iter_messages(bag, topics={topic}):
        if message_type == ODOMETRY_TYPE:
            odometry = ros_cdr.decode_ros2_odometry(topic, timestamp_ns, payload)
            stamp, rate, velocity = (
                odometry.timestamp_ns,
                odometry.angular_velocity,
                odometry.linear_velocity,
            )
        elif message_type in TWIST_TYPES:
            twist = ros_cdr.decode_ros2_twist(topic, timestamp_ns, payload)
            stamp, rate, velocity = (
                twist.timestamp_ns,
                twist.angular_velocity,
                twist.linear_velocity,
            )
        elif message_type == IMU_TYPE:
            imu = ros_cdr.decode_ros2_imu(topic, timestamp_ns, payload)
            stamp, rate, velocity = imu.timestamp_ns, imu.angular_velocity, (0.0, 0.0, 0.0)
        else:
            raise ValueError(f"{topic}: {message_type} is not a velocity source")
        times.append(stamp * 1.0e-9)
        angular.append(rate)
        linear.append(velocity)
    if not times:
        return None
    return lidar_motion_from_body(
        topic, kind, times, angular, linear if kind == "twist" else None, t_body_lidar
    )


def move_to_image_time(
    frame: CameraLidarFrame, motion: LidarMotion
) -> tuple[CameraLidarFrame, float, float] | None:
    """The frame with its points moved to the image time, and the move at the reference.

    Returns ``(frame, rotation_deg, translation_m)``, the last two being the rotation and
    translation applied to a point measured at the scan's reference time; ``None`` when
    the motion has no sample near the frame.
    """

    velocity = motion.velocity_at(0.5 * (frame.time_s + frame.image_time_s))
    if velocity is None:
        return None
    omega, linear = velocity
    gap = frame.time_s - frame.image_time_s
    points = np.asarray(frame.points, dtype=np.float64)
    offsets = frame.point_time_offsets_s
    dt = gap + (np.asarray(offsets, dtype=np.float64) if offsets is not None else 0.0)
    dt = np.broadcast_to(dt, (points.shape[0],))
    moved = _rotate(points, omega[None, :] * dt[:, None])
    if linear is not None:
        moved += linear[None, :] * dt[:, None]
    translation = float(np.linalg.norm(linear) * abs(gap)) if linear is not None else 0.0
    return (
        dataclasses.replace(frame, points=moved.astype(np.float32)),
        float(np.degrees(np.linalg.norm(omega) * abs(gap))),
        translation,
    )


def move_frames_to_image_time(
    frames: Sequence[CameraLidarFrame], motion: LidarMotion
) -> tuple[list[CameraLidarFrame], CameraLidarEdgeMotion]:
    """Every frame moved to its image time (frames without motion are kept as they are)."""

    moved: list[CameraLidarFrame] = []
    rotations: list[float] = []
    translations: list[float] = []
    for frame in frames:
        result = move_to_image_time(frame, motion)
        if result is None:
            moved.append(frame)
            continue
        moved.append(result[0])
        rotations.append(result[1])
        translations.append(result[2])
    twist = motion.linear_m_s is not None
    summary = CameraLidarEdgeMotion(
        source_topic=motion.source_topic,
        kind=motion.kind,
        per_point_time=any(frame.point_time_offsets_s is not None for frame in frames),
        frames_moved=len(rotations),
        frames_without_motion=len(frames) - len(rotations),
        median_rotation_deg=float(np.median(rotations)) if rotations else None,
        max_rotation_deg=float(np.max(rotations)) if rotations else None,
        median_translation_m=float(np.median(translations)) if rotations and twist else None,
        max_translation_m=float(np.max(translations)) if rotations and twist else None,
    )
    return moved, summary


def _rotate(points: FloatArray, rotation_vectors: FloatArray) -> FloatArray:
    """Rotate each point by its own rotation vector (Rodrigues)."""

    angles = np.linalg.norm(rotation_vectors, axis=1)
    out = points.copy()
    turning = angles > 1.0e-12
    if not np.any(turning):
        return out
    axes = rotation_vectors[turning] / angles[turning, None]
    source = points[turning]
    cosines = np.cos(angles[turning])[:, None]
    sines = np.sin(angles[turning])[:, None]
    out[turning] = (
        source * cosines
        + np.cross(axes, source) * sines
        + axes * np.sum(axes * source, axis=1)[:, None] * (1.0 - cosines)
    )
    return out
