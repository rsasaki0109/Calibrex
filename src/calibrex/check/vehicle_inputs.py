"""Bag readers for the vehicle pairs of ``calibrex check``.

The vehicle-pair estimators (``lidar-vehicle``, ``imu-vehicle``, ``ins-lidar``,
``lidar-wheel_odometry``) were built on KITTI text files. These readers give
them the same arrays from a rosbag2: LiDAR odometry from ``PointCloud2`` scans,
an INS pose and body-frame twist from ``nav_msgs/Odometry``, and wheel speed
and yaw rate from a twist or odometry topic. The LiDAR odometry is the one the
KITTI runners use (``IncrementalScanOdometry`` with default options on rigid
sweeps), so a converted KITTI drive reproduces the text-file result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.core.progress import emit_tick
from calibrex.data import ros_cdr
from calibrex.data.rosbag2 import (
    ODOMETRY_TYPE,
    TWIST_COVARIANCE_STAMPED_TYPE,
    TWIST_STAMPED_TYPE,
    iter_topic_messages,
)
from calibrex.solvers.scan_to_scan_odometry import (
    IncrementalScanOdometry,
    ScanOdometryOptions,
    ScanRegistration,
)

FloatArray: TypeAlias = NDArray[np.float64]


MAX_STREAM_GAP_S = 1.0
"""A silence longer than this splits a stream into independent segments.

KITTI's longest in-drive gap is 0.5 s (dropped sweeps); the pause between two
recordings is minutes. Odometry is never chained across such a gap, and motions
never span one.
"""


@dataclass(frozen=True)
class LidarOdometryTrack:
    """Scan stamps (s), poses and per-step registrations.

    The stream is cut into segments at gaps longer than ``MAX_STREAM_GAP_S``;
    ``segments`` holds each segment's ``(start, stop)`` index range, and every
    segment's poses are ``T_first_scan`` relative to its own first scan.
    """

    topic: str
    times_s: tuple[float, ...]
    poses: tuple[FloatArray, ...]
    registrations: tuple[ScanRegistration, ...]
    options: ScanOdometryOptions
    segments: tuple[tuple[int, int], ...] = ()

    def segment_slices(self) -> list[tuple[list[float], list[FloatArray]]]:
        """``(times, poses)`` of every segment (the whole track when none is recorded)."""

        ranges = self.segments or ((0, len(self.times_s)),)
        return [
            (list(self.times_s[start:stop]), list(self.poses[start:stop])) for start, stop in ranges
        ]


def split_at_gaps(times_s: Sequence[float], max_gap_s: float = MAX_STREAM_GAP_S) -> list[slice]:
    """Index ranges of a sorted time series, cut where consecutive samples are further apart."""

    if not len(times_s):
        return []
    cuts = [
        index
        for index in range(1, len(times_s))
        if float(times_s[index]) - float(times_s[index - 1]) > max_gap_s
    ]
    edges = [0, *cuts, len(times_s)]
    return [slice(start, stop) for start, stop in pairwise(edges)]


@dataclass(frozen=True)
class TwistTrack:
    """Time-sorted body-frame linear and angular velocity samples of one topic."""

    topic: str
    frame: str
    times_s: FloatArray
    linear_mps: FloatArray
    angular_rps: FloatArray


@dataclass(frozen=True)
class InsTrack:
    """An INS pose track ``T_world_body`` with its body-frame twist."""

    topic: str
    body_frame: str
    times_s: FloatArray
    poses: FloatArray
    twist: TwistTrack


def read_lidar_odometry(
    bag: Path,
    topic: str,
    options: ScanOdometryOptions | None = None,
    *,
    max_duration_s: float | None = None,
    max_gap_s: float = MAX_STREAM_GAP_S,
) -> LidarOdometryTrack:
    """Run scan-to-scan LiDAR odometry over a ``PointCloud2`` topic, one pass over the bag.

    A gap longer than ``max_gap_s`` between scans starts a new, independent segment
    (a new odometry instance), exactly as the KITTI runners start one per drive.
    """

    opts = options or ScanOdometryOptions()
    odometry = IncrementalScanOdometry(opts)
    times: list[float] = []
    poses: list[FloatArray] = []
    registrations: list[ScanRegistration] = []
    segments: list[tuple[int, int]] = []
    segment_start = 0
    for _conn, timestamp_ns, payload in iter_topic_messages(bag, topic):
        cloud = ros_cdr.decode_ros2_pointcloud2(topic, timestamp_ns, payload)
        time_s = cloud.timestamp_ns * 1.0e-9
        if max_duration_s is not None and times and time_s - times[0] > max_duration_s:
            break
        if times and time_s - times[-1] > max_gap_s:
            poses += odometry.poses
            registrations += odometry.registrations
            segments.append((segment_start, len(times)))
            segment_start = len(times)
            odometry = IncrementalScanOdometry(opts)
        odometry.add(cloud.xyz, time_s)
        times.append(time_s)
        emit_tick(len(times))
    poses += odometry.poses
    registrations += odometry.registrations
    if times:
        segments.append((segment_start, len(times)))
    return LidarOdometryTrack(
        topic=topic,
        times_s=tuple(times),
        poses=tuple(poses),
        registrations=tuple(registrations),
        options=opts,
        segments=tuple(segments),
    )


def _sorted(times: list[float], *columns: list[tuple[float, ...]]) -> tuple[FloatArray, ...]:
    order = np.argsort(np.asarray(times), kind="stable")
    return (
        np.asarray(times, dtype=np.float64)[order],
        *(np.asarray(column, dtype=np.float64)[order] for column in columns),
    )


def read_twist_track(
    bag: Path,
    topic: str,
    message_type: str,
    *,
    max_duration_s: float | None = None,
) -> TwistTrack:
    """Velocity samples of a ``TwistStamped``, ``TwistWithCovarianceStamped`` or ``Odometry`` topic.

    An ``Odometry`` message reports its twist in the child frame, which is the
    frame returned.
    """

    times: list[float] = []
    linear: list[tuple[float, ...]] = []
    angular: list[tuple[float, ...]] = []
    frame = ""
    for _conn, timestamp_ns, payload in iter_topic_messages(bag, topic):
        if message_type == ODOMETRY_TYPE:
            odom = ros_cdr.decode_ros2_odometry(topic, timestamp_ns, payload)
            stamp, frame = odom.timestamp_ns, odom.child_frame_id
            lin, ang = odom.linear_velocity, odom.angular_velocity
        elif message_type in {TWIST_STAMPED_TYPE, TWIST_COVARIANCE_STAMPED_TYPE}:
            twist = ros_cdr.decode_ros2_twist(
                topic,
                timestamp_ns,
                payload,
                with_covariance=message_type == TWIST_COVARIANCE_STAMPED_TYPE,
            )
            stamp, frame = twist.timestamp_ns, twist.frame_id
            lin, ang = twist.linear_velocity, twist.angular_velocity
        else:
            raise ValueError(f"{topic}: {message_type} carries no twist")
        time_s = stamp * 1.0e-9
        if max_duration_s is not None and times and time_s - times[0] > max_duration_s:
            break
        times.append(time_s)
        linear.append(lin)
        angular.append(ang)
    t, lin_array, ang_array = _sorted(times, linear, angular)
    return TwistTrack(topic, frame.lstrip("/"), t, lin_array, ang_array)


def read_ins_track(
    bag: Path,
    topic: str,
    *,
    max_duration_s: float | None = None,
) -> InsTrack:
    """The pose ``T_world_body`` and body-frame twist of a ``nav_msgs/Odometry`` topic."""

    times: list[float] = []
    positions: list[tuple[float, ...]] = []
    quaternions: list[tuple[float, ...]] = []
    linear: list[tuple[float, ...]] = []
    angular: list[tuple[float, ...]] = []
    child = ""
    for _conn, timestamp_ns, payload in iter_topic_messages(bag, topic):
        odom = ros_cdr.decode_ros2_odometry(topic, timestamp_ns, payload)
        time_s = odom.timestamp_ns * 1.0e-9
        if max_duration_s is not None and times and time_s - times[0] > max_duration_s:
            break
        child = odom.child_frame_id
        times.append(time_s)
        positions.append(odom.position)
        quaternions.append(odom.orientation_xyzw)
        linear.append(odom.linear_velocity)
        angular.append(odom.angular_velocity)
    t, pos, quat, lin, ang = _sorted(times, positions, quaternions, linear, angular)
    poses = np.tile(np.eye(4), (len(t), 1, 1))
    if len(t):
        poses[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
        poses[:, :3, 3] = pos
    frame = child.lstrip("/")
    return InsTrack(topic, frame, t, poses, TwistTrack(topic, frame, t, lin, ang))
