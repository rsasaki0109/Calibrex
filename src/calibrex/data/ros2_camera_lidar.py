"""Synchronized image and LiDAR-scan pairs from ROS 2 bags, without ROS.

One pass over a ``sensor_msgs/msg/Image`` topic and a ``sensor_msgs/msg/PointCloud2``
topic yields ``(image, scan)`` pairs for targetless camera-LiDAR calibration.
Scans are taken at a fixed spacing; each is paired with the nearest image to the
scan's *reference time*: the header stamp, or, when the cloud has a per-point time
field and the caller supplies a rule, the time its visible points were measured
(a spinning LiDAR sweeps past the camera's field of view well after the sweep
started).  A pair whose nearest image is further than ``max_gap_s`` is dropped.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.progress import emit_tick
from calibrex.data import ros_cdr
from calibrex.data.ros2_camera_imu import GrayImage, to_gray
from calibrex.data.rosbag2 import iter_messages

PointTimeEncoding = Literal["offset_s", "absolute_ns", "absolute_s"]
ReferenceTimeRule: TypeAlias = Callable[[NDArray[np.float32], NDArray[np.float64] | None], float]
"""``(xyz, seconds after the header stamp per point or None) -> seconds after the header stamp``."""

_IMAGE_WINDOW_NS = 400_000_000


@dataclass(frozen=True)
class CameraLidarFrame:
    """One image paired with one LiDAR scan."""

    time_s: float
    """The scan's reference time (seconds)."""
    image_time_s: float
    image: GrayImage
    points: NDArray[np.float32]
    """``(N, 3)`` finite points in the LiDAR frame."""
    cloud_frame_id: str
    image_frame_id: str

    @property
    def gap_s(self) -> float:
        """Image time minus the scan's reference time."""

        return self.image_time_s - self.time_s


@dataclass
class _PendingScan:
    reference_ns: int
    points: NDArray[np.float32]
    cloud_frame_id: str


def relative_point_times(
    values: NDArray[np.float64] | None, header_ns: int, encoding: PointTimeEncoding
) -> NDArray[np.float64] | None:
    """Per-point seconds after the header stamp for any of the supported encodings."""

    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if encoding == "offset_s":
        return array
    if encoding == "absolute_s":
        return array - header_ns * 1.0e-9
    return (array - header_ns) * 1.0e-9


def load_camera_lidar_frames(
    bag: str | Path,
    image_topic: str,
    lidar_topic: str,
    *,
    spacing_s: float = 2.0,
    max_frames: int = 48,
    max_seconds: float | None = None,
    max_gap_s: float = 0.06,
    point_time: tuple[str, PointTimeEncoding] | None = None,
    reference_time: ReferenceTimeRule | None = None,
) -> list[CameraLidarFrame]:
    """Synchronized frames in time order, at most one scan per ``spacing_s`` seconds."""

    recent: collections.deque[tuple[int, str, bytes]] = collections.deque()
    pending: list[_PendingScan] = []
    frames: list[CameraLidarFrame] = []
    first_ns: int | None = None
    last_selected_ns: int | None = None
    selected = 0
    scanned = 0
    gap_ns = int(max_gap_s * 1.0e9)
    point_time_field = point_time[0] if point_time is not None else None

    def pair(scan: _PendingScan) -> None:
        if not recent:
            return
        best = min(recent, key=lambda item: abs(item[0] - scan.reference_ns))
        if abs(best[0] - scan.reference_ns) > gap_ns:
            return
        message = ros_cdr.decode_ros2_image(image_topic, best[0], best[2])
        if message.data is None:
            return
        frames.append(
            CameraLidarFrame(
                time_s=scan.reference_ns * 1.0e-9,
                image_time_s=best[0] * 1.0e-9,
                image=to_gray(
                    bytes(message.data),
                    height=message.height,
                    width=message.width,
                    step=message.step,
                    encoding=message.encoding,
                ),
                points=scan.points,
                cloud_frame_id=scan.cloud_frame_id,
                image_frame_id=message.frame_id,
            )
        )

    for connection, timestamp_ns, payload in iter_messages(bag, topics={image_topic, lidar_topic}):
        if connection.topic == image_topic:
            header = ros_cdr.decode_ros2_image(
                image_topic, timestamp_ns, payload, include_data=False
            )
            recent.append((header.timestamp_ns, header.frame_id, payload))
            # Pair before evicting: a camera slower than the window would lose the image
            # nearest to a scan.
            still: list[_PendingScan] = []
            for scan in pending:
                if header.timestamp_ns >= scan.reference_ns + gap_ns:
                    pair(scan)
                else:
                    still.append(scan)
            pending = still
            while recent and recent[0][0] < header.timestamp_ns - _IMAGE_WINDOW_NS:
                recent.popleft()
            if selected >= max_frames and not pending:
                break
            continue
        scanned += 1
        emit_tick(scanned)
        if selected >= max_frames:
            continue
        first_ns = timestamp_ns if first_ns is None else first_ns
        if max_seconds is not None and (timestamp_ns - first_ns) * 1.0e-9 > max_seconds:
            selected = max_frames
            continue
        if last_selected_ns is not None and (timestamp_ns - last_selected_ns) * 1.0e-9 < spacing_s:
            continue
        cloud = ros_cdr.decode_ros2_pointcloud2(
            lidar_topic, timestamp_ns, payload, point_time_field=point_time_field
        )
        points = np.asarray(cloud.xyz, dtype=np.float32)
        if points.shape[0] == 0:
            continue
        last_selected_ns = timestamp_ns
        selected += 1
        times = (
            relative_point_times(cloud.point_time_offsets_s, cloud.timestamp_ns, point_time[1])
            if point_time is not None
            else None
        )
        offset = reference_time(points, times) if reference_time is not None else 0.0
        pending.append(
            _PendingScan(
                reference_ns=cloud.timestamp_ns + round(offset * 1.0e9),
                points=points,
                cloud_frame_id=cloud.frame_id,
            )
        )
    for scan in pending:
        pair(scan)
    frames.sort(key=lambda frame: frame.time_s)
    return frames
