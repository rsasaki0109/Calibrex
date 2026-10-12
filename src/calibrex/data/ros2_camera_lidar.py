"""Synchronized image and LiDAR-scan pairs from ROS 2 bags, without ROS.

One pass over a ``sensor_msgs/msg/Image`` topic and a ``sensor_msgs/msg/PointCloud2``
topic yields ``(image, scan)`` pairs for targetless camera-LiDAR calibration.
Scans are taken at a fixed spacing; each is paired with the nearest image to the
scan's *reference time*: the header stamp, or, when the cloud has a per-point time
field and the caller supplies a rule, the time its visible points were measured
(a spinning LiDAR sweeps past the camera's field of view well after the sweep
started).  A pair whose nearest image is further than ``max_gap_s`` is dropped.

A recording shorter than ``max_frames`` scans at ``spacing_s`` would leave most of the
frame budget unused. With ``short_recording_spacing_s`` a denser set of scans is kept
alongside; when the recording (or ``max_seconds``) ends before ``max_frames`` scans were
taken at ``spacing_s``, up to ``max_frames`` of the denser scans, spread evenly over the
recording, are returned instead. A recording long enough for the regular selection gets
exactly the frames it would get without the option. Denser frames are independent views
only when the view changes: a recording whose consecutive denser images differ by less
than ``short_recording_min_view_change`` (median mean absolute difference, as a fraction
of full scale) keeps the regular selection, because repeating a static view adds frames
but no evidence.
"""

from __future__ import annotations

import collections
import itertools
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
    point_time_offsets_s: NDArray[np.float64] | None = None
    """Seconds after ``time_s`` at which each point was measured, when the cloud has a
    per-point time field (``None`` otherwise)."""

    @property
    def gap_s(self) -> float:
        """Image time minus the scan's reference time."""

        return self.image_time_s - self.time_s


@dataclass
class _PendingScan:
    reference_ns: int
    points: NDArray[np.float32]
    cloud_frame_id: str
    point_time_offsets_s: NDArray[np.float64] | None = None


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
    short_recording_spacing_s: float | None = None,
    short_recording_min_view_change: float = 0.05,
    report: dict[str, object] | None = None,
) -> list[CameraLidarFrame]:
    """Synchronized frames in time order, at most one scan per ``spacing_s`` seconds.

    See the module docstring for ``short_recording_spacing_s``. ``report``, when given,
    receives which selection was used (``"selection"``) and, for a short recording, the
    median view change of the denser frames.
    """

    recent: collections.deque[tuple[int, str, bytes]] = collections.deque()
    pending: dict[int, _PendingScan] = {}
    paired: dict[int, CameraLidarFrame] = {}
    regular: list[int] = []
    dense: list[int] = []
    dense_spacing_s = (
        short_recording_spacing_s
        if short_recording_spacing_s is not None and short_recording_spacing_s < spacing_s
        else None
    )
    first_ns: int | None = None
    last_regular_ns: int | None = None
    last_dense_ns: int | None = None
    window_ended = False
    scanned = 0
    gap_ns = int(max_gap_s * 1.0e9)
    point_time_field = point_time[0] if point_time is not None else None

    def regular_full() -> bool:
        return window_ended or len(regular) >= max_frames

    def pair(key: int, scan: _PendingScan) -> None:
        if not recent:
            return
        best = min(recent, key=lambda item: abs(item[0] - scan.reference_ns))
        if abs(best[0] - scan.reference_ns) > gap_ns:
            return
        message = ros_cdr.decode_ros2_image(image_topic, best[0], best[2])
        if message.data is None:
            return
        paired[key] = CameraLidarFrame(
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
            point_time_offsets_s=scan.point_time_offsets_s,
        )

    def forget(keys: set[int]) -> None:
        for key in keys:
            pending.pop(key, None)
            paired.pop(key, None)

    for connection, timestamp_ns, payload in iter_messages(bag, topics={image_topic, lidar_topic}):
        if connection.topic == image_topic:
            header = ros_cdr.decode_ros2_image(
                image_topic, timestamp_ns, payload, include_data=False
            )
            recent.append((header.timestamp_ns, header.frame_id, payload))
            # Pair before evicting: a camera slower than the window would lose the image
            # nearest to a scan.
            for key, scan in list(pending.items()):
                if header.timestamp_ns >= scan.reference_ns + gap_ns:
                    del pending[key]
                    pair(key, scan)
            while recent and recent[0][0] < header.timestamp_ns - _IMAGE_WINDOW_NS:
                recent.popleft()
            if regular_full() and not pending:
                break
            if len(regular) >= max_frames and dense:
                # The regular selection is complete: the dense set is not needed.
                forget(set(dense) - set(regular))
                dense.clear()
            continue
        scanned += 1
        emit_tick(scanned)
        if regular_full():
            continue
        first_ns = timestamp_ns if first_ns is None else first_ns
        if max_seconds is not None and (timestamp_ns - first_ns) * 1.0e-9 > max_seconds:
            window_ended = True
            continue
        take_regular = (
            last_regular_ns is None or (timestamp_ns - last_regular_ns) * 1.0e-9 >= spacing_s
        )
        take_dense = dense_spacing_s is not None and (
            last_dense_ns is None or (timestamp_ns - last_dense_ns) * 1.0e-9 >= dense_spacing_s
        )
        if not (take_regular or take_dense):
            continue
        cloud = ros_cdr.decode_ros2_pointcloud2(
            lidar_topic, timestamp_ns, payload, point_time_field=point_time_field
        )
        points = np.asarray(cloud.xyz, dtype=np.float32)
        if points.shape[0] == 0:
            continue
        if take_regular:
            regular.append(scanned)
            last_regular_ns = timestamp_ns
        if take_dense:
            dense.append(scanned)
            last_dense_ns = timestamp_ns
            if dense_spacing_s is not None and len(dense) > 2 * max_frames:
                # Keep the dense set bounded: double its spacing.
                dense_spacing_s *= 2.0
                kept = dense[::2]
                forget(set(dense) - set(kept) - set(regular))
                dense[:] = kept
        times = (
            relative_point_times(cloud.point_time_offsets_s, cloud.timestamp_ns, point_time[1])
            if point_time is not None
            else None
        )
        offset = reference_time(points, times) if reference_time is not None else 0.0
        pending[scanned] = _PendingScan(
            reference_ns=cloud.timestamp_ns + round(offset * 1.0e9),
            points=points,
            cloud_frame_id=cloud.frame_id,
            point_time_offsets_s=times - round(offset * 1.0e9) * 1.0e-9
            if times is not None
            else None,
        )
    for key, scan in list(pending.items()):
        pair(key, scan)
    chosen = regular
    selection = "regular"
    if dense_spacing_s is not None and len(regular) < max_frames and dense:
        indices = np.unique(np.round(np.linspace(0, len(dense) - 1, max_frames)).astype(int))
        candidates = [dense[index] for index in indices if dense[index] in paired]
        change = _median_view_change([paired[key] for key in candidates])
        if report is not None:
            report["median_view_change"] = change
        if change is not None and change >= short_recording_min_view_change:
            chosen, selection = candidates, "short_recording"
        else:
            selection = "regular_static_view"
    if report is not None:
        report["selection"] = selection
    frames = [paired[key] for key in chosen if key in paired]
    frames.sort(key=lambda frame: frame.time_s)
    return frames


def _median_view_change(frames: list[CameraLidarFrame]) -> float | None:
    """Median mean absolute difference of consecutive images, as a fraction of full scale."""

    ordered = sorted(frames, key=lambda frame: frame.time_s)
    changes = [
        float(
            np.mean(
                np.abs(
                    earlier.image[::4, ::4].astype(np.float32)
                    - later.image[::4, ::4].astype(np.float32)
                )
            )
        )
        / 255.0
        for earlier, later in itertools.pairwise(ordered)
        if earlier.image.shape == later.image.shape
    ]
    return float(np.median(changes)) if changes else None
