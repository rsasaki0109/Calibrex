"""Checkerboard poses from a ROS 2 bag: still segments of the board and their LiDAR scans.

A capture for :mod:`calibrex.evaluation.board_camera_lidar` is a recording in which a
checkerboard is held still at several poses in front of the camera and the LiDAR (or the
rig stops at several places in front of a fixed board). :func:`read_board_poses` finds
those poses in two passes over the bag:

1. **Images.** One image per ``detect_interval_s`` is decoded and searched for the board.
   Consecutive detections whose board centre and normal stay within ``still_m`` and
   ``still_deg`` of the segment's first detection form a *still segment*; a segment
   holding for ``min_still_s`` or longer is a board pose, and its middle detection is the
   camera's view of it. An image without the board (blur, a hand over a corner) does not
   end a segment, and segments at the same pose are merged: near-duplicate poses would make
   the leave-one-out stds too small. The centre and the normal are compared rather than the
   corner numbering, which OpenCV may start from either end of a symmetric board.
2. **Scans.** Every LiDAR scan stamped inside a pose's segment, less ``trim_s`` at either
   end (the board is still settling or starting to move), up to ``max_scans``, is
   accumulated into that pose's cloud. Accumulating a still board densifies sparse or
   non-repetitive scans (a Livox needs several to cover a board).

Nothing here needs the extrinsic: the camera alone finds the poses.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from calibrex.core.board_camera_lidar import BoardCameraLidarArtifact
from calibrex.core.progress import emit_tick
from calibrex.data.livox_ros2 import decode_lidar_scan
from calibrex.data.ros2_camera_imu import decode_gray_message, image_header
from calibrex.data.rosbag2 import iter_messages, require_cdr_connection
from calibrex.evaluation.board_camera_lidar import (
    ROTATION_STD_BOUND_DEG,
    TRANSLATION_STD_BOUND_M,
    UNOBSERVED_AXIS_SHARE,
    BoardAxis,
    BoardCalibration,
    BoardPose,
    BoardView,
    Checkerboard,
    detect_board,
    plane_residuals,
)
from calibrex.evaluation.visual_rotation import CameraModel

_BAG_DELAY_S = 1.0
"""Largest gap between a scan's bag time and its header stamp that is looked for."""


@dataclass(frozen=True)
class BoardCaptureOptions:
    """How still segments are found and how many scans each pose keeps."""

    detect_interval_s: float = 0.25
    still_m: float = 0.01
    """Largest move of the board centre within a still segment."""
    still_deg: float = 0.5
    """Largest turn of the board normal within a still segment."""
    min_still_s: float = 1.0
    trim_s: float = 0.2
    max_scans: int = 30
    max_seconds: float | None = None
    """Only the first seconds of the image stream (``None``: all)."""

    def __post_init__(self) -> None:
        for name in ("detect_interval_s", "still_m", "still_deg", "min_still_s", "trim_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_scans < 1:
            raise ValueError("max_scans must be at least 1")


@dataclass
class BoardSegment:
    """One board pose found in the images: when it was still and what was recorded."""

    pose_id: str
    start_s: float
    end_s: float
    detections: int
    view: BoardView
    scans: int = 0
    points: int = 0


@dataclass
class BoardCapture:
    """The board poses of a bag and what the two passes saw."""

    poses: list[BoardPose]
    segments: list[BoardSegment]
    images: int
    """Images on the topic (within ``max_seconds``)."""
    images_searched: int
    images_with_board: int
    scans: int
    """Scans on the LiDAR topic."""
    notes: list[str] = field(default_factory=list)


def _centre(view: BoardView, board: Checkerboard) -> NDArray[np.float64]:
    middle = np.array(
        [0.5 * (board.columns - 1) * board.square_m, 0.5 * (board.rows - 1) * board.square_m, 0.0]
    )
    return np.asarray(view.rotation @ middle + view.translation, dtype=np.float64)


def _same_pose(
    first: BoardView, second: BoardView, board: Checkerboard, options: BoardCaptureOptions
) -> bool:
    moved = float(np.linalg.norm(_centre(first, board) - _centre(second, board)))
    cosine = abs(float(first.rotation[:, 2] @ second.rotation[:, 2]))
    turned = math.degrees(math.acos(min(1.0, cosine)))
    return moved <= options.still_m and turned <= options.still_deg


@dataclass
class _Run:
    times: list[float]
    views: list[BoardView]

    @property
    def duration_s(self) -> float:
        return self.times[-1] - self.times[0]


def find_still_segments(
    detections: list[tuple[float, BoardView | None]],
    board: Checkerboard,
    options: BoardCaptureOptions,
) -> list[BoardSegment]:
    """Still segments of ``(time, view or None)`` detections in time order.

    A run is anchored at its first detection and lasts while the board stays at that pose;
    an image without the board (``None``: blur, a hand over a corner) does not end it.
    Consecutive runs at the same pose are merged, then runs shorter than ``min_still_s``
    are dropped.
    """

    runs: list[_Run] = []
    for time_s, view in detections:
        if view is None:
            continue
        if runs and _same_pose(runs[-1].views[0], view, board, options):
            runs[-1].times.append(time_s)
            runs[-1].views.append(view)
            continue
        runs.append(_Run([time_s], [view]))
    merged: list[_Run] = []
    for run in runs:
        if merged and _same_pose(merged[-1].views[0], run.views[0], board, options):
            merged[-1].times.extend(run.times)
            merged[-1].views.extend(run.views)
        else:
            merged.append(run)
    segments: list[BoardSegment] = []
    for run in (r for r in merged if r.duration_s >= options.min_still_s):
        halfway = 0.5 * (run.times[0] + run.times[-1])
        middle = int(np.argmin(np.abs(np.asarray(run.times) - halfway)))
        segments.append(
            BoardSegment(
                pose_id=f"{run.times[0]:.2f}s",
                start_s=run.times[0],
                end_s=run.times[-1],
                detections=len(run.views),
                view=run.views[middle],
            )
        )
    return segments


def read_board_poses(
    bag: str | Path,
    image_topic: str,
    lidar_topic: str,
    board: Checkerboard,
    camera: CameraModel,
    options: BoardCaptureOptions | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> BoardCapture:
    """The board poses of ``bag``: still segments in the images, their scans accumulated."""

    opts = options or BoardCaptureOptions()
    report = progress or (lambda message: None)
    detections: list[tuple[float, BoardView | None]] = []
    images = searched = found = 0
    first_ns: int | None = None
    last_searched_ns: int | None = None
    interval_ns = round(opts.detect_interval_s * 1.0e9)
    for connection, timestamp_ns, payload in iter_messages(bag, topics={image_topic}):
        require_cdr_connection(connection)
        stamp_ns, _frame = image_header(connection.message_type, image_topic, timestamp_ns, payload)
        first_ns = stamp_ns if first_ns is None else first_ns
        if opts.max_seconds is not None and (stamp_ns - first_ns) * 1.0e-9 > opts.max_seconds:
            break
        images += 1
        if last_searched_ns is not None and stamp_ns - last_searched_ns < interval_ns:
            continue
        last_searched_ns = stamp_ns
        _stamp, _frame, gray = decode_gray_message(
            connection.message_type, image_topic, timestamp_ns, payload
        )
        view = detect_board(np.asarray(gray), board, camera)
        searched += 1
        found += view is not None
        emit_tick(searched)
        detections.append((stamp_ns * 1.0e-9, view))
    segments = find_still_segments(detections, board, opts)
    report(
        f"board: {found} of {searched} searched image(s) show the board; "
        f"{len(segments)} still pose(s)"
    )
    clouds: list[list[NDArray[np.float32]]] = [[] for _ in segments]
    scans = 0
    if segments:
        windows = [(s.start_s + opts.trim_s, s.end_s - opts.trim_s) for s in segments]
        last_end = max(end for _start, end in windows)
        for connection, timestamp_ns, payload in iter_messages(bag, topics={lidar_topic}):
            scans += 1
            # the bag time only pre-selects (it trails the header stamp by the recording
            # delay); the header stamp decides which pose a scan belongs to
            bag_s = timestamp_ns * 1.0e-9
            if bag_s > last_end + _BAG_DELAY_S:
                break
            near = (start - _BAG_DELAY_S <= bag_s <= end + _BAG_DELAY_S for start, end in windows)
            if not any(near):
                continue
            require_cdr_connection(connection)
            cloud = decode_lidar_scan(connection.message_type, lidar_topic, timestamp_ns, payload)
            stamp_s = cloud.timestamp_ns * 1.0e-9
            hit = next(
                (i for i, (start, end) in enumerate(windows) if start <= stamp_s <= end), None
            )
            if hit is None or len(clouds[hit]) >= opts.max_scans:
                continue
            points = np.asarray(cloud.xyz, dtype=np.float32)
            if points.shape[0]:
                clouds[hit].append(points)
    poses: list[BoardPose] = []
    notes: list[str] = []
    for segment, parts in zip(segments, clouds, strict=True):
        segment.scans = len(parts)
        if not parts:
            notes.append(f"pose {segment.pose_id}: no LiDAR scan while the board was still")
            continue
        cloud = np.concatenate(parts)
        segment.points = int(cloud.shape[0])
        poses.append(BoardPose(segment.pose_id, segment.view, cloud))
    return BoardCapture(poses, segments, images, searched, found, scans, notes)


def _transform_record(matrix: NDArray[np.float64], parent: str, child: str) -> dict[str, object]:
    from scipy.spatial.transform import Rotation

    return {
        "parent_frame": parent,
        "child_frame": child,
        "translation_m": [float(v) for v in matrix[:3, 3]],
        "rotation_quat_xyzw": [float(v) for v in Rotation.from_matrix(matrix[:3, :3]).as_quat()],
    }


def axis_reason(axis: BoardAxis, unobserved_share: float) -> str:
    """Why an axis is, or is not, observed."""

    rotation = axis.unit == "deg"
    bound = ROTATION_STD_BOUND_DEG if rotation else TRANSLATION_STD_BOUND_M
    std = f"leave-one-pose-out std {axis.std:.3g} {axis.unit}"
    if not rotation and unobserved_share >= UNOBSERVED_AXIS_SHARE:
        return (
            "the board normals leave this direction unconstrained, so it keeps the start's "
            "value: tilt the board up and down as well as left and right"
        )
    if axis.observed:
        return f"{std}, within {bound:g} {axis.unit}"
    if not math.isfinite(axis.std):
        return "too few poses for a leave-one-pose-out std"
    return f"{std}, over {bound:g} {axis.unit}: add board poses, nearer and more varied"


def board_artifact(
    result: BoardCalibration,
    board: Checkerboard,
    start: NDArray[np.float64],
    *,
    camera_frame: str,
    lidar_frame: str,
    poses: list[BoardPose],
    capture: BoardCapture | None = None,
    camera: dict[str, object] | None = None,
    options: dict[str, object] | None = None,
    provenance: dict[str, object],
) -> BoardCameraLidarArtifact:
    """The evidence artifact of one board calibration."""

    transform = result.transform_camera_lidar
    share = (
        np.max(np.abs(np.asarray(result.unobserved_directions)), axis=0)
        if result.unobserved_directions
        else np.zeros(3)
    )
    axes = [
        {
            "name": axis.name,
            "unit": axis.unit,
            "value": axis.value,
            "std": axis.std if math.isfinite(axis.std) else None,
            "observed": axis.observed,
            "reason": axis_reason(axis, float(share[index - 3]) if index >= 3 else 0.0),
        }
        for index, axis in enumerate(result.axes)
    ]
    observed = [axis.name for axis in result.axes if axis.observed]
    if transform is None:
        status = "inconclusive"
        reasons = [result.reason]
    else:
        status = "pass" if len(observed) == 6 else "warn" if observed else "inconclusive"
        missing = [axis.name for axis in result.axes if not axis.observed]
        reasons = [
            result.reason,
            f"observed: {', '.join(observed) or 'none'}"
            + (f"; not observed: {', '.join(missing)}" if missing else ""),
        ]
    rejected = dict(result.poses_rejected)
    segments = {segment.pose_id: segment for segment in capture.segments} if capture else {}
    records: list[dict[str, object]] = []
    for pose in poses:
        segment = segments.get(pose.pose_id)
        record: dict[str, object] = {
            "pose_id": pose.pose_id,
            "lidar_points": int(pose.points_lidar.shape[0]),
            "distance_m": float(np.linalg.norm(pose.view.translation)),
            "reprojection_px": pose.view.reprojection_px,
            "used": pose.pose_id in result.poses_used,
            "reason": rejected.get(pose.pose_id),
        }
        if segment is not None:
            record.update(
                start_s=segment.start_s,
                end_s=segment.end_s,
                detections=segment.detections,
                scans=segment.scans,
            )
        if transform is not None:
            found = plane_residuals(pose, board, transform)
            if found is not None:
                record.update(normal_residual_deg=found[1], offset_residual_m=found[2])
        records.append(record)
    for segment in capture.segments if capture else []:
        if segment.scans == 0:
            records.append(
                {
                    "pose_id": segment.pose_id,
                    "start_s": segment.start_s,
                    "end_s": segment.end_s,
                    "detections": segment.detections,
                    "scans": 0,
                    "lidar_points": 0,
                    "distance_m": float(np.linalg.norm(segment.view.translation)),
                    "reprojection_px": segment.view.reprojection_px,
                    "used": False,
                    "reason": "no LiDAR scan while the board was still",
                }
            )
    limitations = [
        "the camera frame must be the optical frame (z forward, x right, y down)",
        "the camera intrinsics are taken as given: a focal-length error biases the board "
        "distances, more for far boards",
    ]
    if result.unobserved_directions:
        limitations.append(
            "the translation along the unobserved direction(s) is the start's value, not a "
            "measurement"
        )
    return BoardCameraLidarArtifact.model_validate(
        {
            "solver_status": result.status,
            "policy_status": status,
            "policy_reasons": reasons,
            "board": {"columns": board.columns, "rows": board.rows, "square_m": board.square_m},
            "transform": _transform_record(transform, camera_frame, lidar_frame)
            if transform is not None
            else None,
            "start_transform": _transform_record(start, camera_frame, lidar_frame),
            "axes": axes,
            "poses": records,
            "images": capture.images if capture else None,
            "images_searched": capture.images_searched if capture else None,
            "images_with_board": capture.images_with_board if capture else None,
            "normal_singular_values": list(result.normal_singular_values)
            if result.normal_singular_values is not None
            else None,
            "unobserved_directions": [list(d) for d in result.unobserved_directions],
            "plane_normal_rmse_deg": result.plane_normal_rmse_deg,
            "plane_offset_rmse_m": result.plane_offset_rmse_m,
            "camera": camera or {},
            "options": options or {},
            "limitations": limitations,
            "provenance": provenance,
        }
    )
