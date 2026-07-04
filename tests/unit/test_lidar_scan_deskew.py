"""Unit tests for pure scan deskew helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from lidar_scan_deskew import (  # noqa: E402
    PoseTrackSample,
    interpolate_pose_track,
    normalize_scan_timestamps,
    rigidify_scan_to_timestamp,
)


def _yaw_pose_matrix(*, x: float, y: float, z: float, yaw_rad: float) -> np.ndarray:
    half = yaw_rad * 0.5
    qz = math.sin(half)
    qw = math.cos(half)
    xx, yy, zz = 0.0, 0.0, 0.0
    xy, xz, yz = 0.0, 0.0, 0.0
    wx, wy, wz = qw * 0.0, qw * 0.0, qw * qz
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[0, 1] = 2.0 * (xy - wz)
    matrix[0, 2] = 2.0 * (xz + wy)
    matrix[1, 0] = 2.0 * (xy + wz)
    matrix[1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[1, 2] = 2.0 * (yz - wx)
    matrix[2, 0] = 2.0 * (xz - wy)
    matrix[2, 1] = 2.0 * (yz + wx)
    matrix[2, 2] = 1.0 - 2.0 * (xx + yy)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3]


def _build_synthetic_track(
    *,
    message_stamp_ns: int,
    span_ns: int,
    linear_speed_mps: float,
    yaw_rate_radps: float,
    sample_count: int = 5,
) -> tuple[tuple[PoseTrackSample, ...], np.ndarray, np.ndarray]:
  timestamps_ns = np.linspace(
      message_stamp_ns - span_ns,
      message_stamp_ns + span_ns,
      sample_count,
      dtype=np.int64,
  )
  track: list[PoseTrackSample] = []
  for stamp_ns in timestamps_ns:
      dt_s = (stamp_ns - timestamps_ns[0]) / 1_000_000_000
      yaw = yaw_rate_radps * dt_s
      x = linear_speed_mps * dt_s
      track.append(
          PoseTrackSample(
              timestamp_ns=int(stamp_ns),
              pose=_yaw_pose_matrix(x=x, y=0.0, z=0.0, yaw_rad=yaw),
          )
      )
  offsets_s = np.linspace(-0.05, 0.0, 4, dtype=np.float64)
  capture_poses = []
  for offset_s in offsets_s:
      capture_ns = message_stamp_ns + round(offset_s * 1_000_000_000)
      capture_pose, _ = interpolate_pose_track(tuple(track), capture_ns)
      capture_poses.append(capture_pose)
  raw_points = np.array(
      [[1.2, -0.3, 0.4], [2.5, 0.1, -0.2], [0.7, 0.9, 0.0], [3.1, 0.2, 0.5]],
      dtype=np.float64,
  )
  world_points = np.vstack(
      [
          _transform_points(pose, raw_points[index : index + 1])[0]
          for index, pose in enumerate(capture_poses)
      ]
  )
  t_msg_pose, _ = interpolate_pose_track(tuple(track), message_stamp_ns)
  expected = _transform_points(np.linalg.inv(t_msg_pose), world_points)
  return tuple(track), offsets_s, expected


def test_rigidify_scan_matches_analytical_motion_compensation() -> None:
    message_stamp_ns = 2_000_000_000
    track, offsets_s, expected = _build_synthetic_track(
        message_stamp_ns=message_stamp_ns,
        span_ns=200_000_000,
        linear_speed_mps=0.8,
        yaw_rate_radps=0.5,
    )
    raw_points = np.array(
        [[1.2, -0.3, 0.4], [2.5, 0.1, -0.2], [0.7, 0.9, 0.0], [3.1, 0.2, 0.5]],
        dtype=np.float64,
    )
    deskewed = rigidify_scan_to_timestamp(
        raw_points,
        offsets_s,
        message_stamp_ns,
        track,
    )
    assert deskewed == pytest.approx(expected, abs=1.0e-9)


def test_rigidify_scan_clamps_capture_time_at_track_edges() -> None:
    message_stamp_ns = 1_000_000_000
    track = (
        PoseTrackSample(
            timestamp_ns=900_000_000,
            pose=_yaw_pose_matrix(x=0.0, y=0.0, z=0.0, yaw_rad=0.0),
        ),
        PoseTrackSample(
            timestamp_ns=1_100_000_000,
            pose=_yaw_pose_matrix(x=1.0, y=0.0, z=0.0, yaw_rad=0.2),
        ),
    )
    raw_points = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    early = rigidify_scan_to_timestamp(
        raw_points,
        np.array([-0.5], dtype=np.float64),
        message_stamp_ns,
        track,
    )
    late = rigidify_scan_to_timestamp(
        raw_points,
        np.array([0.5], dtype=np.float64),
        message_stamp_ns,
        track,
    )
    early_pose, early_clamped = interpolate_pose_track(track, 500_000_000)
    late_pose, late_clamped = interpolate_pose_track(track, 1_600_000_000)
    msg_pose, _ = interpolate_pose_track(track, message_stamp_ns)
    assert early_clamped is True
    assert late_clamped is True
    expected_early = _transform_points(
        np.linalg.inv(msg_pose) @ early_pose,
        raw_points,
    )
    expected_late = _transform_points(
        np.linalg.inv(msg_pose) @ late_pose,
        raw_points,
    )
    assert early == pytest.approx(expected_early, abs=1.0e-9)
    assert late == pytest.approx(expected_late, abs=1.0e-9)


def test_normalize_scan_timestamps_maps_to_unit_interval() -> None:
    offsets = np.array([-0.08, -0.04, 0.0], dtype=np.float64)
    normalized = normalize_scan_timestamps(offsets)
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[-1] == pytest.approx(1.0)
    assert normalized[1] == pytest.approx(0.5)


def test_rigidify_scan_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="time_offsets_s length"):
        rigidify_scan_to_timestamp(
            np.zeros((2, 3), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            0,
            (),
        )
