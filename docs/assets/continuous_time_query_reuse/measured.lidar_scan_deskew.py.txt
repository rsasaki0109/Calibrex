"""Pure scan deskew helpers for rosbag odometry conversion (no kiss-icp dependency)."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from math import acos, isclose, sin, sqrt

import numpy as np

Vector3 = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


def _normalize_quaternion_xyzw(quaternion: QuaternionXYZW) -> QuaternionXYZW:
    x, y, z, w = quaternion
    norm = sqrt(x * x + y * y + z * z + w * w)
    if isclose(norm, 0.0):
        msg = "quaternion norm must be non-zero"
        raise ValueError(msg)
    return (x / norm, y / norm, z / norm, w / norm)


def _quaternion_dot_xyzw(left: QuaternionXYZW, right: QuaternionXYZW) -> float:
    return (
        left[0] * right[0]
        + left[1] * right[1]
        + left[2] * right[2]
        + left[3] * right[3]
    )


def _slerp_quaternion_xyzw(
    left: QuaternionXYZW,
    right: QuaternionXYZW,
    alpha: float,
) -> QuaternionXYZW:
    clamped_alpha = min(1.0, max(0.0, float(alpha)))
    q0 = _normalize_quaternion_xyzw(left)
    q1 = _normalize_quaternion_xyzw(right)
    if _quaternion_dot_xyzw(q0, q1) < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
    dot = min(1.0, max(-1.0, _quaternion_dot_xyzw(q0, q1)))
    if dot > 1.0 - 1.0e-8:
        return _normalize_quaternion_xyzw(
            (
                q0[0] + clamped_alpha * (q1[0] - q0[0]),
                q0[1] + clamped_alpha * (q1[1] - q0[1]),
                q0[2] + clamped_alpha * (q1[2] - q0[2]),
                q0[3] + clamped_alpha * (q1[3] - q0[3]),
            )
        )
    theta = acos(dot)
    sin_theta = sin(theta)
    weight_left = sin((1.0 - clamped_alpha) * theta) / sin_theta
    weight_right = sin(clamped_alpha * theta) / sin_theta
    return _normalize_quaternion_xyzw(
        (
            weight_left * q0[0] + weight_right * q1[0],
            weight_left * q0[1] + weight_right * q1[1],
            weight_left * q0[2] + weight_right * q1[2],
            weight_left * q0[3] + weight_right * q1[3],
        )
    )


def _lerp_vector3(left: Vector3, right: Vector3, alpha: float) -> Vector3:
    clamped_alpha = min(1.0, max(0.0, float(alpha)))
    return (
        left[0] + clamped_alpha * (right[0] - left[0]),
        left[1] + clamped_alpha * (right[1] - left[1]),
        left[2] + clamped_alpha * (right[2] - left[2]),
    )


def _rotate_vector_xyzw(quaternion: QuaternionXYZW, point: Vector3) -> Vector3:
    q = _normalize_quaternion_xyzw(quaternion)
    px, py, pz = point
    qx, qy, qz, qw = q
    tx = 2.0 * (qy * pz - qz * py)
    ty = 2.0 * (qz * px - qx * pz)
    tz = 2.0 * (qx * py - qy * px)
    return (
        px + qw * tx + (qy * tz - qz * ty),
        py + qw * ty + (qz * tx - qx * tz),
        pz + qw * tz + (qx * ty - qy * tx),
    )


def _pose_matrix_from_parts(translation: Vector3, rotation: QuaternionXYZW) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(rotation)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
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
    matrix[0, 3] = translation[0]
    matrix[1, 3] = translation[1]
    matrix[2, 3] = translation[2]
    return matrix


def _parts_from_pose_matrix(matrix: np.ndarray) -> tuple[Vector3, QuaternionXYZW]:
    translation = (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))
    rotation = _quaternion_xyzw_from_rotation_matrix(matrix[:3, :3])
    return translation, rotation


def _quaternion_xyzw_from_rotation_matrix(matrix: np.ndarray) -> QuaternionXYZW:
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    return _normalize_quaternion_xyzw((x, y, z, w))


def _interpolate_pose_matrix(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    left_t, left_q = _parts_from_pose_matrix(left)
    right_t, right_q = _parts_from_pose_matrix(right)
    translation = _lerp_vector3(left_t, right_t, alpha)
    rotation = _slerp_quaternion_xyzw(left_q, right_q, alpha)
    return _pose_matrix_from_parts(translation, rotation)


@dataclass(frozen=True)
class PoseTrackSample:
    """One timestamped ``T_world_sensor`` pose as a 4x4 matrix."""

    timestamp_ns: int
    pose: np.ndarray


def interpolate_pose_track(
    track: tuple[PoseTrackSample, ...],
    timestamp_ns: int,
) -> tuple[np.ndarray, bool]:
    """Return ``(T_world_sensor, was_clamped)`` with linear translation and slerp rotation."""

    if not track:
        return np.eye(4, dtype=np.float64), False
    timestamps = tuple(sample.timestamp_ns for sample in track)
    first_ns = timestamps[0]
    last_ns = timestamps[-1]
    if timestamp_ns <= first_ns:
        return track[0].pose.copy(), timestamp_ns < first_ns
    if timestamp_ns >= last_ns:
        return track[-1].pose.copy(), timestamp_ns > last_ns
    right_idx = bisect.bisect_right(timestamps, timestamp_ns)
    left = track[right_idx - 1]
    right = track[right_idx]
    span_ns = right.timestamp_ns - left.timestamp_ns
    if span_ns <= 0:
        return left.pose.copy(), False
    alpha = (timestamp_ns - left.timestamp_ns) / span_ns
    return _interpolate_pose_matrix(left.pose, right.pose, alpha), False


def normalize_scan_timestamps(time_offsets_s: np.ndarray) -> np.ndarray:
    """Map per-point offsets in seconds to [0, 1] for kiss-icp native deskew."""

    offsets = np.asarray(time_offsets_s, dtype=np.float64).reshape(-1)
    if offsets.size == 0:
        return offsets
    t_min = float(np.min(offsets))
    t_max = float(np.max(offsets))
    if t_max <= t_min:
        return np.zeros_like(offsets)
    return (offsets - t_min) / (t_max - t_min)


def rigidify_scan_to_timestamp(
    points: np.ndarray,
    time_offsets_s: np.ndarray,
    message_stamp_ns: int,
    track: tuple[PoseTrackSample, ...],
) -> np.ndarray:
    """Rigidify points to the message stamp using a pass-1 pose track.

    For each point ``p_i`` captured at ``t_msg + dt_i``:

    ``p_rigid = T_world_sensor(t_msg)^-1 @ T_world_sensor(t_msg + dt_i) @ p_i``
    """

    points_f64 = np.asarray(points, dtype=np.float64)
    if points_f64.ndim != 2 or points_f64.shape[1] != 3:
        msg = "points must have shape (N, 3)"
        raise ValueError(msg)
    offsets = np.asarray(time_offsets_s, dtype=np.float64).reshape(-1)
    if offsets.shape[0] != points_f64.shape[0]:
        msg = "time_offsets_s length must match point count"
        raise ValueError(msg)

    t_msg_pose, _ = interpolate_pose_track(track, message_stamp_ns)
    t_msg_inv = np.linalg.inv(t_msg_pose)
    rigid = np.empty_like(points_f64)
    for index, (point, offset_s) in enumerate(zip(points_f64, offsets, strict=True)):
        capture_ns = message_stamp_ns + round(offset_s * 1_000_000_000)
        capture_pose, _ = interpolate_pose_track(track, capture_ns)
        delta = t_msg_inv @ capture_pose
        homogeneous = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
        transformed = delta @ homogeneous
        rigid[index, 0] = transformed[0]
        rigid[index, 1] = transformed[1]
        rigid[index, 2] = transformed[2]
    return np.ascontiguousarray(rigid, dtype=np.float64)
