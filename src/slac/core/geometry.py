"""Geometry primitives with explicit slac frame conventions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import acos, isclose, sin, sqrt

Vector3 = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


def _tuple3(values: Iterable[float]) -> Vector3:
    data = tuple(float(value) for value in values)
    if len(data) != 3:
        msg = "expected exactly 3 values"
        raise ValueError(msg)
    return data


def _tuple4(values: Iterable[float]) -> QuaternionXYZW:
    data = tuple(float(value) for value in values)
    if len(data) != 4:
        msg = "expected exactly 4 values"
        raise ValueError(msg)
    return data


def normalize_quaternion_xyzw(quaternion: Iterable[float]) -> QuaternionXYZW:
    """Normalize a quaternion in xyzw order."""

    x, y, z, w = _tuple4(quaternion)
    norm = sqrt(x * x + y * y + z * z + w * w)
    if isclose(norm, 0.0):
        msg = "quaternion norm must be non-zero"
        raise ValueError(msg)
    return (x / norm, y / norm, z / norm, w / norm)


def quaternion_xyzw_from_rotation_matrix(values: Iterable[float]) -> QuaternionXYZW:
    """Convert a row-major 3x3 rotation matrix to an xyzw quaternion."""

    matrix = tuple(float(value) for value in values)
    if len(matrix) != 9:
        msg = "expected exactly 9 rotation matrix values"
        raise ValueError(msg)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix
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
    return normalize_quaternion_xyzw((x, y, z, w))


def quaternion_multiply_xyzw(left: QuaternionXYZW, right: QuaternionXYZW) -> QuaternionXYZW:
    """Hamilton product for xyzw quaternions."""

    return normalize_quaternion_xyzw(_quaternion_multiply_raw_xyzw(left, right))


def _quaternion_multiply_raw_xyzw(left: QuaternionXYZW, right: QuaternionXYZW) -> QuaternionXYZW:
    """Hamilton product without normalization."""

    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def quaternion_conjugate_xyzw(quaternion: QuaternionXYZW) -> QuaternionXYZW:
    """Return the inverse rotation for a normalized xyzw quaternion."""

    x, y, z, w = normalize_quaternion_xyzw(quaternion)
    return (-x, -y, -z, w)


def rotate_vector_xyzw(quaternion: QuaternionXYZW, point: Vector3) -> Vector3:
    """Rotate a 3D point by an xyzw quaternion."""

    q = normalize_quaternion_xyzw(quaternion)
    px, py, pz = point
    point_quat: QuaternionXYZW = (px, py, pz, 0.0)
    rotated = _quaternion_multiply_raw_xyzw(
        _quaternion_multiply_raw_xyzw(q, point_quat),
        quaternion_conjugate_xyzw(q),
    )
    return (rotated[0], rotated[1], rotated[2])


def add3(left: Vector3, right: Vector3) -> Vector3:
    """Add two 3D vectors."""

    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def neg3(value: Vector3) -> Vector3:
    """Negate a 3D vector."""

    return (-value[0], -value[1], -value[2])


@dataclass(frozen=True)
class SE3:
    """Rigid transform using the slac `T_parent_child` convention.

    A transform maps a point expressed in the child frame into the parent frame:
    `p_parent = R_parent_child * p_child + t_parent_child`.
    """

    translation_m: Vector3
    rotation_quat_xyzw: QuaternionXYZW

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation_m", _tuple3(self.translation_m))
        object.__setattr__(
            self,
            "rotation_quat_xyzw",
            normalize_quaternion_xyzw(self.rotation_quat_xyzw),
        )

    @classmethod
    def identity(cls) -> SE3:
        """Return identity transform."""

        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    @classmethod
    def from_lists(cls, translation_m: list[float], rotation_quat_xyzw: list[float]) -> SE3:
        """Construct from YAML-friendly lists."""

        return cls(_tuple3(translation_m), _tuple4(rotation_quat_xyzw))

    def inverse(self) -> SE3:
        """Return `T_child_parent`."""

        inv_q = quaternion_conjugate_xyzw(self.rotation_quat_xyzw)
        inv_t = rotate_vector_xyzw(inv_q, neg3(self.translation_m))
        return SE3(inv_t, inv_q)

    def compose(self, other: SE3) -> SE3:
        """Compose transforms.

        If this is `T_a_b` and `other` is `T_b_c`, the result is `T_a_c`.
        """

        rotated_translation = rotate_vector_xyzw(self.rotation_quat_xyzw, other.translation_m)
        translation = add3(self.translation_m, rotated_translation)
        rotation = quaternion_multiply_xyzw(self.rotation_quat_xyzw, other.rotation_quat_xyzw)
        return SE3(translation, rotation)

    def transform_point(self, point_child: Iterable[float]) -> Vector3:
        """Transform a point from child coordinates into parent coordinates."""

        rotated = rotate_vector_xyzw(self.rotation_quat_xyzw, _tuple3(point_child))
        return add3(rotated, self.translation_m)

    def as_dict(self) -> dict[str, list[float]]:
        """Return YAML-friendly transform fields."""

        return {
            "translation_m": list(self.translation_m),
            "rotation_quat_xyzw": list(self.rotation_quat_xyzw),
        }


def _quaternion_dot_xyzw(left: QuaternionXYZW, right: QuaternionXYZW) -> float:
    return (
        left[0] * right[0]
        + left[1] * right[1]
        + left[2] * right[2]
        + left[3] * right[3]
    )


def slerp_quaternion_xyzw(
    left: QuaternionXYZW,
    right: QuaternionXYZW,
    alpha: float,
) -> QuaternionXYZW:
    """Spherical linear interpolation between unit quaternions (shortest arc).

    When the dot product is negative, ``right`` is negated so the interpolation
  follows the shortest rotation path (antipodal ``q`` / ``-q`` equivalence).
    """

    clamped_alpha = min(1.0, max(0.0, float(alpha)))
    q0 = normalize_quaternion_xyzw(left)
    q1 = normalize_quaternion_xyzw(right)
    if _quaternion_dot_xyzw(q0, q1) < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
    dot = min(1.0, max(-1.0, _quaternion_dot_xyzw(q0, q1)))
    if dot > 1.0 - 1.0e-8:
        return normalize_quaternion_xyzw(
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
    return normalize_quaternion_xyzw(
        (
            weight_left * q0[0] + weight_right * q1[0],
            weight_left * q0[1] + weight_right * q1[1],
            weight_left * q0[2] + weight_right * q1[2],
            weight_left * q0[3] + weight_right * q1[3],
        )
    )


def lerp_vector3(left: Vector3, right: Vector3, alpha: float) -> Vector3:
    """Linear interpolation between two 3D points."""

    clamped_alpha = min(1.0, max(0.0, float(alpha)))
    return (
        left[0] + clamped_alpha * (right[0] - left[0]),
        left[1] + clamped_alpha * (right[1] - left[1]),
        left[2] + clamped_alpha * (right[2] - left[2]),
    )


def interpolate_se3(left: SE3, right: SE3, alpha: float) -> SE3:
    """Interpolate transforms with linear translation and quaternion slerp."""

    return SE3(
        lerp_vector3(left.translation_m, right.translation_m, alpha),
        slerp_quaternion_xyzw(left.rotation_quat_xyzw, right.rotation_quat_xyzw, alpha),
    )
