"""Analytic SE(3)/SO(3) manifold operations and Jacobians.

Convention: right-invariant local perturbations with the body frame as the
tangent origin.  A pose ``T`` is perturbed as ``T <- T * Exp(xi)`` where

    xi = (v (m, body frame), w (rad, body frame axis)),
    Exp(xi) = (R(w), V(w) v),

``V`` is the SO(3) left Jacobian and ``w`` is the rotation vector.  The local
coordinates between two poses are

    xi = Log(T_a^{-1} * T_b),

so that ``T_b = T_a * Exp(xi)``.  These are the standard left-invariant
increments used by continuous-time trajectory factors; all Jacobians below are
right-trivialized and verified against finite differences in the test suite.
"""

from __future__ import annotations

import math
from struct import pack
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, normalize_quaternion_xyzw

FloatArray: TypeAlias = NDArray[np.float64]

# Rotation vector axis convention: the rotation vector magnitude is the angle
# in radians and the unit direction is the rotation axis.
# SO(3) exponential: R = I + sin(t)/t w^ + (1-cos t)/t^2 (w^)^2.


def skew(vector: FloatArray) -> NDArray[np.float64]:
    """Return the 3x3 skew-symmetric matrix of a 3-vector."""

    x, y, z = (float(value) for value in vector)
    return np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def rotation_vector_to_matrix(rotation_vector: FloatArray) -> NDArray[np.float64]:
    """Convert a rotation vector (rad) to a 3x3 rotation matrix."""

    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.eye(3, dtype=float)
    axis = vector / angle
    axis_skew = skew(axis)
    return (
        np.eye(3, dtype=float)
        + math.sin(angle) * axis_skew
        + (1.0 - math.cos(angle)) * (axis_skew @ axis_skew)
    )


def rotation_matrix_to_rotation_vector(matrix: FloatArray) -> FloatArray:
    """Convert a 3x3 rotation matrix to a rotation vector (rad)."""

    return so3_log(_matrix_to_quaternion(np.asarray(matrix, dtype=float)))


def _matrix_to_quaternion(matrix: NDArray[np.float64]) -> FloatArray:
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    w = 0.5 * math.sqrt(max(0.0, 1.0 + trace))
    if w > 1.0e-8:
        x = (rotation[2, 1] - rotation[1, 2]) / (4.0 * w)
        y = (rotation[0, 2] - rotation[2, 0]) / (4.0 * w)
        z = (rotation[1, 0] - rotation[0, 1]) / (4.0 * w)
    else:
        x = math.sqrt(
            max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        ) / 2.0
        y = math.sqrt(
            max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        ) / 2.0
        z = math.sqrt(
            max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        ) / 2.0
    return normalize_quaternion_xyzw((x, y, z, w))


def so3_left_jacobian(rotation_vector: FloatArray) -> NDArray[np.float64]:
    """Return the SO(3) left Jacobian V(w) of the exponential map.

    Satisfies ``Exp(w + d) ~= Exp(w) * Exp(V(w) d)`` to first order.
    """

    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.eye(3, dtype=float)
    vector_skew = skew(vector)
    return (
        np.eye(3, dtype=float)
        - (1.0 - math.cos(angle)) / angle**2 * vector_skew
        + (angle - math.sin(angle)) / angle**3 * (vector_skew @ vector_skew)
    )


def so3_left_jacobian_inverse(rotation_vector: FloatArray) -> NDArray[np.float64]:
    """Return the inverse SO(3) left Jacobian V(w)^{-1}."""

    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.eye(3, dtype=float)
    return np.linalg.inv(so3_left_jacobian(vector))


def se3_exp_local(
    translation: FloatArray, rotation_vector: FloatArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (R, t) of Exp(v, w) with t = V(w) v and R = Rot(w)."""

    v = np.asarray(translation, dtype=float).reshape(3)
    w = np.asarray(rotation_vector, dtype=float).reshape(3)
    rotation = rotation_vector_to_matrix(w)
    return rotation, so3_left_jacobian(w) @ v


def se3_log_local(
    rotation: FloatArray, translation: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return (v, w) such that Exp(v, w) = (rotation, translation)."""

    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    t = np.asarray(translation, dtype=float).reshape(3)
    w = rotation_matrix_to_rotation_vector(matrix)
    v = so3_left_jacobian_inverse(w) @ t
    return v, w


def se3_adjoint(
    rotation: FloatArray, translation: FloatArray
) -> NDArray[np.float64]:
    """Return the 6x6 SE(3) adjoint of T = (R, t)."""

    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    t = np.asarray(translation, dtype=float).reshape(3)
    return _se3_blocks(matrix, skew(t) @ matrix, matrix)


def _se3_blocks(
    top_left: FloatArray, top_right: FloatArray, bottom_right: FloatArray,
) -> FloatArray:
    """Assemble fixed 3x3 blocks without general nested-array dispatch."""
    result = np.zeros((6, 6), dtype=float)
    result[:3, :3] = top_left
    result[:3, 3:] = top_right
    result[3:, 3:] = bottom_right
    return result


def _cross3(left: FloatArray, right: FloatArray) -> FloatArray:
    """Cross two fixed FP64 three-vectors without broadcasting."""

    return np.asarray(
        (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ),
        dtype=float,
    )


def se3_left_jacobian(
    translation: FloatArray, rotation_vector: FloatArray
) -> NDArray[np.float64]:
    """Return the 6x6 SE(3) left Jacobian of the exponential map.

    Satisfies ``Exp(xi + d) ~= Exp(xi) * Exp(J * d)`` to first order for the
    convention ``Exp(xi) = (Rot(w), V(w) v)`` with ``xi = (v, w)``.  The
    top-left block is ``Rot(w)^T V(w)`` and the rotation block is ``V(w)``.
    """

    v = np.asarray(translation, dtype=float).reshape(3)
    w = np.asarray(rotation_vector, dtype=float).reshape(3)
    v_block = so3_left_jacobian(w)
    rotation = rotation_vector_to_matrix(w)
    return _se3_jacobian_from_exp(v, w, rotation, v_block)


def _se3_jacobian_from_exp(
    v: FloatArray, w: FloatArray, rotation: FloatArray, v_block: FloatArray,
) -> FloatArray:
    """Use the same rotation and V block as the associated exponential."""
    angle = float(np.linalg.norm(w))
    if angle < 1.0e-12:
        q_block = 0.5 * skew(v)
    else:
        a = (1.0 - math.cos(angle)) / angle**2
        b = (angle - math.sin(angle)) / angle**3
        a_prime = (angle * math.sin(angle) - 2.0 + 2.0 * math.cos(angle)) / angle**3
        b_prime = (angle * (1.0 - math.cos(angle)) - 3.0 * (angle - math.sin(angle))) / angle**4
        cross_w_v = _cross3(w, v)
        cross_w_cross = _cross3(w, cross_w_v)
        d_v: NDArray[np.float64] = np.zeros((3, 3), dtype=float)
        for j in range(3):
            unit = np.zeros(3)
            unit[j] = 1.0
            d_v[:, j] = (
                -a_prime * w[j] / angle * cross_w_v
                - a * _cross3(unit, v)
                + b_prime * w[j] / angle * cross_w_cross
                + b * (_cross3(unit, cross_w_v) + _cross3(w, _cross3(unit, v)))
            )
        q_block = rotation.T @ d_v
    return _se3_blocks(rotation.T @ v_block, q_block, v_block)


def se3_left_jacobian_inverse(
    translation: FloatArray, rotation_vector: FloatArray
) -> NDArray[np.float64]:
    """Return the inverse 6x6 SE(3) left Jacobian."""

    v = np.asarray(translation, dtype=float).reshape(3)
    w = np.asarray(rotation_vector, dtype=float).reshape(3)
    return np.linalg.inv(se3_left_jacobian(v, w))


def se3_exp(origin: SE3, local_delta: FloatArray) -> SE3:
    """Return ``origin * Exp(xi)`` with a 6-vector local delta.

    The first three entries are translation in the body frame and the last
    three are the rotation vector in the body frame.
    """

    delta = np.asarray(local_delta, dtype=float).reshape(6)
    rotation, translation = se3_exp_local(delta[:3], delta[3:])
    origin_rotation, origin_translation = _matrix_pose(origin)
    composed_rotation = origin_rotation @ rotation
    composed_translation = origin_translation + origin_rotation @ translation
    return _pose_from_matrix(composed_rotation, composed_translation)


def se3_log(origin: SE3, target: SE3) -> FloatArray:
    """Return the 6-vector local delta with ``target = origin * Exp(xi)``."""

    origin_rotation, origin_translation = _matrix_pose(origin)
    target_rotation, target_translation = _matrix_pose(target)
    relative_rotation = origin_rotation.T @ target_rotation
    relative_translation = origin_rotation.T @ (
        target_translation - origin_translation
    )
    v, w = se3_log_local(relative_rotation, relative_translation)
    return np.concatenate([v, w])


def _rotation_log(origin: SE3, target: SE3) -> FloatArray:
    """Keep the rotation part of Log without its discarded translation solve."""
    origin_rotation, _origin_translation = _matrix_pose(origin)
    target_rotation, _target_translation = _matrix_pose(target)
    return rotation_matrix_to_rotation_vector(origin_rotation.T @ target_rotation)


def point_transform_jacobian(transform: SE3, point: FloatArray) -> NDArray[np.float64]:
    """Return the 3x6 Jacobian of ``T * p`` w.r.t. a right perturbation."""

    rotation, _ = _matrix_pose(transform)
    p = np.asarray(point, dtype=float).reshape(3)
    return np.hstack([rotation, -rotation @ skew(p)])


def interpolate_screw_jacobians(
    left: SE3,
    right: SE3,
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the 6x6 Jacobians of the screw-interpolated pose.

    The interpolated pose is ``P(a) = left * Exp(a * u)`` with
    ``u = Log(left^{-1} * right)``.  The returned pair is the right-trivialized
    derivative of ``P`` with respect to the left and right knot perturbations.
    """

    return _ScrewSegment(left, right).jacobians(alpha)


class _ScrewSegment:
    """Prepare one fixed knot pair, retaining the original FP64 arithmetic."""

    def __init__(self, left: SE3, right: SE3) -> None:
        self.left = left
        self.u = se3_log(left, right)
        self._origin_rotation, self._origin_translation = _matrix_pose(left)
        self._jacobian_factors: tuple[FloatArray, FloatArray] | None = None
        self._sample_key: bytes | None = None
        self._sample_values: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None

    def _sample(self, alpha: float) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        key = pack("!d", alpha)
        if key != self._sample_key:
            scaled = alpha * self.u
            rotation = rotation_vector_to_matrix(scaled[3:])
            v_block = so3_left_jacobian(scaled[3:])
            self._sample_values = scaled, rotation, v_block @ scaled[:3], v_block
            self._sample_key = key
        assert self._sample_values is not None
        return self._sample_values

    def pose(self, alpha: float) -> SE3:
        _scaled, rotation, translation, _v_block = self._sample(alpha)
        return _pose_from_matrix(
            self._origin_rotation @ rotation,
            self._origin_translation + self._origin_rotation @ translation,
        )

    def _factors(self) -> tuple[FloatArray, FloatArray]:
        if self._jacobian_factors is None:
            v_u, w_u = self.u[:3], self.u[3:]
            rotation, translation = se3_exp_local(v_u, w_u)
            self._jacobian_factors = (
                np.linalg.inv(se3_left_jacobian(v_u, w_u)),
                np.linalg.inv(se3_adjoint(rotation, translation)),
            )
        return self._jacobian_factors

    def jacobians(self, alpha: float) -> tuple[FloatArray, FloatArray]:
        inverse_u, inverse_adjoint_u = self._factors()
        scaled, rotation, translation, v_block = self._sample(alpha)
        v_scaled, w_scaled = scaled[:3], scaled[3:]
        scaled_jacobian = _se3_jacobian_from_exp(v_scaled, w_scaled, rotation, v_block)
        product = scaled_jacobian @ inverse_u
        right_jacobian = alpha * product
        left_jacobian = np.linalg.inv(se3_adjoint(rotation, translation)) - alpha * (
            product @ inverse_adjoint_u
        )
        return left_jacobian, right_jacobian


def _matrix_pose(transform: SE3) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x, y, z, w = transform.rotation_quat_xyzw
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return rotation, np.asarray(transform.translation_m, dtype=float).reshape(3)


def _pose_from_matrix(
    rotation: NDArray[np.float64], translation: NDArray[np.float64]
) -> SE3:
    quaternion = _matrix_to_quaternion(rotation)
    return SE3(
        (float(translation[0]), float(translation[1]), float(translation[2])),
        quaternion,
    )


def so3_exp(rotation_vector: FloatArray) -> FloatArray:
    """Return the xyzw quaternion of a rotation vector."""

    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    axis = vector / angle
    half = angle / 2.0
    s = math.sin(half)
    return np.asarray(
        [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)],
        dtype=float,
    )


def so3_log(quaternion: FloatArray) -> FloatArray:
    """Return the rotation vector (rad) of an xyzw quaternion."""

    x, y, z, w = (float(value) for value in np.asarray(quaternion, dtype=float))
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(vector_norm, w)
    return np.asarray([x, y, z], dtype=float) / vector_norm * angle
