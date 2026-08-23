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
    return np.block(
        [
            [matrix, skew(t) @ matrix],
            [np.zeros((3, 3), dtype=float), matrix],
        ]
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
    angle = float(np.linalg.norm(w))
    v_block = so3_left_jacobian(w)
    rotation = rotation_vector_to_matrix(w)
    if angle < 1.0e-12:
        q_block = 0.5 * skew(v)
    else:
        a = (1.0 - math.cos(angle)) / angle**2
        b = (angle - math.sin(angle)) / angle**3
        a_prime = (angle * math.sin(angle) - 2.0 + 2.0 * math.cos(angle)) / angle**3
        b_prime = (angle * (1.0 - math.cos(angle)) - 3.0 * (angle - math.sin(angle))) / angle**4
        cross_w_v = np.cross(w, v)
        cross_w_cross = np.cross(w, cross_w_v)
        d_v: NDArray[np.float64] = np.zeros((3, 3), dtype=float)
        for j in range(3):
            unit = np.zeros(3)
            unit[j] = 1.0
            d_v[:, j] = (
                -a_prime * w[j] / angle * cross_w_v
                - a * np.cross(unit, v)
                + b_prime * w[j] / angle * cross_w_cross
                + b * (np.cross(unit, cross_w_v) + np.cross(w, np.cross(unit, v)))
            )
        q_block = rotation.T @ d_v
    return np.block(
        [
            [rotation.T @ v_block, q_block],
            [np.zeros((3, 3), dtype=float), v_block],
        ]
    )


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

    u = se3_log(left, right)
    v_u = u[:3]
    w_u = u[3:]
    exp_u_rotation, exp_u_translation = se3_exp_local(v_u, w_u)
    scaled = alpha * u
    v_scaled = scaled[:3]
    w_scaled = scaled[3:]
    exp_scaled_rotation, exp_scaled_translation = se3_exp_local(v_scaled, w_scaled)

    v_scaled_se3 = se3_left_jacobian(v_scaled, w_scaled)
    v_u_se3 = se3_left_jacobian(v_u, w_u)
    adjoint_exp_u = se3_adjoint(exp_u_rotation, exp_u_translation)
    adjoint_exp_scaled = se3_adjoint(
        exp_scaled_rotation, exp_scaled_translation
    )

    right_jacobian = alpha * (v_scaled_se3 @ np.linalg.inv(v_u_se3))
    left_jacobian = np.linalg.inv(adjoint_exp_scaled) - alpha * (
        v_scaled_se3 @ np.linalg.inv(v_u_se3) @ np.linalg.inv(adjoint_exp_u)
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
