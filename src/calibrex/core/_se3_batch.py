"""Private fixed-state SE(3) batches with the scalar arithmetic order."""

from __future__ import annotations

import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import FloatArray, _pose_from_matrix, _ScrewSegment


def _skew_batch(vectors: FloatArray) -> FloatArray:
    output = np.zeros((len(vectors), 3, 3), dtype=float)
    x, y, z = vectors.T
    output[:, 0, 1], output[:, 0, 2] = -z, y
    output[:, 1, 0], output[:, 1, 2] = z, -x
    output[:, 2, 0], output[:, 2, 1] = -y, x
    return output


def _norm_batch(vectors: FloatArray) -> FloatArray:
    # An axis reduction need not share the scalar BLAS dot's rounding.
    return np.asarray([float(np.linalg.norm(row)) for row in vectors], dtype=float)


def _exp_blocks_batch(vectors: FloatArray) -> tuple[FloatArray, FloatArray]:
    angles = _norm_batch(vectors)
    active = ~(angles < 1.0e-12)
    rotation = np.broadcast_to(np.eye(3), (len(vectors), 3, 3)).copy()
    jacobian = rotation.copy()
    v, a = vectors[active], angles[active]
    axis_skew = _skew_batch(v / a[:, None])
    vector_skew = _skew_batch(v)
    sine = np.asarray([math.sin(float(t)) for t in a])
    cosine = np.asarray([math.cos(float(t)) for t in a])
    c2 = np.asarray([(1.0 - math.cos(float(t))) / float(t)**2 for t in a])
    c3 = np.asarray([(float(t) - math.sin(float(t))) / float(t)**3 for t in a])
    rotation[active] = (
        np.eye(3) + sine[:, None, None] * axis_skew
        + (1.0 - cosine)[:, None, None] * (axis_skew @ axis_skew)
    )
    jacobian[active] = (
        np.eye(3) - c2[:, None, None] * vector_skew
        + c3[:, None, None] * (vector_skew @ vector_skew)
    )
    return rotation, jacobian


def _cross_batch(left: FloatArray, right: FloatArray) -> FloatArray:
    return np.stack((
        left[:, 1] * right[:, 2] - left[:, 2] * right[:, 1],
        left[:, 2] * right[:, 0] - left[:, 0] * right[:, 2],
        left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0],
    ), axis=1)


def _jacobian_batch(
    v: FloatArray, w: FloatArray, rotation: FloatArray, v_block: FloatArray,
) -> FloatArray:
    angles = _norm_batch(w)
    active = ~(angles < 1.0e-12)
    q = 0.5 * _skew_batch(v)
    va, wa = v[active], w[active]
    # Keep Python float powers and libm calls, including the cancellation at
    # tiny angles. This execution change does not repair the scalar formula.
    coefficients = np.asarray([(
        (1.0 - math.cos(float(t))) / float(t)**2,
        (float(t) - math.sin(float(t))) / float(t)**3,
        (float(t) * math.sin(float(t)) - 2.0 + 2.0 * math.cos(float(t))) / float(t)**3,
        (float(t) * (1.0 - math.cos(float(t)))
         - 3.0 * (float(t) - math.sin(float(t)))) / float(t)**4,
    ) for t in angles[active]], dtype=float).reshape(-1, 4)
    a, b, ap, bp = (coefficients[:, j, None] for j in range(4))
    angle = angles[active, None]
    wv = _cross_batch(wa, va)
    wwv = _cross_batch(wa, wv)
    derivative = np.zeros((len(va), 3, 3), dtype=float)
    for j in range(3):
        unit = np.broadcast_to(np.eye(3)[j], va.shape)
        derivative[:, :, j] = (
            -ap * wa[:, j, None] / angle * wv - a * _cross_batch(unit, va)
            + bp * wa[:, j, None] / angle * wwv
            + b * (_cross_batch(unit, wv) + _cross_batch(wa, _cross_batch(unit, va)))
        )
    rt = rotation.swapaxes(1, 2)
    q[active] = rt[active] @ derivative
    output = np.zeros((len(v), 6, 6), dtype=float)
    output[:, :3, :3], output[:, :3, 3:], output[:, 3:, 3:] = rt @ v_block, q, v_block
    return output


def _sample_segment(
    segment: _ScrewSegment, alphas: FloatArray, *, jacobians: bool,
) -> tuple[list[SE3], tuple[FloatArray, FloatArray] | None]:
    if not len(alphas):
        return [], (np.empty((0, 6, 6)), np.empty((0, 6, 6))) if jacobians else None
    scaled = alphas[:, None] * segment.u
    rotation, v_block = _exp_blocks_batch(scaled[:, 3:])
    translation = (v_block @ scaled[:, :3, None])[:, :, 0]
    rotations = segment._origin_rotation @ rotation
    translations = segment._origin_translation + (
        segment._origin_rotation @ translation[:, :, None]
    )[:, :, 0]
    poses = [_pose_from_matrix(r, t) for r, t in zip(rotations, translations, strict=True)]
    if not jacobians:
        return poses, None
    inverse_u, inverse_adjoint_u = segment._factors()
    jacobian = _jacobian_batch(scaled[:, :3], scaled[:, 3:], rotation, v_block)
    adjoint = np.zeros((len(alphas), 6, 6), dtype=float)
    adjoint[:, :3, :3] = rotation
    adjoint[:, :3, 3:] = _skew_batch(translation) @ rotation
    adjoint[:, 3:, 3:] = rotation
    product = jacobian @ inverse_u
    right = alphas[:, None, None] * product
    left = np.linalg.inv(adjoint) - alphas[:, None, None] * (product @ inverse_adjoint_u)
    left[alphas == 0.] = np.eye(6)
    right[alphas == 0.] = 0.
    left[alphas == 1.] = 0.
    right[alphas == 1.] = np.eye(6)
    return poses, (left, right)


def _transform_points(poses: list[SE3], points: FloatArray) -> FloatArray:
    if not poses:
        return np.empty((0, 3), dtype=float)
    qx, qy, qz, qw = np.asarray([p.rotation_quat_xyzw for p in poses]).T
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    second_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    cx, cy, cz, cw = -qx/second_norm, -qy/second_norm, -qz/second_norm, qw/second_norm
    px, py, pz = points.T
    x = qw*px + qx*0.0 + qy*pz - qz*py
    y = qw*py - qx*pz + qy*0.0 + qz*px
    z = qw*pz + qx*py - qy*px + qz*0.0
    w = qw*0.0 - qx*px - qy*py - qz*pz
    rx = w*cx + x*cw + y*cz - z*cy
    ry = w*cy - x*cz + y*cw + z*cx
    rz = w*cz + x*cy - y*cx + z*cw
    return np.column_stack((rx, ry, rz)) + np.asarray([p.translation_m for p in poses])


def _point_jacobians(poses: list[SE3], points: FloatArray) -> FloatArray:
    if not poses:
        return np.empty((0, 3, 6), dtype=float)
    x, y, z, w = np.asarray([pose.rotation_quat_xyzw for pose in poses]).T
    rotation = np.empty((len(poses), 3, 3), dtype=float)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y*y + z*z)
    rotation[:, 0, 1] = 2.0 * (x*y - z*w)
    rotation[:, 0, 2] = 2.0 * (x*z + y*w)
    rotation[:, 1, 0] = 2.0 * (x*y + z*w)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x*x + z*z)
    rotation[:, 1, 2] = 2.0 * (y*z - x*w)
    rotation[:, 2, 0] = 2.0 * (x*z - y*w)
    rotation[:, 2, 1] = 2.0 * (y*z + x*w)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x*x + y*y)
    # Negation precedes the matrix product in the scalar point Jacobian.
    return np.concatenate((rotation, (-rotation) @ _skew_batch(points)), axis=2)
