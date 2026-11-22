"""Stereo intrinsics and the cameras' relative rotation without a target.

A stereo rig turned in place (see :mod:`calibrex.evaluation.rotation_self_calibration`)
gives each camera's intrinsics on its own. The two cameras turn together, so between any
two moments the right camera's rotation is the left camera's rotation seen in the right
camera's axes: with ``R_lr`` the right camera's axes in the left camera's frame,
``rotvec(dR_right) = R_lr^T rotvec(dR_left)``. The rotation vectors of matching keyframe
intervals are aligned by a weighted Procrustes fit, which gives ``R_lr``.

The baseline's direction comes from features matched between simultaneous left and right
images: with both intrinsics and ``R_lr`` known, each match ``(x_l, x_r)`` gives
``t . ((R x_l) x x_r) = 0`` for ``X_r = R X_l + t``, so ``t`` is the null vector of the
stacked rows, signed so that the matched points lie in front of both cameras. It needs
near features (disparity of several pixels). Its length is not observable without a
metric reference, so it is taken from ``baseline_m`` when given.

Turning a rig about its middle offsets each lens by half the baseline from the rotation
centre, which lengthens the focal lengths by about 0.8 (b/2)/D at scene distance D.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from calibrex.evaluation.board_camera_lidar import _opencv
from calibrex.evaluation.intrinsics_guide import IntrinsicsEstimate
from calibrex.evaluation.rotation_self_calibration import (
    SelfCalibration,
    undistort_points,
)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class StereoSelfCalibration:
    """Both cameras' self-calibrations and the right camera's pose in the left frame."""

    left: SelfCalibration
    right: SelfCalibration
    rotation_lr: FloatArray
    """The right camera's axes in the left camera's frame (``X_l = R_lr X_r + ...``)."""
    rotation_rms_deg: float
    """RMS angle between matched keyframe intervals' rotations after the fit."""
    rotation_intervals: int
    baseline_direction: FloatArray | None
    """Unit vector ``t / |t|`` with ``X_r = R_lr^T X_l + t``, or ``None`` without matches."""
    baseline_matches: int
    baseline_m: float | None
    warnings: tuple[str, ...]

    def t_cam1_cam0(self) -> FloatArray | None:
        """Kalibr's ``T_cn_cnm1`` (left -> right) when the baseline length is known."""

        if self.baseline_direction is None or self.baseline_m is None:
            return None
        transform = np.eye(4)
        transform[:3, :3] = self.rotation_lr.T
        transform[:3, 3] = self.baseline_m * self.baseline_direction
        return transform


def _params(estimate: IntrinsicsEstimate) -> FloatArray:
    return np.array([estimate.fx, estimate.fy, estimate.cx, estimate.cy, *estimate.distortion])


def relative_rotation(
    left: SelfCalibration, right: SelfCalibration, *, gap: int = 3
) -> tuple[FloatArray, float, int]:
    """``R_lr`` from the keyframe rotations of both cameras on a shared clock.

    The left orientations are interpolated (slerp) at the right keyframe times; intervals
    of ``gap`` right keyframes give rotation-vector pairs, aligned by weighted Procrustes.
    Returns ``(R_lr, RMS residual angle in deg, intervals used)``.
    """

    if left.rotations is None or right.rotations is None:
        raise ValueError("the self-calibrations carry no keyframe rotations")
    lt = np.asarray(left.keyframe_times_s)
    rt = np.asarray(right.keyframe_times_s)
    inside = (rt >= lt[0]) & (rt <= lt[-1])
    if inside.sum() < gap + 2:
        raise ValueError("the left and right recordings do not overlap in time")
    slerp = Slerp(lt, Rotation.from_matrix(left.rotations))
    left_at = slerp(rt[inside]).as_matrix()
    right_at = right.rotations[inside]
    pairs_l, pairs_r = [], []
    for i in range(len(right_at) - gap):
        dl = Rotation.from_matrix(left_at[i].T @ left_at[i + gap]).as_rotvec()
        dr = Rotation.from_matrix(right_at[i].T @ right_at[i + gap]).as_rotvec()
        if np.linalg.norm(dl) > math.radians(2.0):
            pairs_l.append(dl)
            pairs_r.append(dr)
    if len(pairs_l) < 3:
        raise ValueError("the rig did not turn enough to relate the cameras' rotations")
    a, b = np.asarray(pairs_l), np.asarray(pairs_r)
    # b ~ R^T a  ->  a ~ R b: Procrustes of b onto a.
    u, _, vt = np.linalg.svd(a.T @ b)
    fix = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    rotation = u @ fix @ vt
    residual = Rotation.from_rotvec(b).inv() * Rotation.from_rotvec(a @ rotation)
    rms = float(np.degrees(np.sqrt(np.mean(residual.magnitude() ** 2))))
    return rotation, rms, len(a)


MIN_PARALLAX_PX = 2.0
"""The baseline direction needs at least this median parallax between the matches once
the rotation is taken out, and at least ten times the rotation's own error in pixels: a
rotation error tilts the direction by about itself over the parallax angle (the rotation
about the camera's y axis and a translation along x look alike in a narrow field)."""


def baseline_direction(
    pairs: Iterable[tuple[NDArray[np.uint8], NDArray[np.uint8]]],
    left: IntrinsicsEstimate,
    right: IntrinsicsEstimate,
    rotation_lr: FloatArray,
    rotation_error_deg: float = 0.0,
) -> tuple[FloatArray | None, int, float]:
    """``t`` (unit, ``X_r = R_lr^T X_l + t``) from features matched in simultaneous images.

    Each match gives ``t . ((R x_l) x x_r) = 0``; ``t`` is the null vector of the stacked
    rows (iteratively trimmed), signed so the points lie in front of both cameras. Returns
    ``(t or None, matches, median parallax in px)``; ``t`` is ``None`` when the parallax is
    below :data:`MIN_PARALLAX_PX` or ten times ``rotation_error_deg`` in pixels.
    """

    cv2 = _opencv()
    orb = cv2.ORB_create(2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    rotation = rotation_lr.T
    focal = 0.5 * (right.fx + right.fy)
    rows, rays_l, rays_r = [], [], []
    for image_l, image_r in pairs:
        kl, dl = orb.detectAndCompute(image_l, None)
        kr, dr = orb.detectAndCompute(image_r, None)
        if dl is None or dr is None:
            continue
        matches = matcher.match(dl, dr)
        if len(matches) < 20:
            continue
        pl = np.array([kl[m.queryIdx].pt for m in matches], dtype=np.float64)
        pr = np.array([kr[m.trainIdx].pt for m in matches], dtype=np.float64)
        rotated = undistort_points(_params(left), pl, left.model) @ rotation.T
        xr = undistort_points(_params(right), pr, right.model)
        row = np.cross(rotated, xr)
        rows.append(row / np.linalg.norm(row, axis=1, keepdims=True).clip(1e-12))
        rays_l.append(rotated)
        rays_r.append(xr)
    if not rows:
        return None, 0, 0.0
    a = np.concatenate(rows)
    rl, rr = np.concatenate(rays_l), np.concatenate(rays_r)
    parallax = np.linalg.norm(rl[:, :2] / rl[:, 2:3] - rr[:, :2], axis=1) * focal
    keep = parallax < 0.1 * focal  # beyond: mismatches
    t = np.zeros(3)
    for _ in range(5):
        _, _, vt = np.linalg.svd(a[keep])
        t = vt[-1]
        residual = np.abs(a @ t)
        keep &= residual < max(3.0 * float(np.median(residual[keep])), 1e-6)
    median_parallax = float(np.median(parallax[keep]))
    # Sign: depths in front of both cameras. X_r = s_r xr = s_l R xl + t.
    votes = 0
    for x_l, x_r in zip(rl[keep], rr[keep], strict=True):
        depths, *_ = np.linalg.lstsq(np.column_stack([x_l, -x_r]), -t, rcond=None)
        votes += 1 if (depths[0] > 0 and depths[1] > 0) else -1
    if votes < 0:
        t = -t
    needed = max(MIN_PARALLAX_PX, 10.0 * focal * math.radians(rotation_error_deg))
    if median_parallax < needed:
        return None, int(keep.sum()), median_parallax
    return t / np.linalg.norm(t), int(keep.sum()), median_parallax


def combine(
    left: SelfCalibration,
    right: SelfCalibration,
    pairs: Iterable[tuple[NDArray[np.uint8], NDArray[np.uint8]]],
    baseline_m: float | None = None,
) -> StereoSelfCalibration:
    """Relative rotation and baseline direction from two self-calibrated cameras."""

    rotation, rms, intervals = relative_rotation(left, right)
    direction, matches, parallax = baseline_direction(
        pairs, left.estimate, right.estimate, rotation, rms
    )
    warnings = []
    if rms > 0.5:
        warnings.append(
            f"the cameras' rotations agree to {rms:.2f} deg RMS only: are the recordings "
            "synchronized and the cameras rigidly mounted?"
        )
    if direction is None:
        warnings.append(
            f"the left-right matches have {parallax:.1f} px of parallax, too little against "
            f"the rotation's {rms:.2f} deg error: the baseline direction is unknown (a "
            "checkerboard measures it: calibrex intrinsics guide)"
        )
    if baseline_m is None:
        warnings.append(
            "the baseline length is not observable from rotation: give --baseline-m to write "
            "T_cn_cnm1"
        )
    warnings.append(
        "turning the rig about its middle offsets each lens by half the baseline from the "
        "rotation centre: face a distant scene (0.8 (b/2)/D lengthens the focal lengths)"
    )
    return StereoSelfCalibration(
        left,
        right,
        rotation,
        rms,
        intervals,
        direction,
        matches,
        baseline_m,
        tuple(warnings),
    )
