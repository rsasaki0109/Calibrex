"""Sensor-to-vehicle rotation from the vehicle's own motion (non-holonomic constraints).

A road vehicle moves along its forward axis and turns about its vertical axis.
Expressed in the vehicle frame ``V`` (x forward, y left, z up), a sensor's
velocity therefore has no vertical component, and its lateral component is
only the yaw rate times the sensor's longitudinal distance from the rear axle
(``v_y = w_z l``).  Its angular velocity is along z, up to zero-mean pitch and
roll rates from the suspension and the road.  Given a sensor's own motion
(for example LiDAR odometry), these constraints determine ``R_vehicle_sensor``:

* yaw from the lateral velocity (it needs forward motion);
* pitch from the vertical velocity and the rotation axis; and
* roll from the rotation axis (it needs turns).

The rotation is a left perturbation about the vehicle axes.  The lever ``l``
is a nuisance.  The translation of ``T_vehicle_sensor`` is not observable this
way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

FloatArray: TypeAlias = NDArray[np.float64]
VEHICLE_DOFS: tuple[str, str, str] = ("roll", "pitch", "yaw")


@dataclass(frozen=True)
class VehicleMotion:
    """Mean sensor velocity and angular velocity over one short interval."""

    block: int
    start_s: float
    end_s: float
    velocity_mps: FloatArray  # in the sensor frame
    angular_rate_rps: FloatArray  # in the sensor frame


@dataclass(frozen=True)
class VehicleFrameOptions:
    """Noise model and gating."""

    min_speed_mps: float = 2.0
    velocity_sigma_mps: float = 0.05
    velocity_sigma_fraction: float = 0.01
    rate_sigma_rps: float = 0.01
    huber_threshold: float = 1.5


@dataclass(frozen=True)
class VehicleFrameResult:
    """``R_vehicle_sensor``, the lever nuisance, and the analytic covariance."""

    status: Literal["converged", "insufficient_motions"]
    rotation: FloatArray | None
    lever_m: float | None
    covariance: FloatArray | None  # 4x4 over (rotation perturbation, lever)
    motions: int


def initial_vehicle_rotation(motions: Sequence[VehicleMotion]) -> FloatArray:
    """Forward axis from the mean velocity direction, up axis from the turning axis."""

    velocity = np.array([item.velocity_mps for item in motions])
    rates = np.array([item.angular_rate_rps for item in motions])
    forward = np.sum(velocity / np.linalg.norm(velocity, axis=1, keepdims=True), axis=0)
    forward /= np.linalg.norm(forward)
    # Turning axes point both ways (left and right turns); orient them consistently.
    main = np.linalg.svd(rates, full_matrices=False)[2][0]
    up = main - forward * float(main @ forward)
    up /= np.linalg.norm(up)
    left = np.cross(up, forward)
    rotation = np.vstack([forward, left, up])  # rows: vehicle axes in sensor coordinates
    if np.linalg.det(rotation) < 0:
        rotation[1] *= -1.0
    return rotation


def solve_vehicle_frame(
    motions: Sequence[VehicleMotion],
    options: VehicleFrameOptions | None = None,
    *,
    initial: FloatArray | None = None,
    up_hint: FloatArray | None = None,
) -> VehicleFrameResult:
    """Robust least squares over the non-holonomic residuals.

    ``up_hint`` (a sensor-frame vector) resolves whether the turning axis
    points up or down; without it, the initial up axis is taken to be the
    turning axis direction with a positive component along the sensor's z.
    """

    opts = options or VehicleFrameOptions()
    moving = [
        item for item in motions if float(np.linalg.norm(item.velocity_mps)) >= opts.min_speed_mps
    ]
    if len(moving) < 10:
        return VehicleFrameResult("insufficient_motions", None, None, None, len(moving))
    base = initial if initial is not None else initial_vehicle_rotation(moving)
    hint = np.array([0.0, 0.0, 1.0]) if up_hint is None else np.asarray(up_hint, dtype=np.float64)
    if initial is None and float(base[2] @ hint) < 0.0:
        base = np.diag([1.0, -1.0, -1.0]) @ base
    velocity = np.array([item.velocity_mps for item in moving])
    rates = np.array([item.angular_rate_rps for item in moving])
    speed = np.linalg.norm(velocity, axis=1)
    velocity_sigma = opts.velocity_sigma_mps + opts.velocity_sigma_fraction * speed

    def residuals(params: FloatArray) -> FloatArray:
        rotation = Rotation.from_rotvec(params[:3]).as_matrix() @ base
        v = velocity @ rotation.T
        w = rates @ rotation.T
        lateral = (v[:, 1] - w[:, 2] * params[3]) / velocity_sigma
        vertical = v[:, 2] / velocity_sigma
        tilt = w[:, :2] / opts.rate_sigma_rps
        return np.concatenate([lateral, vertical, tilt.ravel()])

    fit = least_squares(
        residuals, np.zeros(4), loss="huber", f_scale=opts.huber_threshold, x_scale="jac"
    )
    jacobian = fit.jac
    raw = residuals(fit.x)
    spread = max(1.4826 * float(np.median(np.abs(raw))), 1e-6)
    information = jacobian.T @ jacobian
    covariance = np.linalg.pinv(information) * spread**2
    rotation = Rotation.from_rotvec(fit.x[:3]).as_matrix() @ base
    return VehicleFrameResult("converged", rotation, float(fit.x[3]), covariance, len(moving))


def motions_from_poses(
    times_s: Sequence[float],
    poses: Sequence[FloatArray],
    *,
    step: int = 2,
    block_duration_s: float = 10.0,
    block_offset: int = 0,
) -> list[VehicleMotion]:
    """Mean velocity and angular velocity over ``step`` frames of a sensor trajectory."""

    times = np.asarray(times_s, dtype=np.float64)
    motions = []
    for index in range(len(times) - step):
        duration = float(times[index + step] - times[index])
        if duration <= 0.0:
            continue
        relative = np.linalg.inv(poses[index]) @ poses[index + step]
        rotation_vector = Rotation.from_matrix(relative[:3, :3]).as_rotvec()
        # The chord between the two poses is expressed in the interval's
        # midpoint orientation; in the start frame it would lean into turns by
        # half the turned angle.
        midpoint = Rotation.from_rotvec(0.5 * rotation_vector).as_matrix()
        motions.append(
            VehicleMotion(
                block=block_offset + int((times[index] - times[0]) // block_duration_s),
                start_s=float(times[index]),
                end_s=float(times[index + step]),
                velocity_mps=np.asarray(midpoint.T @ relative[:3, 3] / duration, dtype=np.float64),
                angular_rate_rps=np.asarray(rotation_vector / duration, dtype=np.float64),
            )
        )
    return motions


def rotation_std_deg(result: VehicleFrameResult) -> FloatArray:
    """Analytic std of roll, pitch, yaw (deg); inf when unsolved."""

    if result.covariance is None:
        return np.full(3, math.inf)
    return np.degrees(np.sqrt(np.clip(np.diag(result.covariance)[:3], 0.0, None)))


def vehicle_residuals(
    motions: Sequence[VehicleMotion],
    rotation: FloatArray,
    lever_m: float,
    options: VehicleFrameOptions | None = None,
) -> FloatArray:
    """Normalized non-holonomic residuals of moving intervals for a fixed rotation and lever."""

    opts = options or VehicleFrameOptions()
    moving = [
        item for item in motions if float(np.linalg.norm(item.velocity_mps)) >= opts.min_speed_mps
    ]
    if not moving:
        return np.zeros(0)
    velocity = np.array([item.velocity_mps for item in moving]) @ rotation.T
    rates = np.array([item.angular_rate_rps for item in moving]) @ rotation.T
    sigma = opts.velocity_sigma_mps + opts.velocity_sigma_fraction * np.linalg.norm(
        velocity, axis=1
    )
    lateral = (velocity[:, 1] - rates[:, 2] * lever_m) / sigma
    vertical = velocity[:, 2] / sigma
    tilt = rates[:, :2] / opts.rate_sigma_rps
    return np.concatenate([lateral, vertical, tilt.ravel()])
