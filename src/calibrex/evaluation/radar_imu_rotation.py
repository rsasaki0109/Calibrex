"""The radar-IMU rotation from a 4D radar's Doppler returns and the IMU.

A radar that reports each return's Doppler (radial) velocity measures its own velocity
in every scan: a static return in direction ``u`` closes at ``-u . v``, so a robust
least-squares fit over a scan's returns gives ``v`` in the radar frame
(:func:`ego_velocity`). The IMU gives the same velocity's changes: between two scan
times ``a`` and ``b`` 1 s apart, within a 10 s window whose start sets a world frame,

``R_wi(b) v_i(b) - R_wi(a) v_i(a) = int_a^b R_wi (f - b_a) dt + g (b - a)``

with ``R_wi`` the gyro-integrated orientation, ``f`` the specific force, ``b_a`` the
accelerometer bias and ``g`` the window's gravity. The IMU's own velocity is the
radar's, turned into the IMU frame and less the lever arm's share:
``v_i = R_ir v_r - omega x p``. Given ``R_ir``, the equations are linear in the lever
arm ``p``, ``b_a`` and every window's gravity, so those are solved in closed form and
only the rotation and a constant clock offset ``t_imu = t_radar + dt`` are searched
(variable projection); the radar velocities are interpolated to the shifted times.

The gyro bias is the mean gyro rate over a still start of the recording (at least
1 s), or zero when the recording starts moving. The lever arm is a nuisance and is not
judged.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from calibrex import __version__
from calibrex.core.imu_lidar_rotation import (
    ImuLidarKnownBadControl,
    ImuLidarPolicyStatus,
    ImuLidarRotationArtifact,
    ImuLidarRotationDofRecord,
    ImuLidarRotationProvenance,
    ImuLidarWindowSummary,
)
from calibrex.core.provenance import git_commit
from calibrex.data.livox_ros2 import ImuSamples
from calibrex.data.radar_pointcloud import DOPPLER_FIELD_NAMES, doppler_field

FloatArray: TypeAlias = NDArray[np.float64]

_AXES: tuple[str, str, str] = ("roll", "pitch", "yaw")

RADAR_LIMITATIONS: tuple[str, ...] = (
    "Only the rotation is judged; the lever arm is fitted as a nuisance and reported "
    "in the options, not judged.",
    "A constant clock offset between the radar and the IMU is fitted (within "
    "0.1 s); a drifting clock is not modelled.",
    "Every return that agrees with the scan's ego velocity is taken as static; a "
    "scene that moves with the sensor (a vehicle's own body, a crowd walking along) "
    "biases the velocity.",
    "The gyro bias is the mean rate over a still start of the recording; a recording "
    "that starts moving uses zero.",
    "The artifact calls the radar frame 'lidar': the rotation is that of T_radar_imu.",
)


@dataclass(frozen=True)
class RadarImuOptions:
    """Ego-velocity, window and acceptance settings (the thresholds are heuristics)."""

    min_range_m: float = 0.1
    ransac_iterations: int = 60
    inlier_mps: float = 0.10
    """A return agrees with the ego velocity when its Doppler is within this."""
    min_inliers: int = 6
    max_condition: float = 30.0
    """A scan whose inlier directions leave the velocity this ill-conditioned is skipped."""
    window_s: float = 10.0
    span_s: float = 1.0
    """Velocity changes are compared between scans this far apart."""
    min_window_scans: int = 20
    still_rate_rps: float = 0.05
    min_still_s: float = 1.0
    estimate_gyro_bias: bool = True
    """Fit a constant correction to the still-start gyro bias (to first order)."""
    robust_scale_mps: float = 0.1
    random_starts: int = 12
    holdout_every: int = 3
    jackknife_groups: int = 8
    rotation_control_deg: float = 1.0
    """Each control turns (or shifts) both ways and averages the held-out chi-square
    increase: a one-sided control also measures how far the held-out optimum lies from
    the training fit, which for a radar's 0.3-0.5 deg stds hides axes the data does
    resolve."""
    time_control_s: float = 0.01
    detection_delta_chi2: float = 9.0
    observable_rotation_std_deg: float = 0.5
    observable_time_offset_std_s: float = 0.002
    max_time_offset_s: float = 0.1
    """The clock offset is searched within this (s)."""
    seed: int = 0


@dataclass(frozen=True)
class RadarVelocities:
    """Per-scan radar ego velocities (radar frame) and how many scans gave one."""

    times_s: FloatArray
    velocities_mps: FloatArray
    scans_read: int
    inlier_counts: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))


def ego_velocity(
    points: FloatArray,
    doppler_mps: FloatArray,
    rng: np.random.Generator,
    options: RadarImuOptions,
) -> tuple[FloatArray, int] | None:
    """The radar's velocity from one scan's returns (RANSAC, then least squares)."""

    opts = options
    rng_m = np.linalg.norm(points, axis=1)
    keep = (rng_m > opts.min_range_m) & np.isfinite(doppler_mps)
    if int(keep.sum()) < opts.min_inliers:
        return None
    design = -points[keep] / rng_m[keep, None]
    measured = doppler_mps[keep]
    count = len(measured)
    samples = np.array([rng.choice(count, 3, replace=False) for _ in range(opts.ransac_iterations)])
    systems = design[samples]
    solvable = np.abs(np.linalg.det(systems)) > 1.0e-3
    if not solvable.any():
        return None
    hypotheses = np.linalg.solve(systems[solvable], measured[samples[solvable]][..., None])[..., 0]
    hits = np.abs(design @ hypotheses.T - measured[:, None]) < opts.inlier_mps
    inliers = hits[:, int(np.argmax(hits.sum(axis=0)))]
    velocity = np.zeros(3)
    for _ in range(2):
        if int(inliers.sum()) < opts.min_inliers:
            return None
        velocity = np.linalg.lstsq(design[inliers], measured[inliers], rcond=None)[0]
        inliers = np.abs(design @ velocity - measured) < opts.inlier_mps
    if (
        int(inliers.sum()) < opts.min_inliers
        or np.linalg.cond(design[inliers]) > opts.max_condition
    ):
        return None
    return np.asarray(velocity, dtype=np.float64), int(inliers.sum())


def read_radar_velocities(
    bag: str | Path,
    topic: str,
    *,
    field_name: str,
    options: RadarImuOptions | None = None,
    max_seconds: float | None = None,
) -> RadarVelocities:
    """Every scan's ego velocity from a ``sensor_msgs/PointCloud2`` radar topic."""

    from calibrex.data import ros_cdr
    from calibrex.data.rosbag2 import iter_messages, require_cdr_connection

    opts = options or RadarImuOptions()
    rng = np.random.default_rng(opts.seed)
    times: list[float] = []
    velocities: list[FloatArray] = []
    counts: list[int] = []
    read = 0
    first: float | None = None
    for connection, timestamp_ns, payload in iter_messages(bag, topics={topic}):
        require_cdr_connection(connection)
        # the Doppler field is read through the per-point time decoder: a float field
        # comes back unscaled, aligned with xyz and with the same rows dropped
        message = ros_cdr.decode_ros2_pointcloud2(
            topic, timestamp_ns, payload, point_time_field=field_name
        )
        stamp = message.timestamp_ns * 1.0e-9
        first = stamp if first is None else first
        if max_seconds is not None and stamp - first > max_seconds:
            break
        read += 1
        if message.point_time_offsets_s is None:
            continue
        found = ego_velocity(
            np.asarray(message.xyz, dtype=np.float64),
            np.asarray(message.point_time_offsets_s, dtype=np.float64),
            rng,
            opts,
        )
        if found is not None:
            times.append(stamp)
            velocities.append(found[0])
            counts.append(found[1])
    order = np.argsort(times, kind="stable")
    return RadarVelocities(
        times_s=np.asarray(times, dtype=np.float64)[order],
        velocities_mps=np.asarray(velocities, dtype=np.float64).reshape(-1, 3)[order],
        scans_read=read,
        inlier_counts=np.asarray(counts, dtype=np.int64)[order],
    )


def still_start_gyro_bias(imu: ImuSamples, options: RadarImuOptions) -> tuple[FloatArray, float]:
    """The mean gyro rate over a still start, and how long the start is still (s)."""

    t, gyro = imu.times_s, imu.gyro_rps
    early = gyro[t < t[0] + 0.5]
    moving = np.flatnonzero(
        np.linalg.norm(gyro - np.median(early, axis=0), axis=1) > options.still_rate_rps
    )
    still_until = float(t[moving[0]]) if len(moving) else float(t[-1])
    still_s = still_until - float(t[0])
    if still_s < options.min_still_s:
        return np.zeros(3), still_s
    return np.asarray(gyro[t < still_until - 0.2].mean(axis=0), dtype=np.float64), still_s


@dataclass(frozen=True)
class _Kinematics:
    """IMU quantities at a window's scan times (world = the window's first IMU sample)."""

    orientation: FloatArray  # R_wi
    force_integral: FloatArray  # int R_wi f dt
    rotation_integral: FloatArray  # int R_wi dt
    rate_skew: FloatArray  # [omega]x
    bias_force: FloatArray  # int [R_wi f]x (int R_wi dt) dt: how a gyro bias moves the force


@dataclass(frozen=True)
class _Window:
    """One window's radar velocities and its densely integrated IMU."""

    stamps_s: FloatArray  # the scans' own stamps
    velocities: FloatArray  # radar frame
    imu_times_s: FloatArray
    attitude: Rotation
    force_integral: FloatArray
    rotation_integral: FloatArray
    bias_force: FloatArray
    rates: FloatArray
    first: NDArray[np.int64]
    second: NDArray[np.int64]

    @property
    def pairs(self) -> int:
        return len(self.first)

    def at(self, offset_s: float) -> _Kinematics:
        """The IMU at the scan times for ``t_imu = t_radar + offset``."""

        times = np.clip(self.stamps_s + offset_s, self.imu_times_s[0], self.imu_times_s[-1])

        def interp(values: FloatArray) -> FloatArray:
            flat = values.reshape(len(self.imu_times_s), -1)
            out = np.stack([np.interp(times, self.imu_times_s, c) for c in flat.T], axis=-1)
            return np.asarray(out.reshape((len(times), *values.shape[1:])), dtype=np.float64)

        return _Kinematics(
            orientation=Slerp(self.imu_times_s, self.attitude)(times).as_matrix(),
            force_integral=interp(self.force_integral),
            rotation_integral=interp(self.rotation_integral),
            rate_skew=_skew(interp(self.rates)),
            bias_force=interp(self.bias_force),
        )


def _skew(w: FloatArray) -> FloatArray:
    out = np.zeros((*w.shape[:-1], 3, 3))
    out[..., 0, 1], out[..., 0, 2], out[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    return out - np.swapaxes(out, -1, -2)


def _window(
    times: FloatArray,
    velocities: FloatArray,
    imu: ImuSamples,
    gyro_bias: FloatArray,
    options: RadarImuOptions,
) -> _Window | None:
    t_imu = imu.times_s
    margin = options.max_time_offset_s
    inside = (times > t_imu[0] + margin) & (times < t_imu[-1] - margin)
    times, velocities = times[inside], velocities[inside]
    if len(times) < options.min_window_scans:
        return None
    lo = max(int(np.searchsorted(t_imu, times[0] - margin)) - 1, 0)
    hi = min(int(np.searchsorted(t_imu, times[-1] + margin)) + 1, len(t_imu) - 1)
    tt = t_imu[lo : hi + 1]
    rates = imu.gyro_rps[lo : hi + 1] - gyro_bias
    force = imu.accel_mps2[lo : hi + 1]
    steps = Rotation.from_rotvec(0.5 * (rates[1:] + rates[:-1]) * np.diff(tt)[:, None])
    chained = [Rotation.identity()]
    for step in steps:
        chained.append(chained[-1] * step)
    attitude = Rotation.concatenate(chained)
    matrices = attitude.as_matrix()
    world_force = np.einsum("nij,nj->ni", matrices, force)
    dt = np.diff(tt)
    force_integral = np.vstack(
        [np.zeros(3), np.cumsum(0.5 * (world_force[1:] + world_force[:-1]) * dt[:, None], 0)]
    )
    rotation_integral = np.concatenate(
        [
            np.zeros((1, 3, 3)),
            np.cumsum(0.5 * (matrices[1:] + matrices[:-1]) * dt[:, None, None], 0),
        ]
    )
    # a gyro bias change d turns the attitude by -(int R_wi dt) d in the world frame, to
    # first order; the force integral then moves by this times d
    sensitivity = np.einsum("nij,njk->nik", _skew(world_force), rotation_integral)
    bias_force = np.concatenate(
        [
            np.zeros((1, 3, 3)),
            np.cumsum(0.5 * (sensitivity[1:] + sensitivity[:-1]) * dt[:, None, None], 0),
        ]
    )
    second = np.searchsorted(times, times + options.span_s)
    keep = second < len(times)
    if int(keep.sum()) < 3:
        return None
    return _Window(
        stamps_s=times,
        velocities=velocities,
        imu_times_s=tt,
        attitude=attitude,
        force_integral=force_integral,
        rotation_integral=rotation_integral,
        bias_force=bias_force,
        rates=rates,
        first=np.flatnonzero(keep),
        second=np.asarray(second, dtype=np.int64)[keep],
    )


def _project(values: FloatArray, dt: FloatArray) -> FloatArray:
    """Remove from per-pair 3-vectors the part a window's gravity ``g dt`` can explain."""

    weight = dt.reshape((-1,) + (1,) * (values.ndim - 1))
    mean = np.sum(weight * values, axis=0) / float(dt @ dt)
    return np.asarray(values - weight * mean, dtype=np.float64)


class _Problem:
    """The windows' equations with every window's gravity eliminated.

    For a rotation ``R_ir`` and clock offset the residual is that of the best lever arm,
    accelerometer bias and gyro-bias correction (and gravities). The gyro-bias
    correction enters to first order: it turns the integrated attitude by
    ``-(int R_wi dt) d``. Each window's rows are projected off its gravity directions
    before the linear solve. The IMU side is recomputed when the offset changes.
    """

    def __init__(self, windows: Sequence[_Window], *, gyro_bias: bool = True) -> None:
        self.windows = tuple(windows)
        self.gyro_bias = gyro_bias
        self._offset: float | None = None
        self._kinematics: list[_Kinematics] = []
        self._dt: list[FloatArray] = []
        self._force: list[FloatArray] = []
        self._fixed: list[FloatArray] = []
        self._fixed_basis = np.zeros((0, 6))

    @property
    def pairs(self) -> int:
        return int(sum(w.pairs for w in self.windows))

    @property
    def unknowns(self) -> int:
        return 9 if self.gyro_bias else 6

    def _setup(self, offset_s: float) -> None:
        if self._offset == offset_s:
            return
        self._kinematics, self._dt, self._force, self._fixed = [], [], [], []
        for w in self.windows:
            a, b = w.first, w.second
            kin = w.at(offset_s)
            dt = w.stamps_s[b] - w.stamps_s[a]
            lever = np.einsum("nij,njk->nik", kin.orientation, kin.rate_skew)
            fixed = np.concatenate(
                [-(lever[b] - lever[a]), kin.rotation_integral[b] - kin.rotation_integral[a]],
                axis=2,
            )
            self._kinematics.append(kin)
            self._dt.append(dt)
            self._force.append(kin.force_integral[b] - kin.force_integral[a])
            self._fixed.append(fixed)
        self._fixed_basis = np.linalg.qr(
            np.vstack(
                [
                    _project(f, dt).reshape(-1, 6)
                    for f, dt in zip(self._fixed, self._dt, strict=True)
                ]
            )
        )[0]
        self._offset = offset_s

    def system(self, radar_to_imu: FloatArray, offset_s: float) -> tuple[FloatArray, FloatArray]:
        """The gravity-projected design and target at a rotation ``R_ir`` and offset."""

        self._setup(offset_s)
        designs: list[FloatArray] = []
        targets: list[FloatArray] = []
        for w, kin, dt, force, fixed in zip(
            self.windows, self._kinematics, self._dt, self._force, self._fixed, strict=True
        ):
            a, b = w.first, w.second
            turned = np.einsum("nij,jk,nk->ni", kin.orientation, radar_to_imu, w.velocities)
            targets.append(_project(force - (turned[b] - turned[a]), dt).ravel())
            if self.gyro_bias:
                moved = np.einsum("nij,njk->nik", _skew(turned), kin.rotation_integral)
                bias = (moved[b] - moved[a]) - (kin.bias_force[b] - kin.bias_force[a])
                designs.append(_project(np.concatenate([fixed, bias], axis=2), dt).reshape(-1, 9))
        target = np.concatenate(targets)
        if not self.gyro_bias:
            return self._fixed_basis, target
        return np.vstack(designs), target

    def at(self, radar_to_imu: FloatArray, offset_s: float) -> FloatArray:
        """Residuals at a rotation ``R_ir`` and clock offset."""

        design, target = self.system(radar_to_imu, offset_s)
        basis = design if not self.gyro_bias else np.linalg.qr(design)[0]
        return np.asarray(basis @ (basis.T @ target) - target, dtype=np.float64)

    def nuisance(self, radar_to_imu: FloatArray, offset_s: float) -> FloatArray:
        """Lever arm (m), accelerometer bias (m/s^2) and gyro-bias correction (rad/s)."""

        self._setup(offset_s)
        target = self.system(radar_to_imu, offset_s)[1]
        if self.gyro_bias:
            design = self.system(radar_to_imu, offset_s)[0]
        else:
            design = np.vstack(
                [
                    _project(f, dt).reshape(-1, 6)
                    for f, dt in zip(self._fixed, self._dt, strict=True)
                ]
            )
        solution = np.linalg.lstsq(design, target, rcond=None)[0]
        return np.asarray(np.r_[solution, np.zeros(9 - len(solution))], dtype=np.float64)


@dataclass(frozen=True)
class _Fit:
    rotation: FloatArray  # R_ir, radar to IMU
    offset_s: float  # t_imu = t_radar + offset
    cost: float
    residuals: FloatArray
    nuisance: FloatArray  # lever arm, accelerometer bias, gyro-bias correction


def _fit(
    problem: _Problem,
    starts: Sequence[FloatArray],
    options: RadarImuOptions,
    offset_s: float = 0.0,
) -> _Fit:
    """Search the rotation from each start at ``offset_s``, then refine it with the offset."""

    def rotation_only(rotvec: FloatArray) -> FloatArray:
        return problem.at(Rotation.from_rotvec(rotvec).as_matrix(), offset_s)

    def with_offset(parameters: FloatArray) -> FloatArray:
        return problem.at(Rotation.from_rotvec(parameters[:3]).as_matrix(), float(parameters[3]))

    robust = {"loss": "soft_l1", "f_scale": options.robust_scale_mps}
    best: Any = None
    for start in starts:
        solution = least_squares(
            rotation_only, np.asarray(start, dtype=np.float64), x_scale=0.1, **robust
        )
        if best is None or solution.cost < best.cost:
            best = solution
    limit = options.max_time_offset_s
    refined = least_squares(
        with_offset,
        np.r_[best.x, offset_s],
        bounds=([-np.inf] * 3 + [-limit], [np.inf] * 3 + [limit]),
        x_scale=[0.1, 0.1, 0.1, 0.01],
        diff_step=[1e-6, 1e-6, 1e-6, 1e-3],
        **robust,
    )
    rotation = Rotation.from_rotvec(refined.x[:3]).as_matrix()
    offset = float(refined.x[3])
    return _Fit(
        rotation, offset, float(refined.cost), problem.at(rotation, offset),
        problem.nuisance(rotation, offset),
    )  # fmt: skip


def _starts(options: RadarImuOptions) -> list[FloatArray]:
    yaws = [
        Rotation.from_euler("z", angle, degrees=True).as_rotvec() for angle in range(0, 360, 90)
    ]
    flips = [
        Rotation.from_euler("x", 180, degrees=True) * Rotation.from_rotvec(item) for item in yaws
    ]
    randoms = Rotation.random(options.random_starts, random_state=options.seed).as_rotvec()
    return [*yaws, *(item.as_rotvec() for item in flips), *randoms]


def _local_deg(rotation: FloatArray, base: FloatArray) -> FloatArray:
    """Small rotation about the radar axes taking ``base`` to ``rotation`` (``T_radar_imu``)."""

    return np.degrees(Rotation.from_matrix(rotation.T @ base).as_rotvec())


def _analytic_std(fit: _Fit, problem: _Problem) -> tuple[FloatArray, float]:
    """Gauss-Newton std of small turns of ``T_radar_imu`` about the radar axes (deg), and
    of the clock offset (s), the offset's correlation with the rotation included.

    The Jacobian is taken by central differences (turns of 0.05 deg, 1 ms shifts), so
    the rotation std is in the same axes as the jackknife spread and the controls.
    """

    step = math.radians(0.05)
    base = fit.rotation.T  # T_radar_imu
    columns = []
    for axis in range(3):
        turn = np.zeros(3)
        turn[axis] = step
        ahead = (Rotation.from_rotvec(turn).as_matrix() @ base).T
        behind = (Rotation.from_rotvec(-turn).as_matrix() @ base).T
        columns.append(
            (problem.at(ahead, fit.offset_s) - problem.at(behind, fit.offset_s)) / (2.0 * step)
        )
    shift = 1.0e-3
    columns.append(
        (
            problem.at(fit.rotation, fit.offset_s + shift)
            - problem.at(fit.rotation, fit.offset_s - shift)
        )
        / (2.0 * shift)
    )
    jacobian = np.column_stack(columns)
    dof = max(len(fit.residuals) - 4, 1)
    scale = float(fit.residuals @ fit.residuals) / dof
    covariance = scale * np.linalg.pinv(jacobian.T @ jacobian)
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return np.degrees(std[:3]), float(std[3])


def _chi2(rotation: FloatArray, offset_s: float, problem: _Problem, variance: float) -> float:
    """Held-out chi-square with the nuisances refit (capped per residual)."""

    residuals = problem.at(rotation, offset_s)
    return float(np.sum(np.minimum(residuals**2 / variance, 25.0)))


@dataclass(frozen=True)
class RadarImuEvaluation:
    """The fit, its spread and controls, ready for an artifact."""

    rotation: FloatArray | None  # R_ir
    offset_s: float
    records: tuple[ImuLidarRotationDofRecord, ...]
    policy_status: ImuLidarPolicyStatus
    policy_reasons: tuple[str, ...]
    windows: int
    train_windows: int
    holdout_windows: int
    jackknife_fits: int
    pairs: int
    lever_arm_m: FloatArray | None
    accel_bias_mps2: FloatArray | None
    gyro_bias_rps: FloatArray | None
    train_rms_mps: float | None
    holdout_rms_mps: float | None


def _control(
    amount: float,
    unit: Literal["deg", "s"],
    delta: float,
    options: RadarImuOptions,
) -> ImuLidarKnownBadControl:
    return ImuLidarKnownBadControl(
        amount=amount,
        unit=unit,
        holdout_delta_chi2=float(delta),
        detected=bool(delta >= options.detection_delta_chi2),
    )


def evaluate_radar_imu_rotation(
    velocities: RadarVelocities,
    imu: ImuSamples,
    options: RadarImuOptions | None = None,
    *,
    reference_rotation: FloatArray | None = None,
    gyro_bias: FloatArray | None = None,
) -> RadarImuEvaluation:
    """Fit on train windows; collect jackknife, control and holdout evidence.

    ``reference_rotation`` (``T_radar_imu``) is compared after the fit and never
    enters it.
    """

    opts = options or RadarImuOptions()
    bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=np.float64)
    # times from the first IMU sample: absolute stamps (~1e9 s) cannot resolve the small
    # clock-offset steps of the search
    origin = float(imu.times_s[0])
    imu = ImuSamples(imu.times_s - origin, imu.gyro_rps, imu.accel_mps2)
    velocities = replace(velocities, times_s=velocities.times_s - origin)
    t = velocities.times_s
    windows: list[_Window] = []
    if len(t):
        for start in np.arange(t[0], t[-1] - opts.window_s / 2.0, opts.window_s):
            inside = (t >= start) & (t < start + opts.window_s)
            built = _window(t[inside], velocities.velocities_mps[inside], imu, bias, opts)
            if built is not None:
                windows.append(built)
    positions = np.arange(len(windows))
    holdout_ids = positions[positions % opts.holdout_every == 1]
    train_ids = positions[positions % opts.holdout_every != 1]
    pairs = sum(item.pairs for item in windows)
    if len(train_ids) < 3:
        return RadarImuEvaluation(
            None, 0.0, (), "inconclusive",
            (f"{len(windows)} radar windows of {opts.window_s:g} s with ego velocities; "
             "at least 5 are needed",),
            len(windows), len(train_ids), len(holdout_ids), 0, pairs, None, None, None, None,
            None,
        )  # fmt: skip
    train_windows = [windows[i] for i in train_ids]
    flag = opts.estimate_gyro_bias
    train = _Problem(train_windows, gyro_bias=flag)
    holdout = (
        _Problem([windows[i] for i in holdout_ids], gyro_bias=flag) if len(holdout_ids) else None
    )
    fit = _fit(train, _starts(opts), opts)
    analytic, analytic_offset = _analytic_std(fit, train)
    groups = min(opts.jackknife_groups, len(train_windows))
    samples: list[FloatArray] = []
    offsets: list[float] = []
    start = Rotation.from_matrix(fit.rotation).as_rotvec()
    for members in np.array_split(np.arange(len(train_windows)), groups):
        dropped = set(members.tolist())
        kept = _Problem(
            [w for i, w in enumerate(train_windows) if i not in dropped], gyro_bias=flag
        )
        refit = _fit(kept, [start], opts, fit.offset_s)
        samples.append(_local_deg(refit.rotation, fit.rotation))
        offsets.append(refit.offset_s)

    def spread(values: FloatArray) -> FloatArray:
        return np.asarray(
            np.sqrt((groups - 1) / groups * np.sum((values - values.mean(0)) ** 2, axis=0)),
            dtype=np.float64,
        )

    jackknife = spread(np.asarray(samples))
    jackknife_offset = float(spread(np.asarray(offsets)))
    variance = max(float(fit.residuals @ fit.residuals) / max(len(fit.residuals) - 4, 1), 1e-12)
    holdout_pairs = 0 if holdout is None else holdout.pairs
    can_control = holdout is not None and holdout_pairs >= 20
    baseline = _chi2(fit.rotation, fit.offset_s, holdout, variance) if holdout is not None else 0.0
    reference_values = (
        None
        if reference_rotation is None
        else Rotation.from_matrix(reference_rotation).as_euler("xyz", degrees=True)
    )
    estimate_euler = Rotation.from_matrix(fit.rotation.T).as_euler("xyz", degrees=True)
    errors = (
        None
        if reference_rotation is None
        else np.degrees(Rotation.from_matrix(fit.rotation.T @ reference_rotation.T).as_rotvec())
    )
    records: list[ImuLidarRotationDofRecord] = []
    for axis, name in enumerate(_AXES):
        reported = max(float(analytic[axis]), float(jackknife[axis]))
        control = None
        if can_control and holdout is not None:
            increases = []
            for sign in (1.0, -1.0):
                turn = np.zeros(3)
                turn[axis] = sign * math.radians(opts.rotation_control_deg)
                moved = (Rotation.from_rotvec(turn).as_matrix() @ fit.rotation.T).T
                increases.append(_chi2(moved, fit.offset_s, holdout, variance) - baseline)
            control = _control(opts.rotation_control_deg, "deg", float(np.mean(increases)), opts)
        records.append(
            ImuLidarRotationDofRecord(
                name=name,  # type: ignore[arg-type]
                unit="deg",
                value=float(estimate_euler[axis]),
                std_analytic=float(analytic[axis]),
                std_jackknife=float(jackknife[axis]),
                std_reported=reported,
                status="estimated"
                if reported <= opts.observable_rotation_std_deg
                else "unobservable",
                reference_value=None if reference_values is None else float(reference_values[axis]),
                error_to_reference=None if errors is None else float(errors[axis]),
                known_bad_control=control,
            )
        )
    offset_reported = max(analytic_offset, jackknife_offset)
    offset_control = None
    if can_control and holdout is not None:
        increases = [
            _chi2(fit.rotation, fit.offset_s + sign * opts.time_control_s, holdout, variance)
            - baseline
            for sign in (1.0, -1.0)
        ]
        offset_control = _control(opts.time_control_s, "s", float(np.mean(increases)), opts)
    records.append(
        ImuLidarRotationDofRecord(
            name="time_offset",
            unit="s",
            value=fit.offset_s,
            std_analytic=analytic_offset,
            std_jackknife=jackknife_offset,
            std_reported=offset_reported,
            status="estimated"
            if offset_reported <= opts.observable_time_offset_std_s
            else "unobservable",
            known_bad_control=offset_control,
        )
    )
    train_rms = float(np.sqrt(np.mean(fit.residuals**2)))
    holdout_rms = (
        None
        if holdout is None
        else float(np.sqrt(np.mean(holdout.at(fit.rotation, fit.offset_s) ** 2)))
    )
    status, reasons = _policy(records, train_rms, holdout_rms, holdout_pairs)
    return RadarImuEvaluation(
        rotation=fit.rotation,
        offset_s=fit.offset_s,
        records=tuple(records),
        policy_status=status,
        policy_reasons=reasons,
        windows=len(windows),
        train_windows=len(train_ids),
        holdout_windows=len(holdout_ids),
        jackknife_fits=groups,
        pairs=pairs,
        lever_arm_m=np.asarray(fit.nuisance[0:3]),
        accel_bias_mps2=np.asarray(fit.nuisance[3:6]),
        gyro_bias_rps=bias + np.asarray(fit.nuisance[6:9]),
        train_rms_mps=train_rms,
        holdout_rms_mps=holdout_rms,
    )


def _policy(
    records: Sequence[ImuLidarRotationDofRecord],
    train_rms: float,
    holdout_rms: float | None,
    holdout_pairs: int,
) -> tuple[ImuLidarPolicyStatus, tuple[str, ...]]:
    if holdout_rms is None or holdout_pairs < 20:
        return "inconclusive", ("too few held-out velocity pairs; holdout evidence is missing",)
    failures: list[str] = []
    warnings: list[str] = []
    if holdout_rms > max(3.0 * train_rms, 0.05):
        failures.append(
            f"held-out velocity residual {holdout_rms:.3f} m/s exceeds three times the "
            f"training residual {train_rms:.3f} m/s"
        )
    for record in records:
        control = record.known_bad_control
        if record.status == "estimated" and control is not None and not control.detected:
            warnings.append(
                f"{record.name} is reported as estimated but a known-bad {control.amount:g} "
                f"{control.unit} turn is not detected on held-out windows "
                f"(delta chi-square {control.holdout_delta_chi2:.1f})"
            )
    unobservable = [record.name for record in records if record.status == "unobservable"]
    notes = (
        ["not constrained by the data, so not calibrated: " + ", ".join(unobservable)]
        if unobservable
        else []
    )
    if failures:
        return "fail", tuple(failures + warnings + notes)
    if warnings:
        return "warn", tuple(warnings + notes)
    if any(name in _AXES for name in unobservable):
        return "inconclusive", tuple(notes)
    return "pass", ("every rotation axis is estimated and held-out windows detect every control",)


def run_radar_imu_rotation(
    bag: str | Path,
    *,
    radar_topic: str,
    imu_topic: str,
    dataset_family: str,
    dataset_license: str,
    acceleration_unit: Literal["mps2", "g"] = "mps2",
    reference_rotation: FloatArray | None = None,
    reference: str | None = None,
    options: RadarImuOptions | None = None,
    max_seconds: float | None = None,
    command: list[str] | None = None,
) -> ImuLidarRotationArtifact:
    """Read a bag's radar and IMU topics and return the rotation evidence."""

    from calibrex.data.ros2_camera_imu import load_ros2_imu
    from calibrex.data.ros_cdr import decode_ros2_pointcloud2
    from calibrex.data.rosbag2 import iter_messages

    opts = options or RadarImuOptions()
    field_name: str | None = None
    for _connection, timestamp_ns, payload in iter_messages(bag, topics={radar_topic}):
        field_name = doppler_field(
            decode_ros2_pointcloud2(radar_topic, timestamp_ns, payload).fields
        )
        break
    if field_name is None:
        raise ValueError(
            f"{radar_topic} has no float Doppler field ({', '.join(DOPPLER_FIELD_NAMES)})"
        )
    velocities = read_radar_velocities(
        bag, radar_topic, field_name=field_name, options=opts, max_seconds=max_seconds
    )
    imu = load_ros2_imu(bag, imu_topic, acceleration_unit=acceleration_unit)
    bias, still_s = still_start_gyro_bias(imu, opts)
    evaluation = evaluate_radar_imu_rotation(
        velocities, imu, opts, reference_rotation=reference_rotation, gyro_bias=bias
    )
    digest = hashlib.sha256(
        f"{Path(bag).name}:{radar_topic}:{imu_topic}:{len(velocities.times_s)}".encode()
    ).hexdigest()
    rotation = evaluation.rotation
    return ImuLidarRotationArtifact(
        method="doppler_velocity_alignment/v0.1",
        sensor_modality="radar",
        solver_status="converged" if rotation is not None else "insufficient_intervals",
        policy_status=evaluation.policy_status,
        policy_reasons=list(evaluation.policy_reasons),
        calibrated_dofs=[item.name for item in evaluation.records if item.status == "estimated"],
        rotation_quat_xyzw=None
        if rotation is None
        else [float(value) for value in Rotation.from_matrix(rotation.T).as_quat()],
        time_offset_s=evaluation.offset_s,
        gyro_bias_rps=[
            float(value)
            for value in (bias if evaluation.gyro_bias_rps is None else evaluation.gyro_bias_rps)
        ],
        dofs=list(evaluation.records),
        options={
            **asdict(opts),
            "doppler_field": field_name,
            "still_start_s": still_s,
            "gyro_bias_source": "still start" if still_s >= opts.min_still_s else "zero",
            "lever_arm_m": None
            if evaluation.lever_arm_m is None
            else [float(value) for value in evaluation.lever_arm_m],
            "accel_bias_mps2": None
            if evaluation.accel_bias_mps2 is None
            else [float(value) for value in evaluation.accel_bias_mps2],
            "train_rms_mps": evaluation.train_rms_mps,
            "holdout_rms_mps": evaluation.holdout_rms_mps,
        },
        windows=ImuLidarWindowSummary(
            scans_read=velocities.scans_read,
            odometry_segments=1,
            unreliable_registrations=velocities.scans_read - len(velocities.times_s),
            windows=evaluation.windows,
            rate_intervals=evaluation.pairs,
            imu_samples=len(imu.times_s),
        ),
        train_windows=evaluation.train_windows,
        holdout_windows=evaluation.holdout_windows,
        jackknife_fits=evaluation.jackknife_fits,
        reference=reference,
        limitations=list(RADAR_LIMITATIONS),
        provenance=ImuLidarRotationProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family=dataset_family,
            sequence_ids=[Path(bag).name],
            stream_profile=f"radar:{radar_topic}+imu:{imu_topic}",
            input_sha256=digest,
            input_digest_scope="bag name, topics and the count of scans with an ego velocity",
            dataset_license=dataset_license,
        ),
    )
