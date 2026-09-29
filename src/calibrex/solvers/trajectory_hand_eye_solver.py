"""Joint trajectory hand-eye calibration with per-DoF observability.

Solves ``A(t + dt) X = X B`` for the extrinsic ``X = T_ref_sensor``, an
optional clock offset ``dt`` (``t_ref = t_sensor + dt``), and an optional
reference trajectory scale ``s`` that multiplies the translations of ``A``.
``A`` is the motion of a reference trajectory (INS/GNSS) interpolated at the
sensor's timestamps and ``B`` is the sensor's own relative motion (LiDAR
odometry).

Rotation and translation are estimated jointly.  For planar vehicle motion
every rotation axis is nearly vertical, so a rotation-first method cannot
recover the yaw of ``X``; the translation equations can, and only the
translation along the axis of motion rotation stays unobservable.  The
reference scale absorbs a systematic length mismatch between the two
trajectories (on KITTI raw the OXTS track is about 3 % shorter than the
LiDAR-observed motion), which would otherwise be pushed into the lever arm.

Residuals are normalized by an explicit noise model: a fixed rotation sigma,
and a translation sigma that grows with the motion length, because heading
errors of either trajectory produce lateral errors proportional to distance.
Each DoF is reported with its data-only standard deviation and classified as
``estimated`` (the data constrain it), ``prior`` (only a declared prior does),
or ``unobservable`` (nothing does, and its value must not be used).

DoFs are expressed in the reference (parent) frame: ``roll``/``pitch``/``yaw``
are left rotation perturbations about the parent x/y/z axes, and ``x``/``y``/
``z`` are the translation of ``X`` in the parent frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

FloatArray: TypeAlias = NDArray[np.float64]
DofName = Literal["roll", "pitch", "yaw", "x", "y", "z", "time_offset", "reference_scale"]
DofStatus = Literal["estimated", "prior", "unobservable"]
DOF_NAMES: tuple[DofName, ...] = (
    "roll",
    "pitch",
    "yaw",
    "x",
    "y",
    "z",
    "time_offset",
    "reference_scale",
)
_TIME = 6
_SCALE = 7


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Timestamped reference poses ``T_world_ref`` (seconds, 4x4).

    Interpolation never bridges a gap longer than ``max_gap_s`` between
    consecutive samples, so concatenated drives or dropped packets cannot
    produce invented motion.
    """

    times_s: FloatArray
    poses: FloatArray
    max_gap_s: float = 1.0

    def __post_init__(self) -> None:
        if len(self.times_s) != len(self.poses) or len(self.times_s) < 2:
            raise ValueError("reference trajectory needs at least two timestamped poses")
        if np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("reference trajectory timestamps must be strictly increasing")
        object.__setattr__(
            self, "_slerp", Slerp(self.times_s, Rotation.from_matrix(self.poses[:, :3, :3]))
        )

    def covers(self, time_s: float) -> bool:
        """Return whether ``time_s`` lies inside the trajectory (no extrapolation)."""

        if not self.times_s[0] <= time_s <= self.times_s[-1]:
            return False
        upper = int(np.searchsorted(self.times_s, time_s, side="left"))
        if upper == 0 or self.times_s[upper] == time_s:
            return True
        return bool(self.times_s[upper] - self.times_s[upper - 1] <= self.max_gap_s)

    def continuous(self, start_s: float, end_s: float) -> bool:
        """Return whether ``[start_s, end_s]`` is covered without any data gap."""

        if not (self.covers(start_s) and self.covers(end_s)):
            return False
        lower = int(np.searchsorted(self.times_s, min(start_s, end_s), side="right")) - 1
        upper = int(np.searchsorted(self.times_s, max(start_s, end_s), side="left"))
        window = self.times_s[max(lower, 0) : upper + 1]
        return bool(window.size < 2 or np.max(np.diff(window)) <= self.max_gap_s)

    def pose_at(self, time_s: float) -> FloatArray:
        """Interpolate ``T_world_ref`` (slerp rotation, linear translation)."""

        if not self.covers(time_s):
            raise ValueError(f"time {time_s} is outside the reference trajectory")
        pose = np.eye(4)
        pose[:3, :3] = self._slerp([time_s]).as_matrix()[0]  # type: ignore[attr-defined]
        for axis in range(3):
            pose[axis, 3] = np.interp(time_s, self.times_s, self.poses[:, axis, 3])
        return pose

    def poses_at(self, times_s: FloatArray) -> FloatArray:
        """Vectorized :meth:`pose_at` for times already checked with :meth:`covers`."""

        times = np.asarray(times_s, dtype=np.float64)
        if times.size and (times.min() < self.times_s[0] or times.max() > self.times_s[-1]):
            raise ValueError("a query time is outside the reference trajectory")
        poses = np.tile(np.eye(4), (times.size, 1, 1))
        poses[:, :3, :3] = self._slerp(times).as_matrix()  # type: ignore[attr-defined]
        for axis in range(3):
            poses[:, axis, 3] = np.interp(times, self.times_s, self.poses[:, axis, 3])
        return poses

    def motion(self, start_s: float, end_s: float) -> FloatArray:
        """Return the relative motion ``T_ref(start)^-1 T_ref(end)``."""

        return np.linalg.inv(self.pose_at(start_s)) @ self.pose_at(end_s)

    def rebased(self, epoch_s: float) -> ReferenceTrajectory:
        """Return the trajectory with ``epoch_s`` subtracted from every timestamp."""

        return ReferenceTrajectory(self.times_s - epoch_s, self.poses, self.max_gap_s)


@dataclass(frozen=True)
class SensorMotion:
    """Relative sensor motion ``T_sensor(start)^-1 T_sensor(end)``."""

    motion_id: str
    block: int
    start_s: float
    end_s: float
    motion: FloatArray


@dataclass(frozen=True)
class DofPrior:
    """A declared Gaussian prior on one absolute translation or time DoF."""

    dof: Literal["x", "y", "z", "time_offset"]
    value: float
    sigma: float
    source: str

    def __post_init__(self) -> None:
        if self.dof not in {"x", "y", "z", "time_offset"}:
            raise ValueError(f"priors are supported on x, y, z, and time_offset, not {self.dof!r}")
        if not (math.isfinite(self.value) and math.isfinite(self.sigma) and self.sigma > 0.0):
            raise ValueError("a DoF prior needs a finite value and a positive sigma")


@dataclass(frozen=True)
class TrajectoryHandEyeOptions:
    """Noise model, nuisance parameters, and observability policy."""

    estimate_time_offset: bool = True
    time_offset_bound_s: float = 0.2
    estimate_reference_scale: bool = True
    reference_scale_bound: float = 0.1
    rotation_sigma_rad: float = math.radians(0.1)
    translation_sigma_floor_m: float = 0.01
    translation_sigma_per_m: float = 0.02
    huber_threshold: float = 1.5
    irls_iterations: int = 5
    yaw_starts_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    observable_rotation_std_deg: float = 0.1
    observable_translation_std_m: float = 0.02
    observable_time_offset_std_s: float = 0.005
    observable_reference_scale_std: float = 0.005
    anchor_sigma: float = 1.0e3
    priors: tuple[DofPrior, ...] = ()

    def active_mask(self) -> NDArray[np.bool_]:
        """Return which of the eight parameters are estimated."""

        mask = np.ones(8, dtype=bool)
        mask[_TIME] = self.estimate_time_offset
        mask[_SCALE] = self.estimate_reference_scale
        return mask


@dataclass(frozen=True)
class DofEstimate:
    """One estimated DoF, its uncertainty, and where its value comes from."""

    name: DofName
    value: float
    std_data: float
    std_total: float
    status: DofStatus


@dataclass(frozen=True)
class MotionResidualStats:
    """Residual norms of ``(A X)^-1 X B`` over motions."""

    motion_count: int
    rotation_rmse_deg: float | None
    translation_rmse_m: float | None
    rotation_median_deg: float | None
    translation_median_m: float | None


@dataclass(frozen=True)
class TrajectoryHandEyeResult:
    """Joint estimate with per-DoF observability."""

    status: Literal["converged", "insufficient_motions"]
    transform: FloatArray | None
    time_offset_s: float
    reference_scale: float
    dofs: tuple[DofEstimate, ...]
    variance_factor: float | None
    information_eigenvalues: tuple[float, ...]
    train: MotionResidualStats
    start_yaw_deg: float | None
    motion_ids: tuple[str, ...] = field(default_factory=tuple)

    def dof(self, name: DofName) -> DofEstimate:
        """Return the estimate of one DoF."""

        return next(item for item in self.dofs if item.name == name)


class TrajectoryHandEyeSolver:
    """Robust joint ``A(t + dt) X = X B`` solver with observability analysis."""

    def solve(
        self,
        reference: ReferenceTrajectory,
        motions: Sequence[SensorMotion],
        options: TrajectoryHandEyeOptions | None = None,
    ) -> TrajectoryHandEyeResult:
        opts = options or TrajectoryHandEyeOptions()
        # Absolute timestamps (~1e9 s) leave only ~0.2 us of float64 resolution,
        # which is below the finite-difference step of the optimizer, so the
        # clock offset would never move.  Solve in epoch-relative time instead.
        epoch = float(reference.times_s[0])
        reference = reference.rebased(epoch)
        motions = [
            replace(item, start_s=item.start_s - epoch, end_s=item.end_s - epoch)
            for item in motions
        ]
        usable = covered_motions(reference, motions, opts)
        if len(usable) < 3:
            return TrajectoryHandEyeResult(
                status="insufficient_motions",
                transform=None,
                time_offset_s=0.0,
                reference_scale=1.0,
                dofs=(),
                variance_factor=None,
                information_eigenvalues=(),
                train=MotionResidualStats(len(usable), None, None, None, None),
                start_yaw_deg=None,
            )
        best: tuple[float, FloatArray, FloatArray, float] | None = None
        for yaw_deg in opts.yaw_starts_deg:
            base = np.eye(4)
            base[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
            params, weights, _ = _irls(reference, usable, base, opts)
            normalized = _normalized_residuals(reference, usable, base, params, opts)
            cost = float(np.sum(weights * normalized**2))
            if best is None or cost < best[0]:
                best = (cost, base, params, yaw_deg)
        assert best is not None
        _, base, params, yaw_deg = best
        _, weights, variance_factor = _irls(reference, usable, base, opts, initial=params)
        transform = compose(base, params)
        dofs, eigenvalues = _observability(
            reference, usable, base, params, weights, variance_factor, opts
        )
        time_offset = float(params[_TIME]) if opts.estimate_time_offset else 0.0
        scale = 1.0 + (float(params[_SCALE]) if opts.estimate_reference_scale else 0.0)
        return TrajectoryHandEyeResult(
            status="converged",
            transform=transform,
            time_offset_s=time_offset,
            reference_scale=scale,
            dofs=dofs,
            variance_factor=variance_factor,
            information_eigenvalues=eigenvalues,
            train=motion_residual_stats(reference, usable, transform, time_offset, scale),
            start_yaw_deg=yaw_deg,
            motion_ids=tuple(item.motion_id for item in usable),
        )


def compose(base: FloatArray, params: FloatArray) -> FloatArray:
    """Apply a left rotation perturbation and a parent-frame translation."""

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix() @ base[:3, :3]
    transform[:3, 3] = base[:3, 3] + params[3:6]
    return transform


def perturb(transform: FloatArray, dof: DofName, amount: float) -> FloatArray:
    """Return ``transform`` moved by ``amount`` (rad or m) along a rotation/translation DoF."""

    index = DOF_NAMES.index(dof)
    if index >= _TIME:
        raise ValueError(f"{dof} is not a rotation or translation DoF")
    params = np.zeros(8)
    params[index] = amount
    return compose(transform, params)


def covered_motions(
    reference: ReferenceTrajectory,
    motions: Sequence[SensorMotion],
    opts: TrajectoryHandEyeOptions,
) -> list[SensorMotion]:
    """Keep motions whose endpoints stay interpolable for every allowed offset."""

    margin = opts.time_offset_bound_s if opts.estimate_time_offset else 0.0
    return [
        item
        for item in motions
        if all(
            reference.continuous(item.start_s + shift, item.end_s + shift)
            for shift in (-margin, 0.0, margin)
        )
    ]


def motion_errors(
    reference: ReferenceTrajectory,
    motions: Sequence[SensorMotion],
    transform: FloatArray,
    time_offset_s: float = 0.0,
    reference_scale: float = 1.0,
) -> FloatArray:
    """Return the ``(rotation vector, translation)`` error of every motion as ``(N, 6)``."""

    if not motions:
        return np.empty((0, 6))
    starts = reference.poses_at(np.array([item.start_s for item in motions]) + time_offset_s)
    ends = reference.poses_at(np.array([item.end_s for item in motions]) + time_offset_s)
    start_rotation_t = np.transpose(starts[:, :3, :3], (0, 2, 1))
    motion_a = np.tile(np.eye(4), (len(motions), 1, 1))
    motion_a[:, :3, :3] = start_rotation_t @ ends[:, :3, :3]
    motion_a[:, :3, 3] = reference_scale * np.einsum(
        "nij,nj->ni", start_rotation_t, ends[:, :3, 3] - starts[:, :3, 3]
    )
    motion_b = np.stack([item.motion for item in motions])
    left = motion_a @ transform
    right = transform @ motion_b
    left_rotation_t = np.transpose(left[:, :3, :3], (0, 2, 1))
    error_rotation = left_rotation_t @ right[:, :3, :3]
    error_translation = np.einsum("nij,nj->ni", left_rotation_t, right[:, :3, 3] - left[:, :3, 3])
    rows = np.empty((len(motions), 6))
    rows[:, :3] = Rotation.from_matrix(error_rotation).as_rotvec()
    rows[:, 3:] = error_translation
    return rows


def motion_residual_stats(
    reference: ReferenceTrajectory,
    motions: Sequence[SensorMotion],
    transform: FloatArray,
    time_offset_s: float = 0.0,
    reference_scale: float = 1.0,
) -> MotionResidualStats:
    """Summarize residual norms, excluding motions outside the reference span."""

    covered = [
        item
        for item in motions
        if reference.continuous(item.start_s + time_offset_s, item.end_s + time_offset_s)
    ]
    if not covered:
        return MotionResidualStats(0, None, None, None, None)
    errors = motion_errors(reference, covered, transform, time_offset_s, reference_scale)
    rotation = np.linalg.norm(errors[:, :3], axis=1)
    translation = np.linalg.norm(errors[:, 3:], axis=1)
    return MotionResidualStats(
        motion_count=len(covered),
        rotation_rmse_deg=math.degrees(float(np.sqrt(np.mean(rotation**2)))),
        translation_rmse_m=float(np.sqrt(np.mean(translation**2))),
        rotation_median_deg=math.degrees(float(np.median(rotation))),
        translation_median_m=float(np.median(translation)),
    )


def motion_sigmas(motions: Sequence[SensorMotion], opts: TrajectoryHandEyeOptions) -> FloatArray:
    """Return the per-component noise sigmas of the declared motion noise model."""

    lengths = np.array([float(np.linalg.norm(item.motion[:3, 3])) for item in motions])
    translation = opts.translation_sigma_floor_m + opts.translation_sigma_per_m * lengths
    sigmas = np.empty((len(motions), 6))
    sigmas[:, :3] = opts.rotation_sigma_rad
    sigmas[:, 3:] = translation[:, None]
    return sigmas.reshape(-1)


def _normalized_residuals(
    reference: ReferenceTrajectory,
    motions: Sequence[SensorMotion],
    base: FloatArray,
    params: FloatArray,
    opts: TrajectoryHandEyeOptions,
) -> FloatArray:
    offset = float(params[_TIME]) if opts.estimate_time_offset else 0.0
    scale = 1.0 + (float(params[_SCALE]) if opts.estimate_reference_scale else 0.0)
    errors = motion_errors(reference, motions, compose(base, params), offset, scale)
    return errors.reshape(-1) / motion_sigmas(motions, opts)


def _prior_rows(base: FloatArray, params: FloatArray, opts: TrajectoryHandEyeOptions) -> FloatArray:
    rows = []
    for prior in opts.priors:
        if prior.dof == "time_offset":
            if not opts.estimate_time_offset:
                continue
            value = float(params[_TIME])
        else:
            axis = "xyz".index(prior.dof)
            value = float(base[axis, 3] + params[3 + axis])
        rows.append((value - prior.value) / prior.sigma)
    return np.array(rows, dtype=np.float64)


def _irls(
    reference: ReferenceTrajectory,
    motions: Sequence[SensorMotion],
    base: FloatArray,
    opts: TrajectoryHandEyeOptions,
    *,
    initial: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, float]:
    """Iteratively reweighted least squares with Huber weights.

    Returns the parameters, the Huber weights, and the variance factor that
    rescales the declared noise model to the observed residual spread.
    """

    mask = opts.active_mask()
    params = np.zeros(8) if initial is None else initial.copy()
    params[~mask] = 0.0
    lower = np.full(8, -np.inf)
    upper = np.full(8, np.inf)
    lower[_TIME], upper[_TIME] = -opts.time_offset_bound_s, opts.time_offset_bound_s
    lower[_SCALE], upper[_SCALE] = -opts.reference_scale_bound, opts.reference_scale_bound
    weights = np.ones(6 * len(motions))
    variance_factor = 1.0
    for _ in range(max(opts.irls_iterations, 1)):
        root_weights = np.sqrt(weights)

        def residuals(
            active: FloatArray,
            root_weights: FloatArray = root_weights,
            variance_factor: float = variance_factor,
        ) -> FloatArray:
            full = np.zeros(8)
            full[mask] = active
            data = root_weights * _normalized_residuals(reference, motions, base, full, opts)
            data /= math.sqrt(variance_factor)
            return np.concatenate([data, _prior_rows(base, full, opts), active / opts.anchor_sigma])

        solution = least_squares(
            residuals, params[mask], bounds=(lower[mask], upper[mask]), x_scale="jac"
        )
        params = np.zeros(8)
        params[mask] = solution.x
        normalized = _normalized_residuals(reference, motions, base, params, opts)
        spread = _robust_spread(normalized)
        variance_factor = spread**2
        scaled = np.abs(normalized) / spread
        weights = np.where(
            scaled <= opts.huber_threshold,
            1.0,
            opts.huber_threshold / np.maximum(scaled, 1e-12),
        )
    return params, weights, variance_factor


def _robust_spread(residuals: FloatArray) -> float:
    median_absolute = float(np.median(np.abs(residuals - np.median(residuals))))
    return max(1.4826 * median_absolute, 1.0e-6)


def _observability(
    reference: ReferenceTrajectory,
    motions: Sequence[SensorMotion],
    base: FloatArray,
    params: FloatArray,
    weights: FloatArray,
    variance_factor: float,
    opts: TrajectoryHandEyeOptions,
) -> tuple[tuple[DofEstimate, ...], tuple[float, ...]]:
    mask = opts.active_mask()
    active = np.flatnonzero(mask)
    steps = np.array([1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6])
    jacobian = np.empty((6 * len(motions), active.size))
    for column, index in enumerate(active):
        delta = np.zeros(8)
        delta[index] = steps[index]
        forward = _normalized_residuals(reference, motions, base, params + delta, opts)
        backward = _normalized_residuals(reference, motions, base, params - delta, opts)
        jacobian[:, column] = (forward - backward) / (2.0 * steps[index])
    jacobian *= np.sqrt(weights)[:, None] / math.sqrt(variance_factor)
    information = jacobian.T @ jacobian
    anchor = np.eye(active.size) / opts.anchor_sigma**2
    prior_information = np.zeros_like(information)
    for prior in opts.priors:
        index = DOF_NAMES.index(prior.dof)
        if mask[index]:
            position = int(np.flatnonzero(active == index)[0])
            prior_information[position, position] += 1.0 / prior.sigma**2
    std_data = np.sqrt(np.diag(np.linalg.inv(information + anchor)))
    std_total = np.sqrt(np.diag(np.linalg.inv(information + prior_information + anchor)))
    eigenvalues = tuple(float(value) for value in np.linalg.eigvalsh(information))
    transform = compose(base, params)
    rotation_threshold = math.radians(opts.observable_rotation_std_deg)
    thresholds = (
        *(rotation_threshold,) * 3,
        *(opts.observable_translation_std_m,) * 3,
        opts.observable_time_offset_std_s,
        opts.observable_reference_scale_std,
    )
    values = (
        *Rotation.from_matrix(transform[:3, :3]).as_euler("xyz"),
        *transform[:3, 3],
        float(params[_TIME]),
        1.0 + float(params[_SCALE]),
    )
    prior_dofs = {prior.dof for prior in opts.priors}
    dofs: list[DofEstimate] = []
    for position, index in enumerate(active):
        name = DOF_NAMES[index]
        status: DofStatus
        if std_data[position] <= thresholds[index]:
            status = "estimated"
        elif name in prior_dofs and std_total[position] <= thresholds[index]:
            status = "prior"
        else:
            status = "unobservable"
        dofs.append(
            DofEstimate(
                name=name,
                value=float(values[index]),
                std_data=float(std_data[position]),
                std_total=float(std_total[position]),
                status=status,
            )
        )
    return tuple(dofs), eigenvalues
