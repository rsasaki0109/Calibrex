"""GNSS antenna lever arm and clock offset from LiDAR odometry and RTK positions.

The antenna phase centre ``l`` is fixed in the LiDAR frame, so within one
window of LiDAR odometry poses ``T_odom_lidar(t) = (R(t), t(t))``::

    g(t_j + dt) - g(t_i + dt) = R_w (t_j - t_i + (R_j - R_i) l)

where ``g`` is the RTK antenna track in ENU and ``R_w`` aligns the window's
odometry frame with ENU.  LiDAR odometry drifts and does not know gravity, so
each window gets its own alignment.  For a trial ``(l, dt)`` every ``R_w`` has
a closed-form weighted Procrustes solution, so the outer robust least squares
only has four unknowns (variable projection).  Covariances come from the
profiled Jacobian, in which each ``R_w`` is re-solved, so they already
marginalize the alignments.

GNSS epochs are used as observed and the smooth, low-noise odometry is
interpolated to them.  Interpolating the noisy GNSS track instead biases the
clock offset: a linear blend of two noisy epochs has less variance than either
epoch, so the optimizer prefers offsets that land between epochs.

The lever arm is observable only through rotation: ``(R_j - R_i) l`` vanishes
for pure translation.  A component is ``estimated`` only when its data-only
standard deviation is below the policy threshold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

FloatArray: TypeAlias = NDArray[np.float64]
LeverArmDofName = Literal["x", "y", "z", "time_offset"]
LeverArmDofStatus = Literal["estimated", "unobservable"]
LEVER_ARM_DOFS: tuple[LeverArmDofName, ...] = ("x", "y", "z", "time_offset")


@dataclass(frozen=True)
class GnssTrackModel:
    """Antenna positions (ENU, metres) with per-epoch sigmas and gap-aware lookup."""

    times_s: FloatArray
    enu_m: FloatArray
    sigma_m: FloatArray
    max_gap_s: float = 0.35

    def __post_init__(self) -> None:
        if len(self.times_s) < 2 or np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("the GNSS track needs strictly increasing timestamps")

    def continuous(self, start_s: float, end_s: float) -> bool:
        """Return whether ``[start_s, end_s]`` is covered without a data gap."""

        low, high = min(start_s, end_s), max(start_s, end_s)
        if low < self.times_s[0] or high > self.times_s[-1]:
            return False
        first = max(int(np.searchsorted(self.times_s, low, side="right")) - 1, 0)
        last = int(np.searchsorted(self.times_s, high, side="left"))
        window = self.times_s[first : last + 1]
        return bool(window.size < 2 or np.max(np.diff(window)) <= self.max_gap_s)

    def positions(self, times_s: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Interpolate positions and sigmas at ``times_s``."""

        positions = np.column_stack(
            [np.interp(times_s, self.times_s, self.enu_m[:, axis]) for axis in range(3)]
        )
        return positions, np.interp(times_s, self.times_s, self.sigma_m)


@dataclass(frozen=True)
class OdometryWindow:
    """A contiguous run of reliable LiDAR odometry poses."""

    window_id: str
    block: int
    times_s: FloatArray
    poses: FloatArray  # (K, 4, 4) T_odom_lidar


@dataclass(frozen=True)
class LeverArmOptions:
    """Pairing, noise model, and observability policy."""

    pair_steps: tuple[int, ...] = (5, 10, 20)
    estimate_time_offset: bool = True
    time_offset_bound_s: float = 0.2
    odometry_sigma_per_m: float = 0.01
    odometry_sigma_floor_m: float = 0.005
    odometry_pose_sigma_m: float = 0.01
    huber_threshold: float = 1.5
    irls_iterations: int = 4
    observable_translation_std_m: float = 0.01
    observable_time_offset_std_s: float = 0.005
    anchor_sigma: float = 1.0e3


@dataclass(frozen=True)
class LeverArmDof:
    """One estimated quantity with its data-only standard deviation."""

    name: LeverArmDofName
    value: float
    std: float
    status: LeverArmDofStatus


@dataclass(frozen=True)
class LeverArmResult:
    """Lever arm (LiDAR frame), clock offset (``t_gnss = t_lidar + dt``), evidence."""

    status: Literal["converged", "insufficient_windows"]
    lever_arm_m: FloatArray | None
    time_offset_s: float
    dofs: tuple[LeverArmDof, ...]
    variance_factor: float | None
    pair_count: int
    window_ids: tuple[str, ...] = field(default_factory=tuple)

    def dof(self, name: LeverArmDofName) -> LeverArmDof:
        """Return one DoF."""

        return next(item for item in self.dofs if item.name == name)


class GnssLeverArmSolver:
    """Robust variable-projection solver for the GNSS lever arm and clock offset."""

    def solve(
        self,
        track: GnssTrackModel,
        windows: Sequence[OdometryWindow],
        options: LeverArmOptions | None = None,
    ) -> LeverArmResult:
        opts = options or LeverArmOptions()
        epoch = float(track.times_s[0])
        rebased = GnssTrackModel(track.times_s - epoch, track.enu_m, track.sigma_m, track.max_gap_s)
        pairs = build_pairs(rebased, windows, opts, epoch=epoch)
        if len(pairs) < 2 or sum(len(item.first) for item in pairs) < 12:
            return LeverArmResult("insufficient_windows", None, 0.0, (), None, 0)
        params, weights, variance = _irls(rebased, pairs, opts)
        dofs = _observability(rebased, pairs, params, weights, variance, opts)
        return LeverArmResult(
            status="converged",
            lever_arm_m=params[:3].copy(),
            time_offset_s=float(params[3]) if opts.estimate_time_offset else 0.0,
            dofs=dofs,
            variance_factor=variance,
            pair_count=sum(len(item.first) for item in pairs),
            window_ids=tuple(item.window_id for item in pairs),
        )


@dataclass(frozen=True)
class WindowPairs:
    """GNSS epoch pairs of one window and the window's odometry for interpolation."""

    window_id: str
    block: int
    epoch_times_s: FloatArray
    gnss_m: FloatArray
    gnss_sigma_m: FloatArray
    first: NDArray[np.int64]
    second: NDArray[np.int64]
    odometry_times_s: FloatArray
    odometry_translations: FloatArray
    odometry_slerp: Slerp

    def lidar_poses(self, times_s: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Interpolate odometry at ``times_s``.

        Also returns ``(1 - a)^2 + a^2`` for the blend fraction ``a`` of each
        query: the variance factor of a linear blend of two independent poses.
        """

        rotations = self.odometry_slerp(times_s).as_matrix()
        translations = np.column_stack(
            [
                np.interp(times_s, self.odometry_times_s, self.odometry_translations[:, axis])
                for axis in range(3)
            ]
        )
        upper = np.clip(
            np.searchsorted(self.odometry_times_s, times_s, side="right"),
            1,
            len(self.odometry_times_s) - 1,
        )
        lower_times = self.odometry_times_s[upper - 1]
        fraction = (times_s - lower_times) / (self.odometry_times_s[upper] - lower_times)
        return rotations, translations, (1.0 - fraction) ** 2 + fraction**2


def build_pairs(
    track: GnssTrackModel,
    windows: Sequence[OdometryWindow],
    opts: LeverArmOptions,
    *,
    epoch: float = 0.0,
) -> list[WindowPairs]:
    """Pair GNSS epochs whose odometry lookups stay inside a window for every offset.

    Epochs are selected once, independent of the trial clock offset, so the
    residual set never changes during the optimization.
    """

    margin = opts.time_offset_bound_s if opts.estimate_time_offset else 0.0
    result: list[WindowPairs] = []
    for window in windows:
        times = window.times_s - epoch
        if len(times) < 2:
            continue
        inside = np.flatnonzero(
            (track.times_s - margin >= times[0]) & (track.times_s + margin <= times[-1])
        )
        if inside.size < 2:
            continue
        epoch_times = track.times_s[inside]
        first: list[int] = []
        second: list[int] = []
        for step in opts.pair_steps:
            for index in range(0, inside.size - step, step):
                end = index + step
                if inside[end] - inside[index] == step and track.continuous(
                    epoch_times[index], epoch_times[end]
                ):
                    first.append(index)
                    second.append(end)
        if len(first) < 3:
            continue
        result.append(
            WindowPairs(
                window_id=window.window_id,
                block=window.block,
                epoch_times_s=epoch_times,
                gnss_m=track.enu_m[inside],
                gnss_sigma_m=track.sigma_m[inside],
                first=np.array(first, dtype=np.int64),
                second=np.array(second, dtype=np.int64),
                odometry_times_s=times,
                odometry_translations=window.poses[:, :3, 3],
                odometry_slerp=Slerp(times, Rotation.from_matrix(window.poses[:, :3, :3])),
            )
        )
    return result


def window_residuals(
    track: GnssTrackModel,
    pairs: Sequence[WindowPairs],
    lever_arm: FloatArray,
    time_offset_s: float,
    opts: LeverArmOptions,
    *,
    normalize: bool = True,
) -> FloatArray:
    """Return residuals with every window alignment profiled out.

    Residuals are divided by their noise sigma unless ``normalize`` is false,
    in which case they are displacement errors in metres.
    """

    del track  # epochs and sigmas are frozen in ``pairs``
    blocks = []
    for window in pairs:
        rotations, translations, blend = window.lidar_poses(window.epoch_times_s - time_offset_s)
        antenna = translations + np.einsum("kij,j->ki", rotations, lever_arm)
        odometry = antenna[window.second] - antenna[window.first]
        gnss = window.gnss_m[window.second] - window.gnss_m[window.first]
        # The pose-noise term depends on the trial offset through the blend
        # fractions; without it the fit is biased toward offsets whose
        # interpolated poses average two noisy poses.
        sigma = np.sqrt(
            window.gnss_sigma_m[window.first] ** 2
            + window.gnss_sigma_m[window.second] ** 2
            + opts.odometry_pose_sigma_m**2 * (blend[window.first] + blend[window.second])
            + (
                opts.odometry_sigma_floor_m
                + opts.odometry_sigma_per_m * np.linalg.norm(odometry, axis=1)
            )
            ** 2
        )
        rotation = weighted_procrustes(odometry, gnss, 1.0 / sigma**2)
        error = gnss - odometry @ rotation.T
        blocks.append((error / sigma[:, None] if normalize else error).reshape(-1))
    return np.concatenate(blocks) if blocks else np.empty(0)


def weighted_procrustes(source: FloatArray, target: FloatArray, weights: FloatArray) -> FloatArray:
    """Rotation ``R`` minimizing ``sum w |target - R source|^2`` (no translation)."""

    covariance = (target * weights[:, None]).T @ source
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(left @ right_t)) or 1.0)])
    return np.asarray(left @ correction @ right_t, dtype=np.float64)


def _unpack(params: FloatArray, opts: LeverArmOptions) -> tuple[FloatArray, float]:
    return params[:3], float(params[3]) if opts.estimate_time_offset else 0.0


def _irls(
    track: GnssTrackModel, pairs: Sequence[WindowPairs], opts: LeverArmOptions
) -> tuple[FloatArray, FloatArray, float]:
    params = np.zeros(4)
    bound = opts.time_offset_bound_s if opts.estimate_time_offset else 1e-12
    lower = np.array([-np.inf, -np.inf, -np.inf, -bound])
    upper = np.array([np.inf, np.inf, np.inf, bound])
    size = int(sum(3 * len(item.first) for item in pairs))
    weights = np.ones(size)
    variance = 1.0
    for _ in range(max(opts.irls_iterations, 1)):
        root = np.sqrt(weights)

        def residuals(
            values: FloatArray, root: FloatArray = root, variance: float = variance
        ) -> FloatArray:
            lever_arm, offset = _unpack(values, opts)
            data = root * window_residuals(track, pairs, lever_arm, offset, opts)
            return np.concatenate([data / math.sqrt(variance), values / opts.anchor_sigma])

        params = least_squares(residuals, params, bounds=(lower, upper), x_scale="jac").x
        raw = window_residuals(track, pairs, *_unpack(params, opts), opts)
        spread = max(1.4826 * float(np.median(np.abs(raw - np.median(raw)))), 1e-6)
        variance = spread**2
        scaled = np.abs(raw) / spread
        weights = np.where(
            scaled <= opts.huber_threshold,
            1.0,
            opts.huber_threshold / np.maximum(scaled, 1e-12),
        )
    return params, weights, variance


def _observability(
    track: GnssTrackModel,
    pairs: Sequence[WindowPairs],
    params: FloatArray,
    weights: FloatArray,
    variance: float,
    opts: LeverArmOptions,
) -> tuple[LeverArmDof, ...]:
    active = 4 if opts.estimate_time_offset else 3
    steps = np.array([1e-4, 1e-4, 1e-4, 1e-4])
    columns = []
    for index in range(active):
        delta = np.zeros(4)
        delta[index] = steps[index]
        forward = window_residuals(track, pairs, *_unpack(params + delta, opts), opts)
        backward = window_residuals(track, pairs, *_unpack(params - delta, opts), opts)
        columns.append((forward - backward) / (2.0 * steps[index]))
    jacobian = np.column_stack(columns) * (np.sqrt(weights)[:, None] / math.sqrt(variance))
    information = jacobian.T @ jacobian + np.eye(active) / opts.anchor_sigma**2
    std = np.sqrt(np.diag(np.linalg.inv(information)))
    thresholds = (
        *(opts.observable_translation_std_m,) * 3,
        opts.observable_time_offset_std_s,
    )
    return tuple(
        LeverArmDof(
            name=LEVER_ARM_DOFS[index],
            value=float(params[index]),
            std=float(std[index]),
            status="estimated" if std[index] <= thresholds[index] else "unobservable",
        )
        for index in range(active)
    )
