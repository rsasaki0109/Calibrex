"""Yaw-only Radar-to-trajectory calibration from paired velocities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from calibrex.core.geometry import Vector3
from calibrex.evaluation.holdout import split_indices

RadarTrajectoryYawStatus = Literal[
    "converged", "max_iterations", "insufficient_pairs", "degenerate_motion"
]


@dataclass(frozen=True)
class RadarTrajectoryVelocityPair:
    """Velocity correspondence at the Radar origin.

    ``trajectory_velocity_at_radar_origin_ego_mps`` must already include any
    lever-arm term ``omega cross translation``. Translation is not estimated here.
    """

    frame_id: str
    velocity_radar_mps: Vector3
    trajectory_velocity_at_radar_origin_ego_mps: Vector3
    weight: float = 1.0


@dataclass(frozen=True)
class RadarTrajectoryYawSolverOptions:
    max_iterations: int = 20
    convergence_tolerance_rad: float = 1.0e-10
    huber_delta_mps: float = 0.5
    min_pair_count: int = 6
    min_planar_speed_mps: float = 0.5
    min_direction_diversity: float = 0.02
    holdout_ratio: float = 0.2
    split_seed: int = 0
    known_bad_margin_mps: float = 0.05


@dataclass(frozen=True)
class RadarYawProbeResult:
    amount_deg: float
    holdout_rmse_mps: float | None
    delta_mps: float | None
    detectable: bool | None


@dataclass(frozen=True)
class RadarTrajectoryYawResult:
    status: RadarTrajectoryYawStatus
    reason: str
    yaw_rad: float | None
    rotation_quat_xyzw: tuple[float, float, float, float] | None
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    train_rmse_mps: float | None
    holdout_rmse_mps: float | None
    direction_diversity: float | None
    inlier_count: int
    iterations: int
    probes: tuple[RadarYawProbeResult, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "yaw_rad": self.yaw_rad,
            "rotation_quat_xyzw": list(self.rotation_quat_xyzw)
            if self.rotation_quat_xyzw is not None
            else None,
            "train_frame_ids": list(self.train_frame_ids),
            "holdout_frame_ids": list(self.holdout_frame_ids),
            "train_rmse_mps": self.train_rmse_mps,
            "holdout_rmse_mps": self.holdout_rmse_mps,
            "direction_diversity": self.direction_diversity,
            "inlier_count": self.inlier_count,
            "iterations": self.iterations,
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "weighted_planar_velocity_alignment_huber/v0.1",
            "estimated_dofs": ["yaw"],
            "unobservable_dofs": ["x", "y", "z", "roll", "pitch"],
            "translation_assumption": "trajectory velocity is lever-arm compensated",
        }


class RadarTrajectoryYawSolver:
    """Estimate ``R_ego_radar`` yaw using robust planar Procrustes alignment."""

    def solve(
        self,
        pairs: Sequence[RadarTrajectoryVelocityPair],
        options: RadarTrajectoryYawSolverOptions | None = None,
    ) -> RadarTrajectoryYawResult:
        solver_options = options or RadarTrajectoryYawSolverOptions()
        eligible = sorted(
            (
                pair
                for pair in pairs
                if pair.weight > 0.0
                and _planar_speed(pair.velocity_radar_mps)
                >= solver_options.min_planar_speed_mps
            ),
            key=lambda pair: pair.frame_id,
        )
        frame_ids = tuple(pair.frame_id for pair in eligible)
        train_indices, holdout_indices = split_indices(
            len(eligible), solver_options.holdout_ratio, seed=solver_options.split_seed
        )
        train = [eligible[index] for index in train_indices]
        holdout = [eligible[index] for index in holdout_indices]
        train_ids = tuple(frame_ids[index] for index in train_indices)
        holdout_ids = tuple(frame_ids[index] for index in holdout_indices)
        if len(train) < solver_options.min_pair_count:
            return _empty_yaw_result("insufficient_pairs", train_ids, holdout_ids)
        diversity = _direction_diversity(train)
        if diversity < solver_options.min_direction_diversity:
            return _empty_yaw_result(
                "degenerate_motion", train_ids, holdout_ids, diversity=diversity
            )

        yaw = 0.0
        status: RadarTrajectoryYawStatus = "max_iterations"
        iterations = 0
        robust_weights = [1.0] * len(train)
        for iteration in range(solver_options.max_iterations):
            estimated = _weighted_yaw(train, robust_weights)
            if estimated is None:
                return _empty_yaw_result(
                    "degenerate_motion", train_ids, holdout_ids, diversity=diversity
                )
            delta = abs(_wrap_angle(estimated - yaw))
            yaw = estimated
            residuals = [_pair_error(pair, yaw) for pair in train]
            robust_weights = [
                _huber_weight(value, solver_options.huber_delta_mps) for value in residuals
            ]
            iterations = iteration + 1
            if delta <= solver_options.convergence_tolerance_rad:
                status = "converged"
                break
        train_rmse = radar_trajectory_yaw_rmse(train, yaw)
        holdout_rmse = radar_trajectory_yaw_rmse(holdout, yaw)
        probes = tuple(
            _probe(holdout, yaw, amount, holdout_rmse, solver_options.known_bad_margin_mps)
            for amount in (-10.0, -5.0, 5.0, 10.0)
        )
        inlier_count = sum(
            _pair_error(pair, yaw) <= solver_options.huber_delta_mps for pair in train
        )
        half = yaw / 2.0
        return RadarTrajectoryYawResult(
            status=status,
            reason=(
                "yaw alignment converged"
                if status == "converged"
                else "maximum IRLS iterations reached"
            ),
            yaw_rad=yaw,
            rotation_quat_xyzw=(0.0, 0.0, math.sin(half), math.cos(half)),
            train_frame_ids=train_ids,
            holdout_frame_ids=holdout_ids,
            train_rmse_mps=train_rmse,
            holdout_rmse_mps=holdout_rmse,
            direction_diversity=diversity,
            inlier_count=inlier_count,
            iterations=iterations,
            probes=probes,
        )


def radar_trajectory_yaw_rmse(
    pairs: Sequence[RadarTrajectoryVelocityPair], yaw_rad: float
) -> float | None:
    if not pairs:
        return None
    residuals = [_pair_error(pair, yaw_rad) for pair in pairs]
    return math.sqrt(sum(value * value for value in residuals) / len(residuals))


def _weighted_yaw(
    pairs: Sequence[RadarTrajectoryVelocityPair], robust_weights: Sequence[float]
) -> float | None:
    cosine_sum = 0.0
    sine_sum = 0.0
    for pair, robust in zip(pairs, robust_weights, strict=True):
        rx, ry, _rz = pair.velocity_radar_mps
        ex, ey, _ez = pair.trajectory_velocity_at_radar_origin_ego_mps
        weight = pair.weight * robust
        cosine_sum += weight * (ex * rx + ey * ry)
        sine_sum += weight * (ey * rx - ex * ry)
    if math.hypot(cosine_sum, sine_sum) < 1.0e-12:
        return None
    return _wrap_angle(math.atan2(sine_sum, cosine_sum))


def _pair_error(pair: RadarTrajectoryVelocityPair, yaw_rad: float) -> float:
    rx, ry, _rz = pair.velocity_radar_mps
    ex, ey, _ez = pair.trajectory_velocity_at_radar_origin_ego_mps
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    dx = cosine * rx - sine * ry - ex
    dy = sine * rx + cosine * ry - ey
    return math.hypot(dx, dy)


def _direction_diversity(pairs: Sequence[RadarTrajectoryVelocityPair]) -> float:
    xx = xy = yy = 0.0
    for pair in pairs:
        x, y, _z = pair.velocity_radar_mps
        norm = math.hypot(x, y)
        ux, uy = x / norm, y / norm
        xx += ux * ux
        xy += ux * uy
        yy += uy * uy
    trace = xx + yy
    root = math.sqrt(max(0.0, (xx - yy) ** 2 + 4.0 * xy * xy))
    maximum = (trace + root) / 2.0
    minimum = (trace - root) / 2.0
    return minimum / maximum if maximum > 0.0 else 0.0


def _probe(
    holdout: Sequence[RadarTrajectoryVelocityPair],
    yaw: float,
    amount_deg: float,
    baseline_rmse: float | None,
    margin: float,
) -> RadarYawProbeResult:
    perturbed = radar_trajectory_yaw_rmse(holdout, yaw + math.radians(amount_deg))
    delta = (
        perturbed - baseline_rmse
        if perturbed is not None and baseline_rmse is not None
        else None
    )
    return RadarYawProbeResult(
        amount_deg=amount_deg,
        holdout_rmse_mps=perturbed,
        delta_mps=delta,
        detectable=delta > margin if delta is not None else None,
    )


def _planar_speed(vector: Vector3) -> float:
    return math.hypot(vector[0], vector[1])


def _huber_weight(residual: float, delta: float) -> float:
    return 1.0 if residual <= delta or residual == 0.0 else delta / residual


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _empty_yaw_result(
    status: Literal["insufficient_pairs", "degenerate_motion"],
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    *,
    diversity: float | None = None,
) -> RadarTrajectoryYawResult:
    return RadarTrajectoryYawResult(
        status=status,
        reason=(
            "not enough train velocity pairs"
            if status == "insufficient_pairs"
            else "train velocity directions lack generalization coverage"
        ),
        yaw_rad=None,
        rotation_quat_xyzw=None,
        train_frame_ids=train_ids,
        holdout_frame_ids=holdout_ids,
        train_rmse_mps=None,
        holdout_rmse_mps=None,
        direction_diversity=diversity,
        inlier_count=0,
        iterations=0,
        probes=(),
    )
