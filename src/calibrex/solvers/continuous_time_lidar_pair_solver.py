"""Continuous-time LiDAR-pair extrinsic and clock-offset refinement.

This module deliberately accepts a supplied odometry trajectory. It profiles a
scalar clock offset while re-solving the six-DoF LiDAR extrinsic at every
candidate, so the temporal estimate cannot be credited with a residual change
that is only caused by holding the extrinsic fixed. A future spline/trajectory
solver can replace :class:`OdometryTrack` without changing the profile/holdout
contract.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from calibrex.core.geometry import SE3, Vector3
from calibrex.core.time import apply_time_offset_ns
from calibrex.data.adaptive_voxel import (
    AdaptiveVoxelPlaneMap,
    AdaptiveVoxelPolicy,
    build_adaptive_voxel_plane_map,
)
from calibrex.data.livox import LivoxPointRecord, VoxelPlaneMap, build_voxel_plane_map
from calibrex.data.odometry_track import OdometryTrack
from calibrex.evaluation.lidar import build_rig_point_to_plane_observations
from calibrex.graph.lidar_point_to_plane import (
    LidarRigPointToPlaneEvaluation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
    FixedTrajectorySe3SolverResult,
)
from calibrex.solvers.robust_lidar_observations import (
    RobustObservationFilterResult,
    filter_lidar_observations_mad,
)

ContinuousTimeLidarPairStatus = Literal[
    "converged",
    "max_iterations",
    "insufficient_constraints",
    "rejected",
]


@dataclass(frozen=True)
class ContinuousTimeLidarPairProblem:
    """Inputs for one continuous-time LiDAR-pair refinement."""

    source_records: Sequence[LivoxPointRecord]
    target_points: Sequence[Vector3]
    target_capture_timestamps_ns: Sequence[int]
    odometry_track: OdometryTrack
    t_base_source: SE3
    initial_t_source_target: SE3
    variable: str
    sensor: str
    voxel_size_m: float
    correspondence_gate_m: float
    train_indices: Sequence[int] | None = None
    holdout_indices: Sequence[int] | None = None


@dataclass(frozen=True)
class ContinuousTimeLidarPairOptions:
    """Controls for deterministic profile optimization over clock offset."""

    initial_time_offset_sec: float = 0.0
    max_abs_time_offset_sec: float = 0.20
    initial_time_step_sec: float = 0.02
    minimum_time_step_sec: float = 0.001
    max_iterations: int = 8
    min_correspondences: int = 6
    max_odometry_extrapolation_s: float = 0.25
    max_source_records: int | None = 6000
    max_target_points_per_split: int | None = 6000
    sampling_seed: int = 0
    holdout_start_fraction: float = 0.8
    outlier_policy: Literal["none", "mad"] = "mad"
    outlier_mad_scale: float = 3.5
    outlier_min_threshold_m: float = 0.02
    outlier_min_inlier_fraction: float = 0.25
    voxel_strategy: Literal["uniform", "adaptive"] = "uniform"
    adaptive_use_uniform_fallback: bool = True
    adaptive_range_reference_m: float = 10.0
    adaptive_range_exponent: float = 0.5
    adaptive_min_voxel_size_m: float = 0.25
    adaptive_max_voxel_size_m: float = 1.5
    adaptive_min_points_per_voxel: int = 3
    adaptive_range_origin_m: Vector3 = (0.0, 0.0, 0.0)
    correspondence_refinement_iterations: int = 1
    solver: FixedTrajectorySe3SolverOptions = field(
        default_factory=FixedTrajectorySe3SolverOptions
    )


@dataclass(frozen=True)
class ContinuousTimeLidarPairIteration:
    """One accepted/rejected time-profile iteration."""

    iteration: int
    candidate_offsets_sec: tuple[float, ...]
    selected_offset_sec: float
    train_rmse_m: float | None
    holdout_rmse_m: float | None
    accepted: bool
    candidate_train_rmse_m: tuple[float | None, ...] = ()
    candidate_holdout_rmse_m: tuple[float | None, ...] = ()


@dataclass(frozen=True)
class ContinuousTimeLidarPairResult:
    """Joint profile result with train/holdout and observability evidence."""

    status: ContinuousTimeLidarPairStatus
    reason: str
    initial_time_offset_sec: float
    estimated_time_offset_sec: float
    initial_t_source_target: SE3
    refined_t_source_target: SE3
    initial_train_rmse_m: float | None
    final_train_rmse_m: float | None
    final_holdout_rmse_m: float | None
    train_correspondence_count: int
    holdout_correspondence_count: int
    outlier_rejected_count: int
    observability: LidarRigPointToPlaneEvaluation | None
    iterations: tuple[ContinuousTimeLidarPairIteration, ...]
    trajectory_model: Literal["piecewise_se3_fixed_odometry"] = (
        "piecewise_se3_fixed_odometry"
    )
    time_offset_sign_convention: str = (
        "positive offset evaluates target capture at odometry time "
        "t_capture + offset"
    )


@dataclass(frozen=True)
class _OffsetEvaluation:
    offset_sec: float
    solver_result: FixedTrajectorySe3SolverResult
    train_factor: LidarRigPointToPlaneFactor
    holdout_factor: LidarRigPointToPlaneFactor | None
    train_correspondence_count: int
    holdout_correspondence_count: int
    holdout_rmse_m: float | None
    outlier_rejected_count: int


@dataclass(frozen=True)
class _FactorBuild:
    factor: LidarRigPointToPlaneFactor
    outlier_rejected_count: int


class ContinuousTimeLidarPairSolver:
    """Profile a scalar clock offset while re-solving a LiDAR extrinsic."""

    def solve(
        self,
        problem: ContinuousTimeLidarPairProblem,
        options: ContinuousTimeLidarPairOptions | None = None,
    ) -> ContinuousTimeLidarPairResult:
        """Return a deterministic continuous-time profile/holdout result."""

        settings = options or ContinuousTimeLidarPairOptions()
        self._validate(problem, settings)
        train_indices, holdout_indices = _resolve_indices(problem)
        initial_offset = _clip(
            settings.initial_time_offset_sec,
            -settings.max_abs_time_offset_sec,
            settings.max_abs_time_offset_sec,
        )
        source_records = _sample_source_records(
            problem.source_records,
            settings.max_source_records,
            seed=settings.sampling_seed,
        )
        source_plane_map: VoxelPlaneMap | None = (
            build_voxel_plane_map(source_records, problem.voxel_size_m)
            if settings.voxel_strategy == "uniform" or settings.adaptive_use_uniform_fallback
            else None
        )
        adaptive_plane_map = (
            build_adaptive_voxel_plane_map(
                source_records,
                AdaptiveVoxelPolicy(
                    base_voxel_size_m=problem.voxel_size_m,
                    range_reference_m=settings.adaptive_range_reference_m,
                    range_exponent=settings.adaptive_range_exponent,
                    min_voxel_size_m=settings.adaptive_min_voxel_size_m,
                    max_voxel_size_m=settings.adaptive_max_voxel_size_m,
                    min_points_per_voxel=settings.adaptive_min_points_per_voxel,
                    range_origin_m=settings.adaptive_range_origin_m,
                ),
            )
            if settings.voxel_strategy == "adaptive"
            else None
        )
        initial = self._evaluate_offset(
            problem,
            settings,
            source_records,
            source_plane_map,
            adaptive_plane_map,
            initial_offset,
            train_indices,
            holdout_indices,
            problem.initial_t_source_target,
        )
        if initial is None:
            return ContinuousTimeLidarPairResult(
                status="insufficient_constraints",
                reason="initial time offset produced too few point-to-plane correspondences",
                initial_time_offset_sec=initial_offset,
                estimated_time_offset_sec=initial_offset,
                initial_t_source_target=problem.initial_t_source_target,
                refined_t_source_target=problem.initial_t_source_target,
                initial_train_rmse_m=None,
                final_train_rmse_m=None,
                final_holdout_rmse_m=None,
                train_correspondence_count=0,
                holdout_correspondence_count=0,
                outlier_rejected_count=0,
                observability=None,
                iterations=(),
            )

        current = initial
        initial_train_rmse = initial.solver_result.final_rmse_m
        step = max(settings.minimum_time_step_sec, settings.initial_time_step_sec)
        trace: list[ContinuousTimeLidarPairIteration] = []
        status: ContinuousTimeLidarPairStatus = "max_iterations"
        reason = "maximum time-profile iterations reached"
        for iteration in range(1, settings.max_iterations + 1):
            candidates = _candidate_offsets(
                current.offset_sec,
                step,
                settings.max_abs_time_offset_sec,
            )
            evaluated: list[_OffsetEvaluation] = []
            for offset in candidates:
                candidate = self._evaluate_offset(
                    problem,
                    settings,
                    source_records,
                    source_plane_map,
                    adaptive_plane_map,
                    offset,
                    train_indices,
                    holdout_indices,
                    current.solver_result.refined_transform,
                )
                if candidate is not None:
                    evaluated.append(candidate)
            if not evaluated:
                step *= 0.5
                if step < settings.minimum_time_step_sec:
                    status = "rejected"
                    reason = "time-profile candidates became insufficiently constrained"
                    break
                continue
            selected = min(
                evaluated,
                key=lambda item: (
                    float("inf")
                    if item.solver_result.final_rmse_m is None
                    else item.solver_result.final_rmse_m,
                    abs(item.offset_sec),
                    item.offset_sec,
                ),
            )
            current_rmse = current.solver_result.final_rmse_m
            selected_rmse = selected.solver_result.final_rmse_m
            accepted = (
                selected_rmse is not None
                and (current_rmse is None or selected_rmse < current_rmse - 1.0e-10)
            )
            trace.append(
                ContinuousTimeLidarPairIteration(
                    iteration=iteration,
                    candidate_offsets_sec=tuple(candidates),
                    candidate_train_rmse_m=tuple(
                        next(
                            (
                                candidate.solver_result.final_rmse_m
                                for candidate in evaluated
                                if candidate.offset_sec == offset
                            ),
                            None,
                        )
                        for offset in candidates
                    ),
                    candidate_holdout_rmse_m=tuple(
                        next(
                            (
                                candidate.holdout_rmse_m
                                for candidate in evaluated
                                if candidate.offset_sec == offset
                            ),
                            None,
                        )
                        for offset in candidates
                    ),
                    selected_offset_sec=selected.offset_sec,
                    train_rmse_m=selected_rmse,
                    holdout_rmse_m=selected.holdout_rmse_m,
                    accepted=accepted,
                )
            )
            if accepted:
                current = selected
            else:
                step *= 0.5
            if step < settings.minimum_time_step_sec:
                status = "converged"
                reason = "time-profile step reached the configured minimum"
                break
        else:
            if step < settings.minimum_time_step_sec:
                status = "converged"
                reason = "time-profile step reached the configured minimum"

        for _ in range(settings.correspondence_refinement_iterations):
            refined = self._evaluate_offset(
                problem,
                settings,
                source_records,
                source_plane_map,
                adaptive_plane_map,
                current.offset_sec,
                train_indices,
                holdout_indices,
                current.solver_result.refined_transform,
            )
            if refined is None:
                break
            current_rmse = current.solver_result.final_rmse_m
            refined_rmse = refined.solver_result.final_rmse_m
            if (
                refined_rmse is None
                or (current_rmse is not None and refined_rmse > current_rmse + 1.0e-10)
            ):
                break
            current = refined

        evaluation = current.train_factor.evaluate(current.solver_result.correction)
        return ContinuousTimeLidarPairResult(
            status=status,
            reason=reason,
            initial_time_offset_sec=initial_offset,
            estimated_time_offset_sec=current.offset_sec,
            initial_t_source_target=problem.initial_t_source_target,
            refined_t_source_target=current.solver_result.refined_transform,
            initial_train_rmse_m=initial_train_rmse,
            final_train_rmse_m=current.solver_result.final_rmse_m,
            final_holdout_rmse_m=current.holdout_rmse_m,
            train_correspondence_count=current.train_correspondence_count,
            holdout_correspondence_count=current.holdout_correspondence_count,
            outlier_rejected_count=current.outlier_rejected_count,
            observability=evaluation,
            iterations=tuple(trace),
        )

    def _evaluate_offset(
        self,
        problem: ContinuousTimeLidarPairProblem,
        settings: ContinuousTimeLidarPairOptions,
        source_records: Sequence[LivoxPointRecord],
        source_plane_map: VoxelPlaneMap | None,
        adaptive_plane_map: AdaptiveVoxelPlaneMap | None,
        offset_sec: float,
        train_indices: Sequence[int],
        holdout_indices: Sequence[int],
        initial_transform: SE3,
    ) -> _OffsetEvaluation | None:
        train_build = self._factor_for_indices(
            problem,
            settings,
            source_records,
            source_plane_map,
            adaptive_plane_map,
            offset_sec,
            train_indices,
            initial_transform,
        )
        if train_build is None:
            return None
        train_factor = train_build.factor
        solver_result = FixedTrajectorySe3ExtrinsicSolver().solve(
            train_factor,
            settings.solver,
        )
        holdout_factor = None
        holdout_rmse = None
        outlier_rejected_count = train_build.outlier_rejected_count
        if holdout_indices:
            holdout_build = self._factor_for_indices(
                problem,
                settings,
                source_records,
                source_plane_map,
                adaptive_plane_map,
                offset_sec,
                holdout_indices,
                initial_transform,
            )
            if holdout_build is not None:
                holdout_factor = holdout_build.factor
                outlier_rejected_count += holdout_build.outlier_rejected_count
                holdout_rmse = _rmse(holdout_factor.residuals(solver_result.correction))
        return _OffsetEvaluation(
            offset_sec=offset_sec,
            solver_result=solver_result,
            train_factor=train_factor,
            holdout_factor=holdout_factor,
            train_correspondence_count=len(train_factor.observations),
            holdout_correspondence_count=(
                len(holdout_factor.observations) if holdout_factor is not None else 0
            ),
            holdout_rmse_m=holdout_rmse,
            outlier_rejected_count=outlier_rejected_count,
        )

    def _factor_for_indices(
        self,
        problem: ContinuousTimeLidarPairProblem,
        settings: ContinuousTimeLidarPairOptions,
        source_records: Sequence[LivoxPointRecord],
        source_plane_map: VoxelPlaneMap | None,
        adaptive_plane_map: AdaptiveVoxelPlaneMap | None,
        offset_sec: float,
        indices: Sequence[int],
        initial_transform: SE3,
    ) -> _FactorBuild | None:
        selected_indices = _select_evenly_spaced_indices(
            indices,
            settings.max_target_points_per_split,
            seed=settings.sampling_seed,
        )
        target_points = [problem.target_points[index] for index in selected_indices]
        target_poses: list[SE3] = []
        for index in selected_indices:
            capture_ns = apply_time_offset_ns(
                problem.target_capture_timestamps_ns[index], offset_sec
            )
            pose, _clamped, extrapolation_s = problem.odometry_track.interpolate(capture_ns)
            if extrapolation_s > settings.max_odometry_extrapolation_s:
                return None
            target_poses.append(pose.compose(problem.t_base_source))
        observations = build_rig_point_to_plane_observations(
            source_records=list(source_records),
            target_points=target_points,
            initial_t_source_target=initial_transform,
            target_t_world_source=target_poses,
            voxel_size_m=problem.voxel_size_m,
            correspondence_gate_m=problem.correspondence_gate_m,
            source_plane_map=source_plane_map,
            adaptive_plane_map=adaptive_plane_map,
        )
        if len(observations) < settings.min_correspondences:
            return None
        outlier_rejected_count = 0
        if settings.outlier_policy == "mad":
            filtered: RobustObservationFilterResult = filter_lidar_observations_mad(
                observations,
                variable=problem.variable,
                t_ego_lidar=initial_transform,
                sensor=problem.sensor,
                mad_scale=settings.outlier_mad_scale,
                minimum_threshold_m=settings.outlier_min_threshold_m,
                minimum_inlier_fraction=settings.outlier_min_inlier_fraction,
                minimum_inliers=settings.min_correspondences,
            )
            observations = list(filtered.observations)
            outlier_rejected_count = filtered.rejected_count
        if len(observations) < settings.min_correspondences:
            return None
        return _FactorBuild(
            factor=LidarRigPointToPlaneFactor(
                variable=problem.variable,
                t_ego_lidar=initial_transform,
                observations=observations,
                sensor=problem.sensor,
            ),
            outlier_rejected_count=outlier_rejected_count,
        )

    @staticmethod
    def _validate(
        problem: ContinuousTimeLidarPairProblem,
        settings: ContinuousTimeLidarPairOptions,
    ) -> None:
        if len(problem.target_points) != len(problem.target_capture_timestamps_ns):
            raise ValueError("target points and capture timestamps must have equal length")
        if not problem.target_points:
            raise ValueError("continuous-time LiDAR pair requires target points")
        if problem.voxel_size_m <= 0.0 or problem.correspondence_gate_m <= 0.0:
            raise ValueError("voxel size and correspondence gate must be positive")
        if settings.max_abs_time_offset_sec < 0.0:
            raise ValueError("max_abs_time_offset_sec must be non-negative")
        if settings.initial_time_step_sec <= 0.0 or settings.minimum_time_step_sec <= 0.0:
            raise ValueError("time-profile steps must be positive")
        if settings.minimum_time_step_sec > settings.initial_time_step_sec:
            raise ValueError("minimum_time_step_sec cannot exceed initial_time_step_sec")
        if not 0.0 < settings.holdout_start_fraction < 1.0:
            raise ValueError("holdout_start_fraction must be in (0, 1)")
        if settings.max_iterations < 1 or settings.min_correspondences < 1:
            raise ValueError("max_iterations and min_correspondences must be positive")
        if settings.outlier_policy not in ("none", "mad"):
            raise ValueError("outlier_policy must be 'none' or 'mad'")
        if settings.voxel_strategy not in ("uniform", "adaptive"):
            raise ValueError("voxel_strategy must be 'uniform' or 'adaptive'")
        if settings.outlier_mad_scale < 0.0:
            raise ValueError("outlier_mad_scale must be non-negative")
        if settings.outlier_min_threshold_m < 0.0:
            raise ValueError("outlier_min_threshold_m must be non-negative")
        if not 0.0 < settings.outlier_min_inlier_fraction <= 1.0:
            raise ValueError("outlier_min_inlier_fraction must be in (0, 1]")
        if settings.adaptive_range_reference_m <= 0.0:
            raise ValueError("adaptive_range_reference_m must be positive")
        if settings.adaptive_range_exponent < 0.0:
            raise ValueError("adaptive_range_exponent must be non-negative")
        if settings.adaptive_min_voxel_size_m <= 0.0:
            raise ValueError("adaptive_min_voxel_size_m must be positive")
        if settings.adaptive_max_voxel_size_m < settings.adaptive_min_voxel_size_m:
            raise ValueError("adaptive voxel size bounds are invalid")
        if settings.adaptive_min_points_per_voxel < 1:
            raise ValueError("adaptive_min_points_per_voxel must be positive")
        if settings.correspondence_refinement_iterations < 0:
            raise ValueError("correspondence_refinement_iterations cannot be negative")
        if (
            settings.max_source_records is not None
            and settings.max_source_records < 1
        ):
            raise ValueError("max_source_records must be positive when configured")
        if (
            settings.max_target_points_per_split is not None
            and settings.max_target_points_per_split < 1
        ):
            raise ValueError(
                "max_target_points_per_split must be positive when configured"
            )


def _sample_source_records(
    records: Sequence[LivoxPointRecord],
    limit: int | None,
    *,
    seed: int = 0,
) -> list[LivoxPointRecord]:
    """Select a stable evenly spaced source subset for repeated profiling."""

    indices = _select_evenly_spaced_indices(range(len(records)), limit, seed=seed)
    return [records[index] for index in indices]


def _select_evenly_spaced_indices(
    indices: Sequence[int],
    limit: int | None,
    *,
    seed: int = 0,
) -> tuple[int, ...]:
    """Return deterministic boundary-preserving indices at most ``limit`` long."""

    if limit is None or len(indices) <= limit:
        return tuple(indices)
    if limit < 1:
        raise ValueError("sampling limit must be positive")
    if limit == 1:
        return (indices[0],)
    if seed != 0:
        selected = {indices[0], indices[-1]}
        interior = list(indices[1:-1])
        remaining = max(0, limit - len(selected))
        if remaining:
            selected.update(random.Random(seed).sample(interior, min(remaining, len(interior))))
        return tuple(sorted(selected))
    last_position = len(indices) - 1
    return tuple(
        indices[(position * last_position) // (limit - 1)]
        for position in range(limit)
    )


def _resolve_indices(
    problem: ContinuousTimeLidarPairProblem,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = len(problem.target_points)
    train = tuple(problem.train_indices or range(count))
    holdout = tuple(problem.holdout_indices or ())
    all_indices = set(train) | set(holdout)
    if not all(0 <= index < count for index in all_indices):
        raise ValueError("train/holdout indices are outside target point range")
    if set(train) & set(holdout):
        raise ValueError("train and holdout indices must be disjoint")
    if not train:
        raise ValueError("continuous-time LiDAR pair requires train indices")
    return train, holdout


def _candidate_offsets(current: float, step: float, bound: float) -> list[float]:
    candidates = [
        _clip(current - step, -bound, bound),
        _clip(current, -bound, bound),
        _clip(current + step, -bound, bound),
    ]
    return list(dict.fromkeys(round(value, 12) for value in candidates))


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


def _rmse(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))
