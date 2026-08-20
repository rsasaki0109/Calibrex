"""Sliding-window fitting with Schur marginalization on knot trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibrex.core.continuous_time_marginalization import (
    KnotMarginalizationPrior,
    marginalize_knot_prefix,
    split_retained_information,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    ContinuousTimeTrajectoryFitResult,
    FitStatus,
    TrajectoryPointToPlaneMeasurement,
    TrajectoryPoseMeasurement,
    _interval_index,
    build_dense_knot_normal_equations,
    fit_continuous_trajectory,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp, se3_log


@dataclass(frozen=True)
class SlidingWindowTrajectoryFitResult:
    """Batch reference and sliding-window knot estimates."""

    batch_result: ContinuousTimeTrajectoryFitResult
    sliding_result: ContinuousTimeTrajectoryFitResult
    window_count: int
    overlap_knot_count: int
    max_overlap_translation_error_m: float
    max_overlap_rotation_error_rad: float


def fit_sliding_window_trajectory(
    problem: ContinuousTimeTrajectoryFitProblem,
    *,
    window_knot_counts: tuple[int, ...],
    overlap_knot_count: int,
    options: ContinuousTimeTrajectoryFitOptions | None = None,
    apply_marginalization: bool = True,
    corrupt_marginalization_translation_m: tuple[float, float, float] | None = None,
    bad_overlap_translation_m: tuple[float, float, float] | None = None,
) -> SlidingWindowTrajectoryFitResult:
    """Fit with overlapping windows and optional Schur-complement knot priors."""

    if len(window_knot_counts) < 2:
        raise ValueError("sliding-window fit requires at least two windows")
    if overlap_knot_count < 1:
        raise ValueError("overlap knot count must be at least one")
    knot_count = len(problem.knot_timestamps)
    expected = sum(window_knot_counts) - overlap_knot_count * (
        len(window_knot_counts) - 1
    )
    if expected != knot_count:
        raise ValueError("window knot counts do not cover the full knot span")
    settings = options or ContinuousTimeTrajectoryFitOptions()
    batch_result = fit_continuous_trajectory(problem, settings)
    sliding_knots = list(problem.initial_knot_poses)
    sliding_status: FitStatus = "max_iterations"
    sliding_iterations = 0
    priors: tuple[KnotMarginalizationPrior, ...] = ()
    start_index = 0
    for window_index, window_size in enumerate(window_knot_counts):
        end_index = start_index + window_size
        window_timestamps = problem.knot_timestamps[start_index:end_index]
        window_initial = tuple(sliding_knots[start_index:end_index])
        if window_index > 0 and bad_overlap_translation_m is not None:
            window_initial = _shift_overlap_initial(
                window_initial,
                overlap_knot_count,
                bad_overlap_translation_m,
            )
        window_poses = _window_pose_gauges(
            window_initial,
            window_timestamps,
            start_index,
            overlap_knot_count=overlap_knot_count,
            is_first_window=window_index == 0,
            has_marginalization_priors=bool(priors) and apply_marginalization,
        )
        window_measurements = _window_point_to_plane_measurements(
            problem.point_to_plane_measurements,
            problem.knot_timestamps,
            window_timestamps[0],
            window_timestamps[-1],
        )
        window_problem = ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=window_timestamps,
            initial_knot_poses=window_initial,
            point_to_plane_measurements=window_measurements,
            pose_measurements=window_poses,
            knot_marginalization_priors=(
                priors
                if apply_marginalization and bad_overlap_translation_m is None
                else ()
            ),
        )
        window_result = fit_continuous_trajectory(window_problem, settings)
        sliding_knots[start_index:end_index] = list(window_result.knot_poses)
        sliding_status = window_result.status
        sliding_iterations += window_result.iterations
        if (
            apply_marginalization
            and window_index < len(window_knot_counts) - 1
        ):
            eliminated = window_size - overlap_knot_count
            if eliminated <= 0:
                raise ValueError("window must eliminate at least one knot")
            hessian, gradient = build_dense_knot_normal_equations(
                window_problem,
                list(window_result.knot_poses),
            )
            marginalized, marginal_gradient = marginalize_knot_prefix(
                hessian,
                gradient,
                eliminated_knot_count=eliminated,
            )
            anchor_poses = tuple(
                window_result.knot_poses[eliminated + offset]
                for offset in range(overlap_knot_count)
            )
            priors = split_retained_information(
                marginalized,
                marginal_gradient,
                retained_knot_count=overlap_knot_count,
                anchor_poses=anchor_poses,
                start_knot_index=0,
            )
            if corrupt_marginalization_translation_m is not None:
                priors = _corrupt_priors(priors, corrupt_marginalization_translation_m)
        start_index += window_size - overlap_knot_count
    sliding_result = ContinuousTimeTrajectoryFitResult(
        status=sliding_status,
        iterations=sliding_iterations,
        knot_poses=tuple(sliding_knots),
        initial_knot_poses=problem.initial_knot_poses,
        final_objective=batch_result.final_objective,
        final_point_rmse=None,
        final_point_to_plane_rmse=batch_result.final_point_to_plane_rmse,
        final_pose_rmse=None,
        final_imu_rotation_rmse_rad=None,
        final_lever_arm_rmse_m_s2=None,
        gyro_bias_rad_s=None,
        lever_arm_body_m=None,
        imu_clock_offset_sec=None,
        accel_bias_body_m_s2=None,
        gravity_world_m_s2=None,
        gyro_scale=None,
        accel_scale=None,
        max_step_translation_m=batch_result.max_step_translation_m,
        max_step_rotation_deg=batch_result.max_step_rotation_deg,
        gradient_norm=batch_result.gradient_norm,
        nonzero_jacobian_blocks=batch_result.nonzero_jacobian_blocks,
    )
    overlap_start = window_knot_counts[0] - overlap_knot_count
    translation_errors, rotation_errors = _overlap_errors(
        batch_result.knot_poses,
        sliding_result.knot_poses,
        overlap_start=overlap_start,
    )
    return SlidingWindowTrajectoryFitResult(
        batch_result=batch_result,
        sliding_result=sliding_result,
        window_count=len(window_knot_counts),
        overlap_knot_count=overlap_knot_count,
        max_overlap_translation_error_m=max(translation_errors) if translation_errors else 0.0,
        max_overlap_rotation_error_rad=max(rotation_errors) if rotation_errors else 0.0,
    )


def _window_pose_gauges(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    global_start_index: int,
    *,
    overlap_knot_count: int,
    is_first_window: bool,
    has_marginalization_priors: bool = False,
) -> tuple[TrajectoryPoseMeasurement, ...]:
    if not is_first_window and has_marginalization_priors:
        return ()
    gauge_index = 0 if is_first_window else overlap_knot_count
    if gauge_index >= len(knots):
        return ()
    return (
        TrajectoryPoseMeasurement(
            measurement_id=f"window-gauge-{global_start_index + gauge_index:02d}",
            timestamp_sec=timestamps[gauge_index],
            pose_world_body=knots[gauge_index],
            weight=100.0,
        ),
    )


def _window_point_to_plane_measurements(
    measurements: tuple[TrajectoryPointToPlaneMeasurement, ...],
    full_timestamps: tuple[float, ...],
    window_start: float,
    window_end: float,
) -> tuple[TrajectoryPointToPlaneMeasurement, ...]:
    selected: list[TrajectoryPointToPlaneMeasurement] = []
    for measurement in measurements:
        if measurement.timestamp_sec < window_start or measurement.timestamp_sec > window_end:
            continue
        if _interval_index(full_timestamps, measurement.timestamp_sec) is None:
            continue
        selected.append(measurement)
    return tuple(selected)


def _overlap_errors(
    reference: tuple[SE3, ...],
    estimate: tuple[SE3, ...],
    *,
    overlap_start: int,
) -> tuple[list[float], list[float]]:
    translations: list[float] = []
    rotations: list[float] = []
    for index in range(overlap_start, len(reference)):
        delta = se3_log(reference[index], estimate[index])
        translations.append(float(np.linalg.norm(delta[:3])))
        rotations.append(float(np.linalg.norm(delta[3:])))
    return translations, rotations


def _corrupt_priors(
    priors: tuple[KnotMarginalizationPrior, ...],
    translation_m: tuple[float, float, float],
) -> tuple[KnotMarginalizationPrior, ...]:
    corrupted: list[KnotMarginalizationPrior] = []
    shift = np.asarray(translation_m, dtype=float)
    for prior in priors:
        anchor = prior.anchor_pose
        translation = np.asarray(anchor.translation_m, dtype=float) + shift
        corrupted.append(
            KnotMarginalizationPrior(
                knot_index=prior.knot_index,
                anchor_pose=SE3(
                    translation_m=tuple(float(value) for value in translation),
                    rotation_quat_xyzw=anchor.rotation_quat_xyzw,
                ),
                information=np.array(prior.information, copy=True),
            )
        )
    return tuple(corrupted)


def _shift_overlap_initial(
    knots: tuple[SE3, ...],
    overlap_knot_count: int,
    translation_m: tuple[float, float, float],
) -> tuple[SE3, ...]:
    shifted = list(knots)
    delta = np.asarray(translation_m, dtype=float)
    for index in range(min(overlap_knot_count, len(shifted))):
        pose = shifted[index]
        translation = np.asarray(pose.translation_m, dtype=float) + delta
        shifted[index] = SE3(
            translation_m=tuple(float(value) for value in translation),
            rotation_quat_xyzw=pose.rotation_quat_xyzw,
        )
    return tuple(shifted)
