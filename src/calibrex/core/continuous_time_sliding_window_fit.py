"""Sliding-window fitting with Schur marginalization on knot trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from calibrex.core.continuous_time_marginalization import (
    JointKnotMarginalizationPrior,
    schur_complement_normal_with_evidence,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    ContinuousTimeTrajectoryFitResult,
    FitStatus,
    TrajectoryPointToPlaneMeasurement,
    _interval_index,
    _knot_normal_evidence,
    _pose_rmse,
    build_dense_knot_normal_equations,
    fit_continuous_trajectory,
    point_to_plane_rmse,
)
from calibrex.core.fit_diagnostics import FitDiagnostics, FitTermination, normalized_gradient_norm
from calibrex.core.geometry import SE3, _tuple3
from calibrex.core.knot_observability import WindowObservabilityEvidence, analyze_knot_information
from calibrex.core.se3_manifold import se3_log


@dataclass(frozen=True)
class SlidingWindowTrajectoryFitResult:
    """Batch reference and sliding-window knot estimates."""

    batch_result: ContinuousTimeTrajectoryFitResult
    sliding_result: ContinuousTimeTrajectoryFitResult
    window_count: int
    overlap_knot_count: int
    max_overlap_translation_error_m: float
    max_overlap_rotation_error_rad: float
    window_results: tuple[ContinuousTimeTrajectoryFitResult, ...] = ()
    window_observability: tuple[WindowObservabilityEvidence, ...] = ()


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
    if any(size <= overlap_knot_count for size in window_knot_counts):
        raise ValueError("every window must extend beyond the overlap")
    knot_count = len(problem.knot_timestamps)
    expected = sum(window_knot_counts) - overlap_knot_count * (
        len(window_knot_counts) - 1
    )
    if expected != knot_count:
        raise ValueError("window knot counts do not cover the full knot span")
    if (problem.point_measurements or problem.imu_preintegration_measurements
            or problem.imu_lever_arm_measurements or problem.knot_marginalization_priors
            or problem.joint_knot_marginalization_priors):
        raise ValueError("sliding windows support only plane and pose observations")
    settings = options or ContinuousTimeTrajectoryFitOptions()
    batch_result = fit_continuous_trajectory(problem, settings)
    sliding_knots = list(problem.initial_knot_poses)
    window_results: list[ContinuousTimeTrajectoryFitResult] = []
    window_observability: list[WindowObservabilityEvidence] = []
    sliding_iterations = 0
    frozen_global_parameters: set[int] = set()
    priors: tuple[JointKnotMarginalizationPrior, ...] = ()
    start_index = 0
    summarized_through: float | None = None
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
        use_priors = bool(priors) and apply_marginalization and bad_overlap_translation_m is None
        window_poses = tuple(
            pose for pose in problem.pose_measurements
            if window_timestamps[0] <= pose.timestamp_sec <= window_timestamps[-1]
            and (not use_priors or summarized_through is None
                 or pose.timestamp_sec > summarized_through)
        )
        window_measurements = _window_point_to_plane_measurements(
            problem.point_to_plane_measurements,
            problem.knot_timestamps,
            window_timestamps[0],
            window_timestamps[-1],
            summarized_through=summarized_through if use_priors else None,
        )
        window_problem = ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=window_timestamps,
            initial_knot_poses=window_initial,
            point_to_plane_measurements=window_measurements,
            pose_measurements=window_poses,
            joint_knot_marginalization_priors=priors if use_priors else (),
        )
        window_result = fit_continuous_trajectory(window_problem, settings)
        sliding_knots[start_index:end_index] = list(window_result.knot_poses)
        window_results.append(window_result)
        frozen_global_parameters.update(
            6 * start_index + index for index in window_result.frozen_parameter_indices
        )
        hessian, gradient = build_dense_knot_normal_equations(
            window_problem, list(window_result.knot_poses),
        )
        if not window_problem.joint_knot_marginalization_priors:
            measurement_information = hessian
        elif window_measurements or window_poses:
            measurement_problem = replace(window_problem, joint_knot_marginalization_priors=())
            measurement_information, _ = build_dense_knot_normal_equations(
                measurement_problem, list(window_result.knot_poses),
            )
        else:
            measurement_information = np.zeros_like(hessian)
        indices = tuple(range(start_index, end_index))
        window_observability.append(WindowObservabilityEvidence(
            window_index=window_index, global_knot_indices=list(indices),
            projected_steps=list(window_result.projected_step_diagnostics),
            frozen_parameters=[
                f"knot[{start_index + index // 6}]."
                f"{('tx', 'ty', 'tz', 'rx', 'ry', 'rz')[index % 6]}"
                for index in window_result.frozen_parameter_indices
            ],
            measurements=analyze_knot_information(measurement_information, indices),
            with_marginal_prior=analyze_knot_information(hessian, indices),
        ))
        sliding_iterations += window_result.iterations
        if (
            apply_marginalization
            and window_index < len(window_knot_counts) - 1
        ):
            eliminated = window_size - overlap_knot_count
            if eliminated <= 0:
                raise ValueError("window must eliminate at least one knot")
            marginalized, marginal_gradient, elimination = schur_complement_normal_with_evidence(
                hessian, gradient, 6 * eliminated,
            )
            window_observability[-1] = window_observability[-1].model_copy(
                update={"elimination": elimination},
            )
            anchor_poses = tuple(
                window_result.knot_poses[eliminated + offset]
                for offset in range(overlap_knot_count)
            )
            priors = (JointKnotMarginalizationPrior(
                knot_indices=tuple(range(overlap_knot_count)),
                anchor_poses=anchor_poses,
                information=0.5 * (marginalized + marginalized.T),
                gradient=marginal_gradient,
            ),)
            summarized_through = window_timestamps[-1]
            if corrupt_marginalization_translation_m is not None:
                priors = _corrupt_priors(priors, corrupt_marginalization_translation_m)
        start_index += window_size - overlap_knot_count
    normal, gradient, objective, block_count = _knot_normal_evidence(problem, sliding_knots)
    scaled_gradient = normalized_gradient_norm(gradient, normal.diagonal(), math.sqrt(objective))
    stationary = scaled_gradient < settings.convergence_gradient_norm
    statuses = {result.status for result in window_results}
    sliding_status: FitStatus
    termination: FitTermination
    if "singular_system" in statuses:
        sliding_status, termination = "singular_system", "singular_system"
    elif "max_iterations" in statuses:
        sliding_status, termination = "max_iterations", "iteration_limit"
    elif "stalled" in statuses or not stationary:
        sliding_status, termination = "stalled", "stationarity_not_confirmed"
    else:
        sliding_status, termination = "converged", "gradient_tolerance"
    pose_rows = [
        (_interval_index(problem.knot_timestamps, pose.timestamp_sec), pose.pose_world_body)
        for pose in problem.pose_measurements
    ]
    if any(interval is None for interval, _pose in pose_rows):
        raise ValueError("pose observation outside the knot domain")
    final_pose_rmse = _pose_rmse(
        problem, [(int(interval), pose) for interval, pose in pose_rows if interval is not None],
        sliding_knots,
    )
    sliding_result = ContinuousTimeTrajectoryFitResult(
        status=sliding_status,
        iterations=sliding_iterations,
        knot_poses=tuple(sliding_knots),
        initial_knot_poses=problem.initial_knot_poses,
        final_objective=objective,
        final_point_rmse=None,
        final_point_to_plane_rmse=point_to_plane_rmse(
            problem.point_to_plane_measurements, problem.knot_timestamps, sliding_knots,
        ),
        final_pose_rmse=final_pose_rmse,
        final_imu_rotation_rmse_rad=None,
        final_lever_arm_rmse_m_s2=None,
        gyro_bias_rad_s=None,
        lever_arm_body_m=None,
        imu_clock_offset_sec=None,
        accel_bias_body_m_s2=None,
        gravity_world_m_s2=None,
        gyro_scale=None,
        accel_scale=None,
        max_step_translation_m=max(result.max_step_translation_m for result in window_results),
        max_step_rotation_deg=max(result.max_step_rotation_deg for result in window_results),
        gradient_norm=float(np.linalg.norm(gradient, ord=np.inf)),
        nonzero_jacobian_blocks=block_count,
        frozen_parameter_indices=tuple(sorted(frozen_global_parameters)),
        projected_step_diagnostics=tuple(
            step for result in window_results for step in result.projected_step_diagnostics
        ),
        diagnostics=FitDiagnostics(
            termination_reason=termination, normalized_gradient_norm=scaled_gradient,
            gradient_tolerance=settings.convergence_gradient_norm, stationary=stationary,
        ),
    )
    overlap_start = window_knot_counts[0] - overlap_knot_count
    translation_errors, rotation_errors = _overlap_errors(
        batch_result.knot_poses,
        sliding_result.knot_poses,
        overlap_start=overlap_start,
    )
    return SlidingWindowTrajectoryFitResult(
        batch_result=batch_result,
        window_results=tuple(window_results),
        window_observability=tuple(window_observability),
        sliding_result=sliding_result,
        window_count=len(window_knot_counts),
        overlap_knot_count=overlap_knot_count,
        max_overlap_translation_error_m=max(translation_errors) if translation_errors else 0.0,
        max_overlap_rotation_error_rad=max(rotation_errors) if rotation_errors else 0.0,
    )


def _window_point_to_plane_measurements(
    measurements: tuple[TrajectoryPointToPlaneMeasurement, ...],
    full_timestamps: tuple[float, ...],
    window_start: float,
    window_end: float,
    *, summarized_through: float | None = None,
) -> tuple[TrajectoryPointToPlaneMeasurement, ...]:
    selected: list[TrajectoryPointToPlaneMeasurement] = []
    for measurement in measurements:
        # The incoming Schur prior contains every previously admitted row,
        # including observations exactly on the previous window's endpoint.
        if summarized_through is not None and measurement.timestamp_sec <= summarized_through:
            continue
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
    priors: tuple[JointKnotMarginalizationPrior, ...],
    translation_m: tuple[float, float, float],
) -> tuple[JointKnotMarginalizationPrior, ...]:
    corrupted: list[JointKnotMarginalizationPrior] = []
    shift = np.asarray(translation_m, dtype=float)
    for prior in priors:
        anchors = tuple(SE3(
            translation_m=_tuple3(np.asarray(anchor.translation_m, dtype=float) + shift),
            rotation_quat_xyzw=anchor.rotation_quat_xyzw,
        ) for anchor in prior.anchor_poses)
        corrupted.append(JointKnotMarginalizationPrior(
            knot_indices=prior.knot_indices, anchor_poses=anchors,
            information=np.array(prior.information, copy=True),
            gradient=np.array(prior.gradient, copy=True),
        ))
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
            translation_m=_tuple3(translation),
            rotation_quat_xyzw=pose.rotation_quat_xyzw,
        )
    return tuple(shifted)
