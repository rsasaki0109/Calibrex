"""Artifact-backed sparse continuous-time trajectory fitting."""

from __future__ import annotations

from pathlib import Path

from calibrex.core.continuous_time_contract import (
    ContinuousTimeTrajectoryContract,
    load_continuous_time_trajectory,
)
from calibrex.core.continuous_time_fit_artifacts import (
    ContinuousTimeFitProvenance,
    ContinuousTimeFittedKnot,
    ContinuousTimeTrajectoryFitResultArtifact,
    ContinuousTimeTrajectoryMeasurements,
    load_continuous_time_measurements,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryPointMeasurement,
    TrajectoryPointToPlaneMeasurement,
    TrajectoryPoseMeasurement,
    fit_continuous_trajectory,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import TransformResult


def run_continuous_time_trajectory_fit(
    trajectory_path: str | Path,
    measurements_path: str | Path,
    *,
    result_id: str | None = None,
    command: list[str] | None = None,
    options: ContinuousTimeTrajectoryFitOptions | None = None,
) -> ContinuousTimeTrajectoryFitResultArtifact:
    """Validate inputs, fit the knots, and package a schema-valid result."""

    trajectory = load_continuous_time_trajectory(trajectory_path)
    measurements = load_continuous_time_measurements(measurements_path)
    trajectory_digest = sha256_path(Path(trajectory_path))
    measurements_digest = sha256_path(Path(measurements_path))
    if trajectory_digest is None or measurements_digest is None:
        raise ValueError("trajectory or measurements file is not readable")

    if trajectory.interpolation != "screw_linear/v0.1":
        raise ValueError(
            "the trajectory fit natively supports the screw_linear/v0.1 "
            "interpolation model; convert the contract first"
        )

    problem = _problem_from_artifacts(trajectory, measurements)
    result = fit_continuous_trajectory(problem, options)

    timestamps = [item.timestamp_sec for item in trajectory.knots]
    initial = [item.transform_world_body.as_se3() for item in trajectory.knots]
    return ContinuousTimeTrajectoryFitResultArtifact(
        fit_id=result_id or f"{trajectory.trajectory_id}-fit",
        trajectory_id=trajectory.trajectory_id,
        interpolation="screw_linear",
        status=result.status,
        iterations=result.iterations,
        final_objective=result.final_objective,
        final_point_rmse=result.final_point_rmse,
        final_point_to_plane_rmse=result.final_point_to_plane_rmse,
        final_pose_rmse=result.final_pose_rmse,
        max_step_translation_m=result.max_step_translation_m,
        max_step_rotation_deg=result.max_step_rotation_deg,
        gradient_norm=result.gradient_norm,
        nonzero_jacobian_blocks=result.nonzero_jacobian_blocks,
        knots=[
            ContinuousTimeFittedKnot(
                knot_index=index,
                timestamp_sec=timestamps[index],
                initial_transform_world_body=TransformResult(
                    parent=trajectory.world_frame,
                    child=trajectory.body_frame,
                    translation_m=list(initial[index].translation_m),
                    rotation_quat_xyzw=list(initial[index].rotation_quat_xyzw),
                ),
                transform_world_body=TransformResult(
                    parent=trajectory.world_frame,
                    child=trajectory.body_frame,
                    translation_m=list(knot.translation_m),
                    rotation_quat_xyzw=list(knot.rotation_quat_xyzw),
                ),
            )
            for index, knot in enumerate(result.knot_poses)
        ],
        provenance=ContinuousTimeFitProvenance(
            generator=__name__,
            generator_version="0.1",
            git_commit=git_commit(),
            command=command or [],
            trajectory_sha256=trajectory_digest,
            measurements_sha256=measurements_digest,
        ),
    )


def _problem_from_artifacts(
    trajectory: ContinuousTimeTrajectoryContract,
    measurements: ContinuousTimeTrajectoryMeasurements,
) -> ContinuousTimeTrajectoryFitProblem:
    """Build a fit problem from schema-valid artifacts."""

    return ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=tuple(item.timestamp_sec for item in trajectory.knots),
        initial_knot_poses=tuple(
            item.transform_world_body.as_se3() for item in trajectory.knots
        ),
        point_measurements=tuple(
            TrajectoryPointMeasurement(
                measurement_id=item.measurement_id,
                timestamp_sec=item.timestamp_sec,
                point_body_m=tuple(item.point_body_m),
                target_world_m=tuple(item.target_world_m),
                weight=item.weight,
            )
            for item in measurements.point_measurements
        ),
        point_to_plane_measurements=tuple(
            TrajectoryPointToPlaneMeasurement(
                measurement_id=item.measurement_id,
                timestamp_sec=item.timestamp_sec,
                point_body_m=tuple(item.point_body_m),
                plane_point_world_m=tuple(item.plane_point_world_m),
                plane_normal_world=tuple(item.plane_normal_world),
                weight=item.weight,
            )
            for item in measurements.point_to_plane_measurements
        ),
        pose_measurements=tuple(
            TrajectoryPoseMeasurement(
                measurement_id=item.measurement_id,
                timestamp_sec=item.timestamp_sec,
                pose_world_body=item.pose_world_body.as_se3(),
                weight=item.weight,
            )
            for item in measurements.pose_measurements
        ),
        interpolation="screw_linear",
    )