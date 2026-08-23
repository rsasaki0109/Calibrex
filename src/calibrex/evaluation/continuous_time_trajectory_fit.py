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
    TrajectoryImuGyroSample,
    TrajectoryImuLeverArmMeasurement,
    TrajectoryImuPreintegrationMeasurement,
    TrajectoryPointMeasurement,
    TrajectoryPointToPlaneMeasurement,
    TrajectoryPoseMeasurement,
    fit_continuous_trajectory,
)
from calibrex.core.geometry import _tuple3
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
        final_imu_rotation_rmse_rad=result.final_imu_rotation_rmse_rad,
        final_lever_arm_rmse_m_s2=result.final_lever_arm_rmse_m_s2,
        gyro_bias_rad_s=(
            list(result.gyro_bias_rad_s) if result.gyro_bias_rad_s is not None else None
        ),
        lever_arm_body_m=(
            list(result.lever_arm_body_m)
            if result.lever_arm_body_m is not None
            else None
        ),
        imu_clock_offset_sec=result.imu_clock_offset_sec,
        accel_bias_body_m_s2=(
            list(result.accel_bias_body_m_s2)
            if result.accel_bias_body_m_s2 is not None
            else None
        ),
        gravity_world_m_s2=(
            list(result.gravity_world_m_s2)
            if result.gravity_world_m_s2 is not None
            else None
        ),
        gyro_scale=(
            list(result.gyro_scale) if result.gyro_scale is not None else None
        ),
        accel_scale=(
            list(result.accel_scale) if result.accel_scale is not None else None
        ),
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
                point_body_m=_tuple3(item.point_body_m),
                target_world_m=_tuple3(item.target_world_m),
                weight=item.weight,
            )
            for item in measurements.point_measurements
        ),
        point_to_plane_measurements=tuple(
            TrajectoryPointToPlaneMeasurement(
                measurement_id=item.measurement_id,
                timestamp_sec=item.timestamp_sec,
                point_body_m=_tuple3(item.point_body_m),
                plane_point_world_m=_tuple3(item.plane_point_world_m),
                plane_normal_world=_tuple3(item.plane_normal_world),
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
        imu_preintegration_measurements=tuple(
            TrajectoryImuPreintegrationMeasurement(
                measurement_id=item.measurement_id,
                gyro_samples=tuple(
                    TrajectoryImuGyroSample(
                        timestamp_sec=sample.timestamp_sec,
                        omega_body_rad_s=_tuple3(sample.omega_body_rad_s),
                    )
                    for sample in item.gyro_samples
                ),
                weight=item.weight,
            )
            for item in measurements.imu_preintegration_measurements
        ),
        imu_lever_arm_measurements=tuple(
            TrajectoryImuLeverArmMeasurement(
                measurement_id=item.measurement_id,
                timestamp_sec=item.timestamp_sec,
                accel_body_m_s2=_tuple3(item.accel_body_m_s2),
                weight=item.weight,
            )
            for item in measurements.imu_lever_arm_measurements
        ),
        initial_gyro_bias_rad_s=_tuple3(measurements.initial_gyro_bias_rad_s),
        initial_lever_arm_body_m=_tuple3(measurements.initial_lever_arm_body_m),
        initial_imu_clock_offset_sec=measurements.initial_imu_clock_offset_sec,
        initial_accel_bias_body_m_s2=_tuple3(measurements.initial_accel_bias_body_m_s2),
        initial_gravity_world_m_s2=_tuple3(measurements.initial_gravity_world_m_s2),
        estimate_gyro_bias=measurements.estimate_gyro_bias,
        estimate_lever_arm=measurements.estimate_lever_arm,
        estimate_imu_clock_offset=measurements.estimate_imu_clock_offset,
        estimate_accel_bias=measurements.estimate_accel_bias,
        estimate_gravity=measurements.estimate_gravity,
        initial_gyro_scale=_tuple3(measurements.initial_gyro_scale),
        initial_accel_scale=_tuple3(measurements.initial_accel_scale),
        estimate_gyro_scale=measurements.estimate_gyro_scale,
        estimate_accel_scale=measurements.estimate_accel_scale,
        interpolation="screw_linear",
    )
