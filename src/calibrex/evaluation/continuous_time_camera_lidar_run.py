"""Artifact-backed continuous-time camera--LiDAR execution."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from calibrex.core.capture_time import ConstantBodyTwist, LidarCaptureTimePolicy
from calibrex.core.continuous_time_camera_lidar_artifacts import (
    ContinuousTimeArtifactProvenance,
    ContinuousTimeCameraLidarProblemArtifact,
    ContinuousTimeCameraLidarResultArtifact,
    ContinuousTimeEvaluationArtifact,
    ContinuousTimeIterationArtifact,
    load_continuous_time_camera_lidar_problem,
)
from calibrex.core.continuous_trajectory import (
    ContinuousTrajectoryPose,
    PiecewiseSE3Trajectory,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import TransformResult
from calibrex.solvers.continuous_time_camera_lidar_solver import (
    ContinuousTimeCameraLidarCapture,
    ContinuousTimeCameraLidarOptions,
    ContinuousTimeCameraLidarResult,
    ContinuousTimeCameraLidarSolver,
    ContinuousTimeEvaluation,
    TimedProbabilisticImageCorrespondence,
)


def run_continuous_time_camera_lidar_problem(
    problem_path: str | Path,
    *,
    result_id: str | None = None,
    command: list[str] | None = None,
    options: ContinuousTimeCameraLidarOptions | None = None,
) -> ContinuousTimeCameraLidarResultArtifact:
    """Validate, solve, and package one continuous-time problem."""

    path = Path(problem_path)
    problem = load_continuous_time_camera_lidar_problem(path)
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"continuous-time problem is not readable: {path}")
    settings = options or ContinuousTimeCameraLidarOptions(
        **cast(dict[str, Any], problem.options)
    )
    trajectory = (
        PiecewiseSE3Trajectory(
            tuple(
                ContinuousTrajectoryPose(
                    timestamp_sec=item.timestamp_sec,
                    transform_world_body=item.transform_world_body.as_se3(),
                )
                for item in problem.body_trajectory.poses
            )
        )
        if problem.body_trajectory is not None
        else None
    )
    captures = tuple(
        _capture_from_artifact(item, trajectory=trajectory)
        for item in problem.captures
    )
    policy = LidarCaptureTimePolicy(
        point_offset_unit=problem.capture_time_policy.point_offset_unit,
        stamp_reference=problem.capture_time_policy.stamp_reference,
        sensor_time_offset_sec=(
            problem.capture_time_policy.sensor_time_offset_sec
        ),
    )
    result = ContinuousTimeCameraLidarSolver().solve(
        captures,
        problem.initial_transform_camera_lidar.as_se3(),
        policy,
        settings,
    )
    return _result_artifact(
        problem,
        result,
        result_id=result_id or f"{problem.problem_id}-result",
        problem_sha256=digest,
        command=command or [],
    )


def _capture_from_artifact(
    value: Any,
    *,
    trajectory: PiecewiseSE3Trajectory | None,
) -> ContinuousTimeCameraLidarCapture:
    return ContinuousTimeCameraLidarCapture(
        capture_id=value.capture_id,
        lidar_message_stamp_sec=value.lidar_message_stamp_sec,
        camera_exposure_time_sec=value.camera_exposure_time_sec,
        correspondences=tuple(
            TimedProbabilisticImageCorrespondence(
                correspondence_id=item.correspondence_id,
                point_lidar_m=_tuple3(item.point_lidar_m),
                capture_offset_sec=item.capture_offset_sec,
                image_mean_px=_tuple2(item.image_mean_px),
                image_covariance_px2=_tuple4(item.image_covariance_px2),
                outlier_probability=item.outlier_probability,
                reliability=item.reliability,
            )
            for item in value.correspondences
        ),
        twist=ConstantBodyTwist(
            linear_velocity_body_mps=_tuple3(
                value.twist.linear_velocity_body_mps
            ),
            angular_velocity_body_radps=_tuple3(
                value.twist.angular_velocity_body_radps
            ),
        ),
        transform_body_lidar=value.transform_body_lidar.as_se3(),
        camera=value.camera,
        rolling_shutter_readout_sec=value.rolling_shutter_readout_sec,
        rolling_shutter_direction=value.rolling_shutter_direction,
        body_trajectory=trajectory,
    )


def _result_artifact(
    problem: ContinuousTimeCameraLidarProblemArtifact,
    result: ContinuousTimeCameraLidarResult,
    *,
    result_id: str,
    problem_sha256: str,
    command: list[str],
) -> ContinuousTimeCameraLidarResultArtifact:
    initial = problem.initial_transform_camera_lidar
    output = TransformResult(
        parent=initial.parent,
        child=initial.child,
        translation_m=list(result.transform_camera_lidar.translation_m),
        rotation_quat_xyzw=list(
            result.transform_camera_lidar.rotation_quat_xyzw
        ),
    )
    reference = problem.reference_transform_camera_lidar
    reference_time = problem.reference_time_offset_sec
    return ContinuousTimeCameraLidarResultArtifact(
        result_id=result_id,
        status=result.status,
        reason=result.reason,
        initial_transform_camera_lidar=initial,
        transform_camera_lidar=output,
        initial_time_offset_sec=result.initial_time_offset_sec,
        estimated_time_offset_sec=result.estimated_time_offset_sec,
        reference_transform_camera_lidar=reference,
        reference_time_offset_sec=reference_time,
        initial_rotation_error_deg=(
            _rotation_error_deg(initial, reference)
            if reference is not None
            else None
        ),
        final_rotation_error_deg=(
            _rotation_error_deg(output, reference)
            if reference is not None
            else None
        ),
        initial_translation_error_m=(
            _translation_error_m(initial, reference)
            if reference is not None
            else None
        ),
        final_translation_error_m=(
            _translation_error_m(output, reference)
            if reference is not None
            else None
        ),
        initial_time_offset_error_sec=(
            abs(result.initial_time_offset_sec - reference_time)
            if reference_time is not None
            else None
        ),
        final_time_offset_error_sec=(
            abs(result.estimated_time_offset_sec - reference_time)
            if reference_time is not None
            else None
        ),
        train_capture_ids=list(result.train_capture_ids),
        holdout_capture_ids=list(result.holdout_capture_ids),
        initial_train_evaluation=_evaluation(
            result.initial_train_evaluation
        ),
        final_train_evaluation=_evaluation(result.final_train_evaluation),
        initial_holdout_evaluation=_evaluation(
            result.initial_holdout_evaluation
        ),
        final_holdout_evaluation=_evaluation(
            result.final_holdout_evaluation
        ),
        time_sensitivity_px_per_sec=result.time_sensitivity_px_per_sec,
        time_observability_rank=result.time_observability_rank,
        trajectory_model=result.trajectory_model,
        trace=[
            ContinuousTimeIterationArtifact(
                evaluation=item.evaluation,
                delta_rotation_deg_xyz=list(
                    item.delta_rotation_deg_xyz
                ),
                delta_translation_m_xyz=list(
                    item.delta_translation_m_xyz
                ),
                time_offset_sec=item.time_offset_sec,
                objective=item.objective,
                accepted=item.accepted,
            )
            for item in result.trace
        ],
        options=cast(
            dict[str, int | float | str | bool],
            asdict(result.options),
        ),
        provenance=ContinuousTimeArtifactProvenance(
            generator=__name__,
            generator_version="0.1",
            git_commit=git_commit(),
            command=command,
            source_sha256=problem_sha256,
        ),
    )


def _evaluation(
    value: ContinuousTimeEvaluation,
) -> ContinuousTimeEvaluationArtifact:
    return ContinuousTimeEvaluationArtifact(**asdict(value))


def _tuple2(values: list[float]) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


def _tuple3(values: list[float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _tuple4(values: list[float]) -> tuple[float, float, float, float]:
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )


def _rotation_error_deg(left: TransformResult, right: TransformResult) -> float:
    left_quaternion = left.as_se3().rotation_quat_xyzw
    right_quaternion = right.as_se3().rotation_quat_xyzw
    dot = abs(
        sum(
            left_value * right_value
            for left_value, right_value in zip(
                left_quaternion,
                right_quaternion,
                strict=True,
            )
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def _translation_error_m(
    left: TransformResult,
    right: TransformResult,
) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(
                left.translation_m,
                right.translation_m,
                strict=True,
            )
        )
    )
