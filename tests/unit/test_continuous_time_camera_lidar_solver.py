from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.capture_time import (
    ConstantBodyTwist,
    LidarCaptureTimePolicy,
    TimedLidarPoint,
    deskew_lidar_points_to_reference,
)
from calibrex.core.continuous_time_camera_lidar_artifacts import (
    ContinuousTimeArtifactProvenance,
    ContinuousTimeBodyTwist,
    ContinuousTimeCameraLidarProblemArtifact,
    ContinuousTimeCaptureArtifact,
    ContinuousTimeCapturePolicyArtifact,
    ContinuousTimeImageCorrespondence,
    ContinuousTimeTrajectoryArtifact,
    ContinuousTimeTrajectoryPoseArtifact,
    load_continuous_time_camera_lidar_problem,
    load_continuous_time_camera_lidar_result,
)
from calibrex.core.continuous_trajectory import (
    ContinuousTrajectoryPose,
    PiecewiseSE3Trajectory,
)
from calibrex.core.geometry import SE3
from calibrex.core.io import write_mapping
from calibrex.core.provenance import sha256_path
from calibrex.core.result import TransformResult
from calibrex.core.validation import validate_file
from calibrex.data.depth import DepthCameraIntrinsics
from calibrex.evaluation.continuous_time_camera_lidar_ablation import (
    run_continuous_time_camera_lidar_ablation,
)
from calibrex.solvers import ContinuousTimeCameraLidarSolver as PublicSolver
from calibrex.solvers.continuous_time_camera_lidar_solver import (
    ContinuousTimeCameraLidarCapture,
    ContinuousTimeCameraLidarOptions,
    ContinuousTimeCameraLidarSolver,
    TimedProbabilisticImageCorrespondence,
    evaluate_continuous_time_camera_lidar,
)


def test_joint_continuous_time_solver_recovers_offset_and_translation() -> None:
    assert PublicSolver is ContinuousTimeCameraLidarSolver
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    captures = _captures(truth, true_offset_sec=0.02)
    initial = SE3((0.05, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    options = ContinuousTimeCameraLidarOptions(
        split_seed=3,
        initial_rotation_step_deg=0.25,
        initial_translation_step_m=0.05,
        initial_time_step_sec=0.01,
        minimum_rotation_step_deg=0.05,
        minimum_translation_step_m=0.005,
        minimum_time_step_sec=0.001,
        max_evaluations=300,
    )

    result = ContinuousTimeCameraLidarSolver().solve(
        captures,
        initial,
        LidarCaptureTimePolicy("seconds", "scan_end"),
        options,
    )

    assert result.status == "converged"
    assert result.transform_camera_lidar.translation_m[0] == pytest.approx(
        0.10, abs=0.006
    )
    assert result.estimated_time_offset_sec == pytest.approx(0.02, abs=0.002)
    assert set(result.train_capture_ids).isdisjoint(result.holdout_capture_ids)
    assert (
        result.final_holdout_evaluation.weighted_reprojection_rmse_px
        < result.initial_holdout_evaluation.weighted_reprojection_rmse_px
    )
    assert result.time_observability_rank == 1
    assert result.as_dict()["clock_convention"] == (
        "lidar_sensor_time + dt_lidar = camera_reference_time"
    )


def test_joint_continuous_time_solver_rejects_static_motion() -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))

    result = ContinuousTimeCameraLidarSolver().solve(
        _captures(truth, true_offset_sec=0.02, static=True),
        truth,
        LidarCaptureTimePolicy("seconds", "scan_end"),
    )

    assert result.status == "degenerate_motion"
    assert result.time_observability_rank == 0
    assert result.trace == ()


def test_continuous_time_ablation_can_freeze_clock_and_point_times() -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    captures = _captures(truth, true_offset_sec=0.02)
    options = ContinuousTimeCameraLidarOptions(
        estimate_time_offset=False,
        use_per_point_time=False,
        use_covariance=False,
        initial_rotation_step_deg=0.25,
        initial_translation_step_m=0.02,
        minimum_rotation_step_deg=0.1,
        minimum_translation_step_m=0.01,
        max_evaluations=150,
    )

    result = ContinuousTimeCameraLidarSolver().solve(
        captures,
        truth,
        LidarCaptureTimePolicy("seconds", "scan_end"),
        options,
    )

    assert result.status == "converged"
    assert result.estimated_time_offset_sec == 0.0
    assert result.options.estimate_time_offset is False
    assert result.options.use_per_point_time is False
    assert result.options.use_covariance is False


def test_rolling_shutter_row_times_change_continuous_time_residual() -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    captures = _captures(truth, true_offset_sec=0.02)
    policy = LidarCaptureTimePolicy("seconds", "scan_end")
    baseline = evaluate_continuous_time_camera_lidar(
        captures,
        truth,
        policy,
        0.02,
    )
    rolling = [
        replace(
            capture,
            rolling_shutter_readout_sec=0.02,
            rolling_shutter_direction="top_to_bottom",
        )
        for capture in captures
    ]
    rolling_evaluation = evaluate_continuous_time_camera_lidar(
        rolling,
        truth,
        policy,
        0.02,
    )
    disabled_evaluation = evaluate_continuous_time_camera_lidar(
        rolling,
        truth,
        policy,
        0.02,
        ContinuousTimeCameraLidarOptions(use_rolling_shutter=False),
    )

    assert baseline.weighted_reprojection_rmse_px == pytest.approx(0.0)
    assert rolling_evaluation.weighted_reprojection_rmse_px is not None
    assert rolling_evaluation.weighted_reprojection_rmse_px > 0.0
    assert disabled_evaluation.weighted_reprojection_rmse_px == pytest.approx(
        0.0
    )


def test_piecewise_trajectory_handles_non_constant_motion() -> None:
    trajectory = PiecewiseSE3Trajectory(
        (
            ContinuousTrajectoryPose(0.0, SE3.identity()),
            ContinuousTrajectoryPose(
                1.0, SE3((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            ),
            ContinuousTrajectoryPose(
                2.0, SE3((3.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            ),
        )
    )
    camera = DepthCameraIntrinsics(
        width=1280,
        height=720,
        fx=600.0,
        fy=600.0,
        cx=640.0,
        cy=360.0,
    )
    message_stamp = 0.9
    exposure = 1.1
    source_points = (
        (-2.0, -1.0, 8.0),
        (-1.0, 1.0, 7.0),
        (0.5, -1.5, 9.0),
        (2.0, 1.0, 10.0),
    )
    offsets = (-0.1, -0.05, 0.0, 0.05)
    correspondences = []
    for index, (point, offset) in enumerate(
        zip(source_points, offsets, strict=True)
    ):
        deskewed = trajectory.transform_static_point_between_body_frames(
            point,
            source_time_sec=message_stamp + offset,
            target_time_sec=exposure,
        )
        correspondences.append(
            TimedProbabilisticImageCorrespondence(
                correspondence_id=str(index),
                point_lidar_m=point,
                capture_offset_sec=offset,
                image_mean_px=(
                    camera.fx * deskewed[0] / deskewed[2] + camera.cx,
                    camera.fy * deskewed[1] / deskewed[2] + camera.cy,
                ),
                image_covariance_px2=(1.0, 0.0, 0.0, 1.0),
            )
        )
    capture = ContinuousTimeCameraLidarCapture(
        capture_id="non-constant",
        lidar_message_stamp_sec=message_stamp,
        camera_exposure_time_sec=exposure,
        correspondences=tuple(correspondences),
        twist=ConstantBodyTwist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        transform_body_lidar=SE3.identity(),
        camera=camera,
        body_trajectory=trajectory,
    )
    policy = LidarCaptureTimePolicy("seconds", "scan_end")

    piecewise = evaluate_continuous_time_camera_lidar(
        [capture], SE3.identity(), policy, 0.0
    )
    constant = evaluate_continuous_time_camera_lidar(
        [replace(capture, body_trajectory=None)],
        SE3.identity(),
        policy,
        0.0,
    )

    assert piecewise.weighted_reprojection_rmse_px == pytest.approx(0.0)
    assert constant.weighted_reprojection_rmse_px is not None
    assert constant.weighted_reprojection_rmse_px > 10.0
    with pytest.raises(ValueError, match="outside"):
        trajectory.pose_at(2.1)


def test_solver_rejects_trajectory_without_clock_search_margin() -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    captures = _captures(truth, true_offset_sec=0.02)
    too_short = PiecewiseSE3Trajectory(
        (
            ContinuousTrajectoryPose(
                99.95, SE3.identity()
            ),
            ContinuousTrajectoryPose(
                101.15, SE3.identity()
            ),
        )
    )
    captures = [
        replace(capture, body_trajectory=too_short) for capture in captures
    ]

    with pytest.raises(ValueError, match="clock search exceeds"):
        ContinuousTimeCameraLidarSolver().solve(
            captures,
            truth,
            LidarCaptureTimePolicy("seconds", "scan_end"),
        )


def test_continuous_time_cli_writes_schema_valid_result(
    tmp_path: Path,
) -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    captures = _captures(truth, true_offset_sec=0.02)
    options = ContinuousTimeCameraLidarOptions(
        split_seed=3,
        initial_rotation_step_deg=0.25,
        initial_translation_step_m=0.05,
        initial_time_step_sec=0.01,
        minimum_rotation_step_deg=0.05,
        minimum_translation_step_m=0.005,
        minimum_time_step_sec=0.001,
        max_evaluations=300,
    )
    problem = _problem(
        captures,
        options,
        reference_transform=truth,
        reference_time_offset_sec=0.02,
    )
    problem_path = tmp_path / "problem.yaml"
    result_path = tmp_path / "result.yaml"
    problem.save(problem_path)

    exit_code = main(
        [
            "camera-lidar",
            "refine-continuous-time",
            str(problem_path),
            "--output",
            str(result_path),
            "--json",
        ]
    )

    assert exit_code == 0
    assert validate_file(
        result_path, "continuous-time-camera-lidar-result"
    ).valid
    result = load_continuous_time_camera_lidar_result(result_path)
    assert result.final_translation_error_m is not None
    assert result.final_translation_error_m < 0.01
    assert result.final_time_offset_error_sec is not None
    assert result.final_time_offset_error_sec < 0.003


def test_cli_attaches_recorded_trajectory_with_digest(
    tmp_path: Path,
) -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    problem_path = tmp_path / "problem.yaml"
    trajectory_path = tmp_path / "trajectory.yaml"
    output_path = tmp_path / "problem-with-trajectory.yaml"
    _problem(
        _captures(truth, true_offset_sec=0.02),
        ContinuousTimeCameraLidarOptions(),
    ).save(problem_path)
    metric = {"value": 0.0, "grade": "pass"}
    write_mapping(
        trajectory_path,
        {
            "schema_version": "slac.trajectory/v0.1",
            "run": {
                "id": "recorded-motion",
                "status": "success",
                "domain": "test",
                "slac_version": "0.0",
                "created_at": "2026-07-31T00:00:00+00:00",
            },
            "odometry_source": {
                "topic": "/odom",
                "message_count": 2,
                "first_timestamp_ns": 99_000_000_000,
                "last_timestamp_ns": 102_000_000_000,
                "time_span_s": 3.0,
            },
            "frame_semantics": {
                "world_frame_id": "world",
                "child_frame_id": "body",
            },
            "pose_count_total": 2,
            "pose_count_recorded": 2,
            "subsampling": {"max_pose_samples": 2},
            "poses": [
                {
                    "timestamp_ns": 99_000_000_000,
                    "translation": [0.0, 0.0, 0.0],
                    "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                {
                    "timestamp_ns": 102_000_000_000,
                    "translation": [3.0, 0.0, 0.0],
                    "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            ],
            "interpolation_health": {
                "interpolation_count": 0,
                "clamp_count": 0,
                "max_extrapolation_s": 0.0,
            },
            "quality": {
                "kinematic_linear_speed_p95_mps": metric,
                "kinematic_linear_speed_max_mps": metric,
                "kinematic_angular_speed_p95_dps": metric,
                "kinematic_angular_speed_max_dps": metric,
                "kinematic_gate_status": "pass",
                "kinematic_gate_reason": "fixture",
                "interpolation_clamp_fraction": metric,
                "interpolation_max_extrapolation_s": metric,
                "interpolation_gate_status": "pass",
                "interpolation_gate_reason": "fixture",
                "cross_segment_rmse_m": metric,
                "cross_segment_first_half_scan_count": 1,
                "cross_segment_second_half_scan_count": 1,
                "cross_segment_first_half_point_count": 1,
                "cross_segment_second_half_point_count": 1,
                "cross_segment_correspondence_count": 1,
                "cross_segment_gate_status": "pass",
                "cross_segment_gate_reason": "fixture",
                "verdict": "pass",
                "verdict_reason": "fixture",
            },
        },
    )

    exit_code = main(
        [
            "camera-lidar",
            "attach-continuous-trajectory",
            str(problem_path),
            str(trajectory_path),
            "--output",
            str(output_path),
            "--json",
        ]
    )

    assert exit_code == 0
    assert validate_file(
        output_path, "continuous-time-camera-lidar-problem"
    ).valid
    payload = load_continuous_time_camera_lidar_problem(output_path)
    assert payload.body_trajectory is not None
    assert payload.body_trajectory.source_sha256 == sha256_path(
        trajectory_path
    )


def test_continuous_time_ablation_uses_paired_capture_splits(
    tmp_path: Path,
) -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    captures = _captures(truth, true_offset_sec=0.02)
    options = ContinuousTimeCameraLidarOptions(
        split_seed=3,
        initial_rotation_step_deg=0.25,
        initial_translation_step_m=0.05,
        initial_time_step_sec=0.01,
        minimum_rotation_step_deg=0.1,
        minimum_translation_step_m=0.01,
        minimum_time_step_sec=0.002,
        max_evaluations=220,
    )
    problem_path = tmp_path / "problem.yaml"
    _problem(
        captures,
        options,
        reference_transform=truth,
        reference_time_offset_sec=0.02,
    ).save(problem_path)

    definition, benchmark = run_continuous_time_camera_lidar_ablation(
        problem_path,
        result_directory=tmp_path / "ablation-results",
        command="pytest continuous-time ablation",
        split_seeds=(3, 7),
        bootstrap_samples=100,
    )

    assert len(definition.methods) == 5
    assert len(definition.trials) == 10
    assert {
        item.name for item in definition.metrics
    } == {
        "holdout_reprojection_rmse_px",
        "rotation_error_deg",
        "translation_error_m",
        "time_offset_abs_error_sec",
    }
    assert len(benchmark.paired_comparisons) == 16
    assert benchmark.reference_method_id == "full_continuous_time"


def test_continuous_time_problem_accepts_pinned_piecewise_trajectory() -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    captures = _captures(truth, true_offset_sec=0.02)
    problem = _problem(captures, ContinuousTimeCameraLidarOptions())
    trajectory = ContinuousTimeTrajectoryArtifact(
        world_frame="world",
        body_frame="body",
        source_sha256="b" * 64,
        poses=[
            ContinuousTimeTrajectoryPoseArtifact(
                timestamp_sec=99.0,
                transform_world_body=TransformResult(
                    parent="world",
                    child="body",
                    translation_m=[0.0, 0.0, 0.0],
                    rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                ),
            ),
            ContinuousTimeTrajectoryPoseArtifact(
                timestamp_sec=102.0,
                transform_world_body=TransformResult(
                    parent="world",
                    child="body",
                    translation_m=[3.0, 0.0, 0.0],
                    rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                ),
            ),
        ],
    )

    enriched = problem.model_copy(update={"body_trajectory": trajectory})

    assert enriched.body_trajectory is not None
    assert enriched.body_trajectory.extrapolation == "reject"


def _captures(
    transform_camera_lidar: SE3,
    *,
    true_offset_sec: float,
    static: bool = False,
) -> list[ContinuousTimeCameraLidarCapture]:
    camera = DepthCameraIntrinsics(
        width=1280,
        height=720,
        fx=600.0,
        fy=600.0,
        cx=640.0,
        cy=360.0,
    )
    points = (
        (-2.0, -1.0, 8.0),
        (-1.0, 1.0, 7.0),
        (0.5, -1.5, 9.0),
        (2.0, 1.0, 10.0),
        (-2.5, 0.5, 12.0),
        (1.5, -0.5, 6.0),
        (0.2, 1.8, 11.0),
        (2.8, -1.2, 13.0),
    )
    offsets = (-0.09, -0.075, -0.06, -0.045, -0.03, -0.02, -0.01, 0.0)
    policy = LidarCaptureTimePolicy("seconds", "scan_end", true_offset_sec)
    captures = []
    for capture_index in range(12):
        scale = 0.0 if static else 1.0 + 0.15 * math.sin(capture_index)
        twist = ConstantBodyTwist(
            (1.2 * scale, -0.3 * scale, 0.1 * scale),
            (0.05 * scale, -0.08 * scale, 0.18 * scale),
        )
        message_stamp = 100.0 + 0.1 * capture_index
        exposure = message_stamp + 0.015
        timed_points = [
            TimedLidarPoint(str(index), point, offsets[index])
            for index, point in enumerate(points)
        ]
        deskewed = deskew_lidar_points_to_reference(
            timed_points,
            message_stamp_sec=message_stamp,
            reference_time_sec=exposure,
            policy=policy,
            twist=twist,
            transform_body_lidar=SE3.identity(),
        )
        correspondences = []
        for source, point in zip(timed_points, deskewed.points, strict=True):
            camera_point = transform_camera_lidar.transform_point(
                point.position_lidar_at_reference_m
            )
            correspondences.append(
                TimedProbabilisticImageCorrespondence(
                    correspondence_id=source.point_id,
                    point_lidar_m=source.position_lidar_m,
                    capture_offset_sec=float(source.capture_offset),
                    image_mean_px=(
                        camera.fx * camera_point[0] / camera_point[2] + camera.cx,
                        camera.fy * camera_point[1] / camera_point[2] + camera.cy,
                    ),
                    image_covariance_px2=(1.0, 0.0, 0.0, 1.0),
                )
            )
        captures.append(
            ContinuousTimeCameraLidarCapture(
                capture_id=f"capture-{capture_index:03d}",
                lidar_message_stamp_sec=message_stamp,
                camera_exposure_time_sec=exposure,
                correspondences=tuple(correspondences),
                twist=twist,
                transform_body_lidar=SE3.identity(),
                camera=camera,
            )
        )
    return captures


def _problem(
    captures: list[ContinuousTimeCameraLidarCapture],
    options: ContinuousTimeCameraLidarOptions,
    *,
    reference_transform: SE3 | None = None,
    reference_time_offset_sec: float | None = None,
) -> ContinuousTimeCameraLidarProblemArtifact:
    initial = SE3((0.05, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    return ContinuousTimeCameraLidarProblemArtifact(
        problem_id="synthetic-continuous-time",
        dataset_id="synthetic",
        dataset_license_spdx="CC0-1.0",
        split_id="test",
        initial_transform_camera_lidar=TransformResult(
            parent="camera",
            child="lidar",
            translation_m=list(initial.translation_m),
            rotation_quat_xyzw=list(initial.rotation_quat_xyzw),
        ),
        reference_transform_camera_lidar=(
            TransformResult(
                parent="camera",
                child="lidar",
                translation_m=list(reference_transform.translation_m),
                rotation_quat_xyzw=list(
                    reference_transform.rotation_quat_xyzw
                ),
            )
            if reference_transform is not None
            else None
        ),
        reference_time_offset_sec=reference_time_offset_sec,
        capture_time_policy=ContinuousTimeCapturePolicyArtifact(
            point_offset_unit="seconds",
            stamp_reference="scan_end",
            sensor_time_offset_sec=0.0,
        ),
        captures=[
            ContinuousTimeCaptureArtifact(
                capture_id=capture.capture_id,
                lidar_message_stamp_sec=capture.lidar_message_stamp_sec,
                camera_exposure_time_sec=capture.camera_exposure_time_sec,
                camera_frame="camera",
                lidar_frame="lidar",
                body_frame="body",
                camera=capture.camera,
                transform_body_lidar=TransformResult(
                    parent="body",
                    child="lidar",
                    translation_m=list(
                        capture.transform_body_lidar.translation_m
                    ),
                    rotation_quat_xyzw=list(
                        capture.transform_body_lidar.rotation_quat_xyzw
                    ),
                ),
                twist=ContinuousTimeBodyTwist(
                    linear_velocity_body_mps=list(
                        capture.twist.linear_velocity_body_mps
                    ),
                    angular_velocity_body_radps=list(
                        capture.twist.angular_velocity_body_radps
                    ),
                ),
                correspondences=[
                    ContinuousTimeImageCorrespondence(
                        correspondence_id=item.correspondence_id,
                        point_lidar_m=list(item.point_lidar_m),
                        capture_offset_sec=item.capture_offset_sec,
                        image_mean_px=list(item.image_mean_px),
                        image_covariance_px2=list(
                            item.image_covariance_px2
                        ),
                        outlier_probability=item.outlier_probability,
                        reliability=item.reliability,
                    )
                    for item in capture.correspondences
                ],
            )
            for capture in captures
        ],
        split_policy="deterministic_disjoint_holdout",
        options=asdict(options),
        provenance=ContinuousTimeArtifactProvenance(
            generator="pytest",
            generator_version="1",
            source_sha256="a" * 64,
        ),
    )
