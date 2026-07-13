import math

import pytest

from calibrex.core.capture_time import (
    ConstantBodyTwist,
    LidarCaptureTimePolicy,
    TimedLidarPoint,
    deskew_lidar_points_to_reference,
)
from calibrex.core.geometry import SE3
from calibrex.solvers.camera_lidar_capture_time_solver import (
    CameraLidarCaptureTimeOptions,
    CameraLidarCaptureTimeSolver,
    CameraLidarTimedCapture,
)


def _captures(
    *,
    count: int = 20,
    true_offset_sec: float = 0.017,
    with_references: bool = True,
    static: bool = False,
) -> list[CameraLidarTimedCapture]:
    transform = SE3(
        (0.16, -0.08, 0.12),
        (0.0, 0.0, math.sin(math.radians(20.0) / 2.0), math.cos(math.radians(20.0) / 2.0)),
    )
    points = (
        TimedLidarPoint("p0", (2.0, -0.5, 0.3), -0.09),
        TimedLidarPoint("p1", (3.2, 1.1, -0.2), -0.06),
        TimedLidarPoint("p2", (1.4, 2.0, 0.8), -0.03),
        TimedLidarPoint("p3", (-1.0, 3.0, 0.5), 0.0),
    )
    truth_policy = LidarCaptureTimePolicy("seconds", "scan_end", true_offset_sec)
    captures: list[CameraLidarTimedCapture] = []
    for index in range(count):
        speed_scale = 0.0 if static else 1.0 + 0.15 * math.sin(index * 0.7)
        twist = ConstantBodyTwist(
            (0.45 * speed_scale, -0.12 * speed_scale, 0.08 * speed_scale),
            (0.15 * speed_scale, -0.21 * speed_scale, 0.32 * speed_scale),
        )
        message_stamp = 100.0 + 0.2 * index
        camera_time = message_stamp + 0.025 + 0.003 * math.cos(index)
        nominal = CameraLidarTimedCapture(
            capture_id=f"capture-{index:03d}",
            lidar_message_stamp_sec=message_stamp,
            camera_exposure_time_sec=camera_time,
            points=points,
            twist=twist,
            transform_body_lidar=transform,
        )
        result = deskew_lidar_points_to_reference(
            points,
            message_stamp_sec=message_stamp,
            reference_time_sec=camera_time,
            policy=truth_policy,
            twist=twist,
            transform_body_lidar=transform,
        )
        captures.append(
            CameraLidarTimedCapture(
                capture_id=nominal.capture_id,
                lidar_message_stamp_sec=message_stamp,
                camera_exposure_time_sec=camera_time,
                points=points,
                twist=twist,
                transform_body_lidar=transform,
                reference_positions_lidar_at_camera_m=(
                    tuple(point.position_lidar_at_reference_m for point in result.points)
                    if with_references
                    else None
                ),
            )
        )
    return captures


def test_capture_time_solver_recovers_truth_on_disjoint_holdout() -> None:
    result = CameraLidarCaptureTimeSolver().solve(
        _captures(),
        LidarCaptureTimePolicy("seconds", "scan_end"),
        CameraLidarCaptureTimeOptions(holdout_ratio=0.25, split_seed=9),
    )

    assert result.status == "converged"
    assert result.estimated_sensor_time_offset_sec == pytest.approx(0.017, abs=1.0e-7)
    assert set(result.train_capture_ids).isdisjoint(result.holdout_capture_ids)
    assert len(result.train_capture_ids) + len(result.holdout_capture_ids) == 20
    assert result.train_evaluation.reference_rmse_m is not None
    assert result.train_evaluation.reference_rmse_m < 1.0e-7
    assert result.holdout_evaluation.reference_rmse_m is not None
    assert result.holdout_evaluation.reference_rmse_m < 1.0e-7
    assert result.time_observability_rank == 1
    assert result.time_sensitivity_rms_mps is not None
    assert result.time_sensitivity_rms_mps > 0.05
    assert len(result.probes) == 6
    assert all(probe.detectable is True for probe in result.probes)
    document = result.as_dict()
    assert document["method"] == "continuous_twist_per_point_time_offset/v0.1"
    assert document["papers"][0]["doi"] == "10.1109/TRO.2016.2596771"
    assert document["clock_convention"] == ("lidar_sensor_time + dt_lidar = camera_reference_time")


def test_capture_time_evidence_only_applies_firing_times_without_claiming_offset() -> None:
    result = CameraLidarCaptureTimeSolver().solve(
        _captures(with_references=False),
        LidarCaptureTimePolicy("seconds", "scan_end"),
    )

    assert result.status == "evidence_only"
    assert result.estimated_sensor_time_offset_sec is None
    assert result.holdout_evaluation.reference_rmse_m is None
    assert result.holdout_evaluation.mean_point_time_span_sec == pytest.approx(0.09)
    assert result.holdout_evaluation.deskew_displacement_rmse_m is not None
    assert result.holdout_evaluation.deskew_displacement_rmse_m > 0.0
    assert all(probe.difference_from_nominal_rmse_m is not None for probe in result.probes)
    assert all(probe.detectable is True for probe in result.probes)


def test_capture_time_solver_rejects_static_motion_as_time_degenerate() -> None:
    result = CameraLidarCaptureTimeSolver().solve(
        _captures(static=True),
        LidarCaptureTimePolicy("seconds", "scan_end"),
    )

    assert result.status == "degenerate_motion"
    assert result.estimated_sensor_time_offset_sec is None
    assert result.time_observability_rank == 0
    assert result.time_sensitivity_rms_mps == pytest.approx(0.0)
    assert len(result.probes) == 6
    assert all(probe.detectable is False for probe in result.probes)


def test_capture_time_solver_is_order_invariant_and_validates_inputs() -> None:
    captures = _captures()
    policy = LidarCaptureTimePolicy("seconds", "scan_end")
    options = CameraLidarCaptureTimeOptions(split_seed=4)
    forward = CameraLidarCaptureTimeSolver().solve(captures, policy, options)
    reverse = CameraLidarCaptureTimeSolver().solve(list(reversed(captures)), policy, options)

    assert forward.estimated_sensor_time_offset_sec == reverse.estimated_sensor_time_offset_sec
    assert forward.train_capture_ids == reverse.train_capture_ids
    assert forward.holdout_capture_ids == reverse.holdout_capture_ids
    with pytest.raises(ValueError, match="capture IDs must be unique"):
        CameraLidarCaptureTimeSolver().solve([captures[0], captures[0]], policy)
    mixed = list(captures)
    mixed[-1] = CameraLidarTimedCapture(
        capture_id=mixed[-1].capture_id,
        lidar_message_stamp_sec=mixed[-1].lidar_message_stamp_sec,
        camera_exposure_time_sec=mixed[-1].camera_exposure_time_sec,
        points=mixed[-1].points,
        twist=mixed[-1].twist,
        transform_body_lidar=mixed[-1].transform_body_lidar,
    )
    with pytest.raises(ValueError, match="every capture or none"):
        CameraLidarCaptureTimeSolver().solve(mixed, policy)


@pytest.mark.parametrize(
    "options",
    [
        CameraLidarCaptureTimeOptions(holdout_ratio=0.0),
        CameraLidarCaptureTimeOptions(coarse_step_sec=0.0),
        CameraLidarCaptureTimeOptions(known_bad_offsets_sec=()),
        CameraLidarCaptureTimeOptions(min_time_sensitivity_mps=0.0),
    ],
)
def test_capture_time_solver_rejects_invalid_options(
    options: CameraLidarCaptureTimeOptions,
) -> None:
    with pytest.raises(ValueError):
        CameraLidarCaptureTimeSolver().solve(
            _captures(), LidarCaptureTimePolicy("seconds", "scan_end"), options
        )
