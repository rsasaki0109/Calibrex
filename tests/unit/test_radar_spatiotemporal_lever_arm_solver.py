import math

import pytest

from calibrex.core.geometry import (
    QuaternionXYZW,
    Vector3,
    quaternion_conjugate_xyzw,
    rotate_vector_xyzw,
)
from calibrex.solvers.radar_spatiotemporal_lever_arm_solver import (
    RadarSpatiotemporalLeverArmOptions,
    RadarSpatiotemporalLeverArmSolver,
    RadarVelocityMeasurement,
    ReferenceKinematicSample,
)


def _linear_velocity(timestamp: float) -> Vector3:
    return (1.0 + 1.7 * timestamp, -0.4 + 0.8 * timestamp, 0.2 - 0.3 * timestamp)


def _angular_velocity(timestamp: float) -> Vector3:
    return (
        0.7 + 0.11 * timestamp,
        -0.5 + 0.17 * timestamp,
        0.4 - 0.09 * timestamp,
    )


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _synthetic_problem(
    translation: Vector3 = (0.42, -0.18, 0.27),
    time_offset: float = 0.023,
) -> tuple[
    list[RadarVelocityMeasurement],
    list[ReferenceKinematicSample],
    QuaternionXYZW,
]:
    yaw = math.radians(17.0)
    rotation = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    inverse = quaternion_conjugate_xyzw(rotation)
    reference = [
        ReferenceKinematicSample(
            timestamp, _linear_velocity(timestamp), _angular_velocity(timestamp)
        )
        for timestamp in (index * 0.01 for index in range(701))
    ]
    measurements = []
    for index in range(120):
        timestamp = 0.3 + index * 0.05
        reference_time = timestamp + time_offset
        lever_velocity = _cross(_angular_velocity(reference_time), translation)
        body_velocity = tuple(
            value + lever
            for value, lever in zip(_linear_velocity(reference_time), lever_velocity, strict=True)
        )
        radar_velocity = rotate_vector_xyzw(inverse, body_velocity)
        measurements.append(
            RadarVelocityMeasurement(f"scan-{index:03d}", timestamp, radar_velocity)
        )
    return measurements, reference, rotation


def test_solver_recovers_lever_arm_clock_offset_and_holdout() -> None:
    measurements, reference, rotation = _synthetic_problem()

    result = RadarSpatiotemporalLeverArmSolver().solve(
        measurements,
        reference,
        rotation,
        RadarSpatiotemporalLeverArmOptions(
            max_abs_time_offset_sec=0.05,
            time_offset_step_sec=0.001,
            holdout_ratio=0.25,
            split_seed=19,
            known_bad_margin_mps=0.005,
        ),
    )

    assert result.status == "converged"
    assert result.translation_body_radar_m == pytest.approx((0.42, -0.18, 0.27), abs=1e-8)
    assert result.time_offset_sec == pytest.approx(0.023, abs=1e-12)
    assert result.train_rmse_mps is not None and result.train_rmse_mps < 1e-10
    assert result.holdout_rmse_mps is not None and result.holdout_rmse_mps < 1e-10
    assert result.lever_arm_rank == 3
    assert result.time_objective_curvature_mps2_per_sec2 is not None
    assert set(result.train_measurement_ids).isdisjoint(result.holdout_measurement_ids)
    assert len(result.probes) == 8
    assert all(probe.detectable is True for probe in result.probes)


def test_solver_rejects_single_axis_angular_motion() -> None:
    measurements, reference, rotation = _synthetic_problem()
    degenerate_reference = [
        ReferenceKinematicSample(
            sample.timestamp_sec,
            sample.linear_velocity_body_mps,
            (0.0, 0.0, 0.8),
        )
        for sample in reference
    ]

    result = RadarSpatiotemporalLeverArmSolver().solve(
        measurements,
        degenerate_reference,
        rotation,
        RadarSpatiotemporalLeverArmOptions(
            max_abs_time_offset_sec=0.05, time_offset_step_sec=0.002
        ),
    )

    assert result.status == "degenerate_motion"
    assert result.translation_body_radar_m is None
    assert result.lever_arm_rank == 2


def test_solver_reports_offset_search_boundary() -> None:
    measurements, reference, rotation = _synthetic_problem(time_offset=0.05)

    result = RadarSpatiotemporalLeverArmSolver().solve(
        measurements,
        reference,
        rotation,
        RadarSpatiotemporalLeverArmOptions(
            max_abs_time_offset_sec=0.05, time_offset_step_sec=0.001
        ),
    )

    assert result.status == "offset_at_boundary"
    assert result.time_offset_sec == pytest.approx(0.05)
