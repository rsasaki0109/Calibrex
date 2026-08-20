"""Tests for continuous-time LiDAR point-to-plane factors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibrex.cli.main import main
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryPointToPlaneMeasurement,
    _point_to_plane_residual,
    fit_continuous_trajectory,
    interpolate_pose_at,
)
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp, se3_log
from calibrex.core.validation import validate_file
from calibrex.evaluation.continuous_time_lidar_point_to_plane import (
    run_synthetic_lidar_point_to_plane_recovery,
)

rng = np.random.default_rng(21)


def test_point_to_plane_jacobian_matches_finite_differences() -> None:
    timestamps = (0.0, 1.0)
    left = se3_exp(SE3.identity(), np.array([0.1, -0.05, 0.02, 0.03, -0.02, 0.04]))
    right = se3_exp(SE3.identity(), np.array([0.4, 0.1, 0.05, 0.01, 0.02, 0.08]))
    measurement = TrajectoryPointToPlaneMeasurement(
        measurement_id="plane0",
        timestamp_sec=0.4,
        point_body_m=(0.2, -0.1, 0.3),
        plane_point_world_m=(0.0, 0.0, 0.0),
        plane_normal_world=(0.1, 0.2, 0.97),
    )
    residual, jacobian_left, jacobian_right = _point_to_plane_residual(
        [left, right], timestamps, 0, measurement
    )
    numeric_left = np.zeros((1, 6))
    numeric_right = np.zeros((1, 6))
    for axis in range(6):
        epsilon = np.zeros(6)
        epsilon[axis] = 1.0e-7
        left_residual, _, _ = _point_to_plane_residual(
            [se3_exp(left, epsilon), right], timestamps, 0, measurement
        )
        right_residual, _, _ = _point_to_plane_residual(
            [left, se3_exp(right, epsilon)], timestamps, 0, measurement
        )
        numeric_left[0, axis] = (float(left_residual[0]) - float(residual[0])) / 1.0e-7
        numeric_right[0, axis] = (
            float(right_residual[0]) - float(residual[0])
        ) / 1.0e-7
    assert np.allclose(jacobian_left, numeric_left, atol=2.0e-5)
    assert np.allclose(jacobian_right, numeric_right, atol=2.0e-5)


def test_point_to_plane_fit_recovers_perturbed_knots() -> None:
    timestamps = tuple(float(index) for index in range(5))
    truth = tuple(
        se3_exp(
            SE3.identity(),
            np.concatenate(
                [
                    np.array([0.2 * index, 0.05 * index, 0.01 * index]),
                    np.array([0.0, 0.0, 0.03 * index]),
                ]
            ),
        )
        for index in range(5)
    )
    planes = (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.5, 0.0), (0.0, 1.0, 0.0)),
        ((0.8, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    measurements: list[TrajectoryPointToPlaneMeasurement] = []
    for sample in range(60):
        timestamp = float(rng.uniform(0.05, 3.95))
        plane_point, plane_normal = planes[sample % 3]
        pose = interpolate_pose_at(truth, timestamps, timestamp)
        world = np.asarray(plane_point, dtype=float)
        if plane_normal == (0.0, 0.0, 1.0):
            world = world + np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.0])
        elif plane_normal == (0.0, 1.0, 0.0):
            world = world + np.array([rng.uniform(-0.3, 0.3), 0.0, rng.uniform(-0.3, 0.3)])
        else:
            world = world + np.array([0.0, rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)])
        body = pose.inverse().transform_point(tuple(float(value) for value in world))
        measurements.append(
            TrajectoryPointToPlaneMeasurement(
                measurement_id=f"p{sample}",
                timestamp_sec=timestamp,
                point_body_m=tuple(float(value) for value in body),
                plane_point_world_m=plane_point,
                plane_normal_world=plane_normal,
            )
        )
    initial = tuple(se3_exp(knot, rng.normal(0.0, 0.03, 6)) for knot in truth)
    result = fit_continuous_trajectory(
        ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=timestamps,
            initial_knot_poses=initial,
            point_to_plane_measurements=tuple(measurements),
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=80),
    )
    assert result.status in {"converged", "max_iterations"}
    errors = [
        float(np.max(np.abs(se3_log(estimate, knot))))
        for estimate, knot in zip(result.knot_poses, truth, strict=True)
    ]
    assert max(errors) < 0.05
    assert result.final_point_to_plane_rmse is not None
    assert result.final_point_to_plane_rmse < 0.01


def test_synthetic_recovery_holdout_and_known_bad_control(tmp_path: Path) -> None:
    artifact = run_synthetic_lidar_point_to_plane_recovery(seed=20260820)
    output = tmp_path / "recovery.yaml"
    artifact.save(output)
    assert validate_file(output, kind="continuous-time-lidar-point-to-plane").valid
    assert artifact.policy_status == "pass"
    assert artifact.holdout_measurement_count >= 1
    assert artifact.known_bad_rmse_delta_m > 0.05
    assert artifact.provenance.seed == 20260820
    assert artifact.interpolation == "screw_linear"


def test_cli_recover_point_to_plane(tmp_path: Path) -> None:
    output = tmp_path / "recovery.yaml"
    exit_code = main(
        [
            "trajectory",
            "recover-point-to-plane",
            "--output",
            str(output),
            "--seed",
            "20260820",
            "--json",
        ]
    )
    assert exit_code == 0
    assert validate_file(output).valid
