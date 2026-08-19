"""Unit tests for the continuous-time manifold foundation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calibrex.core.continuous_time_contract import (
    ContinuousTimeClockSemantics,
    ContinuousTimeKnotDomain,
    ContinuousTimeKnotPose,
    ContinuousTimeTrajectoryContract,
    ContinuousTimeTrajectoryProvenance,
    load_continuous_time_trajectory,
)
from calibrex.core.continuous_time_fit_artifacts import (
    ContinuousTimeTrajectoryMeasurements,
    TrajectoryMeasurementProvenance,
    TrajectoryPointMeasurementArtifact,
    TrajectoryPoseMeasurementArtifact,
    load_continuous_time_measurements,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryPointMeasurement,
    TrajectoryPoseMeasurement,
    fit_continuous_trajectory,
)
from calibrex.core.geometry import SE3
from calibrex.core.result import TransformResult
from calibrex.core.se3_manifold import (
    interpolate_screw_jacobians,
    point_transform_jacobian,
    se3_exp,
    se3_left_jacobian,
    se3_log,
)

rng = np.random.default_rng(7)


def _random_se3() -> SE3:
    translation = rng.normal(0.0, 0.5, 3)
    rotation = rng.normal(0.0, 1.0, 3)
    return se3_exp(SE3.identity(), np.concatenate([translation, rotation]))


def test_se3_exp_log_round_trip() -> None:
    for _ in range(10):
        origin = _random_se3()
        target = _random_se3()
        delta = se3_log(origin, target)
        recovered = se3_exp(origin, delta)
        assert np.allclose(
            recovered.translation_m, target.translation_m, atol=1.0e-9
        )
        assert np.allclose(
            recovered.rotation_quat_xyzw, target.rotation_quat_xyzw, atol=1.0e-9
        )


def test_point_transform_jacobian_matches_finite_differences() -> None:
    for _ in range(5):
        transform = _random_se3()
        point = rng.normal(0.0, 1.0, 3)
        jacobian = point_transform_jacobian(transform, point)
        base = np.asarray(transform.transform_point(point), dtype=float)
        numeric = np.zeros((3, 6))
        for axis in range(6):
            epsilon = np.zeros(6)
            epsilon[axis] = 1.0e-7
            perturbed = se3_exp(transform, epsilon).transform_point(point)
            numeric[:, axis] = (np.asarray(perturbed) - base) / 1.0e-7
        assert np.allclose(jacobian, numeric, atol=1.0e-5)


def test_se3_left_jacobian_satisfies_expansion() -> None:
    for _ in range(5):
        v = rng.normal(0.0, 0.3, 3)
        w = rng.normal(0.0, 1.0, 3)
        jacobian = se3_left_jacobian(v, w)
        xi = np.concatenate([v, w])
        for axis in range(6):
            epsilon = np.zeros(6)
            epsilon[axis] = 1.0e-7
            left = se3_exp(SE3.identity(), xi + epsilon)
            right = se3_exp(se3_exp(SE3.identity(), xi), jacobian @ epsilon)
            delta = se3_log(left, right)
            assert np.max(np.abs(delta)) < 1.0e-6


def test_interpolate_screw_jacobians_match_finite_differences() -> None:
    for alpha in (0.0, 0.25, 0.5, 0.9, 1.0):
        for _ in range(3):
            left = _random_se3()
            right = _random_se3()
            jacobian_left, jacobian_right = interpolate_screw_jacobians(
                left, right, alpha
            )
            interpolated = se3_exp(left, alpha * se3_log(left, right))
            numeric_left = np.zeros((6, 6))
            numeric_right = np.zeros((6, 6))
            for axis in range(6):
                epsilon = np.zeros(6)
                epsilon[axis] = 1.0e-7
                left_perturbed = se3_exp(left, epsilon)
                left_result = se3_exp(
                    left_perturbed, alpha * se3_log(left_perturbed, right)
                )
                numeric_left[:, axis] = (
                    se3_log(interpolated, left_result) / 1.0e-7
                )
                right_perturbed = se3_exp(right, epsilon)
                right_result = se3_exp(
                    left, alpha * se3_log(left, right_perturbed)
                )
                numeric_right[:, axis] = (
                    se3_log(interpolated, right_result) / 1.0e-7
                )
            assert np.allclose(jacobian_left, numeric_left, atol=2.0e-4)
            assert np.allclose(jacobian_right, numeric_right, atol=2.0e-4)


def _contract(tmp_path: Path, *, interpolation: str = "screw_linear/v0.1") -> Path:
    path = tmp_path / "trajectory.yaml"
    contract = ContinuousTimeTrajectoryContract(
        trajectory_id="traj1",
        world_frame="map",
        body_frame="base",
        interpolation=interpolation,  # type: ignore[arg-type]
        knot_domain=ContinuousTimeKnotDomain(
            minimum_time_sec=0.0,
            maximum_time_sec=3.0,
            span_sec=3.0,
        ),
        clock=ContinuousTimeClockSemantics(),
        knots=[
            ContinuousTimeKnotPose(
                timestamp_sec=float(t),
                transform_world_body=TransformResult(
                    parent="map",
                    child="base",
                    translation_m=[0.3 * t, 0.1 * t, 0.02 * t],
                    rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                ),
            )
            for t in range(4)
        ],
        provenance=ContinuousTimeTrajectoryProvenance(
            generator="test",
            generator_version="0.1",
            source_sha256="0" * 64,
        ),
    )
    contract.save(path)
    return path


def test_contract_round_trip_and_knot_domain_validity(tmp_path: Path) -> None:
    path = _contract(tmp_path)
    loaded = load_continuous_time_trajectory(path)
    assert loaded.knot_count == 4
    assert loaded.interpolation == "screw_linear/v0.1"
    assert loaded.knot_domain.span_sec == 3.0
    with pytest.raises(ValueError, match="strictly increasing"):
        bad = loaded.model_copy(deep=True)
        bad.knots[-1].timestamp_sec = 2.0
        bad.knot_domain = ContinuousTimeKnotDomain(
            minimum_time_sec=0.0,
            maximum_time_sec=3.0,
            span_sec=3.0,
        )
        ContinuousTimeTrajectoryContract.model_validate(bad.model_dump())


def test_contract_converts_to_piecewise_adapter(tmp_path: Path) -> None:
    path = _contract(tmp_path, interpolation="piecewise_linear_slerp/v0.1")
    loaded = load_continuous_time_trajectory(path)
    piecewise = loaded.to_piecewise()
    assert piecewise.minimum_time_sec == 0.0
    assert piecewise.maximum_time_sec == 3.0
    assert len(piecewise.poses) == 4
    assert piecewise.pose_at(1.5).translation_m == pytest.approx((0.45, 0.15, 0.03))


def test_screw_contract_rejects_piecewise_conversion(tmp_path: Path) -> None:
    path = _contract(tmp_path)
    loaded = load_continuous_time_trajectory(path)
    with pytest.raises(ValueError, match="cannot be converted"):
        loaded.to_piecewise()


def _measurements(tmp_path: Path) -> Path:
    path = tmp_path / "measurements.yaml"
    measurements = ContinuousTimeTrajectoryMeasurements(
        measurements_id="measurements1",
        world_frame="map",
        body_frame="base",
        pose_measurements=[
            TrajectoryPoseMeasurementArtifact(
                measurement_id=f"pose{index}",
                timestamp_sec=0.5 + index,
                pose_world_body=TransformResult(
                    parent="map",
                    child="base",
                    translation_m=[0.3 * (0.5 + index), 0.1 * (0.5 + index), 0.02 * (0.5 + index)],
                    rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                ),
                weight=4.0,
            )
            for index in range(3)
        ],
        point_measurements=[
            TrajectoryPointMeasurementArtifact(
                measurement_id=f"point{index}",
                timestamp_sec=0.5 + index,
                point_body_m=[0.1, 0.2, 0.3],
                target_world_m=[0.3 * (0.5 + index) + 0.1, 0.2 * (0.5 + index), 0.07],
                weight=1.0,
            )
            for index in range(3)
        ],
        provenance=TrajectoryMeasurementProvenance(
            generator="test",
            generator_version="0.1",
            source_sha256="0" * 64,
        ),
    )
    measurements.save(path)
    return path


def test_measurements_round_trip(tmp_path: Path) -> None:
    path = _measurements(tmp_path)
    loaded = load_continuous_time_measurements(path)
    assert loaded.measurements_id == "measurements1"
    assert len(loaded.pose_measurements) == 3
    assert len(loaded.point_measurements) == 3


def test_fit_recovers_perturbed_knots() -> None:
    timestamps = tuple(float(t) for t in range(6))
    knots = []
    for index in range(6):
        rotation = np.array([0.05, -0.02 + 0.03 * index, 0.01 * index])
        translation = np.array([0.3 * index, 0.1 * index, 0.02 * index])
        knots.append(
            se3_exp(
                SE3.identity(),
                np.concatenate([translation, rotation]),
            )
        )
    points: list[TrajectoryPointMeasurement] = []
    poses: list[TrajectoryPoseMeasurement] = []
    for _ in range(80):
        t = rng.uniform(0.05, 4.95)
        interval = int(t)
        alpha = t - interval
        pose = se3_exp(knots[interval], alpha * se3_log(knots[interval], knots[interval + 1]))
        body = rng.normal(0.0, 0.2, 3)
        target = np.asarray(pose.transform_point(body), dtype=float) + rng.normal(0.0, 0.005, 3)
        points.append(
            TrajectoryPointMeasurement(
                measurement_id=f"p{len(points)}",
                timestamp_sec=float(t),
                point_body_m=tuple(float(x) for x in body),
                target_world_m=tuple(float(x) for x in target),
                weight=1.0,
            )
        )
        pose_noise = rng.normal(0.0, 0.003, 6)
        poses.append(
            TrajectoryPoseMeasurement(
                measurement_id=f"pose{len(poses)}",
                timestamp_sec=float(t),
                pose_world_body=se3_exp(pose, pose_noise),
                weight=4.0,
            )
        )
    initial = [se3_exp(k, rng.normal(0.0, 0.05, 6)) for k in knots]
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=timestamps,
        initial_knot_poses=tuple(initial),
        point_measurements=tuple(points),
        pose_measurements=tuple(poses),
        interpolation="screw_linear",
    )
    result = fit_continuous_trajectory(
        problem, ContinuousTimeTrajectoryFitOptions(max_iterations=60)
    )
    assert result.status in {"converged", "max_iterations"}
    errors = [
        np.max(np.abs(se3_log(estimate, truth)))
        for estimate, truth in zip(result.knot_poses, knots, strict=True)
    ]
    assert max(errors) < 0.03


def test_fit_is_deterministic_given_seed() -> None:
    timestamps = tuple(float(t) for t in range(4))
    knots = tuple(
        se3_exp(SE3.identity(), np.concatenate([np.array([0.3 * i, 0.1 * i, 0.0]), np.zeros(3)]))
        for i in range(4)
    )
    points = tuple(
        TrajectoryPointMeasurement(
            measurement_id=f"p{index}",
            timestamp_sec=0.5 + index,
            point_body_m=(0.1, 0.2, 0.3),
            target_world_m=(0.6 + 0.3 * index, 0.3, 0.07),
            weight=1.0,
        )
        for index in range(3)
    )
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=timestamps,
        initial_knot_poses=knots,
        point_measurements=points,
        interpolation="screw_linear",
    )
    first = fit_continuous_trajectory(problem)
    second = fit_continuous_trajectory(problem)
    assert first.status == second.status
    assert first.iterations == second.iterations
    for left, right in zip(first.knot_poses, second.knot_poses, strict=True):
        assert np.allclose(left.translation_m, right.translation_m)
        assert np.allclose(left.rotation_quat_xyzw, right.rotation_quat_xyzw)


def test_fit_rejects_measurement_outside_domain() -> None:
    timestamps = (0.0, 1.0, 2.0)
    knots = tuple(
        se3_exp(SE3.identity(), np.concatenate([np.array([0.3 * i, 0.0, 0.0]), np.zeros(3)]))
        for i in range(3)
    )
    points = (
        TrajectoryPointMeasurement(
            measurement_id="out",
            timestamp_sec=5.0,
            point_body_m=(0.1, 0.2, 0.3),
            target_world_m=(0.6, 0.3, 0.07),
        ),
    )
    problem = ContinuousTimeTrajectoryFitProblem(
        knot_timestamps=timestamps,
        initial_knot_poses=knots,
        point_measurements=points,
        interpolation="screw_linear",
    )
    with pytest.raises(ValueError, match="outside the knot domain"):
        fit_continuous_trajectory(problem)


def test_fit_artifact_is_schema_valid(tmp_path: Path) -> None:
    from calibrex.core.continuous_time_fit_artifacts import (
        CONTINUOUS_TIME_FIT_SCHEMA_VERSION,
    )
    from calibrex.core.validation import validate_file

    trajectory_path = _contract(tmp_path)
    measurements_path = _measurements(tmp_path)
    from calibrex.evaluation.continuous_time_trajectory_fit import (
        run_continuous_time_trajectory_fit,
    )

    result = run_continuous_time_trajectory_fit(
        trajectory_path,
        measurements_path,
        result_id="fit1",
        command=["pytest"],
    )
    output = tmp_path / "fit.yaml"
    result.save(output)
    verification = validate_file(output, kind="continuous-time-trajectory-fit")
    assert verification.valid
    assert result.schema_version == CONTINUOUS_TIME_FIT_SCHEMA_VERSION
    assert result.trajectory_id == "traj1"
    assert result.interpolation == "screw_linear"
    assert len(result.knots) == 4
    assert len(result.provenance.trajectory_sha256) == 64
    assert len(result.provenance.measurements_sha256) == 64