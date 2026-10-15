"""Independent residual, Jacobian and threaded-fit checks for compiled planes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from calibrex.core.continuous_time_bspline import CubicBSplineTrajectory, so3_exp_batch
from calibrex.core.continuous_time_bspline_fit import (
    BSplineImuSamples,
    BSplineLidarImuFitOptions,
    BSplinePlaneReturns,
    LidarImuCalibration,
    _apply_step,
    _families,
    _linearize,
    _linearize_terms,
)


def _problem(epoch: float, angle: float) -> tuple[CubicBSplineTrajectory, BSplinePlaneReturns]:
    rng = np.random.default_rng(458)
    rotations = [np.eye(3)]
    for _ in range(8):
        axis = rng.normal(size=3)
        axis *= angle / np.linalg.norm(axis)
        rotations.append(rotations[-1] @ so3_exp_batch(axis[None, :])[0])
    trajectory = CubicBSplineTrajectory(epoch, 0.2, np.asarray(rotations), rng.normal(size=(9, 3)))
    times = epoch + np.array([0.0, 0.2, 1.2, 0.39, 0.2, 0.400001, 0.399999, 0.91])
    planes = BSplinePlaneReturns(
        times, rng.normal(size=(8, 3)), rng.normal(size=(8, 3)), rng.normal(size=(8, 3)), 0.03
    )
    return trajectory, planes


def _residual(trajectory: CubicBSplineTrajectory, planes: BSplinePlaneReturns) -> np.ndarray:
    # Public trajectory evaluator, independent of fitter gather and compiled schedule.
    states = trajectory.evaluate(planes.times_sec)
    world = np.einsum("tij,tj->ti", states.rotation_world_body, planes.points_body_m)
    world += states.position_world_m
    return (
        np.sum(planes.plane_normals_world * (world - planes.plane_points_world_m), axis=1)
        / planes.sigma_m
    )


def _calibration() -> LidarImuCalibration:
    return LidarImuCalibration(
        np.zeros(3), np.zeros(3), np.zeros(3), 0.0, np.zeros(3), np.array([0.0, 0.0, -9.81])
    )


@pytest.mark.parametrize("epoch", [0.0, 1.76e9])
@pytest.mark.parametrize("angle", [0.0, 1e-10, 0.3, np.pi - 1e-3])
def test_plane_residual_and_assembled_jacobian_match_native(epoch: float, angle: float) -> None:
    pytest.importorskip("numba")
    from calibrex.solvers.numba_bspline_plane_adapter import _NumbaPlaneEvaluator

    trajectory, planes = _problem(epoch, angle)
    imu = BSplineImuSamples(np.array([epoch + 0.1]), np.zeros((1, 3)), 0.1)
    families = {"plane": _families(planes, imu, imu)["plane"]}
    evaluator = _NumbaPlaneEvaluator(trajectory, planes, 1)
    try:
        expected, base = _linearize(trajectory, _calibration(), families, ())
        actual, residual = _linearize(trajectory, _calibration(), families, (), evaluator)
        np.testing.assert_allclose(residual, _residual(trajectory, planes), atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(residual, base, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(actual.toarray(), expected.toarray(), atol=1e-9, rtol=1e-11)
    finally:
        evaluator.close()


def test_every_control_axis_matches_independent_finite_differences() -> None:
    pytest.importorskip("numba")
    from calibrex.solvers.numba_bspline_plane_adapter import _NumbaPlaneEvaluator

    trajectory, planes = _problem(0.0, 0.4)
    evaluator = _NumbaPlaneEvaluator(trajectory, planes, 1)
    try:
        _, local = evaluator.evaluate(trajectory, derivatives=True)
        epsilon = 1e-6
        for row, segment in enumerate(evaluator.segment):
            for slot in range(4):
                for axis in range(6):
                    step = np.zeros(6 * trajectory.control_point_count)
                    step[6 * (segment + slot) + axis] = epsilon
                    plus, _ = _apply_step(trajectory, _calibration(), step, ())
                    minus, _ = _apply_step(trajectory, _calibration(), -step, ())
                    numeric = (_residual(plus, planes)[row] - _residual(minus, planes)[row]) / (
                        2 * epsilon
                    )
                    assert local[row, 6 * slot + axis] == pytest.approx(numeric, abs=3e-7, rel=1e-7)
    finally:
        evaluator.close()


def test_parallel_slices_are_repeatable_and_control_edits_remain_visible() -> None:
    pytest.importorskip("numba")
    from calibrex.solvers.numba_bspline_plane_adapter import _NumbaPlaneEvaluator

    trajectory, short = _problem(0.0, 0.3)
    index = np.arange(17001) % len(short.times_sec)
    planes = BSplinePlaneReturns(
        short.times_sec[index],
        short.points_body_m[index],
        short.plane_points_world_m[index],
        short.plane_normals_world[index],
        short.sigma_m,
    )
    serial = _NumbaPlaneEvaluator(trajectory, planes, 1)
    threaded = _NumbaPlaneEvaluator(trajectory, planes, 4)
    try:
        assert "workers=4" in threaded.identity
        for _ in range(3):
            for derivatives in (False, True):
                expected = serial.evaluate(trajectory, derivatives=derivatives)
                actual = threaded.evaluate(trajectory, derivatives=derivatives)
                for left, right in zip(expected, actual, strict=True):
                    np.testing.assert_array_equal(left, right)
        trajectory.rotations[1] = so3_exp_batch(np.array([[0.1, -0.2, 0.3]]))[0]
        trajectory.positions[1] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(
            threaded.evaluate(trajectory, derivatives=False)[0],
            _residual(trajectory, planes),
            atol=1e-10,
            rtol=1e-12,
        )
        with pytest.raises(ValueError, match="time domain"):
            threaded.evaluate(replace(trajectory, start_time_sec=1.0), derivatives=False)
    finally:
        serial.close()
        threaded.close()


def test_referenced_near_pi_edges_keep_the_native_guard() -> None:
    pytest.importorskip("numba")
    from calibrex.solvers.numba_bspline_plane_adapter import _NumbaPlaneEvaluator

    rotations = np.stack([np.eye(3)] * 7)
    rotations[-1] = so3_exp_batch(np.array([[np.pi - 1e-7, 0.0, 0.0]]))[0]
    path = CubicBSplineTrajectory(0.0, 0.1, rotations, np.zeros((7, 3)))
    planes = BSplinePlaneReturns(
        np.array([0.01, 0.09]), np.ones((2, 3)), np.zeros((2, 3)), np.ones((2, 3)), 0.1
    )
    evaluator = _NumbaPlaneEvaluator(path, planes, 1)
    try:
        np.testing.assert_allclose(
            evaluator.evaluate(path, derivatives=False)[0], _residual(path, planes), atol=1e-12
        )
        path.rotations[1] = rotations[-1]
        with pytest.raises(ValueError, match="close to pi"):
            evaluator.evaluate(path, derivatives=True)
    finally:
        evaluator.close()


def test_backend_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="backend"):
        BSplineLidarImuFitOptions(plane_backend="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        BSplineLidarImuFitOptions(plane_workers=0)


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("angle", [0.0, 0.3])
@pytest.mark.parametrize("estimate", [(), ("imu_rotation", "gyro_bias", "lever_arm",
                                         "imu_clock_offset", "accel_bias", "gravity")])
def test_direct_normal_matches_full_native_sparse_system(workers, angle, estimate) -> None:
    pytest.importorskip("numba")
    from calibrex.solvers.numba_bspline_plane_adapter import _NumbaPlaneEvaluator

    trajectory, short = _problem(0.0, angle)
    index = np.arange(17001 if workers == 4 else len(short.times_sec)) % len(short.times_sec)
    planes = BSplinePlaneReturns(
        short.times_sec[index], short.points_body_m[index], short.plane_points_world_m[index],
        short.plane_normals_world[index], short.sigma_m,
    )
    times = np.array([0.1, 0.2, 0.400001, 0.399999, 0.91, 0.2])
    imu = BSplineImuSamples(times, np.ones((len(times), 3)), 0.1)
    families = _families(planes, imu, imu)
    evaluator = _NumbaPlaneEvaluator(trajectory, planes, workers)
    try:
        for clock in (0.0, 0.01):
            calibration = replace(_calibration(), imu_clock_offset_sec=clock)
            jacobian, residual = _linearize(trajectory, calibration, families, estimate)
            entries = _linearize_terms(
                trajectory, calibration, {k: v for k, v in families.items() if k != "plane"},
                estimate, evaluator,
            )
            normal, gradient = evaluator.normal_system(trajectory, entries)
            np.testing.assert_allclose(normal.toarray(), (jacobian.T @ jacobian).toarray(),
                                       rtol=1e-11, atol=1e-7)
            np.testing.assert_allclose(gradient, jacobian.T @ residual, rtol=1e-11, atol=1e-7)
            np.testing.assert_array_equal(normal.toarray(), normal.toarray().T)
    finally:
        evaluator.close()


@pytest.mark.parametrize("aligned", [True, False])
def test_imu_sharing_preserves_jacobian_and_is_scoped_to_each_linearization(
    monkeypatch, aligned,
) -> None:
    pytest.importorskip("numba")
    from calibrex.core import continuous_time_bspline_fit as fit
    from calibrex.solvers.numba_bspline_plane_adapter import _NumbaPlaneEvaluator

    trajectory, planes = _problem(0.0, 0.3)
    times = np.array([0.1, 0.2, 0.400001, 0.399999, 0.91, 0.2])
    gyro = BSplineImuSamples(times, np.zeros((len(times), 3)), 0.1)
    accel = BSplineImuSamples(times + (0 if aligned else 2e-5), gyro.values, 0.1)
    families = {k: v for k, v in _families(planes, gyro, accel).items() if k != "plane"}
    estimate = ("imu_rotation", "gyro_bias", "lever_arm", "imu_clock_offset",
                "accel_bias", "gravity")
    evaluator = _NumbaPlaneEvaluator(trajectory, planes, 1)
    calls = []
    original = fit._gather

    def gather(*args, **kwargs):
        calls.append(kwargs.get("derivatives", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(fit, "_gather", gather)
    try:
        for clock in (0.0, 0.01):
            trajectory.positions[1] += [0.1, -0.2, 0.3]
            calibration = replace(_calibration(), imu_clock_offset_sec=clock)
            expected, base = _linearize(trajectory, calibration, families, estimate)
            calls.clear()
            actual, residual = _linearize(trajectory, calibration, families, estimate, evaluator)
            assert len(calls) == (3 if aligned else 6)
            np.testing.assert_array_equal(actual.toarray(), expected.toarray())
            np.testing.assert_array_equal(residual, base)
    finally:
        evaluator.close()
