"""Real-device comparisons with native derivatives, systems and complete fits."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csc_matrix, eye

pytest.importorskip("cupy")
import cupy as cp

from calibrex.core import continuous_time_bspline_fit as fit
from calibrex.core.continuous_time_bspline import CubicBSplineTrajectory, so3_exp_batch
from calibrex.solvers import cuda_bspline_plane_adapter as adapter
from calibrex.solvers.cuda_bspline_plane_adapter import _CudaPlaneEvaluator


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        count = 0
    if not count:
        pytest.skip("requires a CUDA device")


def _problem(*, epoch: float = 0., scale: float = .2) -> fit.BSplineLidarImuProblem:
    rng = np.random.default_rng(4206)
    rotations = np.broadcast_to(np.eye(3), (10, 3, 3)).copy()
    for i in range(1, len(rotations)):
        v = rng.normal(size=(1, 3))
        v *= scale / np.maximum(np.linalg.norm(v), 1e-30)
        rotations[i] = rotations[i-1] @ so3_exp_batch(v)[0]
    path = CubicBSplineTrajectory(epoch, .1, rotations, rng.normal(0, .1, (10, 3)))
    times = epoch + rng.uniform(.002, .698, 500)
    times[:8] = epoch + np.array([0., .1, .7, .099999, .100001, .1, .5, .3])
    states = path.evaluate(times)
    points = rng.normal(size=(500, 3))
    world = np.einsum("tij,tj->ti", states.rotation_world_body, points) + states.position_world_m
    planes = fit.BSplinePlaneReturns(times, points, world + rng.normal(0, .001, world.shape),
                                    rng.normal(size=(500, 3)), .02)
    imu_times = epoch + np.linspace(.02, .68, 140)
    states = path.evaluate(imu_times)
    gravity = np.array([0., 0., -9.81])
    force = np.einsum("tji,tj->ti", states.rotation_world_body,
                      states.linear_acceleration_world_m_s2 - gravity)
    gyro = fit.BSplineImuSamples(imu_times, states.angular_velocity_body_rad_s, .01)
    accel = fit.BSplineImuSamples(imu_times.copy(), force, .1)
    calibration = fit.LidarImuCalibration(np.zeros(3), np.full(3, .001), np.zeros(3), 0.,
                                         np.zeros(3), gravity)
    return fit.BSplineLidarImuProblem(path, planes, gyro, accel, calibration,
                                     gravity_magnitude_m_s2=9.81)


@pytest.mark.parametrize("scale", [0., 1e-10, .2, 3.1])
@pytest.mark.parametrize("epoch", [0., 1.76e9])
def test_plane_residuals_jacobian_and_banded_system_match_native(
    scale: float, epoch: float,
) -> None:
    p = _problem(scale=scale, epoch=epoch)
    path = p.initial_trajectory
    families = {"plane": fit._families(p.planes, p.gyro, p.accel)["plane"]}
    evaluator = _CudaPlaneEvaluator(path, p.planes)
    residual, local = evaluator.evaluate(path, derivatives=True)
    expected, base = fit._linearize(path, p.initial_calibration, families, ())
    actual, device_base = fit._linearize(path, p.initial_calibration, families, (),
                                        plane_evaluator=evaluator)
    np.testing.assert_allclose(residual, base, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(device_base, base, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=2e-11, atol=2e-10)
    assert local.shape == (500, 24)
    size = 6 * path.control_point_count
    imu_normal = eye(size, format="csc") * .3
    imu_gradient = np.arange(size, dtype=float) * .01
    normal, gradient = evaluator.normal_system(path, imu_normal, imu_gradient)
    np.testing.assert_allclose(normal.toarray(), (expected.T @ expected + imu_normal).toarray(),
                               rtol=2e-11, atol=2e-8)
    np.testing.assert_allclose(gradient, expected.T @ base + imu_gradient,
                               rtol=2e-10, atol=2e-8)
    again, repeated = evaluator.normal_system(path, imu_normal, imu_gradient)
    np.testing.assert_array_equal(again.toarray(), normal.toarray())
    np.testing.assert_array_equal(repeated, gradient)
    assert evaluator.statistics()["gpu_plane_derivative_calls"] == 1
    assert evaluator.statistics()["gpu_plane_normal_calls"] == 2


def test_device_plane_derivatives_match_independent_central_differences() -> None:
    p = _problem()
    path = p.initial_trajectory
    planes = replace(p.planes, times_sec=p.planes.times_sec[:20],
                     points_body_m=p.planes.points_body_m[:20],
                     plane_points_world_m=p.planes.plane_points_world_m[:20],
                     plane_normals_world=p.planes.plane_normals_world[:20])
    evaluator = _CudaPlaneEvaluator(path, planes)
    families = {"plane": fit._families(planes, p.gyro, p.accel)["plane"]}
    jacobian, _base = fit._linearize(path, p.initial_calibration, families, (),
                                    plane_evaluator=evaluator)
    dense = jacobian.toarray()
    epsilon = 1e-6
    for control in range(path.control_point_count):
        for axis in range(6):
            step = np.zeros(6 * path.control_point_count)
            step[6 * control + axis] = epsilon
            plus, _ = fit._apply_step(path, p.initial_calibration, step, ())
            minus, _ = fit._apply_step(path, p.initial_calibration, -step, ())
            family = families["plane"]
            above = family.residual(fit._gather(plus, family.times), p.initial_calibration)
            below = family.residual(fit._gather(minus, family.times), p.initial_calibration)
            numeric = ((above-below)/(2*epsilon)).reshape(-1)
            np.testing.assert_allclose(dense[:, 6 * control + axis], numeric,
                                       rtol=1e-6, atol=2e-7)


@pytest.mark.parametrize("iterations", [0, 1, 4])
@pytest.mark.parametrize("prior", [None, 9.81])
def test_complete_fit_keeps_physical_results_and_execution_provenance(
    iterations: int, prior: float | None,
) -> None:
    problem = replace(_problem(), gravity_magnitude_m_s2=prior)
    settings = fit.BSplineLidarImuFitOptions(max_iterations=iterations)
    native = fit.fit_bspline_lidar_imu(problem, settings)
    device = fit.fit_bspline_lidar_imu(problem, replace(settings, plane_backend="cuda"))
    assert (device.status, device.iterations, device.parameter_count, device.residual_count) == (
        native.status, native.iterations, native.parameter_count, native.residual_count,
    )
    assert device.calibration_parameter_names == native.calibration_parameter_names
    for name in ("initial_cost", "final_cost", "residual_variance"):
        assert getattr(device, name) == pytest.approx(getattr(native, name), rel=1e-8, abs=1e-9)
    for block in fit.CALIBRATION_BLOCKS:
        np.testing.assert_allclose(device.calibration.vector(block),
                                   native.calibration.vector(block),
                                   rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(device.trajectory.rotations, native.trajectory.rotations,
                               rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(device.trajectory.positions, native.trajectory.positions,
                               rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(device.calibration_covariance, native.calibration_covariance,
                               rtol=1e-5, atol=1e-11)
    assert device.execution_backend.startswith("calibrex.cuda_bspline_planes/v0.1;")
    assert device.execution_statistics["gpu_plane_normal_calls"] == iterations
    assert device.execution_statistics["gpu_plane_derivative_rows"] > 0
    assert native.execution_backend == "numpy" and native.execution_statistics == {}


def test_schedule_domain_and_unused_near_pi_edges_keep_native_guards() -> None:
    p = _problem()
    path = p.initial_trajectory
    planes = replace(p.planes, times_sec=np.zeros_like(p.planes.times_sec))
    rotations = path.rotations.copy()
    rotations[-1] = rotations[-2] @ so3_exp_batch(np.array([[np.pi-1e-7, 0., 0.]]))[0]
    path = replace(path, rotations=rotations)
    evaluator = _CudaPlaneEvaluator(path, planes)
    evaluator.evaluate(path, derivatives=True)
    with pytest.raises(ValueError, match="time domain"):
        evaluator.evaluate(replace(path, knot_spacing_sec=.2), derivatives=True)
    referenced = replace(planes, times_sec=np.full(len(planes.times_sec), .7))
    with pytest.raises(ValueError, match="too close to pi"):
        _CudaPlaneEvaluator(path, referenced).evaluate(path, derivatives=True)


def test_pose_updates_invalidate_device_residual_and_normal_cache() -> None:
    p = _problem()
    path = p.initial_trajectory
    evaluator = _CudaPlaneEvaluator(path, p.planes)
    before = evaluator.evaluate(path, derivatives=True)[0].copy()
    moved = replace(path, positions=path.positions + .02)
    after = evaluator.evaluate(moved, derivatives=False)[0]
    assert not np.array_equal(before, after)
    native = fit._families(p.planes, p.gyro, p.accel)["plane"]
    np.testing.assert_allclose(after, native.residual(fit._gather(moved, native.times),
                                                    p.initial_calibration).reshape(-1), atol=1e-11)
    evaluator.normal_system(moved, csc_matrix((60, 60)), np.zeros(60))
    np.testing.assert_array_equal(evaluator.evaluate(path, derivatives=True)[0], before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_kernel_buffers_capture_fp64_row_major_measurements(dtype: type, layout: str) -> None:
    p = _problem()

    def copied(a: np.ndarray) -> np.ndarray:
        if layout == "strided":
            result = np.empty((len(a), 6), dtype=dtype)
            result[:, ::2] = a
            return result[:, ::2]
        return np.array(a, dtype=dtype, order=layout)

    planes = replace(p.planes, points_body_m=copied(p.planes.points_body_m),
                     plane_points_world_m=copied(p.planes.plane_points_world_m),
                     plane_normals_world=copied(p.planes.plane_normals_world))
    evaluator = _CudaPlaneEvaluator(p.initial_trajectory, planes)
    family = fit._families(planes, p.gyro, p.accel)["plane"]
    expected = family.residual(fit._gather(p.initial_trajectory, family.times),
                               p.initial_calibration).reshape(-1)
    np.testing.assert_allclose(evaluator.evaluate(p.initial_trajectory, derivatives=True)[0],
                               expected, rtol=1e-10, atol=1e-11)


def test_payload_budget_is_checked_before_any_device_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _problem()
    monkeypatch.setattr(adapter, "_PAYLOAD_BUDGET", 0)

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("uploaded an over-budget plane schedule")

    monkeypatch.setattr(cp, "asarray", unexpected)
    with pytest.raises(ValueError, match="payload budget"):
        _CudaPlaneEvaluator(p.initial_trajectory, p.planes)
