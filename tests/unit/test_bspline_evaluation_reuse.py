"""Shared spline work preserves complete numerical results and bounded lifetimes."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from calibrex.core import continuous_time_bspline_fit as fit
from calibrex.core.continuous_time_bspline import CubicBSplineTrajectory
from calibrex.core.se3_manifold import rotation_vector_to_matrix


@pytest.fixture
def problem() -> fit.BSplineLidarImuProblem:
    rng = np.random.default_rng(375)
    rotations = [np.eye(3)]
    for _ in range(9):
        rotations.append(rotations[-1] @ rotation_vector_to_matrix(rng.normal(0., .08, 3)))
    path = CubicBSplineTrajectory(0., .1, np.array(rotations),
                                np.cumsum(rng.normal(0., .03, (10, 3)), axis=0))
    times = np.linspace(.02, path.end_time_sec - .02, 90)
    # Include both sides of a knot in the finite-difference clock column.
    times[10:13] = [.1 - 2e-7, .1, .1 + 2e-7]
    points = rng.normal(size=(90, 3))
    states = path.evaluate(times)
    world = np.einsum("tij,tj->ti", states.rotation_world_body, points) + states.position_world_m
    planes = fit.BSplinePlaneReturns(times, points, world, rng.normal(size=(90, 3)), .02)
    gravity = np.array([0., 0., -9.81])
    force = np.einsum("tji,tj->ti", states.rotation_world_body,
                      states.linear_acceleration_world_m_s2 - gravity)
    gyro = fit.BSplineImuSamples(times.copy(), states.angular_velocity_body_rad_s, .01)
    accel = fit.BSplineImuSamples(times.copy(), force, .1)
    calibration = fit.LidarImuCalibration(np.zeros(3), np.full(3, .001), np.zeros(3), 0.,
                                         np.zeros(3), gravity)
    return fit.BSplineLidarImuProblem(path, planes, gyro, accel, calibration,
                                     gravity_magnitude_m_s2=9.81)


@pytest.mark.parametrize("different_times", [False, True])
def test_shared_cost_jacobian_and_clock_column_match_native_exactly(
    problem: fit.BSplineLidarImuProblem, different_times: bool,
) -> None:
    if different_times:
        problem = replace(problem, accel=replace(problem.accel,
                                                times_sec=problem.accel.times_sec + 1e-10))
    families = fit._families(problem.planes, problem.gyro, problem.accel)
    path, calibration = problem.initial_trajectory, problem.initial_calibration
    cache = fit._EvaluationCache(families)
    assert fit._cost(path, calibration, families, cache=cache) == fit._cost(
        path, calibration, families,
    )
    actual, residual = fit._linearize(path, calibration, families, fit.CALIBRATION_BLOCKS,
                                      cache=cache)
    expected, native_residual = fit._linearize(path, calibration, families, fit.CALIBRATION_BLOCKS)
    for a, b in ((actual.indptr, expected.indptr), (actual.indices, expected.indices),
                 (actual.data, expected.data), (residual, native_residual)):
        np.testing.assert_array_equal(a, b)
    assert len(cache.batches) == (7 if different_times else 4)


def test_cache_refreshes_rejected_trajectories_clock_changes_and_calibration_values(
    problem: fit.BSplineLidarImuProblem,
) -> None:
    families = fit._families(problem.planes, problem.gyro, problem.accel)
    cache = fit._EvaluationCache(families)
    path, calibration = problem.initial_trajectory, problem.initial_calibration
    family = families["gyro"]
    original = cache.residual(path, calibration, family).copy()
    other = replace(calibration, gyro_bias_rad_s=np.full(3, .003))
    actual = cache.residual(path, other, family)
    np.testing.assert_array_equal(actual, family.residual(fit._gather(path, family.times), other))
    assert not np.array_equal(actual, original)
    shifted = replace(calibration, imu_clock_offset_sec=.001)
    np.testing.assert_array_equal(cache.residual(path, shifted, family),
                                  family.residual(fit._gather(path, family.body_times(shifted)),
                                                  shifted))
    candidate = replace(path, rotations=path.rotations.copy(), positions=path.positions + .1)
    candidate.rotations[1] = candidate.rotations[1] @ rotation_vector_to_matrix(
        np.array([.01, -.02, .03]),
    )
    changed = cache.residual(candidate, calibration, family)
    np.testing.assert_array_equal(changed, family.residual(fit._gather(candidate, family.times),
                                                          calibration))
    assert not np.array_equal(changed, original)
    assert len(cache.batches) == 1 and cache.trajectory is candidate
    np.testing.assert_array_equal(cache.residual(path, calibration, family), original)
    assert len(cache.batches) == 1 and cache.trajectory is path


def test_timestamp_and_clock_signed_zero_are_not_conflated(
    problem: fit.BSplineLidarImuProblem,
) -> None:
    times = problem.gyro.times_sec.copy()
    times[0] = 0.
    negative = times.copy()
    negative[0] = -0.
    families = fit._families(problem.planes, replace(problem.gyro, times_sec=times),
                             replace(problem.accel, times_sec=negative))
    cache = fit._EvaluationCache(families)
    assert cache.groups[id(families["gyro"])] != cache.groups[id(families["accel"])]
    path, calibration = problem.initial_trajectory, problem.initial_calibration
    positive = cache.get(path, calibration, families["gyro"])
    negative_clock = cache.get(path, replace(calibration, imu_clock_offset_sec=-0.),
                               families["gyro"])
    assert positive is not negative_clock


def test_pose_only_derivative_order_stays_separate_from_rate_recursion(
    problem: fit.BSplineLidarImuProblem,
) -> None:
    families = fit._families(problem.planes, problem.gyro, problem.accel)
    families["gyro"] = replace(families["gyro"], on_imu_clock=False)
    cache = fit._EvaluationCache(families)
    assert cache.groups[id(families["plane"])] != cache.groups[id(families["gyro"])]
    path, calibration = problem.initial_trajectory, problem.initial_calibration
    actual, a = fit._linearize(path, calibration, families, fit.CALIBRATION_BLOCKS, cache=cache)
    expected, b = fit._linearize(path, calibration, families, fit.CALIBRATION_BLOCKS)
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    np.testing.assert_array_equal(a, b)


def test_retained_workspace_is_bounded_and_over_budget_rows_remain_exact(
    problem: fit.BSplineLidarImuProblem,
) -> None:
    families = fit._families(problem.planes, problem.gyro, problem.accel)
    cache = fit._EvaluationCache(families, max_bytes=1)
    actual, residual = fit._linearize(problem.initial_trajectory, problem.initial_calibration,
                                      families, fit.CALIBRATION_BLOCKS, cache=cache)
    expected, native = fit._linearize(problem.initial_trajectory, problem.initial_calibration,
                                      families, fit.CALIBRATION_BLOCKS)
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    np.testing.assert_array_equal(residual, native)
    assert not cache.batches


def test_workspace_eviction_stays_within_budget_after_derivative_growth(
    problem: fit.BSplineLidarImuProblem,
) -> None:
    families = fit._families(problem.planes, problem.gyro, problem.accel)
    path, calibration = problem.initial_trajectory, problem.initial_calibration
    cache = fit._EvaluationCache(families)
    first = cache.get(path, calibration, families["gyro"])
    cache.max_bytes = first.retained_bytes() + 1024
    for offset in (0., .001, -.001):
        cache.residual(path, replace(calibration, imu_clock_offset_sec=offset), families["gyro"])
        assert sum(b.retained_bytes() for b in cache.batches.values()) <= cache.max_bytes
    fit._linearize(path, calibration, families, fit.CALIBRATION_BLOCKS, cache=cache)
    assert sum(b.retained_bytes() for b in cache.batches.values()) <= cache.max_bytes


def test_invalid_clock_trials_keep_infinite_cost_and_recover(
    problem: fit.BSplineLidarImuProblem,
) -> None:
    families = fit._families(problem.planes, problem.gyro, problem.accel)
    cache = fit._EvaluationCache(families)
    invalid = replace(problem.initial_calibration, imu_clock_offset_sec=100.)
    assert fit._cost(problem.initial_trajectory, invalid, families, cache=cache) == np.inf
    assert fit._cost(
        problem.initial_trajectory, problem.initial_calibration, families, cache=cache,
    ) == fit._cost(problem.initial_trajectory, problem.initial_calibration, families)


def _plain(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


@pytest.mark.parametrize("iterations", [0, 1, 4])
def test_complete_fit_with_prior_keeps_every_result_field_exact(
    problem: fit.BSplineLidarImuProblem, iterations: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Uncached:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def get(self, *args: object) -> None:
            return None

        def trim(self) -> None:
            pass

        def residual(self, trajectory: CubicBSplineTrajectory,
                     calibration: fit.LidarImuCalibration, family: fit._Family) -> np.ndarray:
            return family.residual(fit._gather(trajectory, family.body_times(calibration)),
                                   calibration)

    options = fit.BSplineLidarImuFitOptions(max_iterations=iterations)
    with monkeypatch.context() as scoped:
        scoped.setattr(fit, "_EvaluationCache", Uncached)
        expected = fit.fit_bspline_lidar_imu(problem, options)
    actual = fit.fit_bspline_lidar_imu(problem, options)
    assert _plain(asdict(actual)) == _plain(asdict(expected))


def test_public_residual_calls_observe_measurement_mutations(
    problem: fit.BSplineLidarImuProblem,
) -> None:
    path, calibration = problem.initial_trajectory, problem.initial_calibration
    first = fit.lidar_imu_residuals(path, calibration, problem.planes, problem.gyro, problem.accel)
    problem.gyro.values[:] += .002
    second = fit.lidar_imu_residuals(path, calibration, problem.planes, problem.gyro, problem.accel)
    assert not np.array_equal(first["gyro"], second["gyro"])
    np.testing.assert_array_equal(first["accel"], second["accel"])


@pytest.mark.parametrize("different_times", [False, True])
def test_holdout_session_keeps_all_signed_controls_and_rms_exact_without_unused_work(
    problem: fit.BSplineLidarImuProblem, different_times: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrex.evaluation.lidar_imu_bag_calibration import _controls, _whitened_rms

    if different_times:
        problem = replace(problem, accel=replace(problem.accel,
                                                times_sec=problem.accel.times_sec + 1e-10))
    path, calibration = problem.initial_trajectory, problem.initial_calibration
    expected = fit.lidar_imu_residuals(path, calibration, problem.planes,
                                     problem.gyro, problem.accel)

    def reference(name: str, value: fit.LidarImuCalibration) -> float:
        residual = fit.lidar_imu_residuals(path, value, problem.planes,
                                         problem.gyro, problem.accel)[name]
        return float(np.sum(residual ** 2))

    controls = _controls(reference, calibration, 9.)
    gathers: list[bool] = []
    calls = {"plane": 0, "gyro": 0, "accel": 0}
    original_gather, original_families = fit._gather, fit._families

    def gather(*args: object, **kwargs: object) -> fit._Gathered:
        gathers.append(kwargs["derivatives"])
        return original_gather(*args, **kwargs)

    def families(*args: object) -> dict[str, fit._Family]:
        result = original_families(*args)
        for name, family in result.items():
            def counted(states: object, value: fit.LidarImuCalibration,
                        name: str = name, family: fit._Family = family) -> np.ndarray:
                calls[name] += 1
                return family.state_residual(states, value)

            result[name] = replace(family, state_residual=counted)
        return result

    monkeypatch.setattr(fit, "_gather", gather)
    monkeypatch.setattr(fit, "_families", families)
    session = fit._FixedTrajectoryResiduals(path, problem.planes, problem.gyro, problem.accel)
    assert _whitened_rms(session.residuals(calibration)) == _whitened_rms(expected)

    def actual(name: str, value: fit.LidarImuCalibration) -> float:
        return float(np.sum(session.residual(name, value) ** 2))

    assert _controls(actual, calibration, 9.) == controls
    assert len(controls) == 32
    assert gathers == [False] * (5 if different_times else 4)
    assert calls == {"plane": 1, "gyro": 15, "accel": 19}


def test_holdout_session_eviction_and_independent_measurements_remain_exact(
    problem: fit.BSplineLidarImuProblem,
) -> None:
    from calibrex.evaluation.lidar_imu_bag_calibration import _controls

    path, calibration = problem.initial_trajectory, problem.initial_calibration
    gyro = replace(problem.gyro, values=problem.gyro.values + .02)
    session = fit._FixedTrajectoryResiduals(path, problem.planes, gyro, problem.accel)
    session.cache.max_bytes = 1

    def actual(name: str, value: fit.LidarImuCalibration) -> float:
        residual = session.residual(name, value)
        assert not session.cache.batches
        return float(np.sum(residual ** 2))

    def reference(name: str, value: fit.LidarImuCalibration) -> float:
        return float(np.sum(fit.lidar_imu_residuals(path, value, problem.planes,
                                                  gyro, problem.accel)[name] ** 2))

    train = fit._FixedTrajectoryResiduals(path, problem.planes, problem.gyro, problem.accel)
    assert not np.array_equal(train.residual("gyro", calibration),
                              session.residual("gyro", calibration))
    assert _controls(actual, calibration, 9.) == _controls(reference, calibration, 9.)


@pytest.mark.parametrize("boundary", ["lower", "upper"])
@pytest.mark.parametrize("cached", [False, True])
def test_clock_boundary_probes_stay_in_shared_domain(problem, monkeypatch, boundary, cached):
    # Keep this step-size comparison away from interior knots; the shared-work
    # tests above separately exercise knot-crossing probes.
    gyro_times, accel_times = problem.gyro.times_sec.copy(), problem.accel.times_sec.copy()
    gyro_times[10:13] = accel_times[10:13] = [.12, .13, .14]
    problem = replace(problem, gyro=replace(problem.gyro, times_sec=gyro_times),
                      accel=replace(problem.accel, times_sec=accel_times))
    path = problem.initial_trajectory
    if boundary == "upper":
        times = problem.gyro.times_sec.copy()
        times[0] = path.start_time_sec
        problem = replace(problem, gyro=replace(problem.gyro, times_sec=times))
        direction = -1
    else:
        times = problem.accel.times_sec.copy()
        times[-1] = path.end_time_sec
        problem = replace(problem, accel=replace(problem.accel, times_sec=times))
        direction = 1
    original = fit._gather
    probes = []

    def gather(trajectory, times, **kwargs):
        assert np.all(times >= trajectory.start_time_sec)
        assert np.all(times <= trajectory.end_time_sec)
        probes.append(times.copy())
        return original(trajectory, times, **kwargs)

    monkeypatch.setattr(fit, "_gather", gather)
    families = fit._families(problem.planes, problem.gyro, problem.accel)
    cache = fit._EvaluationCache(families) if cached else None
    jacobian, residual = fit._linearize(
        path, problem.initial_calibration, families, ("imu_clock_offset",), cache=cache,
    )
    # Independent inward difference at a smaller step checks sign and scale.
    step = direction * 2e-7
    moved = problem.initial_calibration.stepped("imu_clock_offset", np.array([step]))
    shifted = fit.lidar_imu_residuals(path, moved, problem.planes, problem.gyro, problem.accel)
    vector = np.concatenate([shifted[name].ravel() for name in families])
    np.testing.assert_allclose(jacobian.toarray()[:, -1], (vector - residual) / step,
                               rtol=2e-3, atol=.1)
    assert probes
    result = fit.fit_bspline_lidar_imu(problem, fit.BSplineLidarImuFitOptions(max_iterations=0))
    assert np.sign(result.diagnostics.clock_domain.derivative_step_sec) == direction


@pytest.mark.parametrize("estimate", [("imu_clock_offset",), fit.CALIBRATION_BLOCKS])
def test_outside_clock_candidate_is_rejected_before_apply(problem, monkeypatch, estimate):
    problem = replace(problem, estimate=estimate)
    clock_column = 6 * problem.initial_trajectory.control_point_count + sum(
        fit._BLOCK_WIDTH[block] for block in estimate[:estimate.index("imu_clock_offset")]
    )
    def solve(matrix, rhs):
        step = np.zeros_like(rhs)
        step[clock_column] = 100.
        return step

    def forbidden(*args, **kwargs):
        pytest.fail("out-of-domain candidate must be rejected before applying any update")

    monkeypatch.setattr(fit, "spsolve", solve)
    monkeypatch.setattr(fit, "_apply_step", forbidden)
    result = fit.fit_bspline_lidar_imu(problem)
    assert result.status == "stalled"
    assert result.diagnostics.termination_reason == "clock_domain_limit"
    assert result.diagnostics.clock_domain.rejected_steps > 1
    assert result.final_cost == result.initial_cost


def test_clock_domain_rejection_can_retry(problem, monkeypatch):
    original = fit.spsolve
    calls = 0

    def solve(matrix, rhs):
        nonlocal calls
        calls += 1
        if calls == 1:
            step = np.zeros_like(rhs)
            step[6 * problem.initial_trajectory.control_point_count + 9] = 100.
            return step
        return original(matrix, rhs)

    monkeypatch.setattr(fit, "spsolve", solve)
    result = fit.fit_bspline_lidar_imu(problem, fit.BSplineLidarImuFitOptions(max_iterations=2))
    assert calls > 1
    assert result.diagnostics.clock_domain.rejected_steps >= 1
    assert result.final_cost < result.initial_cost


def test_zero_width_clock_domain_requires_fixed_clock(problem):
    times = problem.gyro.times_sec.copy()
    times[0], times[-1] = 0., problem.initial_trajectory.end_time_sec
    problem = replace(problem, gyro=replace(problem.gyro, times_sec=times))
    with pytest.raises(ValueError, match="positive-width"):
        fit.fit_bspline_lidar_imu(problem)
    result = fit.fit_bspline_lidar_imu(
        replace(problem, estimate=("gyro_bias",)), fit.BSplineLidarImuFitOptions(max_iterations=0),
    )
    assert result.diagnostics.clock_domain is None


def test_initial_clock_outside_shared_domain_is_rejected(problem):
    problem = replace(problem, initial_calibration=replace(
        problem.initial_calibration, imu_clock_offset_sec=10.,
    ))
    with pytest.raises(ValueError, match="outside the shared time domain"):
        fit.fit_bspline_lidar_imu(problem)
