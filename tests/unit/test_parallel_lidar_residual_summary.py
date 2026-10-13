"""Bounded shared CPU helpers preserve ordered RMSE, histories and cleanup."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import Event

import numpy as np
import pytest

pytest.importorskip("numba")
from test_compiled_lidar_linearization import _factors

from calibrex.core.geometry import SE3
from calibrex.data.livox import LivoxPointRecord
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack
from calibrex.graph.lidar_point_to_plane import LidarRigPointToPlaneFactor, _ObservationArrays
from calibrex.solvers import continuous_time_lidar_pair_solver as pair
from calibrex.solvers import fixed_trajectory_se3_solver as fixed
from calibrex.solvers import numba_continuous_time_adapter as a


def _bits(value):
    return np.asarray(value, dtype=np.float64).view(np.uint64).tolist()


def _arrays(count, reverse=False):
    rng = np.random.default_rng(639)
    values = [rng.normal(size=(count, 3)) for _ in range(3)]
    values += [rng.normal(size=(count, 7)), rng.uniform(0., 2., size=count)]
    for value in values:
        value.flags.writeable = False
    if reverse:
        values = [v[::-1] for v in values]
    return _ObservationArrays(*values, None)


@pytest.mark.parametrize("count", [32767, 32768, 65537])
@pytest.mark.parametrize("reverse", [False, True])
def test_parallel_summary_matches_native_bits_and_keeps_input_ownership(count, reverse):
    arrays = _arrays(count, reverse)
    transform = SE3((.1, -.2, .3), (.2, -.3, .1, .9))
    original = arrays.points.copy()
    with ThreadPoolExecutor(max_workers=3) as helpers:
        actual = a._numba_parallel_rig_rmse(arrays, transform, helpers)
    if count < 32768 or a._numba_rmse_policy() is None:
        assert actual is None
    else:
        assert _bits(actual) == _bits(a._numba_rig_rmse(arrays, transform))
    assert np.array_equal(arrays.points, original)
    assert not arrays.points.flags.writeable


def test_disjoint_rows_keep_residual_bits_and_both_reduction_policies():
    arrays = _arrays(65537)
    transform = np.asarray([.1, -.2, .3, .2, -.3, .1, .9])
    arguments = (arrays.points, arrays.plane_points, arrays.normals, arrays.poses,
                 arrays.sqrt_weights, transform)
    expected = a._residual_rows(*arguments)
    output = np.empty(len(expected))
    with ThreadPoolExecutor(max_workers=3) as helpers:
        future = helpers.submit(a._residual_rows_into, *arguments, output, 32768, len(output))
        a._residual_rows_into(*arguments, output, 0, 32768)
        future.result()
    assert _bits(output) == _bits(expected)
    for compensated in (False, True):
        assert _bits(a._value_square_sum(output, compensated)) == _bits(
            a._residual_square_sum(*arguments, compensated))


def test_three_concurrent_callers_use_separate_bounded_helpers_without_deadlock():
    arrays = _arrays(65537)
    transforms = [SE3((i*.1, -.2, .3), (.2, -.3, .1, .9)) for i in range(3)]
    expected = [a._numba_rig_rmse(arrays, t) for t in transforms]
    with ThreadPoolExecutor(max_workers=3) as helpers, ThreadPoolExecutor(max_workers=3) as callers:
        results = list(callers.map(lambda t: a._numba_parallel_rig_rmse(arrays, t, helpers),
                                  transforms))
    assert _bits(results) == _bits(expected)


@pytest.mark.parametrize("loss", ["none", "huber"])
def test_complete_optimizer_histories_and_diagnostics_keep_public_values(loss):
    _, reference = _factors(32768)
    with ThreadPoolExecutor(max_workers=3) as helpers:
        parallel = LidarRigPointToPlaneFactor(
            variable=reference.variable, sensor=reference.sensor,
            observations=reference.observations, t_ego_lidar=reference.t_ego_lidar,
            _array_kernel=a.numba_rig_residuals, _array_linearizer=a.numba_rig_linearization,
            _array_normal_kernel=a.numba_rig_normal_equations,
            _array_rmse_kernel=lambda arrays, t: a._numba_parallel_rig_rmse(arrays, t, helpers),
        )
        options = fixed.FixedTrajectorySe3SolverOptions(max_iterations=3, robust_loss=loss)
        assert fixed.FixedTrajectorySe3ExtrinsicSolver().solve(parallel, options) == (
            fixed.FixedTrajectorySe3ExtrinsicSolver().solve(reference, options))
        assert parallel.evaluate() == reference.evaluate()


def test_workspace_mutability_and_unsupported_policy_do_not_submit(monkeypatch):
    arrays = _arrays(32768)
    large = replace(arrays, points=np.broadcast_to(arrays.points[:1], (2097153, 3)))
    writable = replace(arrays, points=arrays.points.copy())
    with ThreadPoolExecutor(max_workers=3) as helpers:
        monkeypatch.setattr(helpers, "submit", lambda *args: pytest.fail("must use CPU fallback"))
        assert a._numba_parallel_rig_rmse(large, SE3.identity(), helpers) is None
        assert a._numba_parallel_rig_rmse(writable, SE3.identity(), helpers) is None
        monkeypatch.setattr(a, "_numba_rmse_policy", lambda: None)
        assert a._numba_parallel_rig_rmse(arrays, SE3.identity(), helpers) is None


def test_failed_caller_joins_helper_and_preserves_original_exception(monkeypatch):
    arrays = _arrays(32768)
    finished = Event()
    def failure(*args):
        if args[-2] == 0:
            raise RuntimeError("caller error")
        finished.set()
        raise ValueError("helper error")
    monkeypatch.setattr(a, "_residual_rows_into", failure)
    with ThreadPoolExecutor(max_workers=3) as helpers:
        with pytest.raises(RuntimeError, match="caller error"):
            a._numba_parallel_rig_rmse(arrays, SE3.identity(), helpers)
        assert finished.is_set()


def test_complete_clock_profile_records_actual_work_and_preserves_logical_counters(monkeypatch):
    records = [LivoxPointRecord(
        (float(x)+.25, float(y)+.25, float(z)+.25, 0.),
        normal_xyz=(float(x)+4., float(y)+4., float(z)+4.),
    ) for x in range(-3, 4) for y in range(-3, 4) for z in range(-3, 4)]
    points = [tuple(float(v) for v in row)
              for row in np.random.default_rng(710).uniform(-3, 3, (65536, 3))]
    samples = [OdometryPoseSample(i*1_000_000_000, SE3(
        (i*.03, -i*.02, i*.01), (.02*i, .03*i, -.01*i, 1.),
    )) for i in range(4)]
    problem = pair.ContinuousTimeLidarPairProblem(
        records, points, [(i*3_000_000) % 2_500_000_000 for i in range(len(points))],
        OdometryTrack(samples), SE3.identity(), SE3.identity(), "extrinsic", "target", 1., 1.5,
    )
    options = pair.ContinuousTimeLidarPairOptions(
        query_backend="numba_cpu", max_target_points_per_split=None, max_iterations=1,
        outlier_policy="none", voxel_strategy="adaptive", adaptive_min_points_per_voxel=1,
        solver=fixed.FixedTrajectorySe3SolverOptions(max_iterations=1),
    )
    monkeypatch.setattr(pair, "cpu_count", lambda: 3)
    reference = pair.ContinuousTimeLidarPairSolver().solve(problem, options)
    other = replace(problem, odometry_track=OdometryTrack(samples))
    monkeypatch.setattr(pair, "cpu_count", lambda: 6)
    actual = pair.ContinuousTimeLidarPairSolver().solve(other, options)
    assert ";summary_parallel_cpu=false;" in reference.query_backend
    assert ";summary_parallel_cpu=true;" in actual.query_backend
    assert asdict(replace(reference, query_backend="same")) == asdict(
        replace(actual, query_backend="same"))
    for name in ("interpolation_count", "clamp_count", "max_extrapolation_s"):
        assert getattr(problem.odometry_track, name) == getattr(other.odometry_track, name)
