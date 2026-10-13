"""Allocation-free internal RMSE preserves Python reduction and public hooks."""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("numba")

from test_compiled_lidar_linearization import _factors

from calibrex.core.geometry import SE3
from calibrex.graph.lidar_point_to_plane import LidarRigPointToPlaneFactor, _ObservationArrays
from calibrex.solvers import fixed_trajectory_se3_solver as fixed
from calibrex.solvers import numba_continuous_time_adapter as adapter


def _bits(value):
    return np.asarray(value, dtype=np.float64).view(np.uint64).tolist()


@pytest.mark.parametrize("count", [1, 64, 8192, 8193])
@pytest.mark.parametrize("correction", [None, (0.1, -0.2, 0.3, 0.04, -0.02, 0.01)])
def test_fused_residual_rmse_matches_native_float_bits(count, correction):
    _, factor = _factors(count)
    transform = factor._corrected_transform(correction)
    arrays = factor._prepared_arrays()
    reference = fixed._rmse(factor.residuals(correction))
    actual = adapter._numba_rig_rmse(arrays, transform)
    if sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)):
        assert _bits(actual[0]) == _bits(reference)
    else:
        assert actual is None


@pytest.mark.parametrize("values", [
    [1e8]+[1.0]*8192,
    [0.0, -0.0, 1e-160, -1e-160],
    [1e154, 1e154],
    [1e308, 1.0],
])
def test_both_ordered_sum_policies_preserve_dynamic_range_underflow_and_overflow(values):
    points = np.zeros((len(values), 3))
    points[:, 0] = values
    arrays = _ObservationArrays(points, np.zeros_like(points), np.tile([1.0, 0.0, 0.0],
        (len(values), 1)), np.tile([0.0]*6+[1.0], (len(values), 1)), np.ones(len(values)), None)
    residuals = adapter.numba_rig_residuals(arrays, SE3.identity())
    # This also captures the first-value promotion from sum's integer zero.
    expected = sum(value*value for value in residuals)
    actual = adapter._residual_square_sum(arrays.points, arrays.plane_points, arrays.normals,
        arrays.poses, arrays.sqrt_weights, np.asarray([0.0]*6+[1.0]),
        sys.version_info[:2] >= (3, 12))
    assert _bits(actual) == _bits(expected)
    plain = 0.0
    for value in residuals:
        plain += value*value
    actual_plain = adapter._residual_square_sum(arrays.points, arrays.plane_points, arrays.normals,
        arrays.poses, arrays.sqrt_weights, np.asarray([0.0]*6+[1.0]), False)
    assert _bits(actual_plain) == _bits(plain)
    assert _bits(adapter._value_square_sum(np.asarray(residuals), False)) == _bits(plain)
    assert _bits(adapter._value_square_sum(np.asarray(residuals),
        sys.version_info[:2] >= (3, 12))) == _bits(expected)
    summary = adapter._numba_values_rmse(np.asarray(residuals))
    if adapter._numba_rmse_policy() is None:
        assert summary is None
    else:
        assert _bits(summary[0]) == _bits(fixed._rmse(residuals))


@pytest.mark.parametrize("loss", ["none", "huber"])
@pytest.mark.parametrize("count", [8191, 8192])
def test_complete_history_and_public_diagnostics_match_with_summary_kernel(count, loss):
    _, reference = _factors(count)
    factor = LidarRigPointToPlaneFactor(variable=reference.variable, sensor=reference.sensor,
        observations=reference.observations, t_ego_lidar=reference.t_ego_lidar,
        _array_kernel=adapter.numba_rig_residuals,
        _array_linearizer=adapter.numba_rig_linearization,
        _array_normal_kernel=adapter.numba_rig_normal_equations,
        _array_rmse_kernel=adapter._numba_rig_rmse)
    options = fixed.FixedTrajectorySe3SolverOptions(max_iterations=3, robust_loss=loss)
    assert fixed.FixedTrajectorySe3ExtrinsicSolver().solve(factor, options) == (
        fixed.FixedTrajectorySe3ExtrinsicSolver().solve(reference, options))
    assert factor.evaluate() == reference.evaluate()
    factor.observations = reference.observations = tuple(reversed(reference.observations))
    factor.t_ego_lidar = reference.t_ego_lidar = SE3((.2, -.4, .1), (.1, .2, -.1, .9))
    assert fixed.FixedTrajectorySe3ExtrinsicSolver().solve(factor, options) == (
        fixed.FixedTrajectorySe3ExtrinsicSolver().solve(reference, options))


@pytest.mark.parametrize("hook", ["instance_residuals", "class_residuals", "correction",
                                  "kernel", "rmse", "sum"])
def test_public_hooks_and_replaced_reductions_preserve_fallback(hook, monkeypatch):
    _, factor = _factors(8192)
    factor._array_rmse_kernel = lambda *args: pytest.fail("hook must bypass summary")
    if hook == "instance_residuals":
        native = factor.residuals
        monkeypatch.setattr(factor, "residuals", lambda correction: native(correction))
    elif hook == "class_residuals":
        native = LidarRigPointToPlaneFactor.residuals
        monkeypatch.setattr(LidarRigPointToPlaneFactor, "residuals", lambda self, c: native(self,c))
    elif hook == "correction":
        native = factor._corrected_transform
        monkeypatch.setattr(factor, "_corrected_transform", lambda c: native(c))
    elif hook == "kernel":
        monkeypatch.setattr(factor, "_array_kernel", lambda *args: [0.0]*8192)
    elif hook == "rmse":
        monkeypatch.setattr(fixed, "_rmse", lambda values: 7.0)
    else:
        monkeypatch.setattr(fixed, "sum", lambda values: 49.0*8192, raising=False)
    actual = fixed._factor_rmse(factor, [0.0]*6)
    assert actual == fixed._rmse(factor.residuals([0.0]*6))


def test_empty_unsupported_runtime_and_normal_equations_keep_native_contract(monkeypatch):
    _, factor = _factors(8192)
    arrays = factor._prepared_arrays()
    for runtime in ((3, 13), (3, 14), (4, 0)):
        with monkeypatch.context() as context:
            context.setattr(adapter.sys, "version_info", runtime)
            assert adapter._numba_rig_rmse(arrays, SE3.identity()) is None
            assert adapter._numba_values_rmse(np.ones(64)) is None
    empty = replace(arrays, points=np.empty((0,3)))
    assert adapter._numba_rig_rmse(empty, SE3.identity()) is None
    complete = factor._compiled_system([0.0]*6, robust_loss="huber", huber_delta_m=.05)
    reduced = factor._compiled_system([0.0]*6, robust_loss="huber", huber_delta_m=.05,
                                      _include_residuals=False)
    assert len(complete[0]) == 8192 and reduced[0] == []
    assert complete[1:] == reduced[1:]


@pytest.mark.parametrize("loss", ["none", "huber"])
def test_complete_dense_profile_preserves_results_and_counters_without_residual_lists(
    loss, monkeypatch,
):
    from test_packed_continuous_time_lidar import _fixture

    from calibrex.data.odometry_track import OdometryTrack
    from calibrex.solvers.continuous_time_lidar_pair_solver import (
        ContinuousTimeLidarPairOptions,
        ContinuousTimeLidarPairProblem,
        ContinuousTimeLidarPairSolver,
    )

    records, points, samples, times = _fixture(10240)
    problem = ContinuousTimeLidarPairProblem(records, points, times, OdometryTrack(samples),
        SE3.identity(), SE3.identity(), "T_base_lidar", "target", 1.0, 1.5)
    options = ContinuousTimeLidarPairOptions(query_backend="numba_cpu", voxel_strategy="adaptive",
        adaptive_min_points_per_voxel=1, max_target_points_per_split=None, max_iterations=1,
        solver=fixed.FixedTrajectorySe3SolverOptions(max_iterations=1, robust_loss=loss))
    calls = []
    native = adapter._numba_rig_rmse
    def observed(arrays, transform):
        calls.append(len(arrays.points))
        return native(arrays, transform)
    monkeypatch.setattr(adapter, "_numba_rig_rmse", observed)
    actual = ContinuousTimeLidarPairSolver().solve(problem, options)
    assert calls and min(calls) >= 8192
    monkeypatch.setattr(adapter, "_numba_rig_rmse", lambda *args: None)
    reference_problem = replace(problem, odometry_track=OdometryTrack(samples))
    reference = ContinuousTimeLidarPairSolver().solve(reference_problem, options)
    assert actual == reference
    assert (problem.odometry_track.interpolation_count, problem.odometry_track.clamp_count,
            problem.odometry_track.max_extrapolation_s) == (
        reference_problem.odometry_track.interpolation_count,
        reference_problem.odometry_track.clamp_count,
        reference_problem.odometry_track.max_extrapolation_s)


def test_custom_compiled_system_receives_existing_arguments_and_complete_residuals(monkeypatch):
    _, factor = _factors(8192)
    calls = []
    native = LidarRigPointToPlaneFactor._compiled_system
    def observed(self, *args, **kwargs):
        assert "_include_residuals" not in kwargs
        result = native(self, *args, **kwargs)
        calls.append(len(result[0]))
        return result
    monkeypatch.setattr(LidarRigPointToPlaneFactor, "_compiled_system", observed)
    fixed.FixedTrajectorySe3ExtrinsicSolver().solve(
        factor, fixed.FixedTrajectorySe3SolverOptions(max_iterations=1))
    assert calls == [8192]
