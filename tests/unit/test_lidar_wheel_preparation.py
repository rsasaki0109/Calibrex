from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from tests.unit.test_lidar_wheel import drive

from calibrex.evaluation.lidar_wheel import evaluate_lidar_wheel
from calibrex.solvers import lidar_wheel_solver as solver
from calibrex.solvers.vehicle_frame_solver import motions_from_poses


def fixture() -> tuple[Any, Any, Any, Any]:
    times, poses, wheel = drive(7)
    options = solver.LidarWheelOptions()
    moving = solver._moving(motions_from_poses(times[:150], poses[:150]), options)
    base = solver.Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
    return moving, wheel, base, options


def test_prepared_residuals_match_every_bit_with_all_six_parameters() -> None:
    moving, wheel, base, opts = fixture()
    evaluate = solver._make_residual_evaluator(moving, wheel, base, opts)
    rng = np.random.default_rng(71)
    for offset in [0.0, -0.0, -0.3, 0.3, *rng.uniform(-0.29, 0.29, 20)]:
        params = np.r_[rng.normal(0, 0.02, 3), 1.0, 1.03, offset]
        assert (
            evaluate(params).tobytes()
            == solver.lidar_wheel_residuals(moving, wheel, params, base, opts).tobytes()
        )


def test_clock_cache_uses_exact_bits_and_evicts_after_eight_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moving, wheel, base, opts = fixture()
    original = solver.WheelOdometry.at
    calls = []

    def counted(*args: Any) -> Any:
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(solver.WheelOdometry, "at", counted)
    evaluate = solver._make_residual_evaluator(moving, wheel, base, opts)
    params = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    for offset in [0.0, -0.0] * 4:
        params[5] = offset
        evaluate(params)
    assert len(calls) == 2
    for index in range(1, 9):
        params[5] = index * 0.001
        evaluate(params)
    params[5] = 0.0
    evaluate(params)
    assert len(calls) == 11


def test_new_solve_context_uses_changed_motion_wheel_and_noise_inputs() -> None:
    moving, wheel, base, opts = fixture()
    params = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.01])
    first = solver._make_residual_evaluator(moving, wheel, base, opts)(params)
    moving[0].velocity_mps[0] += 0.02
    wheel.speed_mps[:] *= 1.01
    opts = replace(opts, velocity_sigma_fraction=0.03)
    actual = solver._make_residual_evaluator(moving, wheel, base, opts)(params)
    assert actual.tobytes() != first.tobytes()
    assert (
        actual.tobytes()
        == solver.lidar_wheel_residuals(moving, wheel, params, base, opts).tobytes()
    )


def test_large_empty_and_derived_wheel_inputs_retain_original_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    moving, wheel, base, opts = fixture()
    params = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.01])
    assert solver._make_residual_evaluator([], wheel, base, opts)(params).size == 0
    monkeypatch.setattr(solver, "_PREPARATION_MAX_BYTES", 0)
    assert solver._make_residual_evaluator(moving, wheel, base, opts)(params).tobytes() == (
        solver.lidar_wheel_residuals(moving, wheel, params, base, opts).tobytes()
    )
    monkeypatch.undo()
    calls = []

    class Derived(solver.WheelOdometry):
        def at(self, times: np.ndarray) -> Any:
            calls.append(1)
            return super().at(times)

    derived = Derived(wheel.times_s, wheel.speed_mps, wheel.yaw_rate_rps)
    evaluate = solver._make_residual_evaluator(moving, derived, base, opts)
    for _ in range(3):
        evaluate(params)
    assert len(calls) == 3


def test_complete_evaluation_and_optimizer_histories_match_uncached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times, poses, wheel = drive(8)
    motions = motions_from_poses(times, poses)
    original = solver.least_squares
    traces: list[str] = []

    def least(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        trace = hashlib.sha256()
        for key in ("x", "fun", "jac", "active_mask"):
            trace.update(result[key].tobytes())
        trace.update(
            json.dumps(
                {k: result[k] for k in ("nfev", "njev", "status", "message", "cost", "optimality")}
            ).encode()
        )
        traces.append(trace.hexdigest())
        return result

    monkeypatch.setattr(solver, "least_squares", least)
    cached = evaluate_lidar_wheel(motions, wheel)
    before = traces.copy()
    traces.clear()
    monkeypatch.setattr(
        solver,
        "_make_residual_evaluator",
        lambda moving, wheel, base, opts: (
            lambda params: solver.lidar_wheel_residuals(moving, wheel, params, base, opts)
        ),
    )
    uncached = evaluate_lidar_wheel(motions, wheel)
    assert before == traces
    for field in ("rotation", "covariance"):
        assert getattr(cached.result, field).tobytes() == getattr(uncached.result, field).tobytes()
    assert cached.records == uncached.records
    assert cached.policy_status == uncached.policy_status
    assert cached.policy_reasons == uncached.policy_reasons
    assert cached.jackknife_fits == uncached.jackknife_fits == 8
    assert (
        cached.train_median == uncached.train_median
        and cached.holdout_median == uncached.holdout_median
    )
