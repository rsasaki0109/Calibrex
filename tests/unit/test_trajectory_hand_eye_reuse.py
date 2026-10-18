from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from pydantic import BaseModel
from tests.unit.test_trajectory_hand_eye import synthetic

from calibrex.evaluation.ins_lidar_hand_eye import InsLidarEvaluation, evaluate_ins_lidar_hand_eye
from calibrex.solvers import trajectory_hand_eye_solver as solver


@pytest.mark.parametrize("estimate_time", [False, True])
@pytest.mark.parametrize("estimate_scale", [False, True])
def test_prepared_motion_residuals_preserve_every_bit(
    estimate_time: bool,
    estimate_scale: bool,
) -> None:
    reference, motions = synthetic(planar=False, duration_s=8)
    options = solver.TrajectoryHandEyeOptions(
        estimate_time_offset=estimate_time,
        estimate_reference_scale=estimate_scale,
    )
    evaluate = solver._make_normalized_evaluator(reference, motions, options)
    rng = np.random.default_rng(51)
    for index in range(30):
        base = np.eye(4)
        base[:3, :3] = solver.Rotation.from_rotvec(rng.normal(0, 0.1, 3)).as_matrix()
        base[:3, 3] = rng.normal(0, 0.3, 3)
        params = rng.normal(0, 0.01, 8)
        if index < 2:
            params = np.zeros(8) * (-1.0 if index else 1.0)
        expected = solver._normalized_residuals(reference, motions, base, params, options)
        actual = evaluate(base, params)
        assert actual.tobytes() == expected.tobytes()
        assert not actual.flags.writeable
        assert evaluate(base.copy(), params.copy()) is actual


def test_clock_cache_keeps_signed_zeros_and_evicts_after_sixteen_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, motions = synthetic(planar=False, duration_s=8)
    options = solver.TrajectoryHandEyeOptions()
    evaluate = solver._make_normalized_evaluator(reference, motions, options)
    original = reference.poses_at
    queries = []

    def poses(times: np.ndarray) -> np.ndarray:
        queries.append(times.copy())
        return original(times)

    monkeypatch.setattr(solver.ReferenceTrajectory, "poses_at", lambda _self, times: poses(times))
    base = np.eye(4)
    for offset in [0.0, -0.0] * 10:
        params = np.zeros(8)
        params[6] = offset
        evaluate(base, params)
    assert len(queries) == 4
    for index in range(1, 17):
        params = np.zeros(8)
        params[6] = index * 0.001
        evaluate(base, params)
    assert len(queries) == 36
    params = np.zeros(8)
    params[3] = 0.123  # New residual key, with an evicted clock state.
    evaluate(base, params)
    assert len(queries) == 38


def test_residual_cache_is_bounded_and_separates_base_transforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, motions = synthetic(planar=False, duration_s=8)
    evaluate = solver._make_normalized_evaluator(
        reference, motions, solver.TrajectoryHandEyeOptions()
    )
    original = solver._motion_error_rows
    calls = []

    def counted(*args: np.ndarray) -> np.ndarray:
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(solver, "_motion_error_rows", counted)
    base = np.eye(4)
    params = np.zeros(8)
    first = evaluate(base, params)
    for _ in range(10):
        assert evaluate(base.copy(), params.copy()) is first
    assert len(calls) == 1
    for index in range(1, 129):
        params[3] = index * 0.001
        evaluate(base, params)
    assert len(calls) == 129
    params[3] = 0.0
    assert evaluate(base, params).tobytes() == first.tobytes()
    assert len(calls) == 130
    base[0, 3] = 0.1
    evaluate(base, params)
    assert len(calls) == 131


def test_prepared_state_does_not_leak_into_changed_inputs_or_options() -> None:
    reference, motions = synthetic(planar=False, duration_s=8)
    options = solver.TrajectoryHandEyeOptions()
    base, params = np.eye(4), np.zeros(8)
    first = solver._make_normalized_evaluator(reference, motions, options)(base, params)
    changed_motions = [replace(m, motion=m.motion.copy()) for m in motions]
    changed_motions[0].motion[0, 3] += 0.2
    changed_reference = solver.ReferenceTrajectory(reference.times_s.copy(), reference.poses.copy())
    changed_reference.poses[:, 1, 3] *= 1.1
    changed_options = replace(options, translation_sigma_per_m=0.03)
    for ref, items, opts in [
        (reference, changed_motions, options),
        (changed_reference, motions, options),
        (reference, motions, changed_options),
    ]:
        result = solver._make_normalized_evaluator(ref, items, opts)(base, params)
        assert (
            result.tobytes()
            == solver._normalized_residuals(ref, items, base, params, opts).tobytes()
        )
        assert result.tobytes() != first.tobytes()


def test_budget_fallback_and_empty_evaluator_preserve_scalar_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, motions = synthetic(planar=False, duration_s=8)
    options = solver.TrajectoryHandEyeOptions()
    monkeypatch.setattr(solver, "_RESIDUAL_CACHE_MAX_BYTES", 0)
    for items in [motions, []]:
        evaluate = solver._make_normalized_evaluator(reference, items, options)
        base, params = np.eye(4), np.zeros(8)
        assert (
            evaluate(base, params).tobytes()
            == solver._normalized_residuals(reference, items, base, params, options).tobytes()
        )


def test_complete_evaluation_and_optimizer_results_match_uncached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, motions = synthetic(planar=False, duration_s=20)
    original_lsq = solver.least_squares
    results = []

    def lsq(*args: object, **kwargs: object) -> object:
        result = original_lsq(*args, **kwargs)
        digest = hashlib.sha256()
        for name in ("x", "fun", "jac", "active_mask"):
            digest.update(np.asarray(result[name]).tobytes())
        results.append((digest.hexdigest(), result.nfev, result.njev, result.status))
        return result

    monkeypatch.setattr(solver, "least_squares", lsq)
    prepared = evaluate_ins_lidar_hand_eye(reference, motions)
    prepared_solves = results.copy()
    results.clear()
    monkeypatch.setattr(solver, "_RESIDUAL_CACHE_MAX_BYTES", 0)
    scalar = evaluate_ins_lidar_hand_eye(reference, motions)

    def encode(value: InsLidarEvaluation) -> str:
        return json.dumps(
            asdict(value),
            allow_nan=False,
            default=lambda a: a.model_dump(mode="json") if isinstance(a, BaseModel) else a.tolist(),
        )

    assert encode(prepared) == encode(scalar)
    assert prepared_solves == results
    assert prepared.jackknife_fits >= 3 and prepared.holdout_blocks
