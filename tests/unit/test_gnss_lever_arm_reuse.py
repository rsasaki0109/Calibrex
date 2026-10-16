from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass, replace
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel
from tests.unit.test_gnss_lidar_lever_arm import EPOCH, LEVER, synthetic

from calibrex.evaluation.gnss_lidar_lever_arm import (
    GnssLidarRunOptions,
    evaluate_gnss_lidar_lever_arm,
)
from calibrex.solvers import gnss_lever_arm_solver as solver


def paired(estimate_time: bool = True) -> tuple[Any, Any, Any]:
    track, windows = synthetic(9, windows=4)
    options = solver.LeverArmOptions(estimate_time_offset=estimate_time)
    # Unequal window sizes ensure batching preserves each covariance's shape.
    windows[0] = replace(windows[0], times_s=windows[0].times_s[:73], poses=windows[0].poses[:73])
    rebased = replace(track, times_s=track.times_s - EPOCH)
    return rebased, solver.build_pairs(rebased, windows, options, epoch=EPOCH), options


@pytest.mark.parametrize("estimate_time", [False, True])
def test_prepared_residuals_keep_every_bit_and_noise_term(estimate_time: bool) -> None:
    track, pairs, options = paired(estimate_time)
    evaluate = solver._make_residual_evaluator(track, pairs, options)
    rng = np.random.default_rng(81)
    for offset in [0.0, -0.0, -0.2, 0.2, *rng.uniform(-0.19, 0.19, 20)]:
        if not estimate_time:
            offset = 0.0
        lever = rng.normal(0.0, 0.4, 6)[::2]  # Deliberately non-contiguous.
        expected = solver.window_residuals(track, pairs, lever, offset, options)
        actual = evaluate(lever, offset)
        assert actual.tobytes() == expected.tobytes()
        assert not actual.flags.writeable
        assert evaluate(lever.copy(), offset) is actual
    for changed in (
        replace(options, odometry_pose_sigma_m=0.2),
        replace(options, odometry_sigma_per_m=0.15, odometry_sigma_floor_m=0.08),
    ):
        expected = solver.window_residuals(track, pairs, LEVER, 0.013, changed)
        assert solver._make_residual_evaluator(track, pairs, changed)(LEVER, 0.013).tobytes() == (
            expected.tobytes()
        )


def test_clock_cache_keeps_signed_zero_and_evicts_after_sixty_four_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track, pairs, options = paired()
    original = solver.WindowPairs.lidar_poses
    calls = []

    def poses(*args: Any) -> Any:
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(solver.WindowPairs, "lidar_poses", poses)
    evaluate = solver._make_residual_evaluator(track, pairs, options)
    for offset in [0.0, -0.0] * 3:
        evaluate(LEVER, offset)
    assert len(calls) == 2 * len(pairs)
    for index in range(1, 65):
        evaluate(LEVER, index * 0.001)
    assert len(calls) == 66 * len(pairs)
    evaluate(LEVER + 0.01, 0.0)  # New residual; the clock state was evicted.
    assert len(calls) == 67 * len(pairs)


def test_residual_cache_uses_exact_lever_bits_and_evicts_after_256_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track, pairs, options = paired()
    original = solver._alignment_rotations
    calls = []

    def align(covariances: np.ndarray) -> np.ndarray:
        calls.append(len(covariances))
        return original(covariances)

    monkeypatch.setattr(solver, "_alignment_rotations", align)
    evaluate = solver._make_residual_evaluator(track, pairs, options)
    first = evaluate(np.zeros(3), 0.0)
    assert evaluate(np.zeros(3), 0.0) is first
    evaluate(-np.zeros(3), 0.0)
    assert len(calls) == 2
    for index in range(1, 256):
        evaluate(np.array([index * 0.001, 0.0, 0.0]), 0.0)
    assert len(calls) == 257
    assert evaluate(np.zeros(3), 0.0).tobytes() == first.tobytes()
    assert len(calls) == 258


def test_each_context_uses_its_own_mutable_inputs_and_options() -> None:
    track, pairs, options = paired()
    first = solver._make_residual_evaluator(track, pairs, options)(LEVER, 0.017)
    pairs[0].gnss_m[3, 0] += 0.2
    pairs[1].gnss_sigma_m[:] *= 2.0
    changed = solver._make_residual_evaluator(track, pairs, options)(LEVER, 0.017)
    assert changed.tobytes() != first.tobytes()
    assert (
        changed.tobytes() == solver.window_residuals(track, pairs, LEVER, 0.017, options).tobytes()
    )


def test_empty_and_large_inputs_use_the_original_path(monkeypatch: pytest.MonkeyPatch) -> None:
    track, pairs, options = paired()
    assert solver._make_residual_evaluator(track, [], options)(LEVER, 0.0).size == 0
    monkeypatch.setattr(solver, "_EVALUATION_CACHE_MAX_BYTES", 0)
    evaluate = solver._make_residual_evaluator(track, pairs, options)
    actual = evaluate(LEVER, 0.01)
    assert actual.tobytes() == solver.window_residuals(track, pairs, LEVER, 0.01, options).tobytes()
    assert actual.flags.writeable  # The original path was retained.


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_batched_procrustes_keeps_rank_deficiency_and_reflections(rank: int) -> None:
    rng = np.random.default_rng(7)
    covariances, expected = [], []
    for _ in range(12):
        source = rng.normal(size=(17, 3))
        source[:, rank:] = 0.0
        target = rng.normal(size=(17, 3))
        weights = rng.uniform(0.1, 10.0, 17)
        covariances.append((target * weights[:, None]).T @ source)
        expected.append(solver.weighted_procrustes(source, target, weights))
    assert (
        solver._alignment_rotations(np.stack(covariances)).tobytes() == np.stack(expected).tobytes()
    )


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


@pytest.mark.parametrize("estimate_time", [False, True])
def test_complete_evaluation_and_all_optimizer_results_match_uncached(
    monkeypatch: pytest.MonkeyPatch, estimate_time: bool
) -> None:
    track, windows = synthetic(6, windows=8)
    options = GnssLidarRunOptions(solver=solver.LeverArmOptions(estimate_time_offset=estimate_time))
    original = solver.least_squares
    traces: list[str] = []

    def least(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        trace = hashlib.sha256()
        for key in ("x", "fun", "jac", "active_mask"):
            trace.update(result[key].tobytes())
        trace.update(
            json.dumps(
                {
                    key: result[key]
                    for key in ("cost", "optimality", "nfev", "njev", "status", "message")
                }
            ).encode()
        )
        traces.append(trace.hexdigest())
        return result

    monkeypatch.setattr(solver, "least_squares", least)
    cached = evaluate_gnss_lidar_lever_arm(track, windows, options, reference_lever_arm=LEVER)
    cached_traces = traces.copy()
    traces.clear()
    monkeypatch.setattr(
        solver,
        "_make_residual_evaluator",
        lambda track, pairs, opts: (
            lambda lever, offset: solver.window_residuals(track, pairs, lever, offset, opts)
        ),
    )
    uncached = evaluate_gnss_lidar_lever_arm(track, windows, options, reference_lever_arm=LEVER)
    assert plain(cached) == plain(uncached)
    assert traces == cached_traces
    assert cached.jackknife_fits == 5
    assert cached.holdout_windows == 3
