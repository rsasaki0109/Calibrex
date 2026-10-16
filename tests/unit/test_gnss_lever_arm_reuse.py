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
        lambda track, pairs, opts, **_kwargs: (
            lambda lever, offset: solver.window_residuals(track, pairs, lever, offset, opts)
        ),
    )
    uncached = evaluate_gnss_lidar_lever_arm(track, windows, options, reference_lever_arm=LEVER)
    assert plain(cached) == plain(uncached)
    assert traces == cached_traces
    assert cached.jackknife_fits == 5
    assert cached.holdout_windows == 3


@pytest.mark.parametrize("estimate_time", [False, True])
def test_combined_pose_lookup_keeps_bits_at_boundaries_and_in_ragged_windows(
    estimate_time: bool,
) -> None:
    track, windows = synthetic(11, windows=4)
    options = solver.LeverArmOptions(estimate_time_offset=estimate_time)
    windows[1] = replace(windows[1], times_s=windows[1].times_s[:71], poses=windows[1].poses[:71])
    rebased = replace(track, times_s=track.times_s - EPOCH)
    pairs = solver.build_pairs(rebased, windows, options, epoch=EPOCH)
    lookup = solver._make_pose_lookup(pairs, windows)
    assert lookup is not None
    evaluate = solver._make_residual_evaluator(rebased, pairs, options, windows=windows)
    for offset in [0.0, -0.0, -0.2, 0.2, 0.0123] if estimate_time else [0.0]:
        actual = lookup(offset)
        expected = [p.lidar_poses(p.epoch_times_s - offset) for p in pairs]
        for index in range(3):
            assert actual[index].tobytes() == np.concatenate([p[index] for p in expected]).tobytes()
        assert (
            evaluate(LEVER, offset).tobytes()
            == solver.window_residuals(rebased, pairs, LEVER, offset, options).tobytes()
        )


def test_combined_pose_lookup_rejects_overlapping_unordered_and_ambiguous_windows() -> None:
    track, windows = synthetic(11, windows=4)
    opts = solver.LeverArmOptions()
    rebased = replace(track, times_s=track.times_s - EPOCH)
    pairs = solver.build_pairs(rebased, windows, opts, epoch=EPOCH)
    for selected, source in (
        ([], windows),
        (pairs, None),
        (pairs, windows[:1]),
        (pairs, [*windows, windows[0]]),
        (list(reversed(pairs)), windows),
        ([pairs[0], pairs[0]], windows),
    ):
        assert solver._make_pose_lookup(selected, source) is None


def test_combined_lookup_preserves_exact_endpoints_on_both_sides_of_a_gap() -> None:
    times = np.arange(11, dtype=np.float64)
    poses = np.tile(np.eye(4), (11, 1, 1))
    poses[:, :3, :3] = solver.Rotation.from_euler("z", times[:, None] * 0.1).as_matrix()
    poses[:, 0, 3] = times**2
    track = solver.GnssTrackModel(times, np.zeros((11, 3)), np.ones(11), max_gap_s=1.1)
    windows = [
        solver.OdometryWindow("a", 0, times[:5], poses[:5]),
        solver.OdometryWindow("b", 1, times[6:], poses[6:]),
    ]
    opts = solver.LeverArmOptions(estimate_time_offset=False, pair_steps=(1,))
    pairs = solver.build_pairs(track, windows, opts)
    lookup = solver._make_pose_lookup(pairs, windows)
    assert lookup is not None
    expected = [p.lidar_poses(p.epoch_times_s) for p in pairs]
    actual = lookup(0.0)
    for index in range(3):
        assert actual[index].tobytes() == np.concatenate([p[index] for p in expected]).tobytes()


@pytest.mark.parametrize("offset", [-0.2001, 0.2001])
def test_combined_lookup_never_interpolates_across_a_window_boundary(offset: float) -> None:
    track_times = np.arange(201, dtype=np.float64) * 0.1
    track = solver.GnssTrackModel(track_times, np.zeros((201, 3)), np.ones(201))
    first = np.r_[0.01 + np.arange(99) * 0.1, 9.9]
    second = np.r_[10.0 + np.arange(100) * 0.1, 19.91]
    windows = [
        solver.OdometryWindow(name, i, t, np.tile(np.eye(4), (len(t), 1, 1)))
        for i, (name, t) in enumerate([("a", first), ("b", second)])
    ]
    pairs = solver.build_pairs(track, windows, solver.LeverArmOptions(pair_steps=(1,)))
    assert len(pairs) == 2
    lookup = solver._make_pose_lookup(pairs, windows)
    assert lookup is not None
    with pytest.raises(ValueError):
        for p in pairs:
            p.lidar_poses(p.epoch_times_s - offset)
    with pytest.raises(ValueError):
        lookup(offset)
