"""Content identity, bounded retention and complete candidate/evidence parity."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import jsonschema
import numpy as np
import pytest
from tests.unit.test_camera_lidar_edge import CAMERA, FAST, _frames, _provenance, _repeated

from calibrex.evaluation import camera_lidar_edge as edge
from calibrex.evaluation.edge_preparation_cache import (
    EdgePreparationCache,
    EdgePreparationCacheDiagnostics,
)


def _assert_inputs_equal(actual, expected):
    assert (actual is None) == (expected is None)
    if actual is None:
        return
    for left, right in zip(actual, expected, strict=True):
        if isinstance(left, np.ndarray):
            assert left.dtype == right.dtype and left.shape == right.shape
            assert left.tobytes() == right.tobytes()
        else:
            assert left == right


@pytest.mark.parametrize("budget", [-1, True, 1.0, "100", None])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match="non-negative integer"):
        EdgePreparationCache(budget)


@pytest.mark.parametrize("shape", [(0, 3), (7, 9)])
def test_every_content_byte_shape_dtype_layout_and_setting_bind_the_key(shape):
    cache = EdgePreparationCache()
    original = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    calls = []

    def run(array, settings=(), namespace="image"):
        def compute():
            calls.append(1)
            return array.copy()

        result = cache._get_or_compute(namespace, array, settings, compute)
        np.testing.assert_array_equal(result, array)

    run(original)
    run(original)
    assert len(calls) == 1 and cache.diagnostics.hits == 1
    run(original.astype(np.float64))
    run(original.reshape(-1))
    run(original, (0.98,))
    run(original, namespace="lidar")
    assert len(calls) == 5
    if original.size:
        modified = original.copy()
        modified.flat[-1] = np.nextafter(modified.flat[-1], np.float32(np.inf))
        run(modified)
        run(np.asfortranarray(original))
        assert len(calls) == 7


def test_both_miss_and_hit_results_can_be_mutated_without_corrupting_retention():
    cache = EdgePreparationCache()
    raw = np.arange(20, dtype=np.float32)
    calls = []

    def compute():
        calls.append(1)
        return raw.copy(), raw.copy(), 4, 0.8

    first = cache._get_or_compute("lidar", raw, (), compute)
    first[0][:] = -99
    second = cache._get_or_compute("lidar", raw, (), compute)
    second[1][:] = -88
    third = cache._get_or_compute("lidar", raw, (), compute)
    assert len(calls) == 1
    np.testing.assert_array_equal(third[0], raw)
    np.testing.assert_array_equal(third[1], raw)
    assert second[0].flags.writeable and third[0].flags.writeable


def test_lru_eviction_zero_results_and_disabled_or_oversize_bypass():
    cache = EdgePreparationCache(max_bytes=2400)
    raw = np.ones(32, dtype=np.float32)
    for i in range(100):
        cache._get_or_compute("image", raw, (i,), lambda: None)
        snapshot = cache.diagnostics
        assert snapshot.retained_bytes <= snapshot.max_bytes
        assert snapshot.entries <= 2
    assert cache.diagnostics.evictions == 98
    for budget in (0, 100):
        tiny = EdgePreparationCache(budget)
        for _ in range(2):
            tiny._get_or_compute("image", raw, (), lambda: raw.copy())
        assert tiny.diagnostics.entries == tiny.diagnostics.retained_bytes == 0
        assert tiny.diagnostics.bypasses == 2
    before = cache.diagnostics
    cache.clear()
    assert cache.diagnostics.entries == cache.diagnostics.retained_bytes == 0
    assert cache.diagnostics.evictions == before.evictions
    jsonschema.validate(
        cache.diagnostics.model_dump(mode="json"),
        EdgePreparationCacheDiagnostics.model_json_schema(),
    )


@pytest.mark.parametrize("settings", [(float("nan"),), (float("inf"),), (object(),)])
def test_unusual_settings_bypass_without_changing_original_errors(settings):
    cache = EdgePreparationCache()
    raw = np.ones(2)

    def fail():
        raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        cache._get_or_compute("image", raw, settings, fail)
    assert cache.diagnostics.bypasses == 1 and cache.diagnostics.entries == 0


def test_compute_failures_are_never_cached():
    cache = EdgePreparationCache()
    raw = np.ones(2)
    calls = []

    def fail():
        calls.append(1)
        raise RuntimeError("original failure")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="original failure"):
            cache._get_or_compute("image", raw, (), fail)
    assert len(calls) == cache.diagnostics.misses == 2
    assert cache.diagnostics.entries == 0


@pytest.mark.parametrize("changed", ["image", "points"])
def test_modifying_the_last_input_element_invalidates_only_that_stage(changed):
    from tests.unit.camera_lidar_scene import true_transform

    frame = _frames(true_transform(), 1)[0]
    options = edge.EdgeAlignmentOptions()
    cache = EdgePreparationCache()
    edge._prepare_edge_inputs(frame, options, cache)
    values = getattr(frame, changed)
    if changed == "image":
        values.flat[-1] ^= 1
    else:
        values.flat[-1] = np.nextafter(values.flat[-1], np.float32(np.inf))
    before = cache.diagnostics
    _assert_inputs_equal(
        edge._prepare_edge_inputs(frame, options, cache), edge._prepare_edge_inputs(frame, options)
    )
    assert cache.diagnostics.misses - before.misses == 1
    assert cache.diagnostics.hits - before.hits == 1


def test_clear_prevents_an_inflight_computation_from_repopulating():
    cache = EdgePreparationCache()
    raw = np.ones(3)
    entered, released = threading.Event(), threading.Event()

    def compute():
        entered.set()
        assert released.wait(10)
        return raw.copy()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(cache._get_or_compute, "image", raw, (), compute)
        assert entered.wait(10)
        cache.clear()
        released.set()
        np.testing.assert_array_equal(pending.result(timeout=10), raw)
    assert cache.diagnostics.entries == cache.diagnostics.retained_bytes == 0
    assert cache.diagnostics.bypasses == 1


def test_clear_during_content_hashing_also_fences_retention(monkeypatch):
    from calibrex.evaluation import edge_preparation_cache as module

    cache = EdgePreparationCache()
    raw = np.ones(3)
    entered, released = threading.Event(), threading.Event()
    original = module.hashlib.sha256

    def hashing(*args, **kwargs):
        entered.set()
        assert released.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(module.hashlib, "sha256", hashing)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(cache._get_or_compute, "image", raw, (), lambda: raw.copy())
        assert entered.wait(10)
        cache.clear()
        released.set()
        np.testing.assert_array_equal(pending.result(timeout=10), raw)
    assert cache.diagnostics.entries == cache.diagnostics.retained_bytes == 0
    assert cache.diagnostics.bypasses == 1


@pytest.mark.parametrize(
    "name,value,expected_misses",
    [
        ("image_max_width", 200, 1),
        ("spill_gamma", 0.95, 1),
        ("spill_radius_px", 12, 1),
        ("edge_background_sigma_px", 12.0, 1),
        ("min_jump_m", 0.5, 1),
        ("jump_exponent", 0.7, 1),
        ("max_edge_points_per_frame", 200, 1),
        ("min_ring_fraction", 0.9, 1),
        ("unstructured_lidar", False, 1),
    ],
)
def test_each_preprocessing_option_invalidates_its_own_stage(name, value, expected_misses):
    from tests.unit.camera_lidar_scene import true_transform

    frame = _frames(true_transform(), 1)[0]
    cache = EdgePreparationCache()
    options = edge.EdgeAlignmentOptions()
    edge._prepare_edge_inputs(frame, options, cache)
    before = cache.diagnostics
    changed = replace(options, **{name: value})
    actual = edge._prepare_edge_inputs(frame, changed, cache)
    _assert_inputs_equal(actual, edge._prepare_edge_inputs(frame, changed))
    assert cache.diagnostics.misses - before.misses == expected_misses
    assert cache.diagnostics.hits - before.hits == 1


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("unstructured", [False, True])
def test_complete_selection_recomputed_for_new_camera_and_candidate(workers, unstructured):
    from tests.unit.camera_lidar_scene import perturbed, true_transform

    truth = true_transform()
    frames = _repeated(truth, unstructured)
    frames.insert(1, replace(frames[0], image=np.zeros_like(frames[0].image), time_s=0.5))
    cache = EdgePreparationCache()
    options = replace(FAST, preparation_workers=workers)
    for camera, candidate in [
        (CAMERA, truth),
        (replace(CAMERA, fx=CAMERA.fx * 1.2), perturbed(truth, (20.0, 3.0, 0))),
        (CAMERA, truth),
    ]:
        expected, dropped = edge.prepare_frames(frames, camera, candidate, options)
        actual, rejected = edge.prepare_frames(
            frames, camera, candidate, options, preparation_cache=cache
        )
        assert dropped == rejected
        assert len(expected) == len(actual)
        for a, b in zip(actual, expected, strict=True):
            assert (a.time_s, a.gap_s, a.scale, a.rings) == (b.time_s, b.gap_s, b.scale, b.rings)
            _assert_inputs_equal((a.edge, a.points, a.weights), (b.edge, b.points, b.weights))
    assert cache.diagnostics.hits > 0


def test_default_estimator_preserves_an_existing_preparation_observer(monkeypatch):
    from tests.unit.camera_lidar_scene import true_transform

    truth = true_transform()
    frames = _frames(truth, 2)
    original = edge.prepare_frames
    observed = []

    def observe(raw_frames, camera, candidate, options):
        observed.append(candidate.copy())
        return original(raw_frames, camera, candidate, options)

    monkeypatch.setattr(edge, "prepare_frames", observe)
    result = edge.estimate_from_frames(
        frames, CAMERA, truth, options=FAST, provenance=_provenance()
    )
    assert result.solver_status == "insufficient_samples"
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0], truth)


@pytest.mark.parametrize("translation", [False, True])
def test_whole_artifact_and_every_score_keep_original_candidate_budgets(monkeypatch, translation):
    from tests.unit.camera_lidar_scene import perturbed, true_transform

    truth = true_transform()
    frames = _frames(truth)
    options = replace(FAST, estimate_translation=translation)
    provenance = _provenance()
    cache = EdgePreparationCache()
    original = edge.EdgeObjective.score
    calls = []

    def score(self, theta):
        value = original(self, theta)
        calls.append((theta.tobytes(), value))
        return value

    monkeypatch.setattr(edge.EdgeObjective, "score", score)
    for offset in [(0.8, -1.2, 0), (-0.4, 0.7, 0.2)]:
        candidate = perturbed(truth, offset)
        calls.clear()
        expected = edge.estimate_from_frames(
            frames, CAMERA, candidate, options=options, provenance=provenance
        )
        trace = calls.copy()
        calls.clear()
        actual = edge.estimate_from_frames(
            frames,
            CAMERA,
            candidate,
            options=options,
            provenance=provenance,
            preparation_cache=cache,
        )
        assert actual.model_dump(mode="json") == expected.model_dump(mode="json")
        assert calls == trace
    assert cache.diagnostics.hits >= 2 * len(frames)
