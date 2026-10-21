"""Non-flat certificates preserve selection, exact scores and every search decision."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tests.unit.test_camera_lidar_edge import (
    CAMERA,
    FAST,
    _frames,
    _provenance,
    _repeated,
    perturbed,
    true_transform,
)
from tools.browser_edge_preparation_probe import _Trace

from calibrex.evaluation import camera_lidar_edge as edge
from calibrex.evaluation.edge_preparation_cache import EdgePreparationCache


def _noisy_sequence():
    rng = np.random.default_rng(10429)
    result = []
    for frame in _frames(true_transform(), 8):
        for sub in range(3):
            image, points = frame.image.copy(), frame.points.copy()
            if sub:
                image = np.clip(
                    image.astype(np.int16) + rng.integers(-1, 2, size=image.shape, dtype=np.int16),
                    0,
                    255,
                ).astype(np.uint8)
                points *= np.float32(1.0 + sub * 0.0001)
            result.append(
                replace(
                    frame,
                    image=image,
                    points=points,
                    time_s=float(len(result)),
                    image_time_s=float(len(result)),
                )
            )
    assert len({f.image.tobytes() for f in result}) == len(result)
    assert len({f.points.tobytes() for f in result}) == len(result)
    return result


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_certified_noisy_images_are_nonflat_under_the_original_filters(factor):
    rng = np.random.default_rng(8891)
    options = replace(FAST, image_max_width=80)
    certified = 0
    for _ in range(12):
        original = rng.integers(0, 256, size=(64, 80), dtype=np.uint8)
        original = np.repeat(np.repeat(original, factor, axis=0), factor, axis=1)
        noise = rng.integers(-1, 2, size=original.shape, dtype=np.int16)
        current = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        response, downscale, percentile = edge._image_edge_response_with_gradient(
            original, max_width=80, gamma=0.98, radius=40, background_sigma=20
        )
        state = edge._RepeatedImageCertificate()
        state.capture(original, downscale, percentile, options)
        state.remember(original[::4, ::4].astype(np.float32))
        assert state.certified_factor(current, options) == downscale == factor
        checked = edge._image_edge_response_with_gradient(
            current, max_width=80, gamma=0.98, radius=40, background_sigma=20
        )
        assert response.size > 0 and checked is not None and checked[2] >= 4.0
        certified += 1
    assert certified == 12


def test_changes_outside_the_view_sample_cannot_certify_a_flat_image():
    original = np.zeros((64, 80), dtype=np.uint8)
    original[1::4, 1::4] = 255
    _, _, percentile = edge._image_edge_response_with_gradient(
        original, max_width=80, gamma=0.98, radius=40
    )
    state = edge._RepeatedImageCertificate()
    options = replace(FAST, image_max_width=80)
    state.capture(original, 1, percentile, options)
    state.remember(original[::4, ::4].astype(np.float32))
    flat = np.zeros_like(original)
    assert np.array_equal(original[::4, ::4], flat[::4, ::4])
    assert state.certified_factor(flat, options) is None
    assert edge.image_edge_response(flat, max_width=80, gamma=0.98, radius=40) is None


def test_uncertain_threshold_and_changed_layout_use_full_preparation():
    original = np.full((64, 80), 42, dtype=np.uint8)
    options = replace(FAST, image_max_width=80)
    state = edge._RepeatedImageCertificate()
    state.capture(original, 1, 4.2, options)
    state.remember(original[::4, ::4].astype(np.float32))
    assert state.certified_factor(original, options) is None
    state.previous_percentile = 100.0
    assert state.certified_factor(original[:, :-1], options) is None
    assert state.certified_factor(original, replace(options, repeated_view_change=None)) is None
    assert state.certified_factor(original, replace(options, image_max_width=30)) is None
    assert state.certified_factor(original.astype(np.float32), options) is None
    assert state.certified_factor(np.ma.array(original, mask=False), options) is None


@pytest.mark.parametrize(
    "changed",
    [
        {"spill_gamma": float("nan")},
        {"spill_gamma": 1.5},
        {"spill_radius_px": 0},
        {"edge_background_sigma_px": float("inf")},
        {"image_max_width": 0},
    ],
)
def test_unusual_parameters_are_not_short_circuited(changed):
    image = np.ones((64, 80), dtype=np.uint8)
    assert edge._certifiable_image_factor(image, replace(FAST, **changed)) is None


def test_reference_pixels_are_owned_and_expire_on_an_unsupported_image():
    image = np.full((64, 80), 42, dtype=np.uint8)
    options = replace(FAST, image_max_width=80)
    state = edge._RepeatedImageCertificate()
    state.capture(image, 1, 100.0, options)
    state.remember(image[::4, ::4].astype(np.float32))
    image[1, 1] = 200
    assert state.previous_pixels[1, 1] == 42
    assert state.certified_factor(image, options) is None
    state.capture(image.astype(np.float32), 1, 100.0, options)
    state.remember(image[::4, ::4].astype(np.float32))
    assert state.previous_pixels is state.previous_percentile is None


@pytest.mark.parametrize("cached", [False, True])
def test_complete_artifacts_arrays_and_all_decisions_equal_full_preparation(monkeypatch, cached):
    frames = _noisy_sequence()
    options = edge.EdgeAlignmentOptions()
    provenance = _provenance()
    snapshots = [(f.image.tobytes(), f.points.tobytes()) for f in frames]

    def sequence():
        cache = EdgePreparationCache() if cached else None
        results = []
        for offset in [(0.8, -1.2, 0.0), (1.3, -1.5, 0.2)]:
            trace = _Trace()
            artifact = edge._drive_edge_workflow(
                edge._estimate_workflow(
                    frames,
                    CAMERA,
                    perturbed(true_transform(), offset),
                    options=options,
                    provenance=provenance,
                    **({"preparation_cache": cache} if cache is not None else {}),
                ),
                observer=trace,
            )
            assert artifact.samples.frames == 8 and artifact.jackknife_fits == 6
            results.append((artifact.model_dump(mode="json"), trace.records, trace.poses))
        if cache is not None:
            assert 0 < cache.diagnostics.retained_bytes <= cache.diagnostics.max_bytes
            assert cache.diagnostics.hits > 0
        return results

    actual = sequence()
    with monkeypatch.context() as reference:
        reference.setattr(edge._RepeatedImageCertificate, "certified_factor", lambda *args: None)
        expected = sequence()
    assert actual == expected
    assert actual[0][0]["start_transform"] != actual[1][0]["start_transform"]
    assert snapshots == [(f.image.tobytes(), f.points.tobytes()) for f in frames]


def test_cache_hits_retain_the_percentile_and_avoid_rejected_image_filters(monkeypatch):
    frames = _noisy_sequence()
    cache = EdgePreparationCache()
    original = edge._image_edge_response_with_gradient
    calls = []

    def observe(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(edge, "_image_edge_response_with_gradient", observe)
    for _ in range(2):
        prepared, dropped = edge.prepare_frames(
            frames, CAMERA, true_transform(), FAST, preparation_cache=cache
        )
        assert len(prepared) == 8 and len(dropped) == 16
        assert all("repeats the last kept view" in reason for reason in dropped)
    assert len(calls) == 8
    assert cache.diagnostics.entries == 32  # 24 distinct clouds and 8 kept images.


def test_legacy_two_field_image_entries_cannot_supply_gradient_metadata():
    frame = _frames(true_transform(), 1)[0]
    cache = EdgePreparationCache()
    settings = (
        FAST.image_max_width,
        FAST.spill_gamma,
        FAST.spill_radius_px,
        FAST.edge_background_sigma_px,
    )
    legacy = cache._get_or_compute(
        "image-edge-response/v1",
        frame.image,
        settings,
        lambda: edge.image_edge_response(
            frame.image,
            max_width=FAST.image_max_width,
            gamma=FAST.spill_gamma,
            radius=FAST.spill_radius_px,
            background_sigma=FAST.edge_background_sigma_px,
        ),
    )
    prepared, dropped = edge.prepare_frames(
        [frame, replace(frame, time_s=1.0, image_time_s=1.0)],
        CAMERA,
        true_transform(),
        FAST,
        preparation_cache=cache,
    )
    assert len(prepared) == 1 and "repeats the last kept view" in dropped[0]
    assert prepared[0].edge.tobytes() == legacy[0].tobytes()
    assert cache.diagnostics.misses == 3  # legacy image, current image, depth.


@pytest.mark.parametrize("changed_frame", [0, 1])
def test_progress_mutation_keeps_the_processed_image_and_its_certificate_bound(
    monkeypatch, changed_frame
):
    first, different = _frames(true_transform(), 2)

    def run(certified):
        frames = [
            replace(first, image=first.image.copy()),
            replace(first, image=first.image.copy(), time_s=1.0, image_time_s=1.0),
        ]
        if changed_frame == 0:
            frames[1].image.fill(0)

        def mutate(index, total):
            if index == changed_frame + 1:
                frames[changed_frame].image[:] = 0 if changed_frame == 0 else different.image

        with monkeypatch.context() as context:
            context.setattr(edge, "emit_tick", mutate)
            if not certified:
                context.setattr(
                    edge._RepeatedImageCertificate, "certified_factor", lambda *args: None
                )
            prepared, dropped = edge.prepare_frames(frames, CAMERA, true_transform(), FAST)
        return [(f.time_s, f.edge.tobytes(), f.points.tobytes()) for f in prepared], dropped

    expected = run(False)
    assert run(True) == expected


@pytest.mark.parametrize("unstructured", [False, True])
def test_depth_visibility_and_flat_rejection_keep_their_original_priority(
    monkeypatch, unstructured
):
    frames = _repeated(true_transform(), unstructured)
    frames.insert(1, replace(frames[0], image=np.zeros_like(frames[0].image), time_s=0.5))
    for candidate in [true_transform(), perturbed(true_transform(), (20.0, 3.0, 0.0))]:
        actual, rejected = edge.prepare_frames(frames, CAMERA, candidate, FAST)
        with monkeypatch.context() as reference:
            reference.setattr(
                edge._RepeatedImageCertificate, "certified_factor", lambda *args: None
            )
            expected, dropped = edge.prepare_frames(frames, CAMERA, candidate, FAST)
        assert rejected == dropped
        assert "image has no edges" in dropped[0]
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert (a.time_s, a.scale, a.rings) == (b.time_s, b.scale, b.rings)
            for x, y in [(a.edge, b.edge), (a.points, b.points), (a.weights, b.weights)]:
                assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
