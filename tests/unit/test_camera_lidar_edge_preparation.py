"""Bounded independent preparation with native frame selection and all evidence."""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import jsonschema
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

from calibrex.core.camera_lidar_edge import camera_lidar_edge_json_schema
from calibrex.evaluation import camera_lidar_edge as edge


@pytest.mark.parametrize("workers", [0, -1, 33, True, 1.0, "4"])
def test_invalid_worker_budget_is_rejected(workers):
    with pytest.raises(ValueError, match="edge preparation workers"):
        replace(FAST, preparation_workers=workers)


@pytest.mark.parametrize("workers", [2, 4, 32])
@pytest.mark.parametrize("unstructured", [False, True])
def test_ordered_selection_and_prepared_bits_equal(workers, unstructured):
    truth = true_transform()
    frames = _repeated(truth, unstructured)
    frames.insert(1, replace(frames[0], image=np.zeros_like(frames[0].image), time_s=0.5))
    snapshots = [(frame.image.tobytes(), frame.points.tobytes()) for frame in frames]
    expected, dropped = edge.prepare_frames(frames, CAMERA, truth, FAST)
    actual, rejected = edge.prepare_frames(
        frames, CAMERA, truth, replace(FAST, preparation_workers=workers)
    )
    assert rejected == dropped
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert (left.time_s, left.gap_s, left.scale, left.rings) == (
            right.time_s,
            right.gap_s,
            right.scale,
            right.rings,
        )
        for array, native in (
            (left.edge, right.edge),
            (left.points, right.points),
            (left.weights, right.weights),
        ):
            assert array.dtype == native.dtype and array.shape == native.shape
            assert array.tobytes() == native.tobytes()
    assert [(frame.image.tobytes(), frame.points.tobytes()) for frame in frames] == snapshots


def test_completion_order_does_not_change_frame_order(monkeypatch):
    truth = true_transform()
    frames = _frames(truth, 4)
    ready = threading.Event()
    original = edge._prepare_edge_inputs
    completed = []

    def reordered(frame, options):
        if frame.time_s == frames[0].time_s:
            assert ready.wait(timeout=10)
        result = original(frame, options)
        completed.append(frame.time_s)
        if frame.time_s == frames[1].time_s:
            ready.set()
        return result

    monkeypatch.setattr(edge, "_prepare_edge_inputs", reordered)
    actual, _ = edge.prepare_frames(frames, CAMERA, truth, replace(FAST, preparation_workers=4))
    assert completed.index(frames[1].time_s) < completed.index(frames[0].time_s)
    assert [frame.time_s for frame in actual] == [frame.time_s for frame in frames]


def test_queue_is_bounded_and_early_close_shuts_down(monkeypatch):
    from concurrent import futures

    submitted, shutdown = [], []

    class Immediate:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class Executor:
        def __init__(self, *, max_workers, thread_name_prefix):
            assert max_workers == 3

        def submit(self, function, frame, options):
            submitted.append(frame)
            return Immediate(None)

        def shutdown(self, *, wait, cancel_futures):
            shutdown.append((wait, cancel_futures))

    monkeypatch.setattr(futures, "ThreadPoolExecutor", Executor)
    frames = list(range(100))
    pending = edge._ordered_edge_inputs(frames, replace(FAST, preparation_workers=3))
    assert next(pending) is None
    assert len(submitted) == 3
    assert next(pending) is None
    assert len(submitted) == 4
    pending.close()
    assert shutdown == [(True, True)]


@pytest.mark.parametrize("consumer_error", [False, True])
def test_worker_and_consumer_errors_propagate_without_leaking_threads(monkeypatch, consumer_error):
    truth = true_transform()
    frames = _frames(truth, 4)

    def failure(*args, **kwargs):
        raise RuntimeError("injected preparation failure")

    if consumer_error:
        monkeypatch.setattr(edge.EdgeCamera, "project", failure)
    else:
        monkeypatch.setattr(edge, "_prepare_edge_inputs", failure)
    with pytest.raises(RuntimeError, match="injected preparation failure"):
        edge.prepare_frames(frames, CAMERA, truth, replace(FAST, preparation_workers=4))
    assert not any(
        thread.name.startswith("calibrex-offline-edge") for thread in threading.enumerate()
    )


def test_single_worker_and_empty_input_need_no_executor(monkeypatch):
    from concurrent import futures

    def unexpected(*args, **kwargs):
        raise AssertionError("unexpected executor")

    monkeypatch.setattr(futures, "ThreadPoolExecutor", unexpected)
    truth = true_transform()
    edge.prepare_frames(_frames(truth, 1), CAMERA, truth, FAST)
    assert edge.prepare_frames([], CAMERA, truth, replace(FAST, preparation_workers=4)) == ([], [])


@pytest.mark.parametrize("translation", [False, True])
def test_complete_estimation_and_controls_keep_native_bits(translation):
    truth = true_transform()
    frames = _frames(truth)
    candidate = perturbed(truth, (0.8, -1.2, 0.0))
    options = replace(FAST, estimate_translation=translation)
    provenance = _provenance()
    native = edge.estimate_from_frames(
        frames, CAMERA, candidate, options=options, provenance=provenance
    ).model_dump(mode="json")
    parallel = edge.estimate_from_frames(
        frames,
        CAMERA,
        candidate,
        options=replace(options, preparation_workers=4),
        provenance=provenance,
    ).model_dump(mode="json")
    jsonschema.validate(parallel, camera_lidar_edge_json_schema())
    assert parallel["options"].pop("preparation_workers") == 4
    assert native["options"].pop("preparation_workers") == 1
    assert json.dumps(parallel, sort_keys=True) == json.dumps(native, sort_keys=True)
