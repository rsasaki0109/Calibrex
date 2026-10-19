"""Ordered preparation, bounded worker selection and CPU/CUDA setup overlap."""

from __future__ import annotations

import sys
from threading import Barrier, Event, get_ident
from types import ModuleType

import numpy as np
import pytest

from calibrex.solvers.levinson_online_preparation import (
    LevinsonOnlineImage,
    prepare_levinson_online_frames,
)
from calibrex.solvers.levinson_thrun_online_solver import (
    LevinsonCameraModel,
    levinson_edge_response,
)


def _image(index: int) -> LevinsonOnlineImage:
    return LevinsonOnlineImage(
        str(index),
        np.arange(48.0).reshape(6, 8) + index,
        np.array([[0.0, 0.0, 5.0]]),
        np.ones(1),
        LevinsonCameraModel(8, 6, 2.0, 2.0, 4.0, 3.0),
    )


def test_sequential_loading_stays_on_caller():
    called = []

    def load(index):
        called.append(get_ident())
        return _image(index)

    frames = prepare_levinson_online_frames([2, 0, 1], load, max_radius=2)
    assert [frame.frame_id for frame in frames] == ["2", "0", "1"]
    assert called == [get_ident()] * 3


def test_parallel_loading_is_concurrent_and_ordered():
    barrier = Barrier(3, timeout=5)
    main = get_ident()

    def load(index):
        assert get_ident() != main
        barrier.wait()
        return _image(index)

    actual = prepare_levinson_online_frames([2, 0, 1], load, workers=3, max_radius=2)
    expected = prepare_levinson_online_frames([2, 0, 1], _image, max_radius=2)
    assert [frame.frame_id for frame in actual] == ["2", "0", "1"]
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            left.edge_response.view(np.uint64), right.edge_response.view(np.uint64)
        )


def test_cuda_setup_overlaps_loading_and_evaluation_stays_on_caller(monkeypatch):
    loading = Event()
    initialized = Event()
    main = get_ident()
    evaluated = []

    class Runtime:
        def __init__(self):
            assert get_ident() == main
            assert loading.wait(5)
            initialized.set()

        def evaluate(self, image, **options):
            assert get_ident() == main
            evaluated.append(True)
            return levinson_edge_response(image, **options)

    module = ModuleType("calibrex.solvers.cuda_online_edge_adapter")
    module.CudaEdgeImageRuntime = Runtime
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def load(index):
        loading.set()
        assert initialized.wait(5)
        return _image(index)

    frames = prepare_levinson_online_frames(
        [2, 0, 1], load, workers=3, evaluation_backend="cuda", max_radius=2
    )
    assert [frame.frame_id for frame in frames] == ["2", "0", "1"]
    assert len(evaluated) == 3


@pytest.mark.parametrize("workers", [0, -1, 33, True, 1.5])
def test_invalid_workers_do_not_read_inputs(workers):
    def fail(index):
        raise AssertionError("invalid configuration must not read inputs")

    with pytest.raises(ValueError, match="workers"):
        prepare_levinson_online_frames([0], fail, workers=workers)


def test_loader_exception_is_preserved():
    def fail(index):
        raise OSError("input probe")

    with pytest.raises(OSError, match="input probe"):
        prepare_levinson_online_frames([0, 1, 2], fail, workers=3)
