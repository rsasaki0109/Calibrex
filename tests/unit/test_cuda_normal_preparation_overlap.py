"""Exact normal preparation across streams, abandoned targets and failed repairs."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers.cuda_scan_odometry_adapter import CudaOdometryTarget
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions, ScanRegistration


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


def _scene() -> np.ndarray:
    rng = np.random.default_rng(516)
    faces = []
    for axis in range(3):
        face = rng.uniform(-4., 4., (1200, 3))
        face[:, axis] = 3.
        faces.append(face)
    return np.concatenate(faces)


def _exact(left: ScanRegistration, right: ScanRegistration) -> None:
    np.testing.assert_array_equal(left.transform.view(np.uint64), right.transform.view(np.uint64))
    assert {k: v for k, v in vars(left).items() if k != "transform"} == {
        k: v for k, v in vars(right).items() if k != "transform"
    }


def test_priming_keeps_snapshots_and_registration_bits_across_streams() -> None:
    points = _scene()
    source = points - [.05, -.02, .01]
    opts = ScanOdometryOptions()
    ordinary = CudaOdometryTarget(points, opts)
    expected = ordinary.register(source, np.eye(4))
    producer, consumer = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
    with producer:
        target = CudaOdometryTarget(points, opts)
        target._prime_normals(len(source))
        pending = target._pending_normals
        assert pending is not None
        target._prime_normals(len(source))
        assert target._pending_normals is pending
    points[:] = 1e4
    with consumer:
        actual = target.register(source, np.eye(4))
        assert cp.cuda.get_current_stream() == consumer
    _exact(expected, actual)
    assert target._pending_normals is None and target._all_done
    assert target.statistics()["gpu_async_normal_batches"] == 1
    assert {k: v for k, v in target.statistics().items() if k != "gpu_async_normal_batches"} == {
        k: v for k, v in ordinary.statistics().items() if k != "gpu_async_normal_batches"
    }
    first = target.statistics().copy()
    target._prime_normals(len(source))
    assert target.statistics() == first


@pytest.mark.parametrize("rows", [0, 1, 359])
def test_sparse_source_hints_do_not_schedule_unneeded_full_normals(rows: int) -> None:
    target = CudaOdometryTarget(_scene(), ScanOdometryOptions())
    target._prime_normals(rows)
    assert target._pending_normals is None
    assert target.statistics()["gpu_normal_rows"] == 0


@pytest.mark.parametrize("count,primed", [(65536, True), (65537, False)])
def test_async_payload_is_limited_to_one_normal_chunk(count: int, primed: bool) -> None:
    points = np.random.default_rng(13).normal(size=(count, 3))
    target = CudaOdometryTarget(points, ScanOdometryOptions(normal_neighbours=4))
    target._prime_normals(count)
    assert (target._pending_normals is not None) == primed
    if primed:
        target._finish_normal_preparation()
        assert target._all_done
        assert target.statistics()["gpu_normal_kernel_calls"] == 1
    else:
        assert target.statistics()["gpu_normal_rows"] == 0


def test_failed_native_repair_invalidates_rows_and_can_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.repeat(np.column_stack((np.arange(40.), np.zeros((40, 2)))), 2, axis=0)
    target = CudaOdometryTarget(points, ScanOdometryOptions(normal_neighbours=4))
    repair = target._repair_normals

    def fail(*args: object) -> None:
        raise RuntimeError("injected native repair failure")

    monkeypatch.setattr(target, "_repair_normals", fail)
    target._prime_normals(len(points))
    with pytest.raises(RuntimeError, match="injected native"):
        target.register(points, np.eye(4))
    assert target._pending_normals is None and not target._all_done
    assert not cp.asnumpy(target.done).any()
    monkeypatch.setattr(target, "_repair_normals", repair)
    target.register(points, np.eye(4))
    assert target._all_done and cp.asnumpy(target.done).all()
    assert target.statistics()["gpu_normal_rows"] == 2 * len(points)
    assert target.statistics()["scalar_normal_repairs"] == len(points)


def test_failed_async_submission_drains_writes_and_retains_retry_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _scene()
    target = CudaOdometryTarget(points, ScanOdometryOptions())
    download = cp.asnumpy

    def fail_after_download(*args: object, **kwargs: object) -> object:
        download(*args, **kwargs)
        raise RuntimeError("injected submission failure after queued download")

    with monkeypatch.context() as patch:
        patch.setattr(cp, "asnumpy", fail_after_download)
        with pytest.raises(RuntimeError, match="injected submission"):
            target._prime_normals(len(points))
    assert target._pending_normals is None and not target._all_done
    assert not cp.asnumpy(target.done).any()
    ordinary = CudaOdometryTarget(points, ScanOdometryOptions())
    _exact(ordinary.register(points, np.eye(4)), target.register(points, np.eye(4)))
    assert target._all_done
    assert target.statistics()["gpu_normal_rows"] == 2 * len(points)


def test_abandoned_target_waits_before_releasing_gpu_and_pinned_buffers() -> None:
    target = CudaOdometryTarget(_scene(), ScanOdometryOptions())
    target._prime_normals(len(target.host))
    pending = target._pending_normals
    assert pending is not None
    event = pending.ready
    waited = []

    class ObservedEvent:
        def synchronize(self) -> None:
            waited.append(True)
            event.synchronize()

    pending.ready = ObservedEvent()
    reference = weakref.ref(target)
    del target
    gc.collect()
    assert reference() is None and waited == [True]
    assert event.done


def test_empty_source_after_priming_retains_empty_registration_semantics() -> None:
    points = _scene()
    ordinary = CudaOdometryTarget(points, ScanOdometryOptions())
    target = CudaOdometryTarget(points, ScanOdometryOptions())
    target._prime_normals(len(points))
    _exact(ordinary.register(np.empty((0, 3)), np.eye(4)),
           target.register(np.empty((0, 3)), np.eye(4)))
