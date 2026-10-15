"""Conservative pruning, FP64 membership, stream ownership and native fallbacks."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers.cuda_lidar_map_adapter import _CudaTree
from calibrex.solvers.cuda_scan_odometry_adapter import CudaOdometryTarget, execution_identity
from calibrex.solvers.cuda_spatial_interval_adapter import _CudaIntervalTree
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


@pytest.mark.parametrize("scale,shift", [
    (1e-160, 0.), (1e-24, 0.), (1e-8, 0.), (1., 0.), (1e4, 0.),
    (1e24, 0.), (1e45, 0.), (1e-5, 1e4), (1., 1e4),
])
def test_query_distances_and_membership_match_native_bits(scale: float, shift: float) -> None:
    rng = np.random.default_rng(120)
    points = rng.normal(size=(128, 3)) * scale + shift
    queries = cp.asarray(rng.normal(size=(32, 3)) * scale + shift)
    native, intervals = _CudaTree(points), _CudaIntervalTree(points)
    for k in (1, 2, 11, 17):
        expected = native.query(queries, k)
        actual = intervals.query(queries, k)
        np.testing.assert_array_equal(cp.asnumpy(actual[0]).view(np.uint64),
                                      cp.asnumpy(expected[0]).view(np.uint64))
        np.testing.assert_array_equal(cp.asnumpy(actual[1]), cp.asnumpy(expected[1]))
    assert intervals.query_rows == 128
    if scale == 1. and shift == 0.:
        assert intervals.interval_query_rows == 128


def test_duplicate_ties_and_cross_stream_queries_preserve_membership() -> None:
    points = np.repeat(np.random.default_rng(12).normal(size=(32, 3)), 4, axis=0)
    expected = _CudaTree(points).query(cp.asarray(points[::4]), 17)
    producer, consumer = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
    with producer:
        tree = _CudaIntervalTree(points)
    with consumer:
        actual = tree.query(cp.asarray(points[::4]), 17)
        assert cp.cuda.get_current_stream() == consumer
        np.testing.assert_array_equal(cp.asnumpy(actual[0]).view(np.uint64),
                                      cp.asnumpy(expected[0]).view(np.uint64))
        np.testing.assert_array_equal(cp.asnumpy(actual[1]), cp.asnumpy(expected[1]))
    assert tree.interval_query_rows == 32


@pytest.mark.parametrize("count", [31, 65537])
def test_interval_payload_eligibility_retains_native_tree(count: int) -> None:
    tree = _CudaIntervalTree(np.random.default_rng(10).normal(size=(count, 3)))
    assert tree.interval_arrays is None and not tree.select_intervals(True)


def test_abandoned_tree_drains_its_preparation_writes() -> None:
    tree = _CudaIntervalTree(np.random.default_rng(10).normal(size=(3600, 3)))
    event, waited = tree._ready, []
    assert event is not None

    class ObservedEvent:
        def synchronize(self) -> None:
            waited.append(True)
            event.synchronize()

    tree._ready = ObservedEvent()
    reference = weakref.ref(tree)
    del tree
    gc.collect()
    assert reference() is None and waited == [True] and event.done


def test_registration_matches_native_kernel_and_counts_real_interval_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(216)
    faces = []
    for axis in range(3):
        points = rng.uniform(-4., 4., (1000, 3))
        points[:, axis] = 3.
        faces.append(points)
    points = np.concatenate(faces)
    source = points - [.05, -.01, .02]
    native = CudaOdometryTarget(points, ScanOdometryOptions())
    choose = native.tree.select_intervals
    monkeypatch.setattr(native.tree, "select_intervals", lambda enabled: choose(False))
    native.tree.use_intervals = False
    expected = native.register(source, np.eye(4))
    target = CudaOdometryTarget(points, ScanOdometryOptions())
    actual = target.register(source, np.eye(4))
    np.testing.assert_array_equal(actual.transform.view(np.uint64),
                                  expected.transform.view(np.uint64))
    assert {k: v for k, v in vars(actual).items() if k != "transform"} == {
        k: v for k, v in vars(expected).items() if k != "transform"
    }
    counters = target.statistics()
    assert 0 < counters["gpu_interval_query_rows"] <= counters["gpu_query_rows"]
    assert 0 < counters["gpu_interval_system_calls"] == counters["gpu_fused_system_calls"]
    assert {k: v for k, v in counters.items() if not k.startswith("gpu_interval_")} == {
        k: v for k, v in native.statistics().items() if not k.startswith("gpu_interval_")
    }


def test_wide_initial_transform_uses_original_fused_kernel() -> None:
    points = np.random.default_rng(21).normal(size=(128, 3))
    target = CudaOdometryTarget(points, ScanOdometryOptions())
    initial = np.eye(4)
    initial[:3, 3] = 20.
    target.register(points, initial)
    assert target.statistics()["gpu_interval_system_calls"] == 0
    assert target.statistics()["gpu_fused_system_calls"] > 0


def test_registration_without_priming_waits_for_capture_across_streams() -> None:
    points = np.random.default_rng(26).normal(size=(1000, 3))
    expected = CudaOdometryTarget(points, ScanOdometryOptions()).register(points, np.eye(4))
    producer, consumer = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
    with producer:
        target = CudaOdometryTarget(points, ScanOdometryOptions())
    with consumer:
        actual = target.register(points, np.eye(4))
        assert cp.cuda.get_current_stream() == consumer
    np.testing.assert_array_equal(actual.transform.view(np.uint64),
                                  expected.transform.view(np.uint64))
    assert {k: v for k, v in vars(actual).items() if k != "transform"} == {
        k: v for k, v in vars(expected).items() if k != "transform"
    }


def test_frozen_source_hashes_match_actual_execution_identity() -> None:
    from tools.benchmark_bspline_lidar_imu import _scan_spatial_hashes

    query, system = _scan_spatial_hashes(Path(__file__).resolve().parents[2])
    identity = execution_identity()
    assert f"query_sha256={query};system_sha256={system};" in identity
    assert identity.endswith(";spatial_pruning=fp32-intervals-v1")
