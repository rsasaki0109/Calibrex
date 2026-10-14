"""Device scheduling preserves original rows, exact terms and registration evidence."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers.cuda_scan_odometry_adapter import CudaOdometryTarget
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


def _scene() -> np.ndarray:
    rng = np.random.default_rng(218)
    points = rng.uniform(-5, 5, (6000, 3))
    for axis in range(3):
        points[2000 * axis : 2000 * (axis + 1), axis] = 4
    return points


@pytest.mark.parametrize("count", [0, 31, 4095, 4096, 8193])
def test_order_is_a_bijection_and_skips_spatial_sort_for_small_sources(count: int) -> None:
    target = CudaOdometryTarget(_scene(), ScanOdometryOptions())
    source = np.random.default_rng(19).normal(size=(count, 3))
    order = cp.asnumpy(target._source_order(source, cp.asarray(source)))
    np.testing.assert_array_equal(np.sort(order), np.arange(count))
    if count < 4096:
        np.testing.assert_array_equal(order, np.arange(count))
    statistics = target.statistics()
    assert statistics["gpu_source_order_calls"] == int(count >= 4096)
    assert statistics["gpu_source_order_kernel_calls"] == int(count >= 4096)
    assert statistics["gpu_source_order_rows"] == (count if count >= 4096 else 0)


def test_duplicate_codes_keep_source_order_and_repeat_deterministically() -> None:
    target = CudaOdometryTarget(_scene(), ScanOdometryOptions())
    source = np.full((5000, 3), 1.25)
    points = cp.asarray(source)
    for _ in range(2):
        np.testing.assert_array_equal(cp.asnumpy(target._source_order(source, points)),
                                      np.arange(len(source)))
    assert target.statistics()["gpu_source_order_calls"] == 2


@pytest.mark.parametrize("translation", [0., .15, 1.5])
def test_spatial_scheduling_keeps_every_fused_output_bit(translation: float) -> None:
    host = _scene()
    options = ScanOdometryOptions()
    target = CudaOdometryTarget(host, options)
    _ = target.normals
    source = host + np.random.default_rng(20).normal(0, .002, host.shape)
    points = cp.asarray(source)
    transform = np.eye(4)
    transform[:3, 3] = [translation, -.1, .05]
    matrix = cp.asarray(transform)
    spatial = target._source_order(source, points)
    assert not np.array_equal(cp.asnumpy(spatial), np.arange(len(source)))
    expected = None
    for order in (cp.arange(len(source), dtype=cp.int64), spatial):
        uncertain = cp.empty(len(source), dtype=cp.float64)
        terms = cp.empty((31, len(source)), dtype=cp.float64)
        moved = cp.empty_like(points)
        distances = cp.empty((len(source), 2), dtype=cp.float64)
        indices = cp.empty((len(source), 2), dtype=cp.int64)
        pending = cp.empty(len(source), dtype=cp.int64)
        target.fused_kernel(((len(source) + 127) // 128,), (128,), (
            order, points, matrix, target.tree.data, *target.tree.arrays,
            target._normals, target._planar, target.done,
            np.int32(len(source)), np.int32(len(host)),
            np.float64(options.max_correspondence_m), np.float64(options.huber_delta_m),
            np.float64(1e-12), uncertain, terms, moved, distances, indices, pending,
        ))
        actual = [cp.asnumpy(value) for value in
                  (uncertain, terms, moved, distances, indices, pending, cp.sum(terms, axis=1))]
        if expected is None:
            expected = actual
        for before, after in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(before.view(np.uint8), after.view(np.uint8))


def test_registration_reuses_one_order_and_keeps_exact_uncanonicalized_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _scene()
    options = ScanOdometryOptions()
    source = host + np.array([.13, -.06, .08])
    target = CudaOdometryTarget(host, options)
    actual = target.register(source, np.eye(4))
    stats = target.statistics()
    assert stats["gpu_source_order_calls"] == 1
    assert stats["gpu_source_order_rows"] == len(source)
    assert stats["gpu_fused_system_calls"] > 1
    monkeypatch.setattr(target, "_source_order",
                        lambda host, points: cp.arange(len(points), dtype=cp.int64))
    expected = target.register(source, np.eye(4))
    np.testing.assert_array_equal(actual.transform.view(np.uint64),
                                  expected.transform.view(np.uint64))
    for field in ("converged", "iterations", "correspondences", "rmse_m",
                  "hessian_min_eigenvalue", "execution_backend"):
        assert getattr(actual, field) == getattr(expected, field)
