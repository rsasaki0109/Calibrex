"""Leaf-coordinate packing preserves every fused output and reduction bit."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers.cuda_lidar_map_adapter import _CudaTree
from calibrex.solvers.cuda_scan_odometry_adapter import _FUSED_SOURCE, CudaOdometryTarget
from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


@pytest.mark.parametrize("count", [0, 1, 257, 8193])
def test_packed_coordinates_preserve_original_ids_and_payload(count: int) -> None:
    points = np.random.default_rng(221).normal(size=(count, 3))
    tree = _CudaTree(points)
    assert cp.asnumpy(tree.data).tobytes() == points.tobytes()
    assert tree.packed.flags.c_contiguous
    assert cp.asnumpy(tree.packed).tobytes() == points[cp.asnumpy(tree.arrays[0])].tobytes()
    assert tree.bytes == tree.data.nbytes + tree.packed.nbytes + sum(
        array.nbytes for array in tree.arrays
    )


@pytest.mark.parametrize("translation", [0.0, 0.02, 1.5])
def test_packed_fused_kernel_preserves_all_outputs_and_system_bits(translation: float) -> None:
    rng = np.random.default_rng(222)
    points = rng.normal(size=(8193, 3))
    holder = CudaOdometryTarget(points, ScanOdometryOptions())
    statistics = holder.statistics()
    assert statistics["gpu_packed_reference_rows"] == len(points)
    assert statistics["gpu_packed_reference_bytes"] == points.nbytes
    holder._ensure_all()
    original = _FUSED_SOURCE
    for axis in range(3):
        suffix = "" if axis == 0 else f"+{axis}"
        packed = f"packed_reference[3*v{suffix}]"
        assert original.count(packed) == 1
        original = original.replace(packed, f"reference[3*position{suffix}]")
    baseline = cp.RawKernel(original, "project_query_system3", options=("--fmad=false",))
    matrix = np.eye(4)
    matrix[:3, 3] = [translation, -translation, translation]
    moving = cp.asarray(points[::2])
    count = len(moving)
    results = []
    for kernel in (baseline, holder.fused_kernel):
        uncertain = cp.empty(count, dtype=cp.float64)
        terms = cp.empty((31, count), dtype=cp.float64)
        moved = cp.empty_like(moving)
        distances = cp.empty((count, 2), dtype=cp.float64)
        indexes = cp.empty((count, 2), dtype=cp.int64)
        pending = cp.empty(count, dtype=cp.int64)
        kernel(((count + 127) // 128,), (128,), (
            moving, cp.asarray(matrix), holder.tree.data, holder.tree.packed,
            *holder.tree.arrays, holder._normals, holder._planar, holder.done,
            np.int32(count), np.int32(len(points)), np.float64(1.5), np.float64(0.1),
            np.float64(1e-12), uncertain, terms, moved, distances, indexes, pending,
        ))
        results.append([cp.asnumpy(value).tobytes() for value in (
            uncertain, terms, moved, distances, indexes, pending, cp.sum(terms, axis=1),
        )])
    assert results[0] == results[1]
