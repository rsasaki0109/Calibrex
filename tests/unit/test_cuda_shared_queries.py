"""Register candidates retain the complete generic traversal's FP64 bytes."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers.cuda_lidar_map_adapter import _CudaTree


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


@pytest.mark.parametrize("k", [11])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
@pytest.mark.parametrize("count", [0, 1, 4097])
def test_register_queries_equal_generic_bytes_with_ties_and_missing_neighbors(
    k: int,
    scale: float,
    count: int,
) -> None:
    rng = np.random.default_rng(229)
    reference = rng.normal(size=(count, 3)) * scale
    reference[: count // 9] = 0
    tree = _CudaTree(reference)
    points = cp.asarray(np.concatenate((np.zeros((13, 3)), rng.normal(size=(244, 3)) * scale)))
    expected_distance = cp.empty((len(points), k), dtype=cp.float64)
    expected_index = cp.empty((len(points), k), dtype=cp.int64)
    tree.kernel(
        ((len(points) + 127) // 128,),
        (128,),
        (
            points,
            tree.data,
            *tree.arrays,
            np.int32(len(points)),
            np.int32(count),
            np.int32(k),
            expected_distance,
            expected_index,
        ),
    )
    distance, index = tree.query(points, k)
    assert cp.asnumpy(distance).tobytes() == cp.asnumpy(expected_distance).tobytes()
    assert cp.asnumpy(index).tobytes() == cp.asnumpy(expected_index).tobytes()
    assert tree.query_calls == tree.register_query_calls == 1
    assert tree.query_rows == tree.register_query_rows == len(points)


def test_shared_module_queries_keep_independent_tree_snapshots() -> None:
    points = np.random.default_rng(230).normal(size=(257, 3))
    first = _CudaTree(points)
    shifted = points + 10
    second = _CudaTree(shifted)
    for tree, values in ((first, points), (second, shifted), (first, points)):
        distances, indices = tree.query(cp.asarray(values), 11)
        np.testing.assert_array_equal(cp.asnumpy(distances)[:, 0], 0)
        np.testing.assert_array_equal(cp.asnumpy(indices)[:, 0], np.arange(len(values)))
    assert first.spatial_ops is second.spatial_ops
    first.query(cp.asarray(points), 2)
    assert first.query_calls == 3 and first.register_query_calls == 2
