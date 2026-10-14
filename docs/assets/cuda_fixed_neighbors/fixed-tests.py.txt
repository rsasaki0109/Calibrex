"""Specialized queries retain exhaustive distances, ties and missing neighbors."""

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


@pytest.mark.parametrize("k", range(1, 18))
def test_fixed_queries_match_exhaustive_reference_and_generic_bytes(k: int) -> None:
    rng = np.random.default_rng(617)
    reference = rng.normal(size=(129, 3))
    reference = np.concatenate((reference, reference[:17], np.zeros((9, 3))))
    points = np.concatenate((reference[::5], rng.normal(size=(23, 3)), [[0, 0, 0]]))
    tree = _CudaTree(reference)
    queries = cp.asarray(points)
    distances, indexes = tree.query(queries, k)
    generic_distances = cp.empty_like(distances)
    generic_indexes = cp.empty_like(indexes)
    tree.kernel(
        ((len(points) + 127) // 128,), (128,),
        (queries, tree.data, *tree.arrays, np.int32(len(points)), np.int32(len(reference)),
         np.int32(k), generic_distances, generic_indexes),
    )
    delta = points[:, None] - reference
    squared = delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2
    expected = np.argsort(squared, axis=1, kind="stable")[:, :k]
    actual_distances = cp.asnumpy(distances)
    actual_indexes = cp.asnumpy(indexes)
    np.testing.assert_array_equal(actual_indexes, expected)
    np.testing.assert_array_equal(actual_distances,
                                  np.sqrt(np.take_along_axis(squared, expected, axis=1)))
    assert actual_distances.tobytes() == cp.asnumpy(generic_distances).tobytes()
    assert actual_indexes.tobytes() == cp.asnumpy(generic_indexes).tobytes()
    assert tree.fixed_query_calls == tree.query_calls == 1
    assert tree.fixed_query_rows == tree.query_rows == len(points)
    first_kernel = tree._fixed_kernels[k]
    tree.query(queries, k)
    assert tree._fixed_kernels[k] is first_kernel


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_fixed_queries_keep_infinite_missing_neighbors(count: int, scale: float) -> None:
    reference = np.arange(3 * count, dtype=float).reshape(count, 3) * scale
    tree = _CudaTree(reference)
    queries = cp.asarray([[0.0, 0.0, 0.0], [scale, -scale, scale]])
    distances, indexes = tree.query(queries, 17)
    actual_distances, actual_indexes = cp.asnumpy(distances), cp.asnumpy(indexes)
    assert np.isinf(actual_distances[:, count:]).all()
    assert (actual_indexes[:, count:] == count).all()
    assert np.isfinite(actual_distances[:, :count]).all()
    assert tree.fixed_query_calls == 1
    empty_d, empty_i = tree.query(cp.empty((0, 3)), 17)
    assert empty_d.shape == empty_i.shape == (0, 17)
    assert tree.fixed_query_calls == 1
