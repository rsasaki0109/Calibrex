"""Candidate queries keep exhaustive neighbors; runtime adapters stay unchanged."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp
from tools.benchmark_cuda_spatial_candidates import _heap_source

from calibrex.solvers.cuda_lidar_map_adapter import _SOURCE, _CudaTree


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


@pytest.mark.parametrize("k", [1, 2, 11, 17])
@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
@pytest.mark.parametrize("sorted_output", [False, True])
def test_heap_queries_keep_exhaustive_neighbors_and_sorted_output(
    k: int, scale: float, sorted_output: bool,
) -> None:
    rng = np.random.default_rng(214)
    reference = rng.normal(size=(257, 3)) * scale
    reference = np.concatenate((reference, reference[:23], [[-scale, 0, 0], [scale, 0, 0]]))
    points = np.concatenate((reference[::7], rng.normal(size=(143, 3)) * scale,
                             [[0, 0, 0], [100 * scale, 0, 0]]))
    tree = _CudaTree(reference)
    distances = cp.empty((len(points), k), dtype=cp.float64)
    indexes = cp.empty((len(points), k), dtype=cp.int64)
    kernel = cp.RawKernel(_heap_source(_SOURCE, sorted_output=sorted_output), "query3",
                         options=("--fmad=false",))
    kernel(((len(points) + 127) // 128,), (128,),
           (cp.asarray(points), tree.data, tree.packed, *tree.arrays, np.int32(len(points)),
            np.int32(len(reference)), np.int32(k), distances, indexes))
    delta = points[:, None] - reference
    squared = delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2
    expected = np.argsort(squared, axis=1, kind="stable")[:, :k]
    np.testing.assert_array_equal(cp.asnumpy(indexes), expected)
    np.testing.assert_array_equal(cp.asnumpy(distances),
                                  np.sqrt(np.take_along_axis(squared, expected, axis=1)))
    native_distances, native_indexes = tree.query(cp.asarray(points), k)
    np.testing.assert_array_equal(cp.asnumpy(distances), cp.asnumpy(native_distances))
    np.testing.assert_array_equal(cp.asnumpy(indexes), cp.asnumpy(native_indexes))


def test_heap_source_rejects_an_unknown_query_shader() -> None:
    with pytest.raises(ValueError, match="requires review"):
        _heap_source("unrecognized query")


def test_query_policy_generation_round_trips() -> None:
    sorted_source = _heap_source(_SOURCE, sorted_output=True)
    assert sorted_source != _SOURCE
    assert _heap_source(sorted_source) == _SOURCE
    assert _heap_source(_SOURCE) == _SOURCE
    assert _heap_source(sorted_source, sorted_output=True) == sorted_source
