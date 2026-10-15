"""Real-device queries against exhaustive FP64 distances, including box boundaries."""

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


@pytest.mark.parametrize("k", [1, 2, 11, 17])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_queries_keep_exhaustive_neighbors_on_thin_oblique_clouds(k: int, scale: float) -> None:
    rng = np.random.default_rng(174)
    t = rng.uniform(-20, 20, 517)
    reference = np.column_stack((t, 2 * t, -3 * t))
    reference += rng.normal(0, .01, reference.shape)
    reference *= scale
    queries = rng.uniform(-40, 40, (63, 3)) * scale
    delta = queries[:, None] - reference
    squared = delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2
    expected = np.argsort(squared, axis=1, kind="stable")[:, :k]
    tree = _CudaTree(reference)
    distances, indexes = tree.query(cp.asarray(queries), k)
    np.testing.assert_array_equal(cp.asnumpy(indexes), expected)
    np.testing.assert_allclose(
        cp.asnumpy(distances), np.sqrt(np.take_along_axis(squared, expected, axis=1)),
        rtol=2e-15, atol=0,
    )


def test_queries_keep_ties_and_adjacent_floats_at_subtree_box_edges() -> None:
    # A point just beyond a box must not cause an equal-distance branch to disappear.
    reference = np.array([[x, y, z] for x in range(-3, 4) for y in range(-3, 4)
                          for z in (-1., 0., 1.)])
    reference = np.concatenate((reference, reference[:20]))
    queries = np.array([[x, y, z] for x in (.5, np.nextafter(.5, 0),
                                         np.nextafter(.5, 1), 100.)
                        for y in (-.5, .5, 0.) for z in (-1., 0., 1.)])
    delta = queries[:, None] - reference
    squared = delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2
    expected = np.argsort(squared, axis=1, kind="stable")[:, :17]
    distances, indexes = _CudaTree(reference).query(cp.asarray(queries), 17)
    np.testing.assert_array_equal(cp.asnumpy(indexes), expected)
    np.testing.assert_array_equal(
        cp.asnumpy(distances), np.sqrt(np.take_along_axis(squared, expected, axis=1)),
    )
