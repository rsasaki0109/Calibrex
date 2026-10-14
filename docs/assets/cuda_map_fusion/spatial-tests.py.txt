"""Device construction, exact bounds and lazy native repair snapshots."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers import cuda_lidar_map_adapter as adapter
from calibrex.solvers.cuda_lidar_map_adapter import _CudaTree


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 63, 64, 65, 4097, 65537])
def test_device_bvh_has_exact_leaf_and_parent_bounds_without_native_build(count: int) -> None:
    points = np.random.default_rng(215).normal(size=(count, 3))
    points[:count // 7] = 0
    tree = _CudaTree(points)
    assert tree._cpu is None and tree.native_tree_builds == 0
    assert tree.capture_method == "device_morton_bvh"
    order, axis, _split, children, spans, boxes = (cp.asnumpy(value) for value in tree.arrays)
    np.testing.assert_array_equal(np.sort(order), np.arange(count))
    leaves = np.flatnonzero(axis == -1)
    internal = np.flatnonzero(axis >= 0)
    assert len(axis) == 2 * len(leaves) - 1
    np.testing.assert_array_equal(spans[0], [0, count])
    valid = leaves[spans[leaves, 1] > spans[leaves, 0]]
    valid = valid[np.argsort(spans[valid, 0])]
    empty = leaves[spans[leaves, 1] == spans[leaves, 0]]
    if count:
        ordered = points[order]
        np.testing.assert_array_equal(boxes[valid, 0],
                                      np.minimum.reduceat(ordered, spans[valid, 0], axis=0))
        np.testing.assert_array_equal(boxes[valid, 1],
                                      np.maximum.reduceat(ordered, spans[valid, 0], axis=0))
    assert np.isposinf(boxes[empty, 0]).all()
    assert np.isneginf(boxes[empty, 1]).all()
    left, right = children[internal].T
    np.testing.assert_array_equal(np.sort(children[internal].ravel()), np.arange(1, len(axis)))
    np.testing.assert_array_equal(spans[internal, 0], np.minimum(spans[left, 0], spans[right, 0]))
    np.testing.assert_array_equal(spans[internal, 1], np.maximum(spans[left, 1], spans[right, 1]))
    np.testing.assert_array_equal(boxes[internal, 0], np.minimum(boxes[left, 0], boxes[right, 0]))
    np.testing.assert_array_equal(boxes[internal, 1], np.maximum(boxes[left, 1], boxes[right, 1]))
    assert tree.build_kernel_calls == 3 + int(count > 0) + int(count > 1)
    split_nodes = internal[axis[internal] < 3]
    for node in split_nodes:
        dimension = axis[node]
        left, right = children[node]
        assert boxes[left, 1, dimension] <= _split[node] <= boxes[right, 0, dimension]


def test_query_and_native_repair_keep_a_snapshot_and_build_native_only_once(monkeypatch) -> None:
    points = np.random.default_rng(216).normal(size=(129, 3))
    captured = points.copy()
    calls = []
    native = adapter.cKDTree

    def measured(*args, **kwargs):
        calls.append(1)
        return native(*args, **kwargs)

    monkeypatch.setattr(adapter, "cKDTree", measured)
    tree = _CudaTree(points)
    points[:] = 0
    distances, indexes = tree.query(cp.asarray(captured[:13]), 2)
    assert calls == []
    np.testing.assert_array_equal(cp.asnumpy(indexes)[:, 0], np.arange(13))
    np.testing.assert_array_equal(cp.asnumpy(distances)[:, 0], 0)
    expected = native(captured)
    for actual, baseline in zip(tree.cpu.query(captured[:13], k=2),
                                expected.query(captured[:13], k=2), strict=True):
        np.testing.assert_array_equal(actual, baseline)
    assert tree.cpu is tree.cpu and calls == [1] and tree.native_tree_builds == 1


def test_empty_device_tree_keeps_native_missing_neighbor_sentinels() -> None:
    tree = _CudaTree(np.empty((0, 3)))
    distances, indexes = tree.query(cp.asarray([[0., 0., 0.]]), 17)
    assert np.isposinf(cp.asnumpy(distances)).all()
    np.testing.assert_array_equal(cp.asnumpy(indexes), np.zeros((1, 17), dtype=np.int64))
