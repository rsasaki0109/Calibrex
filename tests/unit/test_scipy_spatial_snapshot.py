"""CPU-only topology validation and compatibility fallback checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.spatial import cKDTree

from calibrex.solvers.scipy_spatial_snapshot import capture_spatial_tree


@pytest.mark.parametrize("count", [1, 17, 1001])
@pytest.mark.parametrize("collapsed", [False, True])
def test_bulk_capture_matches_every_public_node_and_exact_subtree_bounds(
    count: int, collapsed: bool,
) -> None:
    points = np.random.default_rng(73).normal(size=(count, 3))
    if collapsed:
        points[:, 1:] = 0
        points[:count // 2] = 0
    tree = cKDTree(points)
    (axes, splits, children, spans), bounds, method = capture_spatial_tree(tree)
    assert method == ("validated_serialized" if np.dtype(np.intp).itemsize == 8 else "public_nodes")
    remaining = [(0, tree.tree)]
    while remaining:
        index, node = remaining.pop()
        assert axes[index] == node.split_dim
        assert splits[index] == node.split
        assert spans[index].tolist() == [node.start_idx, node.end_idx]
        selected = points[tree.indices[node.start_idx:node.end_idx]]
        np.testing.assert_array_equal(bounds[index], [selected.min(axis=0), selected.max(axis=0)])
        if node.split_dim >= 0:
            remaining.extend([(children[index, 0], node.lesser),
                              (children[index, 1], node.greater)])


@pytest.mark.parametrize("damage", ["layout", "child", "span", "split", "axis"])
def test_unknown_or_inconsistent_serialization_uses_the_public_fallback(damage: str) -> None:
    class AlteredTree(cKDTree):
        def __getstate__(self) -> tuple[Any, ...]:
            state = list(super().__getstate__())
            raw = state[0].copy()
            if damage == "layout":
                raw = raw[:-1]
            elif damage == "child":
                raw.view(np.int64)[7] = -1
            elif damage == "span":
                raw.view(np.int64)[4] -= 1
            elif damage == "split":
                raw.view(np.float64)[2] = 1e20
            else:
                raw.view(np.int64)[0] = 5
            state[0] = raw
            return tuple(state)

    points = np.random.default_rng(6).normal(size=(135, 3))
    expected = capture_spatial_tree(cKDTree(points))
    actual = capture_spatial_tree(AlteredTree(points))
    assert actual[2] == "public_nodes"
    for a, b in zip(actual[0], expected[0], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(actual[1], expected[1])
