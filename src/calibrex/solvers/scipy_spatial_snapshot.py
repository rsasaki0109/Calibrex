"""Validated bulk capture of SciPy's tree, with a public-node fallback.

The optional CUDA adapters need flat topology, never native pointers. The
64-bit serialized node layout follows SciPy's BSD-licensed ckdtreenode:
https://github.com/scipy/scipy/blob/v1.17.1/scipy/spatial/ckdtree/src/ckdtree_decl.h
Unknown layouts or inconsistent metadata use the public node attributes.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

Topology: TypeAlias = tuple[NDArray[np.int32], NDArray[np.float64],
                           NDArray[np.int32], NDArray[np.int64]]


def _serialized_topology(tree: cKDTree) -> Topology:
    raw = tree.__getstate__()[0]
    if (np.dtype(np.intp).itemsize != 8 or not isinstance(raw, np.ndarray)
            or raw.ndim != 1 or raw.dtype != np.dtype("S1")
            or not raw.flags.c_contiguous or raw.size != tree.size * 72):
        raise ValueError("unrecognized SciPy node layout")
    integers = raw.view(np.int64).reshape(-1, 9)
    axes, counts = integers[:, 0], integers[:, 1]
    spans, children = integers[:, 3:5], integers[:, 7:9]
    if (np.any((axes < -1) | (axes > 2))
            or np.any(counts != spans[:, 1] - spans[:, 0])
            or np.any(children[axes >= 0] >= tree.size)
            or np.any(children[axes >= 0] <= np.flatnonzero(axes >= 0)[:, None])):
        raise ValueError("inconsistent SciPy node layout")
    return (axes.astype(np.int32), raw.view(np.float64).reshape(-1, 9)[:, 2].copy(),
            children.astype(np.int32), spans.copy())


def _public_topology(tree: cKDTree) -> Topology:
    axes: list[int] = []
    splits: list[float] = []
    children: list[list[int]] = []
    spans: list[list[int]] = []

    def append(node: Any, depth: int = 0) -> int:
        if depth >= 63:
            raise ValueError("CUDA spatial tree exceeds the bounded traversal stack")
        index = len(axes)
        axes.append(node.split_dim)
        splits.append(node.split)
        children.append([0, 0])
        spans.append([node.start_idx, node.end_idx])
        if node.split_dim >= 0:
            children[index] = [append(node.lesser, depth + 1), append(node.greater, depth + 1)]
        return index

    append(tree.tree)
    return (np.asarray(axes, dtype=np.int32), np.asarray(splits, dtype=np.float64),
            np.asarray(children, dtype=np.int32), np.asarray(spans, dtype=np.int64))


def _subtree_bounds(tree: cKDTree, topology: Topology) -> NDArray[np.float64]:
    """Validate the partition and compute tight boxes in batches, one level at a time."""
    axes, splits, children, spans = topology
    count = len(axes)
    internal = np.flatnonzero(axes >= 0)
    if (spans[0].tolist() != [0, tree.n] or not np.isfinite(splits).all()
            or np.any(spans[:, 0] < 0) or np.any(spans[:, 1] > tree.n)
            or np.any(spans[:, 0] >= spans[:, 1])
            or not np.array_equal(np.sort(children[internal].ravel()), np.arange(1, count))):
        raise ValueError("inconsistent SciPy tree partition")
    left, right = children[internal].T
    if (not np.array_equal(spans[left, 0], spans[internal, 0])
            or not np.array_equal(spans[left, 1], spans[right, 0])
            or not np.array_equal(spans[right, 1], spans[internal, 1])):
        raise ValueError("inconsistent SciPy child spans")
    levels: list[NDArray[np.int32]] = []
    level = np.array([0], dtype=np.int32)
    while len(level):
        if len(levels) >= 63:
            raise ValueError("CUDA spatial tree exceeds the bounded traversal stack")
        parents = level[axes[level] >= 0]
        levels.append(parents)
        level = children[parents].ravel()
    leaves = np.flatnonzero(axes < 0)
    if (not np.array_equal(spans[leaves[:-1], 1], spans[leaves[1:], 0])
            or spans[leaves[0], 0] != 0 or spans[leaves[-1], 1] != tree.n):
        raise ValueError("inconsistent SciPy leaf partition")
    ordered = tree.data[tree.indices]
    bounds = np.empty((count, 2, 3), dtype=np.float64)
    bounds[leaves, 0] = np.minimum.reduceat(ordered, spans[leaves, 0], axis=0)
    bounds[leaves, 1] = np.maximum.reduceat(ordered, spans[leaves, 0], axis=0)
    for parents in reversed(levels):
        left, right = children[parents].T
        bounds[parents, 0] = np.minimum(bounds[left, 0], bounds[right, 0])
        bounds[parents, 1] = np.maximum(bounds[left, 1], bounds[right, 1])
        axis = axes[parents]
        if (np.any(bounds[left, 1, axis] > splits[parents])
                or np.any(bounds[right, 0, axis] < splits[parents])):
            raise ValueError("inconsistent SciPy split geometry")
    return bounds


def capture_spatial_tree(tree: cKDTree) -> tuple[Topology, NDArray[np.float64], str]:
    """Capture validated flat topology and bounds without importing a GPU runtime."""
    try:
        topology = _serialized_topology(tree)
        bounds = _subtree_bounds(tree, topology)
        return topology, bounds, "validated_serialized"
    except (ValueError, TypeError, IndexError, AttributeError):
        topology = _public_topology(tree)
        return topology, _subtree_bounds(tree, topology), "public_nodes"
