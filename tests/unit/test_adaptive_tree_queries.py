"""Spatial candidate pruning preserves original ordered adaptive associations."""

import math
from dataclasses import replace

import numpy as np
import pytest

from calibrex.data.adaptive_voxel import (
    AdaptivePlaneQueryContext,
    AdaptiveVoxelPlane,
    AdaptiveVoxelPlaneMap,
    AdaptiveVoxelPolicy,
    _SparsePlaneCells,
    build_adaptive_voxel_plane_map,
)
from calibrex.data.livox import LivoxPointRecord


@pytest.mark.parametrize("count", [0, 63, 64, 4097])
@pytest.mark.parametrize("gate", [0.001, 0.5, 1.5, 100.0])
def test_tree_matches_scalar_plane_identities_for_range_adaptive_levels(count, gate):
    rng = np.random.default_rng(319)
    records = [
        LivoxPointRecord(point=(*point, 0.0), normal_xyz=(0.1, -0.2, 0.9))
        for point in rng.uniform(-5.0, 5.0, (1000, 3)).tolist()
    ]
    planes = build_adaptive_voxel_plane_map(records, AdaptiveVoxelPolicy(min_points_per_voxel=1))
    context = AdaptivePlaneQueryContext(planes)
    assert context._tree is not None
    points = rng.uniform(-6.0, 6.0, (count, 3))
    actual = context.nearest_planes(points, correspondence_gate_m=gate)
    expected = [
        planes.nearest_plane(tuple(point), correspondence_gate_m=gate) for point in points.tolist()
    ]
    assert all(a is b for a, b in zip(actual, expected, strict=True))


def _cell(key, centroid):
    return AdaptiveVoxelPlane(0, key, centroid, (0.0, 0.0, 1.0), 3, 1.0, 1.0)


def test_exact_and_roundoff_close_ties_and_inclusive_gates_use_scalar_order():
    cells = [
        _cell((-1, 0, 0), (-0.5, 0.0, 0.0)),
        _cell((0, 0, 0), (0.5, 0.0, 0.0)),
        _cell((1, 0, 0), (math.nextafter(1.0, math.inf), 0.0, 0.0)),
    ]
    planes = AdaptiveVoxelPlaneMap(
        tuple(cells), (0.0, 0.0, 0.0), {0: _SparsePlaneCells({cell.key: cell for cell in cells})}
    )
    context = AdaptivePlaneQueryContext(planes)
    assert context._tree is not None
    points = np.asarray(
        [(0.0, 0.0, 0.0), (math.nextafter(0.0, 1.0), 0.0, 0.0), (0.75, 0.0, 0.0), (0.5, 0.0, 0.0)]
        * 20
    )
    for gate in (0.5, math.nextafter(0.5, 0.0), math.nextafter(0.5, math.inf)):
        actual = context.nearest_planes(points, correspondence_gate_m=gate)
        expected = [
            planes.nearest_plane(tuple(point), correspondence_gate_m=gate)
            for point in points.tolist()
        ]
        assert all(a is b for a, b in zip(actual, expected, strict=True))


def test_manual_mappings_and_out_of_cell_centroids_keep_original_search_box():
    near = _cell((0, 0, 0), (0.5, 0.0, 0.0))
    outside = _cell((100, 0, 0), (0.1, 0.0, 0.0))
    for mapping in (
        {near.key: near, outside.key: outside},
        _SparsePlaneCells({near.key: near, outside.key: outside}),
    ):
        planes = AdaptiveVoxelPlaneMap((near, outside), (0.0, 0.0, 0.0), {0: mapping})
        context = AdaptivePlaneQueryContext(planes)
        assert context._tree is None
        assert context.nearest_planes(np.zeros((100, 3)), correspondence_gate_m=1.0) == [near] * 100


def test_extreme_distance_and_level_size_inconsistency_use_scalar_fallback():
    cell = _cell((0, 0, 0), (1e-200, 0.0, 0.0))
    planes = AdaptiveVoxelPlaneMap(
        (cell,),
        (0.0, 0.0, 0.0),
        {0: _SparsePlaneCells({cell.key: cell})},
    )
    assert (
        AdaptivePlaneQueryContext(planes).nearest_planes(
            np.zeros((100, 3)),
            correspondence_gate_m=1.0,
        )
        == [cell] * 100
    )
    changed = replace(cell, voxel_size_m=2.0)
    inconsistent = replace(planes, _index={0: _SparsePlaneCells({cell.key: changed})})
    assert AdaptivePlaneQueryContext(inconsistent)._tree is None
