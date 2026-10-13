"""Exact selection and bounded sparse-box work for adaptive plane queries."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping

import pytest

from calibrex.data.adaptive_voxel import (
    AdaptiveVoxelPlane,
    AdaptiveVoxelPlaneMap,
    AdaptiveVoxelPolicy,
    _SparsePlaneCells,
    build_adaptive_voxel_plane_map,
)
from calibrex.data.livox import LivoxPointRecord


def _cell(key, *, level=0, size=0.5, centroid=None):
    return AdaptiveVoxelPlane(
        level=level,
        key=key,
        centroid=centroid or tuple((value + 0.5) * size for value in key),
        normal=(0.0, 0.0, 1.0),
        support_count=3,
        voxel_size_m=size,
        range_m=1.0,
    )


def _maps(index, origin=(0.0, 0.0, 0.0)):
    cells = tuple(cell for group in index.values() for cell in group.values())
    return (
        AdaptiveVoxelPlaneMap(cells, origin, index),
        AdaptiveVoxelPlaneMap(
            cells, origin,
            {level: _SparsePlaneCells(group) for level, group in index.items()},
        ),
    )


@pytest.mark.parametrize("seed", range(6))
def test_sparse_boxes_match_legacy_cell_identity_for_random_levels_and_gates(seed):
    rng = random.Random(seed)
    index = {}
    for level, size in [(2, 1.0), (-1, 0.25), (1, 0.5)]:
        keys = [tuple(rng.randrange(-18, 19) for _ in range(3)) for _ in range(130)]
        index[level] = {
            key: _cell(
                key, level=level, size=size,
                centroid=tuple((value + rng.random()) * size for value in key),
            )
            for key in keys
        }
    legacy, sparse = _maps(index, origin=(0.2, -0.3, 0.7))
    for _ in range(45):
        point = tuple(rng.uniform(-6.0, 6.0) for _ in range(3))
        gate = rng.choice([0.01, 0.25, 0.75, 1.5])
        assert sparse.nearest_plane(point, correspondence_gate_m=gate) is (
            legacy.nearest_plane(point, correspondence_gate_m=gate)
        )


def test_sparse_box_yields_only_occupied_cells_in_original_offset_order():
    keys = [
        (-2, 1, -3), (-2, -1, 4), (-2, -1, -4),
        (1, -1, -4), (1, 1, 4), (100, 0, 0),
    ]
    mapping = _SparsePlaneCells({key: _cell(key) for key in reversed(keys)})
    assert isinstance(mapping, Mapping)
    assert list(mapping) == list(reversed(keys))
    assert list(mapping.nearby((0, 0, 0), 4)) == [
        mapping[key] for key in sorted(keys) if max(abs(value) for value in key) <= 4
    ]
    assert list(mapping.nearby((1000, 1000, 1000), 1)) == []
    with pytest.raises(KeyError):
        _ = mapping[(99, 99, 99)]


@pytest.mark.parametrize("gate", [0.5, 1.0, math.nextafter(1.0, 0.0), 2.0])
def test_ties_keep_lexical_key_then_original_level_order_and_inclusive_gate(gate):
    negative = _cell((-1, 0, 0), level=4, centroid=(-1.0, 0.0, 0.0))
    positive = _cell((1, 0, 0), level=4, centroid=(1.0, 0.0, 0.0))
    other_level = _cell((0, 0, 0), level=-2, centroid=(0.0, 1.0, 0.0))
    legacy, sparse = _maps({4: {positive.key: positive, negative.key: negative},
                           -2: {other_level.key: other_level}})
    expected = negative if gate >= 1.0 else None
    assert legacy.nearest_plane((0.0, 0.0, 0.0), correspondence_gate_m=gate) is expected
    assert sparse.nearest_plane((0.0, 0.0, 0.0), correspondence_gate_m=gate) is expected


@pytest.mark.parametrize("point,gate", [
    ((0.0, 0.0, 0.0), 0.0),
    ((0.0, 0.0, 0.0), -1.0),
    ((0.0, 0.0, 0.0), float("nan")),
    ((0.0, 0.0, 0.0), float("inf")),
    ((float("nan"), 0.0, 0.0), 1.0),
    ((float("inf"), 0.0, 0.0), 1.0),
])
def test_invalid_query_keeps_legacy_exception(point, gate):
    legacy, sparse = _maps({0: {(0, 0, 0): _cell((0, 0, 0))}})
    with pytest.raises((ValueError, OverflowError)) as original:
        legacy.nearest_plane(point, correspondence_gate_m=gate)
    with pytest.raises(type(original.value), match=str(original.value)):
        sparse.nearest_plane(point, correspondence_gate_m=gate)


def test_nonfinite_centroids_and_large_integer_keys_keep_legacy_selection():
    key = (2**70, 0, 0)
    finite = _cell(key, size=1.0, centroid=(float(2**70), 0.0, 0.0))
    invalid = _cell((key[0], 0, -1), size=1.0, centroid=(float("nan"), 0.0, 0.0))
    legacy, sparse = _maps({0: {invalid.key: invalid, finite.key: finite}})
    point = (float(2**70), 0.0, 0.0)
    assert legacy.nearest_plane(point, correspondence_gate_m=1.0) is finite
    assert sparse.nearest_plane(point, correspondence_gate_m=1.0) is finite


def test_builder_owns_sparse_cells_and_query_never_performs_empty_cell_lookups(monkeypatch):
    records = [
        LivoxPointRecord(
            point=(float(x), float(y), 0.0),
            normal_xyz=(0.0, 0.0, 1.0),
        )
        for x in range(-10, 11)
        for y in range(-10, 11)
    ]
    built = build_adaptive_voxel_plane_map(
        records, AdaptiveVoxelPolicy(min_points_per_voxel=1),
    )
    assert built.cells
    assert all(isinstance(group, _SparsePlaneCells) for group in built._index.values())
    legacy = AdaptiveVoxelPlaneMap(
        built.cells, built.origin_m,
        {level: dict(group) for level, group in built._index.items()},
    )
    expected = legacy.nearest_plane((1.0, 1.0, 0.05), correspondence_gate_m=1.5)

    def forbidden_lookup(*args, **kwargs):
        raise AssertionError("occupied box queries must avoid empty cell lookups")

    monkeypatch.setattr(_SparsePlaneCells, "get", forbidden_lookup)
    assert built.nearest_plane((1.0, 1.0, 0.05), correspondence_gate_m=1.5) is expected


def test_sparse_mapping_keeps_its_own_snapshot():
    cell = _cell((0, 0, 0))
    original = {cell.key: cell}
    sparse = _SparsePlaneCells(original)
    original.clear()
    assert sparse[cell.key] is cell
    assert list(sparse.nearby(cell.key, 1)) == [cell]


def test_empty_map_and_missing_level_keep_legacy_behavior():
    for index in ({}, {0: {}}):
        for mapping in (index, {level: _SparsePlaneCells(group) for level, group in index.items()}):
            plane_map = AdaptiveVoxelPlaneMap((), (0.0, 0.0, 0.0), mapping)
            if not index:
                assert plane_map.nearest_plane((0.0, 0.0, 0.0), correspondence_gate_m=1.0) is None
            else:
                with pytest.raises(ValueError, match="adaptive voxel level has no cells"):
                    plane_map.nearest_plane((0.0, 0.0, 0.0), correspondence_gate_m=1.0)
