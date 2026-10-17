"""Packed geometry and hybrid keys retain identities, shadows and exact ties."""

import math

import numpy as np
import pytest

from calibrex.data.adaptive_voxel import AdaptiveVoxelPlane
from calibrex.data.livox import (
    LivoxPointRecord,
    VoxelPlaneQueryContext,
    _PlaneGeometry,
    build_voxel_plane_map,
)
from calibrex.evaluation.lidar import _batched_plane_choices


@pytest.mark.parametrize("count", [0, 7, 8, 63, 64, 4097])
@pytest.mark.parametrize("gate", [0.001, 0.5, 1.5])
def test_captured_geometries_retain_original_match_values(count, gate):
    rng = np.random.default_rng(632)
    records = [
        LivoxPointRecord((*point, 0.0), (0.1, -0.2, 0.9))
        for point in rng.uniform(-4, 4, (500, 3)).tolist()
    ]
    context = VoxelPlaneQueryContext(build_voxel_plane_map(records, 0.5))
    points = rng.uniform(-4, 4, (count, 3))
    actual = context._nearest_geometries(points, voxel_size_m=0.5, correspondence_gate_m=gate)
    expected = context.nearest_matches(points, voxel_size_m=0.5, correspondence_gate_m=gate)
    if count < 8:
        assert actual is None
        return
    assert actual is not None
    assert [(p.centroid, p.normal) if p is not None else None for p in actual] == [
        (p.centroid, p.normal) if p is not None else None for p in expected
    ]


def test_custom_capture_queries_keep_public_dispatch(monkeypatch):
    context = VoxelPlaneQueryContext(
        build_voxel_plane_map([LivoxPointRecord((0.1, 0.2, 0.3, 0.0), (0.1, -0.2, 0.9))], 0.5)
    )
    monkeypatch.setattr(context, "nearest_matches", lambda *args, **kwargs: [None] * 64)
    assert (
        context._nearest_geometries(np.zeros((64, 3)), voxel_size_m=0.5, correspondence_gate_m=1.0)
        is None
    )


def _adaptive(center, normal, key=0):
    return AdaptiveVoxelPlane(0, (key, 0, 0), center, normal, 3, 0.5, 1.0)


def _reference(point, left, right):
    candidates = [
        plane for plane in (left, right) if plane is not None and plane.normal is not None
    ]
    return (
        min(
            candidates,
            key=lambda plane: (
                abs(
                    plane.normal[0] * (point[0] - plane.centroid[0])
                    + plane.normal[1] * (point[1] - plane.centroid[1])
                    + plane.normal[2] * (point[2] - plane.centroid[2])
                ),
                math.dist(point, plane.centroid),
            ),
        )
        if candidates
        else None
    )


def test_batched_hybrid_residual_distance_ties_and_missing_candidates_keep_identity():
    rng = np.random.default_rng(811)
    points = rng.normal(size=(257, 3))
    left, right = [], []
    for row in range(len(points)):
        center, normal = tuple(rng.normal(size=3).tolist()), tuple(rng.normal(size=3).tolist())
        left.append(_PlaneGeometry(center, normal) if row % 11 else None)
        right.append(_adaptive(tuple(rng.normal(size=3).tolist()), normal) if row % 13 else None)
    points[:5] = 0.0
    for row, (lc, rc) in enumerate(
        (
            ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            ((0.0, 2.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 1.0, 0.0), (0.0, math.nextafter(1.0, 0.0), 0.0)),
            ((-0.0, 0.0, 0.0), (0.0, -0.0, 0.0)),
            ((1e-160, 0.0, 0.0), (1e-160, 1e-161, 0.0)),
        )
    ):
        left[row] = _PlaneGeometry(lc, (0.0, 0.0, 1.0))
        right[row] = _adaptive(rc, (0.0, 0.0, 1.0))
    actual = _batched_plane_choices(points, left, right)
    assert actual is not None
    expected = [
        _reference(tuple(point), a, b)
        for point, a, b in zip(points.tolist(), left, right, strict=True)
    ]
    assert all(a is b for a, b in zip(actual, expected, strict=True))


def test_nonfinite_score_and_extreme_coordinates_keep_scalar_selection():
    left = [_PlaneGeometry((0.0,) * 3, (math.nan, 0.1, 0.2))] * 64
    right = [_adaptive((0.0,) * 3, (0.1, 0.2, 0.9))] * 64
    assert _batched_plane_choices(np.zeros((64, 3)), left, right) is None
    assert _batched_plane_choices(np.full((64, 3), 1e100), left, right) is None
    assert _batched_plane_choices(np.zeros((63, 3)), left[:63], right[:63]) is None
