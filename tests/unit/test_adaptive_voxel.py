"""Tests for range-adaptive solid-state voxel planes."""

from __future__ import annotations

from calibrex.core.geometry import SE3
from calibrex.data.adaptive_voxel import (
    AdaptiveVoxelPlane,
    AdaptiveVoxelPlaneMap,
    AdaptiveVoxelPolicy,
    adaptive_voxel_size_m,
    build_adaptive_voxel_plane_map,
)
from calibrex.data.livox import LivoxPointRecord, build_voxel_plane_map
from calibrex.evaluation.lidar import build_rig_point_to_plane_observations


def test_adaptive_voxel_size_grows_with_range() -> None:
    policy = AdaptiveVoxelPolicy(
        base_voxel_size_m=0.5,
        range_reference_m=10.0,
        range_exponent=0.5,
        min_voxel_size_m=0.2,
        max_voxel_size_m=2.0,
    )

    assert adaptive_voxel_size_m(2.0, policy) < adaptive_voxel_size_m(20.0, policy)


def test_adaptive_plane_map_preserves_plane_query() -> None:
    records = [
        LivoxPointRecord(
            point=(10.0 + 0.1 * x, 0.1 * y, 0.0),
            normal_xyz=(0.0, 0.0, 1.0),
        )
        for x in range(6)
        for y in range(6)
    ]
    plane_map = build_adaptive_voxel_plane_map(
        records,
        AdaptiveVoxelPolicy(
            base_voxel_size_m=0.5,
            range_reference_m=10.0,
            min_voxel_size_m=0.25,
            max_voxel_size_m=1.0,
            min_points_per_voxel=1,
        ),
    )

    match = plane_map.nearest_plane(
        (10.2, 0.2, 0.01),
        correspondence_gate_m=0.5,
    )
    assert match is not None
    assert match.normal == (0.0, 0.0, 1.0)
    assert match.support_count >= 1


def test_adaptive_plane_map_uses_wide_geometric_fallback_without_normals() -> None:
    records = [
        LivoxPointRecord(point=(0.0, 0.0, 0.0)),
        LivoxPointRecord(point=(0.01, 0.0, 0.05)),
        LivoxPointRecord(point=(0.02, 0.0, 0.10)),
        LivoxPointRecord(point=(0.0, 1.0, 0.0)),
        LivoxPointRecord(point=(1.0, 1.0, 0.0)),
        LivoxPointRecord(point=(1.0, 0.0, 0.0)),
    ]
    plane_map = build_adaptive_voxel_plane_map(
        records,
        AdaptiveVoxelPolicy(
            base_voxel_size_m=2.0,
            range_reference_m=10.0,
            min_voxel_size_m=2.0,
            max_voxel_size_m=2.0,
            min_points_per_voxel=3,
        ),
    )

    assert len(plane_map.cells) == 1
    normal = plane_map.cells[0].normal
    assert abs(normal[2]) > 0.99


def test_hybrid_plane_selection_prefers_lower_initial_residual() -> None:
    source_records = [
        LivoxPointRecord(point=(0.0, 0.0, 0.0, 0.0)),
        LivoxPointRecord(point=(0.1, 0.0, 0.0, 0.0)),
        LivoxPointRecord(point=(0.0, 0.1, 0.0, 0.0)),
    ]
    adaptive_cell = AdaptiveVoxelPlane(
        level=0,
        key=(0, 0, 0),
        centroid=(0.0, 0.0, 0.4),
        normal=(0.0, 0.0, 1.0),
        support_count=3,
        voxel_size_m=0.5,
        range_m=0.4,
    )
    adaptive_map = AdaptiveVoxelPlaneMap(
        cells=(adaptive_cell,),
        origin_m=(0.0, 0.0, 0.0),
        _index={0: {(0, 0, 0): adaptive_cell}},
    )

    observations = build_rig_point_to_plane_observations(
        source_records=source_records,
        target_points=[(0.1, 0.1, 0.05)],
        initial_t_source_target=SE3.identity(),
        voxel_size_m=0.5,
        correspondence_gate_m=1.0,
        source_plane_map=build_voxel_plane_map(source_records, 0.5),
        adaptive_plane_map=adaptive_map,
    )

    assert len(observations) == 1
    assert observations[0].plane_point_world_m[2] == 0.0
