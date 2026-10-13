"""Exact geometry capture, ordered world transforms and bounded pose reuse."""

from __future__ import annotations

import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from calibrex.core.geometry import SE3, interpolate_se3
from calibrex.data import livox
from calibrex.data.adaptive_voxel import (
    AdaptiveVoxelPolicy,
    _NeighborhoodCache,
    _SparsePlaneCells,
    build_adaptive_voxel_plane_map,
)
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack, _PreparedInterval
from calibrex.evaluation import lidar
from calibrex.graph.lidar_point_to_plane import _array_world_points
from calibrex.solvers.continuous_time_lidar_pair_solver import _QueryReuse


def _bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64).tolist()


def _match_bits(match):
    if match is None:
        return None
    return (match.voxel_key, _bits((*match.centroid, *match.normal, match.centroid_distance_m)))


def _records():
    return [
        livox.LivoxPointRecord(
            point=(x + 0.25, y + 0.25, z + 0.25, 0.0),
            normal_xyz=(float(x + 4), float(y + 4), float(z + 4)),
        )
        for x in range(-3, 4) for y in range(-3, 4) for z in range(-3, 4)
    ]


@pytest.mark.parametrize("count", [0, 63, 64, 4097])
@pytest.mark.parametrize("budget", [0, 8 * 1024 * 1024])
@pytest.mark.parametrize("gate", [0.0, 1.5, math.inf, math.nan])
def test_captured_uniform_matches_are_bit_exact_with_scalar_queries(
    count, budget, gate, monkeypatch,
):
    planes = livox.build_voxel_plane_map(_records(), 1.0)
    monkeypatch.setattr(livox, "_DENSE_PLANE_INDEX_MAX_BYTES", budget)
    context = livox.VoxelPlaneQueryContext(planes)
    points = np.random.default_rng(8103).uniform(-4.0, 4.0, (count, 3))
    actual = context.nearest_matches(points, voxel_size_m=1.0, correspondence_gate_m=gate)
    expected = [
        livox.nearest_voxel_plane_match(tuple(point), planes, voxel_size_m=1.0,
                                       correspondence_gate_m=gate)
        for point in points.tolist()
    ]
    assert [_match_bits(value) for value in actual] == [_match_bits(value) for value in expected]


def test_geometry_capture_computes_fallback_once_and_observes_new_capture(monkeypatch):
    records = [
        livox.LivoxPointRecord(point=(*point, 0.0))
        for point in [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)]
    ]
    planes = livox.build_voxel_plane_map(records, 1.0)
    original = livox._fallback_plane_normal
    calls = []

    def fallback(points):
        calls.append(points)
        return original(points)

    monkeypatch.setattr(livox, "_fallback_plane_normal", fallback)
    context = livox.VoxelPlaneQueryContext(planes)
    assert len(calls) == 1
    options = {"voxel_size_m": 1.0, "correspondence_gate_m": math.inf}
    before = context.nearest_match((0.0, 0.0, 0.0), **options)
    for _ in range(3):
        context.nearest_matches(np.zeros((100, 3)), **options)
    assert len(calls) == 1
    planes[(0, 0, 0)].add(livox.LivoxPointRecord(
        point=(0.2, 0.2, 0.2, 0.0), normal_xyz=(1.0, 0.0, 0.0),
    ))
    assert _match_bits(context.nearest_match((0.0, 0.0, 0.0), **options)) == _match_bits(before)
    fresh = livox.VoxelPlaneQueryContext(planes).nearest_match((0.0, 0.0, 0.0), **options)
    assert fresh is not None and fresh.normal == (1.0, 0.0, 0.0)
    assert _match_bits(fresh) != _match_bits(before)
    planes.clear()
    assert _match_bits(context.nearest_match((0.0, 0.0, 0.0), **options)) == _match_bits(before)


def test_closest_normal_less_plane_is_rejected_and_first_tie_is_preserved():
    first = livox._VoxelPlane(count=1, x_sum=-0.5, normal_x_sum=1.0, normal_count=1)
    second = livox._VoxelPlane(count=1, x_sum=0.5, normal_y_sum=1.0, normal_count=1)
    planes = {(0, 0, 0): second, (-1, 0, 0): first}
    options = {"voxel_size_m": 1.0, "correspondence_gate_m": 1.0}
    context = livox.VoxelPlaneQueryContext(planes)
    assert all(match.voxel_key == (-1, 0, 0) for match in context.nearest_matches(
        np.zeros((65, 3)), **options,
    ))
    planes[(0, 0, 0)] = livox._VoxelPlane(count=1)
    context = livox.VoxelPlaneQueryContext(planes)
    assert context.nearest_matches(np.zeros((65, 3)), **options) == [None] * 65


@pytest.mark.parametrize("count", [0, 1, 63, 64, 4097])
def test_moving_world_coordinates_preserve_hamilton_normalization_bits(count):
    rng = np.random.default_rng(887)
    points = rng.normal(size=(count, 3))
    poses = [SE3(tuple(rng.normal(size=3)), tuple(rng.normal(size=4))) for _ in range(count)]
    transform = SE3((0.3, -0.2, 0.1), (0.11, -0.2, 0.3, 0.83))
    pose_array = np.asarray([
        (*pose.translation_m, *pose.rotation_quat_xyzw) for pose in poses
    ], dtype=np.float64).reshape(-1, 7)
    expected = np.asarray([
        pose.compose(transform).transform_point(point)
        for pose, point in zip(poses, points.tolist(), strict=True)
    ], dtype=np.float64).reshape(-1, 3)
    actual = _array_world_points(points, pose_array, transform)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("moving", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_hybrid_observations_match_the_legacy_scalar_builder(moving, adaptive, monkeypatch):
    records = _records()
    planes = livox.build_voxel_plane_map(records, 1.0)
    adaptive_map = build_adaptive_voxel_plane_map(
        records, AdaptiveVoxelPolicy(min_points_per_voxel=1),
    ) if adaptive else None
    points = np.random.default_rng(37).uniform(-2.0, 2.0, (120, 3)).tolist()
    poses = [SE3((i / 1000, 0.01, 0.02), (0.001 * i, 0.1, -0.2, 0.9))
             for i in range(len(points))] if moving else None
    options = {
        "source_records": records, "target_points": points,
        "target_t_world_source": poses, "source_plane_map": planes,
        "adaptive_plane_map": adaptive_map, "voxel_size_m": 1.0,
        "initial_t_source_target": SE3((0.1, -0.2, 0.3), (0.1, 0.2, -0.1, 0.8)),
    }
    actual = lidar.build_rig_point_to_plane_observations(
        **options, source_plane_queries=livox.VoxelPlaneQueryContext(planes),
    )
    monkeypatch.setattr(lidar, "_batched_world_coordinates", lambda *args: None)
    # A nonempty per-point pose list selects the exact scalar legacy route.
    if poses is None:
        options["target_t_world_source"] = [SE3.identity() for _ in points]
    expected = lidar.build_rig_point_to_plane_observations(**options)
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert _bits((*left.point_lidar_m, *left.plane_point_world_m,
                      *left.plane_normal_world, left.weight)) == (
            _bits((*right.point_lidar_m, *right.plane_point_world_m,
                   *right.plane_normal_world, right.weight))
        )
        assert _bits((*left.t_world_ego.translation_m, *left.t_world_ego.rotation_quat_xyzw)) == (
            _bits((*right.t_world_ego.translation_m, *right.t_world_ego.rotation_quat_xyzw))
        )


@pytest.mark.parametrize("size", [0, 1, 8, 8192])
def test_odometry_lru_preserves_exact_poses_and_all_query_counters(size):
    rng = np.random.default_rng(670)
    samples = [
        OdometryPoseSample(i * 1000, SE3(tuple(rng.normal(size=3)), tuple(rng.normal(size=4))))
        for i in range(8)
    ]
    samples.insert(3, replace(samples[2]))
    uncached = OdometryTrack(samples, cache_size=0)
    cached = OdometryTrack(samples, cache_size=size)
    queries = [-100, 0, 1, 100, 1999, 2000, 2100, 6999, 7000, 7100] * 3
    for timestamp in queries:
        left, right = cached.interpolate(timestamp), uncached.interpolate(timestamp)
        assert left[1:] == right[1:]
        assert _bits((*left[0].translation_m, *left[0].rotation_quat_xyzw)) == (
            _bits((*right[0].translation_m, *right[0].rotation_quat_xyzw))
        )
        assert cached.interpolation_count == uncached.interpolation_count
        assert cached.clamp_count == uncached.clamp_count
        assert cached.max_extrapolation_s == uncached.max_extrapolation_s
        assert len(cached._pose_cache) <= size


def test_odometry_lru_invalidates_sample_and_timestamp_replacements():
    def sample(x):
        return OdometryPoseSample(x * 1000, SE3((float(x), 0.0, 0.0), (0, 0, 0, 1)))
    track = OdometryTrack([sample(0), sample(2)])
    old = track.interpolate(1000)[0]
    assert track.interpolate(1000)[0] is old
    track.samples = (sample(0), sample(4))
    track._timestamps_ns = (0, 4000)
    new = track.interpolate(1000)[0]
    assert new is not old
    assert new.translation_m == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        OdometryTrack([], cache_size=-1)


def test_source_pose_reuse_respects_pose_and_base_identity_and_bounded_size():
    context = _QueryReuse(None)
    pose, base = SE3.identity(), SE3((1.0, 0.0, 0.0), (0, 0, 0, 1))
    first = context.source_pose(1, pose, base)
    assert context.source_pose(1, pose, base) is first
    changed = context.source_pose(1, pose, SE3.identity())
    assert changed.translation_m == (0.0, 0.0, 0.0)
    assert context.source_pose(1, SE3.identity(), base) is not first
    for timestamp in range(8200):
        context.source_pose(timestamp, pose, base)
    assert len(context.poses) <= 8192


@pytest.mark.parametrize("budget", [0, 600, 2048])
def test_adaptive_neighborhood_cache_keeps_exact_order_under_eviction(budget):
    cache = _NeighborhoodCache(budget)
    records = _records()
    original = build_adaptive_voxel_plane_map(
        records, AdaptiveVoxelPolicy(min_points_per_voxel=1),
    )
    for level, cells in original._index.items():
        captured = _SparsePlaneCells(cells, cache=cache, level=level)
        for key in [(0, 0, 0), (20, 20, 20), (-1, 1, 1)] * 3:
            for radius in (1, 3, 7):
                assert list(captured.nearby(key, radius)) == [
                    cell for coordinate, cell in sorted(cells.items())
                    if all(abs(coordinate[i] - key[i]) <= radius for i in range(3))
                ]
                assert 0 <= cache.retained_bytes <= budget


def test_adaptive_cached_queries_remain_picklable_and_safe_for_parallel_readers():
    plane_map = build_adaptive_voxel_plane_map(
        _records(), AdaptiveVoxelPolicy(min_points_per_voxel=1),
    )
    points = [(x / 10, y / 10, 0.1) for x in range(-10, 10) for y in range(-10, 10)]
    def query(point):
        return plane_map.nearest_plane(point, correspondence_gate_m=1.5)
    expected = list(map(query, points))
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(query, points * 2)) == expected * 2
    restored = pickle.loads(pickle.dumps(plane_map))
    assert [
        restored.nearest_plane(point, correspondence_gate_m=1.5) for point in points
    ] == expected


def test_odometry_cached_parallel_queries_keep_counters_and_pickle_values():
    track = OdometryTrack([
        OdometryPoseSample(0, SE3.identity()),
        OdometryPoseSample(1000, SE3((1.0, 0.0, 0.0), (0.1, 0.2, 0.3, 0.7))),
    ], cache_size=8)
    timestamps = [0, 100, 300, 999, 1100] * 100
    reference = OdometryTrack(list(track.samples), cache_size=0)
    expected = [reference.interpolate(timestamp) for timestamp in timestamps]
    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(track.interpolate, timestamps))
    assert actual == expected
    assert track.interpolation_count == reference.interpolation_count
    assert track.clamp_count == reference.clamp_count
    assert track.max_extrapolation_s == reference.max_extrapolation_s
    restored = pickle.loads(pickle.dumps(track))
    assert restored.interpolate(300) == track.interpolate(300)


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 0.001, 0.5, 0.999, 1.0, 1.1, math.nan])
@pytest.mark.parametrize("rotations", [
    ((0.0, -0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0)),
    ((0.1, 0.2, 0.3, 0.7), (0.1, 0.2, 0.3, 0.7)),
    ((0.0, 0.0, 0.0, 1.0), (1.0e-5, 0.0, 0.0, 1.0)),
    ((0.1, 0.2, 0.3, 0.7), (-0.4, 0.1, 0.2, 0.5)),
])
def test_prepared_interval_keeps_exact_shortest_arc_slerp_and_clamp_bits(alpha, rotations):
    left = SE3((0.1, -0.2, -0.0), rotations[0])
    right = SE3((3.1, 0.1, 0.5), rotations[1])
    expected = interpolate_se3(left, right, alpha)
    actual = _PreparedInterval.capture(left, right).interpolate(alpha)
    assert _bits((*actual.translation_m, *actual.rotation_quat_xyzw)) == (
        _bits((*expected.translation_m, *expected.rotation_quat_xyzw))
    )
