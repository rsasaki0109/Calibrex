"""Optional fused CPU geometry retains scalar pose/world bits and diagnostics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

pytest.importorskip("numba")

from calibrex.core.geometry import SE3
from calibrex.core.time import apply_time_offset_ns
from calibrex.data.livox import LivoxPointRecord
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers import continuous_time_lidar_pair_solver as solver_module
from calibrex.solvers.continuous_time_lidar_pair_solver import (
    ContinuousTimeLidarPairOptions,
    ContinuousTimeLidarPairProblem,
    ContinuousTimeLidarPairSolver,
)
from calibrex.solvers.fixed_trajectory_se3_solver import FixedTrajectorySe3SolverOptions
from calibrex.solvers.numba_continuous_time_adapter import (
    NumbaContinuousTimeQueryContext,
    _CapturedTimestampRows,
    numba_rig_linearization,
    numba_rig_normal_equations,
    numba_rig_residuals,
)


@pytest.mark.parametrize("mode", ["general", "nlerp", "antipodal"])
@pytest.mark.parametrize("offset", [0., 0.0000000005, -0.0200000005])
def test_prepared_timestamp_rows_match_legacy_geometry_and_every_query_counter(mode, offset):
    samples = _samples(mode)
    left, right = OdometryTrack(samples), OdometryTrack(samples)
    old = NumbaContinuousTimeQueryContext.capture(left)
    new = NumbaContinuousTimeQueryContext.capture(right)
    timestamps = [0, 500_000_000, 1_000_000_000, 5_000_000_000]*17
    owner = _CapturedTimestampRows.capture(timestamps)
    assert owner is not None and not owner.values.flags.writeable
    points = np.random.default_rng(12).normal(size=(len(timestamps), 3))
    base = SE3((.1, -.2, .3), (.2, -.1, .4, .8))
    for iteration in range(3):
        transform = SE3((.1*iteration, -.2, .3), (.2, .1, -.3, .8))
        shifted = owner.shifted(apply_time_offset_ns(0, offset), int(new.times[0]))
        assert shifted is not None and not shifted.values.flags.writeable
        expected = old._evaluate_rows(left, [apply_time_offset_ns(t, offset) for t in timestamps],
                                      points+iteration, base, transform, .1)
        actual = new._evaluate_prepared_rows(right, shifted, points+iteration, base, transform, .1)
        assert actual[-1] == expected[-1]
        for a, b in zip(actual[:2], expected[:2], strict=True):
            assert a.view(np.uint64).tolist() == b.view(np.uint64).tolist()
        for name in ("interpolation_count", "clamp_count", "max_extrapolation_s"):
            assert getattr(left, name) == getattr(right, name)
    assert len(new._pose_cache) == 1


def test_prepared_query_keys_own_snapshots_validate_integer_bounds_and_charge_lru():
    original = [0, 500_000_000, 1_000_000_000, 5_000_000_000]*128
    owner = _CapturedTimestampRows.capture(original)
    original[0] = 99
    assert owner.values[0] == 0
    first, second = owner.shifted(0, 0), owner.shifted(0, 0)
    assert first == second and hash(first) == hash(second)
    assert first != _CapturedTimestampRows.capture(original).shifted(0, 0)
    for times in ([], [2**63], [-(2**63)], [0.0], [np.int64(0)]):
        assert _CapturedTimestampRows.capture(times) is None
    assert owner.shifted(2**63, 0) is None
    assert owner.shifted(2**53+1, 0) is None
    edge = _CapturedTimestampRows.capture([2**63-2])
    assert edge.shifted(2, 2**63-2) is None
    track = OdometryTrack(_samples())
    context = NumbaContinuousTimeQueryContext.capture(track)
    context._pose_cache_limit_bytes = 60_000
    points = np.zeros((len(original), 3))
    for shift in (0, 1, 2):
        result = context._evaluate_prepared_rows(track, owner.shifted(shift, 0), points,
                                                 SE3.identity(), SE3.identity(), .1)
        assert result[-1]
        assert 0 < context._pose_cache_bytes <= context._pose_cache_limit_bytes
        assert len(context._pose_cache) == 1
    assert context._evaluate_prepared_rows(track, owner.shifted(0, 1), points,
                                          SE3.identity(), SE3.identity(), .1) is None


def test_prepared_extrapolation_failure_keeps_the_exact_visited_prefix():
    left, right = OdometryTrack(_samples()), OdometryTrack(_samples())
    old = NumbaContinuousTimeQueryContext.capture(left)
    new = NumbaContinuousTimeQueryContext.capture(right)
    times = [1_000_000_000, -200_000_000, 2_000_000_000]
    prepared = _CapturedTimestampRows.capture(times).shifted(0, 0)
    points = np.ones((3, 3))
    old_result = old._evaluate_rows(left, times, points, SE3.identity(), SE3.identity(), .01)
    new_result = new._evaluate_prepared_rows(
        right, prepared, points, SE3.identity(), SE3.identity(), .01,
    )
    assert old_result[-1] is new_result[-1] is False
    for name in ("interpolation_count", "clamp_count", "max_extrapolation_s"):
        assert getattr(left, name) == getattr(right, name)
    assert right.interpolation_count == 2 and right.clamp_count == 1


def _samples(mode="general"):
    samples = []
    for i in range(6):
        angle = i * 1e-7 if mode == "nlerp" else i * 0.13
        sign = -1.0 if mode == "antipodal" and i % 2 else 1.0
        samples.append(
            OdometryPoseSample(
                i * 1_000_000_000,
                SE3(
                    (i * 0.3, math.sin(i), -i * 0.1),
                    (0.0, 0.0, sign * math.sin(angle), sign * math.cos(angle)),
                ),
            )
        )
    return samples


def _bits(poses):
    return (
        np.asarray(
            [(*pose.translation_m, *pose.rotation_quat_xyzw) for pose in poses], dtype=np.float64
        )
        .view(np.uint64)
        .tolist()
    )


@pytest.mark.parametrize("mode", ["general", "nlerp", "antipodal"])
@pytest.mark.parametrize("count", [0, 1, 64, 4097])
def test_fused_geometry_matches_scalar_bits_and_all_query_counters(mode, count):
    samples = _samples(mode)
    scalar, fused = OdometryTrack(samples), OdometryTrack(samples)
    context = NumbaContinuousTimeQueryContext.capture(fused)
    assert context is not None
    rng = np.random.default_rng(831)
    times = rng.integers(-100_000_000, 5_100_000_001, count).tolist()
    points = [tuple(point) for point in rng.uniform(-4.0, 4.0, (count, 3)).tolist()]
    base = SE3((0.12, -0.23, 0.34), (0.2, -0.1, 0.4, 0.8))
    transform = SE3((-0.25, 0.16, 0.32), (-0.1, 0.3, 0.2, 0.9))
    result = context.evaluate(fused, times, points, base, transform, 1.0)
    assert result is not None
    actual, coordinates, usable = result
    expected = [scalar.interpolate(timestamp)[0].compose(base) for timestamp in times]
    assert usable and _bits(actual) == _bits(expected)
    world = np.asarray(
        [
            pose.compose(transform).transform_point(point)
            for pose, point in zip(expected, points, strict=True)
        ]
    ).reshape((-1, 3))
    assert coordinates.view(np.uint64).tolist() == world.view(np.uint64).tolist()
    assert fused.interpolation_count == scalar.interpolation_count
    assert fused.clamp_count == scalar.clamp_count
    assert fused.max_extrapolation_s == scalar.max_extrapolation_s


def test_extrapolation_failure_counts_only_the_visited_prefix():
    track = OdometryTrack(_samples())
    context = NumbaContinuousTimeQueryContext.capture(track)
    assert context is not None
    result = context.evaluate(
        track,
        [1_000_000_000, -200_000_000, 2_000_000_000],
        [(1.0, 0.0, 0.0)] * 3,
        SE3.identity(),
        SE3.identity(),
        0.01,
    )
    assert result is not None and not result[-1]
    assert track.interpolation_count == 2 and track.clamp_count == 1
    assert track.max_extrapolation_s == 0.2


def test_cached_pose_batches_keep_world_bits_with_changed_points_extrinsics_and_base():
    track = OdometryTrack(_samples())
    reference = OdometryTrack(_samples())
    cached = NumbaContinuousTimeQueryContext.capture(track)
    uncached = NumbaContinuousTimeQueryContext.capture(reference)
    assert cached is not None and uncached is not None
    uncached._pose_cache_limit_bytes = 0
    times = [-100_000_000, 0, 1_230_000_000, 5_100_000_000] * 20
    rng = np.random.default_rng(101)
    previous = None
    for iteration in range(4):
        base = SE3((0.1 if iteration < 3 else 0.2, -0.2, 0.3), (0.2, -0.1, 0.4, 0.8))
        extrinsic = SE3((iteration * 0.1, -0.2, 0.3), (0.1, 0.2, -0.1, 0.9))
        points = [tuple(point) for point in rng.normal(size=(len(times), 3)).tolist()]
        actual = cached.evaluate(track, times, points, base, extrinsic, 0.2)
        expected = uncached.evaluate(reference, times, points, base, extrinsic, 0.2)
        assert actual is not None and expected is not None
        assert actual[2] and expected[2] and _bits(actual[0]) == _bits(expected[0])
        assert actual[1].view(np.uint64).tolist() == expected[1].view(np.uint64).tolist()
        if previous is not None and iteration < 3:
            assert all(a is b for a, b in zip(actual[0], previous, strict=True))
        previous = actual[0]
        assert track.interpolation_count == reference.interpolation_count
        assert track.clamp_count == reference.clamp_count
        assert track.max_extrapolation_s == reference.max_extrapolation_s
    assert len(cached._pose_cache) == 2 and not uncached._pose_cache
    assert all(not batch.rows.flags.writeable for batch in cached._pose_cache.values())


def test_cached_batch_lower_tolerance_replays_only_visited_failure_prefix():
    track = OdometryTrack(_samples())
    context = NumbaContinuousTimeQueryContext.capture(track)
    assert context is not None
    times = [1_000_000_000, -200_000_000, 2_000_000_000]
    points = [(1.0, 0.0, 0.0)] * 3
    identity = SE3.identity()
    assert context.evaluate(track, times, points, identity, identity, 0.3)[-1]
    result = context.evaluate(track, times, points, identity, identity, 0.1)
    assert result is not None and not result[-1]
    assert track.interpolation_count == 5 and track.clamp_count == 2
    assert track.max_extrapolation_s == 0.2


def test_pose_cache_budget_eviction_zero_budget_and_custom_diagnostics(monkeypatch):
    track = OdometryTrack(_samples())
    context = NumbaContinuousTimeQueryContext.capture(track)
    assert context is not None
    identity = SE3.identity()
    points = [(1.0, 0.0, 0.0)] * 64
    query = [1_000_000_000] * 64
    context.evaluate(track, query, points, identity, identity, 0.1)
    size = context._pose_cache_bytes
    assert size > 0
    context._pose_cache_limit_bytes = (size + 1024) * 2
    context.evaluate(track, [2_000_000_000] * 64, points, identity, identity, 0.1)
    context.evaluate(track, query, points, identity, identity, 0.1)
    context.evaluate(track, [3_000_000_000] * 64, points, identity, identity, 0.1)
    assert len(context._pose_cache) == 2
    assert [key[0][0] for key in context._pose_cache] == [1_000_000_000, 3_000_000_000]
    assert context._pose_cache_bytes == sum(
        batch.size_bytes for batch in context._pose_cache.values()
    )
    assert context._pose_cache_bytes <= context._pose_cache_limit_bytes
    context._pose_cache_limit_bytes = 0
    result = context.evaluate(track, query, points, identity, identity, 0.1)
    assert result is not None
    assert result[0][0] is not next(iter(context._pose_cache.values())).poses[0]
    context._pose_cache_limit_bytes = size - 1
    context.evaluate(track, [4_000_000_000] * 64, points, identity, identity, 0.1)
    assert len(context._pose_cache) == 2  # An oversized new entry is not retained.
    calls = []
    monkeypatch.setattr(track, "_record_extrapolation", calls.append)
    context._pose_cache_limit_bytes = (size + 1024) * 2
    context.evaluate(track, query, points, identity, identity, 0.1)
    assert calls == [0.0] * 64


def test_replaced_samples_duplicate_timestamps_and_huge_integer_ranges_use_scalar_fallback():
    track = OdometryTrack(_samples())
    context = NumbaContinuousTimeQueryContext.capture(track)
    assert context is not None
    identity = SE3.identity()
    assert context.evaluate(track, [0], [(0.0, 0.0, 0.0)], identity, identity, 0.2)[-1]
    track.samples = tuple(replace(sample, pose=SE3.identity()) for sample in track.samples)
    result = context.evaluate(
        track,
        [0],
        [(0.0, 0.0, 0.0)],
        SE3.identity(),
        SE3.identity(),
        0.2,
    )
    assert result is None
    duplicate = _samples()
    duplicate.insert(1, duplicate[0])
    assert NumbaContinuousTimeQueryContext.capture(OdometryTrack(duplicate)) is None
    huge = [replace(sample, timestamp_ns=sample.timestamp_ns * 2**70) for sample in _samples()]
    assert NumbaContinuousTimeQueryContext.capture(OdometryTrack(huge)) is None


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("count", [64, 4097])
def test_fused_residuals_and_numerical_jacobians_match_numpy_bits(shared, count):
    rng = np.random.default_rng(330)
    points = rng.normal(size=(count, 3))
    observations = [
        LidarPointToPlaneObservation(
            tuple(point),
            (0.2, -0.3, 0.1),
            (0.1, -0.2, 0.9),
            SE3((0.1, 0.2, -0.3), (0.1, 0.2, -0.3, 0.8))
            if shared
            else SE3(tuple(point), (float(point[0]), float(point[1]), 0.1, 0.9)),
            weight=0.4 + i / max(1, count),
        )
        for i, point in enumerate(points.tolist())
    ]
    transform = SE3((0.1, -0.2, 0.3), (0.2, 0.1, -0.3, 0.8))
    baseline = LidarRigPointToPlaneFactor(
        variable="extrinsic", t_ego_lidar=transform, observations=observations
    )
    compiled = LidarRigPointToPlaneFactor(
        variable="extrinsic",
        t_ego_lidar=transform,
        observations=observations,
        _array_kernel=numba_rig_residuals,
        _array_linearizer=numba_rig_linearization,
        _array_normal_kernel=numba_rig_normal_equations,
    )
    correction = (0.01, -0.02, 0.03, -0.01, 0.005, 0.02)
    expected = np.asarray(baseline.residuals(correction)).view(np.uint64).tolist()
    actual = np.asarray(compiled.residuals(correction)).view(np.uint64).tolist()
    assert actual == expected
    a, b = baseline.linearize(correction), compiled.linearize(correction)
    for name in ("residuals_m", "jacobian", "hessian", "gradient"):
        expected_values = np.asarray(getattr(a, name)).view(np.uint64).tolist()
        actual_values = np.asarray(getattr(b, name)).view(np.uint64).tolist()
        assert actual_values == expected_values


def test_complete_profile_matches_scalar_geometry_and_all_diagnostics(monkeypatch):
    records = [
        LivoxPointRecord(point=(*point, 0.0), normal_xyz=normal)
        for i in range(-3, 4)
        for j in range(-3, 4)
        for point, normal in (
            ((i * 0.5, j * 0.5, 0.0), (0.0, 0.0, 1.0)),
            ((1.5, i * 0.5, j * 0.5), (1.0, 0.0, 0.0)),
            ((i * 0.5, -1.5, j * 0.5), (0.0, 1.0, 0.0)),
        )
    ]
    track = OdometryTrack(_samples())
    truth = SE3((0.03, -0.02, 0.01), (0.1, 0.2, -0.1, 0.9))
    points, times = [], []
    for timestamp in (5_000_000, 1_005_000_000, 2_005_000_000, 3_005_000_000):
        inverse = track.interpolate(timestamp + 10_000_000)[0].compose(truth).inverse()
        for record in records[::4]:
            points.append(inverse.transform_point(record.point[:3]))
            times.append(timestamp)
    track.interpolation_count = track.clamp_count = 0
    track.max_extrapolation_s = 0.0
    split = len(points) * 3 // 4
    problem = ContinuousTimeLidarPairProblem(
        records,
        points,
        times,
        track,
        SE3.identity(),
        SE3.identity(),
        "extrinsic",
        "lidar",
        0.5,
        1.0,
        tuple(range(split)),
        tuple(range(split, len(points))),
    )
    options = ContinuousTimeLidarPairOptions(
        max_iterations=2,
        minimum_time_step_sec=0.01,
        voxel_strategy="adaptive",
        adaptive_min_points_per_voxel=1,
        solver=FixedTrajectorySe3SolverOptions(max_iterations=3),
    )
    capture = solver_module.adaptive_voxel_module.AdaptivePlaneQueryContext
    monkeypatch.delattr(solver_module.adaptive_voxel_module, "AdaptivePlaneQueryContext")
    reference = ContinuousTimeLidarPairSolver().solve(problem, options)
    monkeypatch.setattr(
        solver_module.adaptive_voxel_module, "AdaptivePlaneQueryContext", capture, raising=False
    )
    other = replace(problem, odometry_track=OdometryTrack(list(track.samples)))
    result = ContinuousTimeLidarPairSolver().solve(
        other,
        replace(options, query_backend="numba_cpu"),
    )
    assert result.query_backend.startswith("calibrex.numba_continuous_time/")
    assert json.dumps(asdict(replace(result, query_backend="python")), sort_keys=True) == (
        json.dumps(asdict(reference), sort_keys=True)
    )
    assert other.odometry_track.interpolation_count == track.interpolation_count
    assert other.odometry_track.clamp_count == track.clamp_count
    assert other.odometry_track.max_extrapolation_s == track.max_extrapolation_s
