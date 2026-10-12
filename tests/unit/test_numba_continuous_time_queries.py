"""Optional fused CPU geometry retains scalar pose/world bits and diagnostics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

pytest.importorskip("numba")

from calibrex.core.geometry import SE3
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
    numba_rig_linearization,
    numba_rig_normal_equations,
    numba_rig_residuals,
)


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


def test_replaced_samples_duplicate_timestamps_and_huge_integer_ranges_use_scalar_fallback():
    track = OdometryTrack(_samples())
    context = NumbaContinuousTimeQueryContext.capture(track)
    assert context is not None
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
