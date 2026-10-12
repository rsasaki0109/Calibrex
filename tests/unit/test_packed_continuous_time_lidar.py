"""Packed preparation retains native association, arithmetic and public observation APIs."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

pytest.importorskip("numba")

from calibrex.core.geometry import SE3
from calibrex.data.adaptive_voxel import (
    AdaptivePlaneQueryContext,
    AdaptiveVoxelPlane,
    AdaptiveVoxelPlaneMap,
    AdaptiveVoxelPolicy,
    _SparsePlaneCells,
    build_adaptive_voxel_plane_map,
)
from calibrex.data.livox import LivoxPointRecord, VoxelPlaneQueryContext, build_voxel_plane_map
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack
from calibrex.evaluation.lidar import build_rig_point_to_plane_observations
from calibrex.evaluation.packed_lidar_observations import _PackedObservationBuilder
from calibrex.graph import lidar_point_to_plane as graph
from calibrex.graph.lidar_point_to_plane import LidarRigPointToPlaneFactor, _PackedObservations
from calibrex.solvers import numba_continuous_time_adapter as adapter
from calibrex.solvers.continuous_time_lidar_pair_solver import (
    ContinuousTimeLidarPairOptions,
    ContinuousTimeLidarPairProblem,
    ContinuousTimeLidarPairSolver,
)
from calibrex.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
)
from calibrex.solvers.numba_continuous_time_adapter import (
    NumbaContinuousTimeQueryContext,
    numba_rig_linearization,
    numba_rig_normal_equations,
    numba_rig_residuals,
)
from calibrex.solvers.robust_lidar_observations import (
    _filter_packed_factor_mad,
    filter_lidar_observations_mad,
)


def _fixture(count=4097):
    records = [LivoxPointRecord(
        (float(x)+0.25, float(y)+0.25, float(z)+0.25, 0.0),
        normal_xyz=(float(x)+4.0, float(y)+4.0, float(z)+4.0),
    ) for x in range(-3, 4) for y in range(-3, 4) for z in range(-3, 4)]
    points = [tuple(float(value) for value in row)
              for row in np.random.default_rng(710).uniform(-3.0, 3.0, (count, 3))]
    samples = [OdometryPoseSample(i*1_000_000_000, SE3(
        (i*0.03, -i*0.02, i*0.01), (0.02*i, 0.03*i, -0.01*i, 1.0),
    )) for i in range(4)]
    times = [i*3_000_000 for i in range(count)]
    times = [time % 2_500_000_000 for time in times]
    return records, points, samples, times


def _factor(observations):
    return LidarRigPointToPlaneFactor(
        variable="T_base_lidar", sensor="target", t_ego_lidar=SE3.identity(),
        observations=observations, _array_kernel=numba_rig_residuals,
        _array_linearizer=numba_rig_linearization, _array_normal_kernel=numba_rig_normal_equations,
    )


@pytest.mark.parametrize("strategy", ["uniform", "adaptive", "hybrid"])
@pytest.mark.parametrize("count", [64, 4097])
def test_packed_matches_all_native_observations_and_solver_history(strategy, count):
    records, points, samples, times = _fixture(count)
    uniform_map = build_voxel_plane_map(records, 1.0) if strategy != "adaptive" else None
    uniform = VoxelPlaneQueryContext(uniform_map) if uniform_map is not None else None
    adaptive_map = build_adaptive_voxel_plane_map(
        records, AdaptiveVoxelPolicy(min_points_per_voxel=1),
    ) if strategy != "uniform" else None
    adaptive = AdaptivePlaneQueryContext(adaptive_map) if adaptive_map is not None else None
    builder = _PackedObservationBuilder.capture(points, uniform, adaptive)
    assert builder is not None
    packed_track, native_track = OdometryTrack(samples), OdometryTrack(samples)
    packed_context = NumbaContinuousTimeQueryContext.capture(packed_track)
    native_context = NumbaContinuousTimeQueryContext.capture(native_track)
    batch = packed_context._evaluate_rows(
        packed_track, times, builder.points, SE3.identity(), SE3.identity(), 0.1,
    )
    poses, world, usable = native_context.evaluate(
        native_track, times, points, SE3.identity(), SE3.identity(), 0.1,
    )
    assert usable and batch[-1]
    np.testing.assert_array_equal(world.view(np.uint64), batch[1].view(np.uint64))
    packed = builder.build(np.arange(count), batch[0], batch[1], voxel_size_m=1.0,
                           correspondence_gate_m=1.5)
    native = build_rig_point_to_plane_observations(
        source_records=records, target_points=points, initial_t_source_target=SE3.identity(),
        target_t_world_source=poses, voxel_size_m=1.0, correspondence_gate_m=1.5,
        source_plane_map=uniform_map, adaptive_plane_map=adaptive_map, source_plane_queries=uniform,
        _adaptive_queries=adaptive, _prepared_world_coordinates=world,
    )
    assert packed is not None
    assert packed.materialize() == tuple(native)
    assert len(packed) == len(native)
    left, right = _factor(packed), _factor(native)
    assert left.linearize([0.01]*6) == right.linearize([0.01]*6)
    for key in ("residuals_m", "jacobian", "hessian", "gradient"):
        np.testing.assert_array_equal(
            np.asarray(getattr(left.linearize([0.01]*6), key)).view(np.uint64),
            np.asarray(getattr(right.linearize([0.01]*6), key)).view(np.uint64),
        )
    assert type(left._observations) is _PackedObservations
    options = FixedTrajectorySe3SolverOptions(max_iterations=3)
    assert FixedTrajectorySe3ExtrinsicSolver().solve(left, options) == (
        FixedTrajectorySe3ExtrinsicSolver().solve(right, options)
    )
    assert type(left._observations) is _PackedObservations
    assert left.evaluate([0.01]*6) == right.evaluate([0.01]*6)
    assert left.observations == tuple(native) and type(left.observations) is tuple
    # The public replacement contract invalidates numeric preparation.
    left.observations = tuple(reversed(native))
    right.observations = tuple(reversed(native))
    assert left.linearize() == right.linearize()
    assert (packed_track.interpolation_count, packed_track.clamp_count,
            packed_track.max_extrapolation_s) == (
                native_track.interpolation_count, native_track.clamp_count,
                native_track.max_extrapolation_s,
            )


@pytest.mark.parametrize("floor,scale", [(0.25, 3.5), (1.0, 0.0), (0.25, 0.0)])
def test_packed_mad_preserves_inclusive_threshold_fallback_order_and_arrays(floor, scale):
    records, points, samples, times = _fixture()
    uniform = VoxelPlaneQueryContext(build_voxel_plane_map(records, 1.0))
    builder = _PackedObservationBuilder.capture(points, uniform, None)
    context = NumbaContinuousTimeQueryContext.capture(OdometryTrack(samples))
    batch = context._evaluate_rows(OdometryTrack(samples), times, builder.points,
                                   SE3.identity(), SE3.identity(), 0.1)
    # Context snapshots must match the same track instance.
    assert batch is None
    track = OdometryTrack(samples)
    context = NumbaContinuousTimeQueryContext.capture(track)
    batch = context._evaluate_rows(track, times, builder.points,
                                   SE3.identity(), SE3.identity(), 0.1)
    packed = builder.build(np.arange(len(points)), batch[0], batch[1], voxel_size_m=1.0,
                           correspondence_gate_m=1.5)
    left, right = _factor(packed), _factor(packed.materialize())
    kwargs = {"mad_scale": scale, "minimum_threshold_m": 0.02,
              "minimum_inlier_fraction": floor, "minimum_inliers": 6}
    rejected = _filter_packed_factor_mad(left, **kwargs)
    expected = filter_lidar_observations_mad(
        right.observations, variable=right.variable, sensor=right.sensor,
        t_ego_lidar=right.t_ego_lidar, _factor=right, **kwargs,
    )
    assert rejected == expected.rejected_count
    assert type(left._observations) is _PackedObservations
    assert left.linearize() == right.linearize()
    assert left.observations == right.observations


@pytest.mark.parametrize("strategy", ["uniform", "adaptive"])
@pytest.mark.parametrize("mad", [False, True])
def test_complete_profile_result_and_odometry_telemetry_match_unpacked(strategy, mad, monkeypatch):
    records, points, samples, times = _fixture(1024)
    problem = ContinuousTimeLidarPairProblem(
        records, points, times, OdometryTrack(samples), SE3.identity(), SE3.identity(),
        "T_base_lidar", "target", 1.0, 1.5,
    )
    options = ContinuousTimeLidarPairOptions(
        query_backend="numba_cpu", voxel_strategy=strategy, adaptive_min_points_per_voxel=1,
        outlier_policy="mad" if mad else "none", max_iterations=3,
        solver=FixedTrajectorySe3SolverOptions(max_iterations=3),
    )
    actual = ContinuousTimeLidarPairSolver().solve(problem, options)
    other = replace(problem, odometry_track=OdometryTrack(samples))
    monkeypatch.setattr(_PackedObservationBuilder, "capture", classmethod(lambda *args: None))
    expected = ContinuousTimeLidarPairSolver().solve(other, options)
    assert asdict(actual) == asdict(expected)
    assert json.dumps(asdict(actual), sort_keys=True) == (
        json.dumps(asdict(expected), sort_keys=True)
    )
    assert problem.odometry_track.interpolation_count == other.odometry_track.interpolation_count
    assert problem.odometry_track.clamp_count == other.odometry_track.clamp_count
    assert problem.odometry_track.max_extrapolation_s == other.odometry_track.max_extrapolation_s


def test_native_constructor_and_query_overrides_keep_unpacked_semantics(monkeypatch):
    records, points, _, _ = _fixture(64)
    uniform = VoxelPlaneQueryContext(build_voxel_plane_map(records, 1.0))
    builder = _PackedObservationBuilder.capture(points, uniform, None)
    assert builder is not None
    uniform.nearest_matches = lambda *args, **kwargs: []
    assert builder.build(np.arange(64), np.zeros((64, 7)), np.zeros((64, 3)),
                         voxel_size_m=1.0, correspondence_gate_m=1.5) is None
    monkeypatch.setattr(graph.LidarPointToPlaneObservation, "__post_init__", lambda self: None)
    assert _PackedObservationBuilder.capture(points, None, None) is None


@pytest.mark.parametrize("gate", [1e-200, 2e-170, 3e-170])
def test_underflowing_kd_distances_keep_exact_scalar_identity_and_gate(gate, monkeypatch):
    far = AdaptiveVoxelPlane(0, (0, 0, 0), (3e-170, 0.0, 0.0), (1.0, 0.0, 0.0), 1, 1.0, 0.0)
    near = replace(far, level=1, centroid=(2e-170, 0.0, 0.0))
    plane_map = AdaptiveVoxelPlaneMap((far, near), (0.0,)*3,
        {0: _SparsePlaneCells({far.key: far}), 1: _SparsePlaneCells({near.key: near})})
    context = AdaptivePlaneQueryContext(plane_map)
    expected = plane_map.nearest_plane((0.0,)*3, correspondence_gate_m=gate)
    assert expected is (near if gate >= 2e-170 else None)
    # Successful tiny-distance repair must avoid the broad scalar voxel traversal.
    monkeypatch.setattr(AdaptiveVoxelPlaneMap, "nearest_plane",
                        lambda *args, **kwargs: pytest.fail("unexpected broad repair"))
    assert all(plane is expected for plane in context.nearest_planes(
        np.zeros((64, 3)), correspondence_gate_m=gate,
    ))


def test_row_only_cache_preserves_failed_prefix_custom_diagnostics_and_budget():
    _, points, samples, _ = _fixture(64)
    track = OdometryTrack(samples)
    context = NumbaContinuousTimeQueryContext.capture(track)
    xyz = np.asarray(points)
    identity = SE3.identity()
    times = [0]*63 + [3_050_000_000]
    assert context._evaluate_rows(track, times, xyz, identity, identity, 0.1)[-1]
    assert track.interpolation_count == 64 and track.clamp_count == 1
    assert context._pose_cache_bytes <= context._pose_cache_limit_bytes
    assert all(not value.rows.flags.writeable and not value.poses
               for value in context._pose_cache.values())
    before = track.interpolation_count
    assert not context._evaluate_rows(track, times, xyz, identity, identity, 0.01)[-1]
    assert track.interpolation_count-before == 64 and track.clamp_count == 2
    # Public SE3 output remains available after an internal row-only cache hit.
    result = context.evaluate(track, times, points, identity, identity, 0.1)
    assert len(result[0]) == 64 and all(type(pose) is SE3 for pose in result[0])
    calls = []
    track._record_extrapolation = calls.append
    context._evaluate_rows(track, times, xyz, identity, identity, 0.01)
    assert calls == [0.0]*63 + [0.05]
    context._pose_cache_limit_bytes = 0
    size = context._pose_cache_bytes
    context._evaluate_rows(track, [1_500_000_000]*64, xyz, identity, identity, 0.1)
    assert context._pose_cache_bytes == size


def test_fused_kernel_preserves_caller_thread_mask_after_failure(monkeypatch):
    from numba import get_num_threads, set_num_threads

    from calibrex.graph.lidar_point_to_plane import _ObservationArrays

    previous = get_num_threads()
    mask = min(previous, 2)
    set_num_threads(mask)
    arrays = _ObservationArrays(np.zeros((4096, 3)), np.zeros((4096, 3)),
        np.ones((4096, 3)), np.tile([0.0]*6+[1.0], (4096, 1)), np.ones(4096), None)
    observed = []

    def failing(*args):
        observed.append(get_num_threads())
        raise RuntimeError("kernel failure")

    monkeypatch.setattr(adapter, "_fused_jacobian_rows", failing)
    try:
        with pytest.raises(RuntimeError, match="kernel failure"):
            adapter.numba_rig_linearization(arrays, [SE3.identity()]*13, [1e-4]*6)
        assert observed == [mask] and get_num_threads() == mask
        with pytest.raises(ZeroDivisionError):
            adapter.numba_rig_linearization(arrays, [SE3.identity()]*13, [0.0]*6)
        assert get_num_threads() == mask
    finally:
        set_num_threads(previous)


def test_parallel_clock_candidates_keep_order_complete_results_and_counters(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from numba import get_num_threads

    records, points, samples, times = _fixture(16640)
    problem = ContinuousTimeLidarPairProblem(
        records, points, times, OdometryTrack(samples), SE3.identity(), SE3.identity(),
        "T_base_lidar", "target", 1.0, 1.5,
    )
    options = ContinuousTimeLidarPairOptions(
        query_backend="numba_cpu", voxel_strategy="adaptive", adaptive_min_points_per_voxel=1,
        max_target_points_per_split=None,
        max_iterations=2, solver=FixedTrajectorySe3SolverOptions(max_iterations=2),
    )
    submit = ThreadPoolExecutor.submit
    calls = []
    masks = []
    caller_mask = get_num_threads()

    def observed_submit(self, fn, *args, **kwargs):
        calls.append(args)

        def observed_task():
            masks.append(get_num_threads())
            return fn(*args, **kwargs)

        return submit(self, observed_task)

    monkeypatch.setattr(ThreadPoolExecutor, "submit", observed_submit)
    actual = ContinuousTimeLidarPairSolver().solve(problem, options)
    assert len(calls) == 6
    assert len(masks) == 6 and get_num_threads() == caller_mask
    other = replace(problem, odometry_track=OdometryTrack(samples))
    monkeypatch.setattr(_PackedObservationBuilder, "capture", classmethod(lambda *args: None))
    expected = ContinuousTimeLidarPairSolver().solve(other, options)
    assert len(calls) == 6
    assert json.dumps(asdict(actual), sort_keys=True) == (
        json.dumps(asdict(expected), sort_keys=True)
    )
    assert problem.odometry_track.interpolation_count == other.odometry_track.interpolation_count
    assert problem.odometry_track.clamp_count == other.odometry_track.clamp_count
    assert problem.odometry_track.max_extrapolation_s == other.odometry_track.max_extrapolation_s


@pytest.mark.parametrize("guard", ["custom_factor", "custom_solver", "small_sample_cap"])
def test_custom_methods_and_small_sample_caps_keep_serial_profile(guard, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    records, points, samples, times = _fixture(16640)
    problem = ContinuousTimeLidarPairProblem(
        records, points, times, OdometryTrack(samples), SE3.identity(), SE3.identity(),
        "T_base_lidar", "target", 1.0, 1.5,
    )
    if guard == "custom_factor":
        original = LidarRigPointToPlaneFactor.linearize
        monkeypatch.setattr(LidarRigPointToPlaneFactor, "linearize",
                            lambda self, *args, **kwargs: original(self, *args, **kwargs))
    elif guard == "custom_solver":
        original = FixedTrajectorySe3ExtrinsicSolver.solve
        monkeypatch.setattr(FixedTrajectorySe3ExtrinsicSolver, "solve",
                            lambda self, *args, **kwargs: original(self, *args, **kwargs))
    monkeypatch.setattr(ThreadPoolExecutor, "submit",
                        lambda *args, **kwargs: pytest.fail("guard must keep serial evaluation"))
    result = ContinuousTimeLidarPairSolver().solve(problem, ContinuousTimeLidarPairOptions(
        query_backend="numba_cpu", max_iterations=1,
        max_target_points_per_split=64 if guard == "small_sample_cap" else None,
        solver=FixedTrajectorySe3SolverOptions(max_iterations=1),
    ))
    assert result.iterations
