"""Initial projections preserve native bits, ownership, MAD and public hooks."""
from dataclasses import asdict, replace

import numpy as np
import pytest

pytest.importorskip("numba")
from test_packed_continuous_time_lidar import _fixture

from calibrex.core.geometry import SE3
from calibrex.data.livox import VoxelPlaneQueryContext, build_voxel_plane_map
from calibrex.data.odometry_track import OdometryTrack
from calibrex.evaluation.packed_lidar_observations import _PackedObservationBuilder
from calibrex.graph import lidar_point_to_plane as graph
from calibrex.solvers import numba_continuous_time_adapter as a
from calibrex.solvers.continuous_time_lidar_pair_solver import (
    ContinuousTimeLidarPairOptions,
    ContinuousTimeLidarPairProblem,
    ContinuousTimeLidarPairSolver,
)
from calibrex.solvers.fixed_trajectory_se3_solver import FixedTrajectorySe3SolverOptions
from calibrex.solvers.robust_lidar_observations import _filter_packed_factor_mad


def _prepared(count=8192, transform=None):
    transform = transform or SE3((.1, -.2, .3), (.2, -.3, .1, .9))
    records, points, samples, times = _fixture(count)
    track = OdometryTrack(samples)
    context = a.NumbaContinuousTimeQueryContext.capture(track)
    uniform = VoxelPlaneQueryContext(build_voxel_plane_map(records, 1.))
    builder = _PackedObservationBuilder.capture(points, uniform, None)
    batch = context._evaluate_rows(track, times, builder.points, SE3.identity(), transform, .1)
    packed = builder.build(np.arange(count), batch[0], batch[1], voxel_size_m=1.,
                           correspondence_gate_m=1.5, _array_packer=lambda *args:
                           a._pack_rig_observations(*args, _initial_transform=transform))
    return packed, transform


@pytest.mark.parametrize("transform", [SE3.identity(), SE3((.1, -.2, .3), (.2, -.3, .1, .9)),
                                      SE3((-.3, .1, .2), (.7, .4, -.2, .3))])
def test_projected_residuals_match_every_native_bit_before_and_after_selection(transform):
    packed, transform = _prepared(transform=transform)
    for observations in (packed, packed.select(np.arange(len(packed)-1, -1, -2))):
        arrays = observations.arrays
        actual = graph._initial_residual_values(arrays, transform)
        expected = np.asarray(a.numba_rig_residuals(arrays, transform))
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
        assert not actual.flags.writeable
        assert a._numba_values_rmse(actual) == a._numba_rig_rmse(arrays, transform)


def test_cache_binds_every_input_identity_mutability_and_exact_transform_bits():
    packed, transform = _prepared()
    arrays = packed.arrays
    for field in ("points", "plane_points", "normals", "poses", "sqrt_weights"):
        changed = getattr(arrays, field).copy()
        changed.flags.writeable = False
        replaced = replace(arrays, **{field: changed})
        assert graph._initial_residual_values(replaced, transform) is None
        value = getattr(arrays, field)
        value.flags.writeable = True
        assert graph._initial_residual_values(arrays, transform) is None
        value.flags.writeable = False
    changed_transform = SE3((np.nextafter(transform.translation_m[0], np.inf),
                             *transform.translation_m[1:]), transform.rotation_quat_xyzw)
    assert graph._initial_residual_values(arrays, changed_transform) is None
    assert graph._initial_residual_values(arrays, transform) is not None
    arrays.initial.values.flags.writeable = True
    assert graph._initial_residual_values(arrays, transform) is None


def test_oversized_projection_cache_is_not_borrowed():
    packed, transform = _prepared()
    arrays = packed.arrays
    n = 2097153
    inputs = tuple(np.broadcast_to(value[:1], (n, *value.shape[1:])) for value in (
        arrays.points, arrays.plane_points, arrays.normals, arrays.poses, arrays.sqrt_weights))
    values = np.broadcast_to(arrays.initial.values[:1], (n,))
    initial = graph._InitialResiduals(inputs, graph._transform_key(transform), values)
    large = graph._ObservationArrays(*inputs, None, initial)
    assert graph._initial_residual_values(large, transform) is None


@pytest.mark.parametrize("floor,scale", [(1., 0.), (.25, 3.5)])
def test_mad_avoids_list_roundtrip_and_preserves_fallback_order(floor, scale, monkeypatch):
    packed, transform = _prepared()
    cached = graph.LidarRigPointToPlaneFactor(
        variable="extrinsic", t_ego_lidar=transform, observations=packed,
        _array_kernel=a.numba_rig_residuals, _initial_array_kernel=graph._initial_residual_values,
    )
    native = graph.LidarRigPointToPlaneFactor(
        variable="extrinsic", t_ego_lidar=transform, observations=packed,
        _array_kernel=a.numba_rig_residuals,
    )
    kwargs = {"mad_scale": scale, "minimum_threshold_m": .02,
              "minimum_inlier_fraction": floor, "minimum_inliers": 6}
    expected = _filter_packed_factor_mad(native, **kwargs)
    monkeypatch.setattr(cached, "_array_kernel", lambda *args: pytest.fail("must borrow rows"))
    cached._captured_array_kernel = cached._array_kernel
    assert _filter_packed_factor_mad(cached, **kwargs) == expected
    for field in ("points", "plane_points", "normals", "poses", "sqrt_weights"):
        np.testing.assert_array_equal(getattr(cached._prepared_arrays(), field).view(np.uint64),
                                      getattr(native._prepared_arrays(), field).view(np.uint64))
    assert graph._initial_residual_values(cached._prepared_arrays(), transform) is not None


def test_public_residual_overrides_are_still_called_by_mad():
    packed, transform = _prepared()
    factor = graph.LidarRigPointToPlaneFactor(
        variable="extrinsic", t_ego_lidar=transform, observations=packed,
        _array_kernel=a.numba_rig_residuals, _initial_array_kernel=graph._initial_residual_values,
    )
    calls = []
    def custom(correction=None):
        calls.append(correction)
        return [0.]*len(packed)
    factor.residuals = custom
    assert factor._compiled_initial_residuals() is None
    assert _filter_packed_factor_mad(factor, mad_scale=3.5, minimum_threshold_m=.02,
                                   minimum_inlier_fraction=.25, minimum_inliers=6) == 0
    assert calls == [None]


@pytest.mark.parametrize("mad", [False, True])
def test_complete_clock_profile_and_counters_match_without_projection_cache(mad, monkeypatch):
    records, points, samples, times = _fixture(16384)
    problem = ContinuousTimeLidarPairProblem(records, points, times, OdometryTrack(samples),
        SE3.identity(), SE3.identity(), "extrinsic", "target", 1., 1.5)
    options = ContinuousTimeLidarPairOptions(query_backend="numba_cpu",
        max_target_points_per_split=None, max_iterations=1, outlier_policy="mad" if mad else "none",
        solver=FixedTrajectorySe3SolverOptions(max_iterations=2))
    actual = ContinuousTimeLidarPairSolver().solve(problem, options)
    other = replace(problem, odometry_track=OdometryTrack(samples))
    native = a._pack_rig_observations
    def uncached(*args, **kwargs):
        kwargs.pop("_initial_transform", None)
        return native(*args, **kwargs)
    monkeypatch.setattr(a, "_pack_rig_observations", uncached)
    expected = ContinuousTimeLidarPairSolver().solve(other, options)
    assert ";initial_projection_reused=true;" in actual.query_backend
    assert ";initial_projection_reused=false;" in expected.query_backend
    assert asdict(replace(actual, query_backend="same")) == asdict(
        replace(expected, query_backend="same"))
    for name in ("interpolation_count", "clamp_count", "max_extrapolation_s"):
        assert getattr(problem.odometry_track, name) == getattr(other.odometry_track, name)
