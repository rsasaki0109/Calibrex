"""Actual-device exact finite differences and complete native optimizer parity."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("numba")
import cupy as cp

from calibrex.core.geometry import SE3
from calibrex.data.livox import LivoxPointRecord
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers.continuous_time_lidar_pair_solver import (
    ContinuousTimeLidarPairOptions,
    ContinuousTimeLidarPairProblem,
    ContinuousTimeLidarPairSolver,
)
from calibrex.solvers.cuda_continuous_time_adapter import (
    CudaRigLinearizer,
    cuda_continuous_time_identity,
)
from calibrex.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
)
from calibrex.solvers.numba_continuous_time_adapter import (
    _numba_rig_rmse,
    _numba_values_rmse,
    numba_rig_linearization,
    numba_rig_normal_equations,
    numba_rig_residuals,
)


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


def _factors(count: int = 4097):
    rng = np.random.default_rng(211)
    observations = [LidarPointToPlaneObservation(
        tuple(point), (0.1, 0.2, -0.3), (0.1, -0.2, 0.9),
        SE3(tuple(point), (0.1, 0.2, -0.3, 0.8)),
        weight=0.0 if i % 17 == 0 else 0.4+i/count,
    ) for i, point in enumerate(rng.normal(size=(count, 3)).tolist())]
    kwargs = {"variable": "extrinsic", "observations": observations,
              "t_ego_lidar": SE3((0.1, -0.2, 0.3), (0.1, 0.2, -0.3, 0.8)),
              "_array_normal_kernel": numba_rig_normal_equations}
    gpu = CudaRigLinearizer()
    return (
        LidarRigPointToPlaneFactor(**kwargs, _array_kernel=numba_rig_residuals,
                                  _array_linearizer=numba_rig_linearization),
        LidarRigPointToPlaneFactor(**kwargs, _array_kernel=gpu.residuals,
                                  _array_linearizer=gpu.linearize), gpu,
    )


def _bits(value):
    return np.asarray(value, dtype=np.float64).view(np.uint64).tolist()


@pytest.mark.parametrize("count", [64, 65, 4096, 4097, 8193])
@pytest.mark.parametrize("correction", [None, (0.1, -0.2, 0.3, 0.04, -0.02, 0.01)])
def test_residuals_jacobian_and_ordered_system_match_every_bit(count, correction):
    cpu, factor, gpu = _factors(count)
    assert _bits(cpu.residuals(correction)) == _bits(factor.residuals(correction))
    left, right = cpu.linearize(correction), factor.linearize(correction)
    for name in ("residuals_m", "jacobian", "hessian", "gradient"):
        assert _bits(getattr(left, name)) == _bits(getattr(right, name))
    assert cpu.evaluate(correction) == factor.evaluate(correction)
    assert gpu.linearization_calls == 1 and gpu.linearization_cache_hits == 1
    assert gpu.residual_calls == 1 and gpu.rows == 2*count


@pytest.mark.parametrize("loss", ["none", "huber"])
def test_complete_optimizer_history_and_diagnostics_match(loss):
    cpu, factor, gpu = _factors()
    options = FixedTrajectorySe3SolverOptions(max_iterations=3, robust_loss=loss)
    solver = FixedTrajectorySe3ExtrinsicSolver()
    assert solver.solve(cpu, options) == solver.solve(factor, options)
    assert gpu.linearization_calls > 0


def test_payload_replacement_and_perturbation_step_changes_do_not_reuse_stale_values():
    cpu, factor, gpu = _factors()
    factor.linearize()
    first_payload = gpu._payload
    for item in (cpu, factor):
        item.observations = tuple(reversed(item.observations))
        item.t_ego_lidar = SE3((0.4, -0.3, 0.2), (0.2, -0.3, 0.1, 0.9))
    assert cpu.linearize(translation_step_m=2e-4, rotation_step_rad=2e-5) == factor.linearize(
        translation_step_m=2e-4, rotation_step_rad=2e-5,
    )
    assert gpu._payload is not first_payload


def test_cached_arrays_remain_owned_and_writable_without_contaminating_later_calls():
    cpu, factor, gpu = _factors()
    arrays = factor._prepared_arrays()
    assert arrays is not None
    transforms = [factor._corrected_transform(None)]*13
    steps = [1e-4]*3 + [1e-5]*3
    expected = numba_rig_linearization(arrays, transforms, steps)
    first = gpu.linearize(arrays, transforms, steps)
    first[0][:] = 123.0
    first[1][:] = -456.0
    second = gpu.linearize(arrays, transforms, steps)
    for left, right in zip(second, expected, strict=True):
        assert left.flags.writeable
        np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
    second[0][:] = -789.0
    third = gpu.linearize(arrays, transforms, steps)
    np.testing.assert_array_equal(third[0].view(np.uint64), expected[0].view(np.uint64))
    assert gpu.linearization_calls == 1 and gpu.linearization_cache_hits == 2
    assert gpu.linearization_cache_bytes == sum(v.nbytes for v in expected)
    assert cpu.evaluate() == factor.evaluate()


def test_cache_uses_transform_and_step_bits_and_evicts_the_previous_entry():
    _, factor, gpu = _factors(64)
    arrays = factor._prepared_arrays()
    assert arrays is not None
    transforms = [factor._corrected_transform(None)]*13
    steps = [1e-4]*3 + [1e-5]*3
    gpu.linearize(arrays, transforms, steps)
    changed = list(steps)
    changed[0] = np.nextafter(changed[0], np.inf)
    gpu.linearize(arrays, transforms, changed)
    gpu.linearize(arrays, transforms, steps)
    transforms[0] = SE3((-0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    gpu.linearize(arrays, transforms, steps)
    assert gpu.linearization_calls == 4 and gpu.linearization_cache_hits == 0


def test_cache_budget_bypass_and_payload_replacement_keep_complete_numerical_results(monkeypatch):
    from calibrex.solvers import cuda_continuous_time_adapter as adapter

    _, factor, gpu = _factors(64)
    arrays = factor._prepared_arrays()
    assert arrays is not None
    transforms = [factor._corrected_transform(None)]*13
    steps = [1e-4]*3 + [1e-5]*3
    expected = numba_rig_linearization(arrays, transforms, steps)
    monkeypatch.setattr(adapter, "_LINEARIZATION_CACHE_BYTES", 64*56-1)
    for _ in range(2):
        actual = gpu.linearize(arrays, transforms, steps)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
    assert gpu.linearization_calls == 2 and gpu.linearization_cache_bytes == 0
    monkeypatch.setattr(adapter, "_LINEARIZATION_CACHE_BYTES", 64*56)
    gpu.linearize(arrays, transforms, steps)
    gpu.linearize(arrays, transforms, steps)
    assert gpu.linearization_cache_hits == 1
    gpu.linearize(replace(arrays), transforms, steps)
    assert gpu.linearization_calls == 4 and gpu.linearization_cache_hits == 1
    fork = gpu.fork()
    assert fork.linearization_cache_bytes == 0 and fork._cached_result is None


def test_concurrent_differences_and_cache_hits_keep_independent_writable_results():
    _, factor, gpu = _factors(65)
    arrays = factor._prepared_arrays()
    assert arrays is not None
    steps = [1e-4]*3 + [1e-5]*3
    transforms = [[factor._corrected_transform([value]*6)]*13 for value in (0.0, 0.01)]
    jobs = [transforms[0]]*3 + [transforms[1]]*3 + [transforms[0]]*3
    expected = [numba_rig_linearization(arrays, t, steps) for t in jobs]
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda t: gpu.linearize(arrays, t, steps), jobs))
    for actual, reference in zip(results, expected, strict=True):
        for left, right in zip(actual, reference, strict=True):
            np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
    assert gpu.linearization_calls + gpu.linearization_cache_hits == len(jobs)
    results[0][0][:] = 123.0
    np.testing.assert_array_equal(results[1][0].view(np.uint64), expected[1][0].view(np.uint64))


def test_owned_stream_and_device_restore_callers_cuda_context():
    _, factor, gpu = _factors()
    device = cp.cuda.runtime.getDevice()
    with cp.cuda.Stream(non_blocking=True) as stream:
        factor.linearize()
        assert cp.cuda.get_current_stream().ptr == stream.ptr
        assert cp.cuda.runtime.getDevice() == device
    assert gpu._stream.ptr != stream.ptr


def test_fork_shares_compiled_kernels_and_owns_payload_and_stream():
    cpu, factor, gpu = _factors()
    fork = gpu.fork()
    assert fork._linearize is gpu._linearize and fork._residuals is gpu._residuals
    assert fork._stream.ptr != gpu._stream.ptr
    arrays = factor._prepared_arrays()
    assert arrays is not None
    assert fork.residuals(arrays, factor._corrected_transform(None)) == cpu.residuals()
    assert fork._arrays is arrays and gpu._arrays is None


def test_concurrent_callers_cannot_cross_contaminate_an_owned_payload():
    cpu, factor, gpu = _factors()
    other_cpu, other, _ = _factors(8193)
    jobs = [(factor._prepared_arrays(), factor._corrected_transform(None)),
            (other._prepared_arrays(), other._corrected_transform(None))]*3
    with ThreadPoolExecutor(max_workers=3) as executor:
        values = list(executor.map(lambda args: gpu.residuals(*args), jobs))
    expected = [cpu.residuals(), other_cpu.residuals()]*3
    assert values == expected
    assert gpu.residual_calls == 6 and gpu.rows == 3*(4097+8193)


def test_input_budget_mutability_and_zero_steps_are_checked_before_launch():
    _, factor, gpu = _factors()
    arrays = factor._prepared_arrays()
    assert arrays is not None
    with pytest.raises(ZeroDivisionError):
        factor.linearize(translation_step_m=0.0)
    with pytest.raises(ValueError, match="thirteen transforms"):
        gpu.linearize(arrays, [], [1e-4]*6)
    with pytest.raises(ValueError, match="immutable"):
        gpu.residuals(replace(arrays, points=arrays.points.copy()), SE3.identity())
    with pytest.raises(ValueError, match="matching columns"):
        wrong = arrays.points[:-1].copy()
        wrong.flags.writeable = False
        gpu.residuals(replace(arrays, points=wrong), SE3.identity())
    with pytest.raises(ValueError, match="payload budget"):
        gpu.residuals(replace(arrays,
            points=np.broadcast_to(arrays.points[0], (1_000_000, 3)),
            plane_points=np.broadcast_to(arrays.plane_points[0], (1_000_000, 3)),
            normals=np.broadcast_to(arrays.normals[0], (1_000_000, 3)),
            poses=np.broadcast_to(arrays.poses[0], (1_000_000, 7)),
            sqrt_weights=np.broadcast_to(arrays.sqrt_weights[0], (1_000_000,)),
        ), SE3.identity())


def test_captured_summary_reuses_payload_and_cached_values_without_new_uploads():
    _cpu, factor, gpu = _factors(8193)
    arrays = factor._prepared_arrays()
    initial = factor._corrected_transform(None)
    assert gpu._captured_residuals(arrays, initial) is None
    assert gpu._arrays is None and gpu._payload == () and gpu.residual_calls == 0
    factor.linearize()
    payload = gpu._payload
    cached = gpu._captured_residuals(arrays, initial)
    assert cached is gpu._cached_result[0] and not cached.flags.writeable
    assert gpu.residual_calls == 0
    assert _bits(_numba_values_rmse(cached)) == _bits(_numba_rig_rmse(arrays, initial))
    transform = factor._corrected_transform((.1, -.2, .3, .04, -.02, .01))
    fresh = gpu._captured_residuals(arrays, transform)
    assert gpu._payload is payload and gpu.residual_calls == 1
    assert not fresh.flags.writeable
    assert _bits(_numba_values_rmse(fresh)) == _bits(_numba_rig_rmse(arrays, transform))
    assert gpu.fork()._captured_residuals(arrays, transform) is None
    # Retaining old host rows remains safe across later payload/cache eviction.
    expected = cached.copy()
    factor.observations = tuple(reversed(factor.observations))
    replacement = factor._prepared_arrays()
    assert gpu._captured_residuals(replacement, transform) is None
    assert gpu._payload is payload and gpu.residual_calls == 1
    factor.linearize()
    np.testing.assert_array_equal(cached.view(np.uint64), expected.view(np.uint64))


def test_captured_summary_preserves_caller_stream_and_serializes_shared_context():
    _cpu, factor, gpu = _factors(8193)
    arrays = factor._prepared_arrays()
    factor.linearize()
    transforms = [factor._corrected_transform([value]*6) for value in (.01, .02, .03)]
    caller = cp.cuda.Stream(non_blocking=True)
    with caller:
        before = cp.cuda.get_current_stream().ptr
        gpu._captured_residuals(arrays, transforms[0])
        assert cp.cuda.get_current_stream().ptr == before
        assert gpu._captured_residuals(replace(arrays), transforms[0]) is None
        assert cp.cuda.get_current_stream().ptr == before
    with ThreadPoolExecutor(max_workers=3) as executor:
        values = list(executor.map(lambda t: gpu._captured_residuals(arrays, t), transforms))
    for value, transform in zip(values, transforms, strict=True):
        assert _bits(value) == _bits(numba_rig_residuals(arrays, transform))
    assert gpu.residual_calls == 4 and gpu.linearization_calls == 1


@pytest.mark.parametrize("loss", ["none", "huber"])
def test_captured_gpu_summaries_keep_complete_optimizer_history_and_diagnostics(loss):
    cpu, factor, gpu = _factors(8193)
    def summary(arrays, transform):
        values = gpu._captured_residuals(arrays, transform)
        return _numba_rig_rmse(arrays, transform) if values is None else _numba_values_rmse(values)
    factor._array_rmse_kernel = summary
    options = FixedTrajectorySe3SolverOptions(max_iterations=3, robust_loss=loss)
    assert FixedTrajectorySe3ExtrinsicSolver().solve(cpu, options) == (
        FixedTrajectorySe3ExtrinsicSolver().solve(factor, options))
    assert cpu.evaluate() == factor.evaluate()
    assert gpu.residual_calls > 0


def test_identity_binds_kernel_and_declares_cpu_trajectory_and_reduction():
    identity = cuda_continuous_time_identity()
    assert identity.startswith("calibrex.cuda_continuous_time/v0.7;fp64=true;fma=false;")
    assert ";trajectory_cpu=true;normal_cpu=true;standalone_residual_cpu=true;" in identity
    assert len(identity.rsplit("=", 1)[1]) == 64
    assert ";associations_gpu=false;" in identity
    assert ";linearization_cache=last;linearization_cache_bytes=67108864;" in identity
    actual = cuda_continuous_time_identity(associations_gpu=True)
    assert ";associations_gpu=true;" in actual and ";association_kernel_sha256=" in actual
    assert ";captured_summary_gpu=false;summary_reduction_cpu=true;" in identity
    assert ";captured_summary_gpu=true;summary_reduction_cpu=true;" in (
        cuda_continuous_time_identity(captured_summary_gpu=True))


def test_gpu_queries_are_recorded_even_when_no_correspondences_reach_the_optimizer(monkeypatch):
    samples = [OdometryPoseSample(i*1_000_000_000, SE3.identity()) for i in range(2)]
    problem = ContinuousTimeLidarPairProblem(
        [LivoxPointRecord((0., 0., 0., 0.), (1., 0., 0.))], [(20., 0., 0.)]*9000,
        [0]*9000, OdometryTrack(samples), SE3.identity(), SE3.identity(), "extrinsic", "target",
        1., .5,
    )
    def unexpected(*args, **kwargs):
        raise AssertionError("an empty factor cannot request CUDA differences")
    monkeypatch.setattr(CudaRigLinearizer, "linearize", unexpected)
    result = ContinuousTimeLidarPairSolver().solve(problem, ContinuousTimeLidarPairOptions(
        query_backend="cuda", max_target_points_per_split=None,
    ))
    assert result.status == "insufficient_constraints"
    assert result.query_backend.startswith("calibrex.cuda_continuous_time/v0.7;")
    assert ";captured_summary_gpu=false;" in result.query_backend
    assert ";associations_gpu=true;" in result.query_backend


@pytest.mark.parametrize("count", [1024, 18000])
def test_complete_clock_profile_keeps_results_telemetry_and_actual_backend(count):
    records = [LivoxPointRecord(
        (float(x)+0.25, float(y)+0.25, float(z)+0.25, 0.0),
        normal_xyz=(float(x)+4.0, float(y)+4.0, float(z)+4.0),
    ) for x in range(-3, 4) for y in range(-3, 4) for z in range(-3, 4)]
    points = [tuple(float(v) for v in row)
              for row in np.random.default_rng(710).uniform(-3, 3, (count, 3))]
    samples = [OdometryPoseSample(i*1_000_000_000, SE3(
        (i*0.03, -i*0.02, i*0.01), (0.02*i, 0.03*i, -0.01*i, 1.0),
    )) for i in range(4)]
    problem = ContinuousTimeLidarPairProblem(
        records, points, [(i*3_000_000) % 2_500_000_000 for i in range(count)],
        OdometryTrack(samples), SE3.identity(), SE3.identity(), "T_base_lidar", "target", 1., 1.5,
    )
    options = ContinuousTimeLidarPairOptions(
        query_backend="numba_cpu", max_target_points_per_split=None, max_iterations=2,
        voxel_strategy="adaptive", adaptive_min_points_per_voxel=1,
        solver=FixedTrajectorySe3SolverOptions(max_iterations=2),
    )
    other = replace(problem, odometry_track=OdometryTrack(samples))
    cpu = ContinuousTimeLidarPairSolver().solve(problem, options)
    gpu = ContinuousTimeLidarPairSolver().solve(other, replace(options, query_backend="cuda"))
    expected_prefix = "calibrex.cuda_continuous_time/" if count == 18000 else (
        "calibrex.numba_continuous_time/"
    )
    assert gpu.query_backend.startswith(expected_prefix)
    if count == 18000:
        assert ";captured_summary_gpu=true;summary_reduction_cpu=true;" in gpu.query_backend
    assert asdict(replace(cpu, query_backend="same")) == asdict(replace(gpu, query_backend="same"))
    for name in ("interpolation_count", "clamp_count", "max_extrapolation_s"):
        assert getattr(problem.odometry_track, name) == getattr(other.odometry_track, name)
