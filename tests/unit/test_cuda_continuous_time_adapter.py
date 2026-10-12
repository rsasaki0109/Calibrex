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
    assert gpu.linearization_calls == 2 and gpu.residual_calls == 1
    assert gpu.rows == 3*count


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


def test_identity_binds_kernel_and_declares_cpu_trajectory_and_reduction():
    identity = cuda_continuous_time_identity()
    assert identity.startswith("calibrex.cuda_continuous_time/v0.4;fp64=true;fma=false;")
    assert ";trajectory_cpu=true;normal_cpu=true;standalone_residual_cpu=true;" in identity
    assert len(identity.rsplit("=", 1)[1]) == 64


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
    assert asdict(replace(cpu, query_backend="same")) == asdict(replace(gpu, query_backend="same"))
    for name in ("interpolation_count", "clamp_count", "max_extrapolation_s"):
        assert getattr(problem.odometry_track, name) == getattr(other.odometry_track, name)
