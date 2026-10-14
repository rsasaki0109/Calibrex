"""Actual-device registration, temporal chaining and boundary parity checks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers.cuda_scan_odometry_adapter import CudaOdometryTarget
from calibrex.solvers.scan_to_scan_odometry import (
    ScanOdometryOptions,
    ScanRegistration,
    estimate_normals,
    register_point_to_plane,
    run_scan_to_scan_odometry,
)


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


def _scene() -> np.ndarray:
    rng = np.random.default_rng(42)
    parts = []
    for axis in range(3):
        face = rng.uniform(-5, 5, (2000, 3))
        face[:, axis] = 4.0
        parts.append(face)
    return np.concatenate(parts)


def _same(cpu: ScanRegistration, gpu: ScanRegistration) -> None:
    assert (cpu.converged, cpu.iterations, cpu.correspondences) == (
        gpu.converged,
        gpu.iterations,
        gpu.correspondences,
    )
    np.testing.assert_allclose(cpu.transform, gpu.transform, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        [cpu.rmse_m, cpu.hessian_min_eigenvalue],
        [gpu.rmse_m, gpu.hessian_min_eigenvalue],
        rtol=1e-6,
        atol=1e-9,
    )
    assert gpu.execution_backend.startswith("calibrex.cuda_scan_odometry/v0.1;")


def test_cuda_registration_recovers_known_motion_with_native_quality() -> None:
    target = _scene()
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_rotvec([0.01, -0.02, 0.03]).as_matrix()
    truth[:3, 3] = [0.3, -0.2, 0.1]
    source = (target - truth[:3, 3]) @ truth[:3, :3]
    opts = ScanOdometryOptions()
    cpu = register_point_to_plane(source, target, np.eye(4), opts)
    gpu = register_point_to_plane(source, target, np.eye(4), opts, backend="cuda")
    assert gpu.converged
    np.testing.assert_allclose(gpu.transform, truth, atol=1e-6)
    _same(cpu, gpu)


@pytest.mark.parametrize("window", [1, 3])
@pytest.mark.parametrize("quantum", [0., 1e-9])
def test_cuda_temporal_odometry_keeps_gap_scaling_and_local_maps(
    window: int, quantum: float,
) -> None:
    target = _scene()
    scans, times = [], []
    for i in (0, 1, 2, 4, 5):
        pose = Rotation.from_rotvec([0.004 * i, -0.003 * i, 0.005 * i]).as_matrix()
        scans.append((target - np.array([0.08, -0.02, 0.03]) * i) @ pose)
        times.append(0.1 * i)
    opts = ScanOdometryOptions(
        voxel_size_m=0.15, local_map_scans=window, geometry_quantum_m=quantum,
    )
    cpu = run_scan_to_scan_odometry(scans, opts, times_s=times)
    gpu = run_scan_to_scan_odometry(scans, opts, times_s=times, backend="cuda")
    assert cpu.all_converged == gpu.all_converged
    np.testing.assert_allclose(cpu.poses, gpu.poses, rtol=1e-6, atol=1e-9)
    for a, b in zip(cpu.registrations, gpu.registrations, strict=True):
        _same(a, b)
    assert gpu.execution_statistics["gpu_query_rows"] > sum(len(p) for p in scans)


def test_cuda_normals_repair_ties_and_degenerate_planarity() -> None:
    target = np.array([[float(i), 0.0, 0.0] for i in range(40)])
    target = np.concatenate((target, target))
    opts = ScanOdometryOptions(normal_neighbours=4)
    native, planar = estimate_normals(target, cKDTree(target), opts)
    prepared = CudaOdometryTarget(target, opts)
    with cp.cuda.Device(prepared.device_id):
        prepared._ensure(cp.arange(len(target)))
    assert cp.asnumpy(prepared.planar).tolist() == planar.tolist()
    np.testing.assert_array_equal(cp.asnumpy(prepared.normals), native)
    assert prepared.statistics()["scalar_normal_repairs"] == len(target)
    assert prepared.statistics()["scalar_query_repairs"] > 0


def test_cuda_snapshots_inputs_and_reports_insufficient_correspondences() -> None:
    target = _scene()
    source = target.copy()
    opts = ScanOdometryOptions()
    prepared = CudaOdometryTarget(target, opts)
    target[:] = 1e4
    gpu = prepared.register(source, np.eye(4))
    cpu = register_point_to_plane(source, source, np.eye(4), opts)
    _same(cpu, gpu)
    empty = prepared.register(np.empty((0, 3)), np.eye(4))
    native = register_point_to_plane(np.empty((0, 3)), source, np.eye(4), opts)
    _same(native, empty)
    assert not empty.converged and empty.correspondences == 0


def test_cuda_rejects_unsupported_caches_and_invalid_geometry() -> None:
    target = _scene()
    opts = ScanOdometryOptions()
    with pytest.raises(ValueError, match="own tree"):
        register_point_to_plane(
            target, target, np.eye(4), backend="cuda", target_tree=cKDTree(target)
        )
    with pytest.raises(ValueError, match="neighbors"):
        CudaOdometryTarget(target, replace(opts, normal_neighbours=17))
    target[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite bounded"):
        CudaOdometryTarget(target, opts)


@pytest.mark.parametrize("gate", [np.nextafter(.5, 0.), .5, np.nextafter(.5, np.inf)])
def test_cuda_correspondence_gate_keeps_native_strict_boundary(gate: float) -> None:
    target = _scene()
    row = np.flatnonzero((np.abs(target[:2000, 1]) < 1) & (np.abs(target[:2000, 2]) < 1))[0]
    source = target[[row]] + np.array([.5, 0., 0.])
    options = ScanOdometryOptions(max_correspondence_m=gate)
    cpu = register_point_to_plane(source, target, np.eye(4), options)
    gpu = register_point_to_plane(source, target, np.eye(4), options, backend="cuda")
    _same(cpu, gpu)
    assert gpu.correspondences == int(gate > .5)


def test_cuda_deskew_rounds_share_target_and_preserve_trajectory() -> None:
    from calibrex.solvers.scan_to_scan_odometry import IncrementalScanOdometry

    target = _scene()
    opts = ScanOdometryOptions(voxel_size_m=0.15, deskew_iterations=2)
    cpu, gpu = IncrementalScanOdometry(opts), IncrementalScanOdometry(opts, backend="cuda")
    offsets = np.linspace(-0.02, 0.02, len(target))
    for i in range(4):
        cloud = target - np.array([0.05, -0.02, 0.01]) * i
        cpu.add(cloud, i * 0.1, offsets)
        gpu.add(cloud, i * 0.1, offsets)
    np.testing.assert_allclose(cpu.poses, gpu.poses, rtol=1e-6, atol=1e-9)
    for a, b in zip(cpu.registrations, gpu.registrations, strict=True):
        _same(a, b)


def test_cuda_native_registration_repair_preserves_close_stopping_decision() -> None:
    target = _scene()
    source = target - np.array([.1, 0., 0.])
    options = ScanOdometryOptions(max_iterations=1, convergence_translation_m=.1)
    probe = register_point_to_plane(source, target, np.eye(4), options)
    options = replace(
        options, convergence_translation_m=float(np.linalg.norm(probe.transform[:3, 3])),
    )
    prepared = CudaOdometryTarget(target, options)
    gpu = prepared.register(source, np.eye(4))
    cpu = register_point_to_plane(source, target, np.eye(4), options)
    _same(cpu, gpu)
    np.testing.assert_array_equal(cpu.transform, gpu.transform)
    assert prepared.statistics()["scalar_registration_repairs"] == 1


def test_cuda_thin_neighborhoods_keep_native_normal_directions() -> None:
    rng = np.random.default_rng(36)
    points = np.column_stack((np.linspace(-10, 10, 200),
                              rng.uniform(-1e-5, 1e-5, 200),
                              rng.uniform(-1e-8, 1e-8, 200)))
    options = ScanOdometryOptions()
    native, planar = estimate_normals(points, cKDTree(points), options)
    target = CudaOdometryTarget(points, options)
    np.testing.assert_array_equal(cp.asnumpy(target.normals), native)
    np.testing.assert_array_equal(cp.asnumpy(target.planar), planar)
    assert target.statistics()["scalar_normal_repairs"] == len(points)


def test_cuda_registration_computes_only_hit_normals_and_reuses_them() -> None:
    points = _scene()
    source = points[:400]
    options = ScanOdometryOptions()
    target = CudaOdometryTarget(points, options)
    assert target.statistics()["gpu_normal_rows"] == 0
    native = register_point_to_plane(source, points, np.eye(4), options)
    _same(native, target.register(source, np.eye(4)))
    first = target.statistics().copy()
    assert 0 < first["gpu_normal_rows"] < len(points)
    assert first["gpu_system_refresh_calls"] > 0
    _same(native, target.register(source, np.eye(4)))
    assert target.statistics()["gpu_normal_rows"] == first["gpu_normal_rows"]
    assert target.statistics()["gpu_system_refresh_calls"] == first["gpu_system_refresh_calls"]


def test_cuda_out_of_gate_points_do_not_compute_normals() -> None:
    target_points = _scene()
    source = target_points[:400] + 100
    options = ScanOdometryOptions()
    target = CudaOdometryTarget(target_points, options)
    native = register_point_to_plane(source, target_points, np.eye(4), options)
    _same(native, target.register(source, np.eye(4)))
    assert target.statistics()["gpu_normal_rows"] == 0
    assert target.statistics()["gpu_system_refresh_calls"] == 0


def test_cuda_motion_grid_returns_identical_native_motion() -> None:
    points = _scene()
    source = points - np.array([0.12, -0.08, 0.03])
    options = ScanOdometryOptions(motion_translation_quantum_m=1e-6,
                                  motion_rotation_quantum_rad=1e-7)
    target = CudaOdometryTarget(points, options)
    native = register_point_to_plane(source, points, np.eye(4), options)
    actual = target.register(source, np.eye(4))
    _same(native, actual)
    np.testing.assert_array_equal(native.transform, actual.transform)


def test_cuda_repairs_an_ambiguous_motion_grid_boundary() -> None:
    points = _scene()
    source = points - np.array([0.12, 0, 0])
    options = ScanOdometryOptions(max_iterations=1)
    probe = register_point_to_plane(source, points, np.eye(4), options)
    options = replace(options, motion_translation_quantum_m=2 * abs(probe.transform[0, 3]))
    target = CudaOdometryTarget(points, options)
    native = register_point_to_plane(source, points, np.eye(4), options)
    actual = target.register(source, np.eye(4))
    _same(native, actual)
    np.testing.assert_array_equal(native.transform, actual.transform)
    assert target.statistics()["scalar_motion_repairs"] == 1


@pytest.mark.parametrize("scale", [1e-6, 1., 1e6])
def test_fused_normals_keep_native_planes_and_directions_without_library_eigh(
    monkeypatch: pytest.MonkeyPatch, scale: float,
) -> None:
    rotation = Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
    points = (_scene() @ rotation.T) * scale
    options = ScanOdometryOptions()
    expected, planar = estimate_normals(points, cKDTree(points), options)
    target = CudaOdometryTarget(points, options)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("normal preparation invoked the batched library eigensolver")

    monkeypatch.setattr(cp.linalg, "eigh", forbidden)
    actual = cp.asnumpy(target.normals)
    np.testing.assert_array_equal(cp.asnumpy(target.planar), planar)
    np.testing.assert_allclose(np.abs(np.sum(actual * expected, axis=1)), 1., atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1., atol=1e-14)
    assert target.statistics()["gpu_normal_kernel_calls"] > 0
    assert target.statistics()["scalar_normal_repairs"] < len(points)
    first = target.statistics().copy()
    target._ensure_all()
    assert target.statistics() == first


def test_completing_partially_cached_normals_does_not_recompute_rows() -> None:
    points = _scene()
    target = CudaOdometryTarget(points, ScanOdometryOptions())
    target._ensure(cp.asarray([0, 0, 1, 10, 10]))
    assert target.statistics()["gpu_normal_rows"] == 3
    target._ensure_all()
    assert target.statistics()["gpu_normal_rows"] == len(points)
    first = target.statistics().copy()
    target._ensure(cp.asarray([0, 0, 5]))
    assert target.statistics() == first


def test_complete_normal_preparation_crosses_chunk_boundaries() -> None:
    points = np.random.default_rng(67).uniform(-5, 5, (66001, 3))
    points[:, 0] = 4
    target = CudaOdometryTarget(points, ScanOdometryOptions())
    normals = cp.asnumpy(target.normals)
    np.testing.assert_allclose(np.abs(normals[:, 0]), 1., atol=1e-12)
    np.testing.assert_allclose(normals[:, 1:], 0., atol=1e-12)
    assert cp.asnumpy(target.planar).all()
    assert target.statistics()["gpu_normal_rows"] == len(points)
    assert target.statistics()["gpu_normal_kernel_calls"] == 2


@pytest.mark.parametrize("low", [.2 - 1e-9, .2, .2 + 1e-9])
def test_fused_normal_classification_repairs_repeated_upper_root_boundaries(low: float) -> None:
    # Six symmetric axis points have diagonal covariance (low, 1, 1).
    points = np.concatenate((np.diag(np.sqrt([3 * low, 3., 3.])),
                             -np.diag(np.sqrt([3 * low, 3., 3.]))))
    target = CudaOdometryTarget(points, ScanOdometryOptions(normal_neighbours=6))
    rows = cp.asarray([0], dtype=cp.int64)
    indexes = cp.asarray([[0, 1, 2, 3, 4, 5, 6]], dtype=cp.int64)
    # Distinct boundary distances isolate numerical classification from tie repair.
    distances = cp.asarray([[1., 2., 3., 4., 5., 6., 7.]])
    flags = cp.empty(1, dtype=cp.uint8)
    target.normal_kernel((1,), (128,), (
        target.tree.data, indexes, distances, rows, np.int32(1), np.int32(6), np.float64(.2),
        target._normals, target._planar, target.done, flags,
    ))
    assert cp.asnumpy(flags).tolist() == [1]
