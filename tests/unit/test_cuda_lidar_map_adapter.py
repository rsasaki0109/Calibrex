"""Optional real-device geometry, calibration and uncertainty comparisons."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("cupy")
import cupy as cp

from calibrex.evaluation.lidar_lidar_map import LidarLidarMapOptions, evaluate_lidar_lidar_map
from calibrex.solvers.cuda_lidar_map_adapter import CudaMapRuntime, _CudaSample, _CudaTree
from calibrex.solvers.lidar_lidar_map_solver import (
    MapSample,
    MapSampleOptions,
    _jacobian,
    chi_square,
    solve_map_extrinsic,
)


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        count = 0
    if count == 0:
        pytest.skip("requires a CUDA device")


@pytest.mark.parametrize("count", [3, 63, 64, 4097])
@pytest.mark.parametrize("k", [1, 2, 11])
def test_specialized_queries_preserve_nearest_distances_and_neighbors(count: int, k: int) -> None:
    rng = np.random.default_rng(72)
    reference = rng.normal(size=(count, 3))
    queries = rng.normal(size=(149, 3))
    tree = _CudaTree(reference)
    distances, indexes = tree.query(cp.asarray(queries), k)
    expected, positions = tree.cpu.query(queries, k=k)
    actual = cp.asnumpy(distances).reshape(np.shape(expected))
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-15)
    assert cp.asnumpy(indexes).reshape(np.shape(positions)).tolist() == positions.tolist()


def _samples() -> list[MapSample]:
    rng = np.random.default_rng(63)
    points = []
    for axis in range(3):
        face = rng.uniform(-3, 3, (1500, 3))
        face[:, axis] = axis + 0.2
        points.append(face)
    world = np.concatenate(points)
    return [MapSample(float(i), i, np.eye(4), world.copy(), world.copy()) for i in range(3)]


@pytest.mark.parametrize("moving_pose", [False, True])
def test_full_solve_covariance_and_controls_preserve_cpu_outcomes(moving_pose: bool) -> None:
    initial = np.eye(4)
    initial[:3, 3] = [0.08, -0.07, 0.05]
    options = MapSampleOptions(iterations=12)
    cpu_samples, gpu_samples = _samples(), _samples()
    if moving_pose:
        for samples in (cpu_samples, gpu_samples):
            for index, sample in enumerate(samples):
                sample.reference_pose[:3, :3] = Rotation.from_rotvec(
                    [0.05 * index, -0.1 * index, 0.15 * index],
                ).as_matrix()
                sample.reference_pose[:3, 3] = [0.3 * index, 0.2 * index, 0.1 * index]
                sample.target_points = (
                    sample.target_points - sample.reference_pose[:3, 3]
                ) @ sample.reference_pose[:3, :3]
    cpu = solve_map_extrinsic(cpu_samples, initial, options)
    gpu = solve_map_extrinsic(gpu_samples, initial, options, backend="cuda")
    assert cpu.status == gpu.status and cpu.iterations == gpu.iterations
    assert cpu.correspondences == gpu.correspondences
    assert gpu.execution_backend.startswith("calibrex.cuda_lidar_map/v0.1;")
    np.testing.assert_allclose(gpu.transform, cpu.transform, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(gpu.covariance, cpu.covariance, atol=1e-13, rtol=1e-5)
    assert gpu.sigma_m == pytest.approx(cpu.sigma_m, rel=1e-6, abs=1e-10)
    for axis in range(3):
        moved = np.eye(4)
        moved[axis, 3] = 0.05
        baseline = chi_square(cpu_samples, moved, 0.01, options)
        candidate = chi_square(gpu_samples, moved, 0.01, options, backend="cuda")
        assert candidate[2] == baseline[2]
        assert candidate[:2] == pytest.approx(baseline[:2], rel=1e-6, abs=1e-9)


def test_tied_neighborhoods_gate_boundaries_and_snapshots_keep_native_selection() -> None:
    world = np.asarray([[float(x), float(y), 0.0] for x in range(-4, 5) for y in range(-4, 5)])
    points = np.asarray([[0.5, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]])
    sample = MapSample(0., 0, np.eye(4), points, world)
    for gate in (0.5, math.nextafter(0.5, 0.), math.nextafter(0.5, math.inf)):
        options = MapSampleOptions(max_correspondence_m=gate)
        runtime = CudaMapRuntime([sample], options)
        expected, _, _ = sample.residuals(np.eye(4), options)
        actual = runtime.residuals(sample, np.eye(4))
        np.testing.assert_allclose(actual, expected, atol=1e-13)
        sample.target_points[:] += 10
        np.testing.assert_allclose(runtime.residuals(sample, np.eye(4)), actual, atol=1e-13)
        sample.target_points[:] -= 10
    with pytest.raises(ValueError, match="2 through 16"):
        CudaMapRuntime([sample], replace(options, normal_neighbours=17))


def test_invalid_coordinates_custom_methods_and_snapshot_membership_are_explicit() -> None:
    sample = _samples()[0]
    sample.target_points[0, 0] = math.nan
    with pytest.raises(ValueError, match="finite bounded"):
        CudaMapRuntime([sample], MapSampleOptions())
    sample = _samples()[0]
    runtime = CudaMapRuntime([sample], MapSampleOptions())
    with pytest.raises(ValueError, match="does not belong"):
        runtime.residuals(_samples()[0], np.eye(4))
    sample.residuals = lambda *_: None
    with pytest.raises(ValueError, match="native MapSample"):
        CudaMapRuntime([sample], MapSampleOptions())


def test_complete_blocked_evaluation_retains_quality_and_jackknife_policy() -> None:
    initial = np.eye(4)
    initial[:3, 3] = [0.08, -0.07, 0.05]
    original = _samples()[0]
    def samples():
        return [MapSample(float(i), i, np.eye(4), original.target_points.copy(),
                          original.map_points.copy()) for i in range(9)]
    options = LidarLidarMapOptions(jackknife_groups=3)
    reference = evaluate_lidar_lidar_map(samples(), initial, options)
    result = evaluate_lidar_lidar_map(samples(), initial, options, backend="cuda")
    assert result.policy_status == reference.policy_status
    assert result.policy_reasons == reference.policy_reasons
    assert result.jackknife_fits == reference.jackknife_fits == 3
    assert result.train_blocks == reference.train_blocks
    assert result.holdout_blocks == reference.holdout_blocks
    assert result.holdout_correspondences == reference.holdout_correspondences
    assert result.holdout_median_m == pytest.approx(reference.holdout_median_m, abs=1e-10)
    np.testing.assert_allclose(result.result.transform, reference.result.transform, atol=2e-8)
    for actual, expected in zip(result.records, reference.records, strict=True):
        assert actual.name == expected.name and actual.status == expected.status
        assert actual.known_bad_control.detected == expected.known_bad_control.detected
        assert actual.std_reported == pytest.approx(expected.std_reported, rel=1e-5, abs=1e-10)


@pytest.mark.parametrize("k", [2, 3, 10, 16])
@pytest.mark.parametrize("scale", [1e-9, 1., 1e9])
def test_lazy_normals_match_native_geometry_without_library_gpu_eigh(
    k: int, scale: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = np.random.default_rng(76).normal(size=(137, 3)) * scale
    sample = MapSample(0., 0, np.eye(4), world.copy(), world.copy())
    options = MapSampleOptions(normal_neighbours=k)
    sample.prepare(options)
    native_normals, native_planar = sample._normals.copy(), sample._planar.copy()
    sample._normals = sample._planar = sample._tree = None

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("map normals called the batched GPU eigensolver")

    monkeypatch.setattr(cp.linalg, "eigh", forbidden)
    prepared = _CudaSample(sample, options)
    assert prepared.normal_rows == 0
    prepared._ensure(cp.asarray([0, 0, 1, 10, 10], dtype=cp.int64))
    assert prepared.normal_rows == 3
    prepared._ensure(cp.arange(len(world)))
    assert prepared.normal_rows == len(world)
    assert prepared.normal_kernel_calls == 2
    actual = cp.asnumpy(prepared.normals)
    signs = np.sign(np.sum(actual * native_normals, axis=1))
    np.testing.assert_allclose(actual * signs[:, None], native_normals, atol=2e-9, rtol=0)
    np.testing.assert_array_equal(cp.asnumpy(prepared.planar), native_planar)
    first = prepared.tree.query_rows
    prepared._ensure(cp.arange(len(world)))
    assert prepared.tree.query_rows == first


def test_only_reached_normals_are_computed_and_preexisting_native_policy_is_retained() -> None:
    world = np.random.default_rng(77).normal(size=(400, 3))
    sample = MapSample(0., 0, np.eye(4), world[:3].copy(), world.copy())
    runtime = CudaMapRuntime([sample], MapSampleOptions())
    runtime.residuals(sample, np.eye(4))
    assert runtime.statistics()["gpu_normal_rows"] == 3
    first = runtime.statistics()
    runtime.residuals(sample, np.eye(4))
    assert runtime.statistics()["gpu_normal_rows"] == first["gpu_normal_rows"]
    sample.prepare(MapSampleOptions(normal_neighbours=3, max_planarity_ratio=1.))
    prepared = _CudaSample(sample, MapSampleOptions())
    prepared._ensure(cp.arange(len(world)))
    np.testing.assert_array_equal(cp.asnumpy(prepared.normals), sample._normals)
    np.testing.assert_array_equal(cp.asnumpy(prepared.planar), sample._planar)
    assert prepared.normal_rows == 0


@pytest.mark.parametrize("ratio", [.2 - 1e-9, .2, .2 + 1e-9])
def test_planarity_boundaries_and_degenerate_normals_use_native_repairs(ratio: float) -> None:
    world = np.concatenate((np.diag(np.sqrt([.6, 3., 3.])),
                            -np.diag(np.sqrt([.6, 3., 3.]))))
    sample = MapSample(0., 0, np.eye(4), world.copy(), world.copy())
    options = MapSampleOptions(normal_neighbours=6, max_planarity_ratio=ratio)
    sample.prepare(options)
    normals, planar = sample._normals.copy(), sample._planar.copy()
    sample._normals = sample._planar = sample._tree = None
    prepared = _CudaSample(sample, options)
    prepared._ensure(cp.arange(len(world)))
    assert prepared.normal_repairs == len(world)
    np.testing.assert_array_equal(cp.asnumpy(prepared.normals), normals)
    np.testing.assert_array_equal(cp.asnumpy(prepared.planar), planar)


def test_native_repair_failure_does_not_publish_unrepaired_cached_normals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = np.zeros((3, 3))
    sample = MapSample(0., 0, np.eye(4), world.copy(), world.copy())
    prepared = _CudaSample(sample, MapSampleOptions())
    original = np.linalg.eigh

    def failure(*args: object, **kwargs: object) -> None:
        raise RuntimeError("native repair failed")

    monkeypatch.setattr(np.linalg, "eigh", failure)
    with pytest.raises(RuntimeError, match="native repair failed"):
        prepared._ensure(cp.arange(3))
    assert not cp.asnumpy(prepared.done).any()
    assert prepared.normal_rows == 0
    monkeypatch.setattr(np.linalg, "eigh", original)
    prepared._ensure(cp.arange(3))
    assert cp.asnumpy(prepared.done).all()
    assert prepared.normal_repairs == prepared.normal_rows == 3


def test_lazy_normal_completion_crosses_chunks_and_keeps_map_planarity_without_a_floor() -> None:
    world = np.random.default_rng(78).uniform(-5, 5, (66001, 3))
    world[:, 0] = 4.
    sample = MapSample(0., 0, np.eye(4), world[:1].copy(), world)
    prepared = _CudaSample(sample, MapSampleOptions())
    prepared._ensure(cp.arange(len(world)))
    assert prepared.normal_rows == len(world)
    assert prepared.normal_kernel_calls == 2
    np.testing.assert_allclose(np.abs(cp.asnumpy(prepared.normals)[:, 0]), 1., atol=1e-12)
    assert cp.asnumpy(prepared.planar).all()


def test_fused_projection_residual_and_weighted_jacobian_preserve_native_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples()
    rng = np.random.default_rng(79)
    for index, sample in enumerate(samples):
        pose = sample.reference_pose
        pose[:3, :3] = Rotation.from_rotvec([.02 * index, -.03 * index, .04 * index]).as_matrix()
        pose[:3, 3] = [.1 * index, -.2 * index, .05 * index]
        sample.target_points = (sample.map_points - pose[:3, 3]) @ pose[:3, :3]
        sample.target_points += rng.normal(0., .01, sample.target_points.shape)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec([.01, -.02, .03]).as_matrix()
    extrinsic[:3, 3] = [.02, -.03, .04]
    options = MapSampleOptions()
    hessian, gradient, residuals = np.zeros((6, 6)), np.zeros(6), []
    for sample in samples:
        residual, normals, points = sample.residuals(extrinsic, options)
        jacobian = _jacobian(sample.reference_pose, extrinsic, normals, points)
        weight = np.minimum(1., options.huber_m / np.maximum(np.abs(residual), 1e-12))
        hessian += jacobian.T @ (jacobian * weight[:, None])
        gradient += jacobian.T @ (weight * residual)
        residuals.append(residual)
        sample._tree = sample._normals = sample._planar = None
    original = _CudaTree.query

    def query(tree: _CudaTree, points: object, k: int) -> object:
        assert k != 2, "registration used a separate query dispatch"
        return original(tree, points, k)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("registration used the unfused Jacobian assembly")

    monkeypatch.setattr(_CudaTree, "query", query)
    monkeypatch.setattr(cp, "cross", forbidden)
    monkeypatch.setattr(cp, "column_stack", forbidden)
    runtime = CudaMapRuntime(samples, options)
    actual_hessian, actual_gradient, actual_residuals = runtime.normal_equations(samples, extrinsic)
    np.testing.assert_allclose(actual_hessian, hessian, rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(np.abs(actual_residuals), np.abs(np.concatenate(residuals)),
                               rtol=1e-9, atol=1e-10)
    statistics = runtime.statistics()
    assert statistics["gpu_projection_query_calls"] == 3
    assert statistics["gpu_residual_kernel_calls"] == 3
    assert statistics["gpu_jacobian_kernel_calls"] == 3
    assert statistics["gpu_projection_query_rows"] + statistics["gpu_normal_rows"] == (
        statistics["gpu_query_rows"]
    )


@pytest.mark.parametrize("empty", [False, True])
def test_fused_registration_handles_empty_and_completely_gated_targets(empty: bool) -> None:
    sample = _samples()[0]
    sample.target_points = np.empty((0, 3)) if empty else sample.target_points + 100.
    runtime = CudaMapRuntime([sample], MapSampleOptions())
    assert runtime.residuals(sample, np.eye(4)).shape == (0,)
    hessian, gradient, residual = runtime.normal_equations([sample], np.eye(4))
    np.testing.assert_array_equal(hessian, np.zeros((6, 6)))
    np.testing.assert_array_equal(gradient, np.zeros(6))
    assert residual.shape == (0,)
    assert runtime.statistics()["gpu_normal_rows"] == 0
    assert runtime.statistics()["gpu_residual_kernel_calls"] == 0
    assert runtime.statistics()["gpu_jacobian_kernel_calls"] == 0


@pytest.mark.parametrize("gate", [.6, math.nextafter(.6, 0.), math.nextafter(.6, math.inf)])
def test_fused_native_tie_repairs_keep_selected_normals_gate_and_source_order(gate: float) -> None:
    world = np.asarray([[float(x), float(y), 0.] for x in range(-4, 5) for y in range(-4, 5)])
    points = np.asarray([[.5, 0., .125], [3., 3., .25], [20., 20., 0.],
                         [.5, 0., -.125], [0., 0., .6]])
    sample = MapSample(0., 0, np.eye(4), points, world)
    options = MapSampleOptions(max_correspondence_m=gate)
    sample.prepare(options)
    # Distinct caller-captured normals expose a wrong member at an exact tie;
    # coplanar normals would conceal it in the residual comparison.
    sample._normals[:] = 0.
    for i, point in enumerate(world):
        sample._normals[i, int(point[0]) % 3] = 1.
    sample._planar[:] = True
    expected, _, _ = sample.residuals(np.eye(4), options)
    runtime = CudaMapRuntime([sample], options)
    actual = runtime.residuals(sample, np.eye(4))
    np.testing.assert_array_equal(actual, expected)
    assert runtime.statistics()["scalar_query_repairs"] >= 2
    assert runtime.statistics()["gpu_normal_rows"] == 0
