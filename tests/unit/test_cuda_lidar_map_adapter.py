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
from calibrex.solvers.cuda_lidar_map_adapter import CudaMapRuntime, _CudaTree
from calibrex.solvers.lidar_lidar_map_solver import (
    MapSample,
    MapSampleOptions,
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
