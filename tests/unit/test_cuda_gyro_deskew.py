"""Quantized device gyro sweeps preserve every native voxel coordinate."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("cupy")
import cupy as cp

from calibrex.solvers.cuda_scan_odometry_adapter import CudaScanPreprocessor
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries, _GyroRotationModel
from calibrex.solvers.scan_to_scan_odometry import (
    IncrementalScanOdometry,
    ScanOdometryOptions,
    preprocess_scan,
)


@pytest.fixture(scope="module", autouse=True)
def require_cuda() -> None:
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        available = 0
    if not available:
        pytest.skip("requires a CUDA device")


def _model(epoch: float = 0.) -> _GyroRotationModel:
    rng = np.random.default_rng(1225)
    times = epoch + np.arange(0, 2, .005)
    gyro = GyroSeries(times, rng.normal(0, .8, (len(times), 3)))
    return _GyroRotationModel(gyro, Rotation.from_rotvec([.2, -.3, .1]).as_matrix(),
                              np.array([.01, -.02, .03]), .003)


@pytest.mark.parametrize("epoch", [0., 1.76e9])
@pytest.mark.parametrize("quantum", [1e-9, 1e-6])
@pytest.mark.parametrize("count", [257, 4097])
def test_device_sweeps_match_native_voxel_bytes_and_reuse_rotations(
    epoch: float, quantum: float, count: int,
) -> None:
    rng = np.random.default_rng(1226)
    points = rng.normal(0, 10, (count, 4))
    offsets = rng.uniform(-.03, .1, count).astype(np.float32)
    saved = points.copy()
    model = _model(epoch)
    pre = CudaScanPreprocessor()
    opts = ScanOdometryOptions(voxel_size_m=.03, geometry_quantum_m=quantum)
    stamp = epoch + .7
    rotated = model.rotate_points(stamp, offsets, points)
    for delta in (0., .02, -.01):
        motion = np.eye(4)
        motion[:3, 3] = [delta, -delta, delta]
        expected = preprocess_scan(rotated + np.outer(offsets.astype(float) / .1,
                                                     motion[:3, 3]), opts)
        actual = pre.gyro_deskew(model, stamp, offsets, points, motion, .1, opts)
        assert actual is not None and actual.tobytes() == expected.tobytes()
    stats = pre.statistics()
    assert stats["gpu_gyro_rotation_calls"] == 1
    assert stats["gpu_gyro_rotation_rows"] == count
    assert stats["gpu_gyro_translation_calls"] == 3
    assert np.array_equal(points, saved)


def test_coordinate_half_grids_are_repaired_and_next_scan_captures_changed_bias() -> None:
    times = np.arange(0, 2, .005)
    gyro = GyroSeries(times, np.zeros((len(times), 3)))
    model = _GyroRotationModel(gyro, np.eye(3), np.zeros(3), 0)
    points = np.array([[5 + (i + .5) * 1e-9, 4., 3.] for i in range(257)])
    offsets = np.linspace(0, .1, len(points))
    opts = ScanOdometryOptions(voxel_size_m=.03, geometry_quantum_m=1e-9)
    pre = CudaScanPreprocessor()
    for bias in (0., .01):
        pre.begin_scan()
        model.bias[:] = bias
        expected = preprocess_scan(model.rotate_points(.7, offsets, points), opts)
        actual = pre.gyro_deskew(model, .7, offsets, points, np.eye(4), .1, opts)
        assert actual is not None and actual.tobytes() == expected.tobytes()
    assert pre.statistics()["gpu_gyro_rotation_calls"] == 2
    assert pre.statistics()["native_gyro_point_repairs"] >= len(points)


@pytest.mark.parametrize("kind", ["zero_grid", "subclass", "empty", "one_point",
                                  "outside", "bias", "prefix", "long_offset"])
def test_unsupported_gyro_sweeps_decline_without_device_work(kind: str) -> None:
    model = _model()
    points = np.ones((3, 3))
    offsets = np.linspace(0, .1, len(points))
    quantum = 0. if kind == "zero_grid" else 1e-9
    if kind == "subclass":
        class Custom(_GyroRotationModel):
            pass
        model = Custom(model.gyro, model.rotation, model.bias, model.time_offset_s)
    elif kind in ("empty", "one_point"):
        points = points[:0 if kind == "empty" else 1]
        offsets = offsets[:len(points)]
    elif kind == "outside":
        offsets += 10
    elif kind == "bias":
        model.bias[:] = 2
    elif kind == "prefix":
        orientations = model.gyro._orientations.copy()
        orientations[:, 0, 0] += .01
        object.__setattr__(model.gyro, "_orientations", orientations)
    elif kind == "long_offset":
        offsets = np.array([0., .1, .3])
    pre = CudaScanPreprocessor()
    assert pre.gyro_deskew(model, .7, offsets, points, np.eye(4), .1,
                           ScanOdometryOptions(geometry_quantum_m=quantum)) is None
    assert pre.statistics()["gpu_gyro_rotation_calls"] == 0


def test_temporal_cuda_chain_matches_native_point_rotation_without_motion_grids() -> None:
    class Native(_GyroRotationModel):
        pass

    model = _model()
    native = Native(model.gyro, model.rotation, model.bias, model.time_offset_s)
    rng = np.random.default_rng(1227)
    parts = []
    for axis in range(3):
        face = rng.uniform(-5, 5, (700, 3))
        face[:, axis] = 4.
        parts.append(face)
    world = np.concatenate(parts)
    offsets = np.linspace(0, .08, len(world))
    options = ScanOdometryOptions(voxel_size_m=.15, geometry_quantum_m=1e-9,
                                  local_map_scans=3, deskew_iterations=2)
    estimators = [IncrementalScanOdometry(options, rotation_model=rotation, backend="cuda")
                  for rotation in (native, model)]
    for stamp in (.3, .4, .5, .6):
        points = world + np.array([0.04 * stamp, -.02 * stamp, .03 * stamp])
        for estimator in estimators:
            estimator.add(points, stamp, offsets)
    old, new = (estimator.result() for estimator in estimators)
    for a, b in zip(old.poses, new.poses, strict=True):
        assert a.tobytes() == b.tobytes()
    for a, b in zip(old.registrations, new.registrations, strict=True):
        assert (a.converged, a.iterations, a.correspondences, a.rmse_m,
                a.hessian_min_eigenvalue) == (b.converged, b.iterations, b.correspondences,
                                             b.rmse_m, b.hessian_min_eigenvalue)
    assert old.execution_statistics["gpu_gyro_rotation_calls"] == 0
    assert new.execution_statistics["gpu_gyro_rotation_calls"] == 3
    assert new.execution_statistics["gpu_gyro_translation_calls"] == 9
