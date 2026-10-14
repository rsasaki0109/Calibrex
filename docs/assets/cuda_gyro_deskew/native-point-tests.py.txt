"""Check fused native gyro point rotation against the preserved matrix callback."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.solvers import scan_to_scan_odometry as odometry
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries, _GyroRotationModel


@pytest.mark.parametrize("epoch", [0.0, 1.76e9])
@pytest.mark.parametrize("bias_scale", [0.0, 0.01, 0.5])
def test_fused_points_match_matrix_reference_and_preserve_inputs(
    epoch: float, bias_scale: float
) -> None:
    rng = np.random.default_rng(931)
    times = epoch + np.arange(0, 8.0, 0.005)
    gyro = GyroSeries(times, rng.normal(0, 0.8, (len(times), 3)))
    mounting = Rotation.from_rotvec([0.2, -0.3, 0.1]).as_matrix()
    bias = np.array([0.5, -0.2, 0.3]) * bias_scale
    model = _GyroRotationModel(gyro, mounting, bias, 0.003)
    offsets = rng.uniform(-0.03, 0.1, 217)
    points = rng.normal(0, 10, (217, 4))
    saved_points, saved_offsets = points.copy(), offsets.copy()
    for start in (epoch + 0.1, epoch + 6.7):
        reference = np.einsum("nij,nj->ni", model(start, offsets), points[:, :3])
        assert np.allclose(
            model.rotate_points(start, offsets, points), reference, rtol=2e-12, atol=3e-11
        )
    assert np.array_equal(points, saved_points)
    assert np.array_equal(offsets, saved_offsets)
    # The original callback observes subsequent changes to the caller's bias.
    bias[:] += 0.02
    reference = np.einsum("nij,nj->ni", model(epoch + 0.1, offsets), points[:, :3])
    assert np.allclose(model.rotate_points(epoch + 0.1, offsets, points), reference, atol=3e-11)


@pytest.mark.parametrize("kind", ["extrapolated", "large_bias", "non_rotation", "empty"])
def test_unusual_inputs_use_the_matrix_reference(kind: str) -> None:
    times = np.arange(0, 1, 0.01)
    gyro = GyroSeries(times, np.tile([0.2, 0.1, -0.1], (len(times), 1)))
    mounting = np.eye(3) if kind != "non_rotation" else np.diag([1, 2, 1])
    bias = np.zeros(3) if kind != "large_bias" else np.array([2, 0, 0])
    model = _GyroRotationModel(gyro, mounting, bias, 0)
    offsets = np.array([]) if kind == "empty" else np.array([-0.01, 0, 0.03])
    points = np.ones((len(offsets), 3))
    start = -0.03 if kind == "extrapolated" else 0.5
    reference = np.einsum("nij,nj->ni", model(start, offsets), points)
    assert np.array_equal(model.rotate_points(start, offsets, points), reference)


def test_non_orthogonal_prefix_uses_the_matrix_reference() -> None:
    times = np.arange(0, 1, 0.01)
    gyro = GyroSeries(times, np.tile([0.2, -0.1, 0.3], (len(times), 1)))
    # Do not mutate the shared content-keyed gyro table cache.
    orientations = gyro._orientations.copy()
    orientations[:, 0, 0] += 0.01
    object.__setattr__(gyro, "_orientations", orientations)
    model = _GyroRotationModel(gyro, np.eye(3), np.array([0.1, 0, 0]), 0)
    offsets = np.array([0, 0.03, 0.06])
    points = np.ones((len(offsets), 3))
    reference = np.einsum("nij,nj->ni", model(0.5, offsets), points)
    assert np.array_equal(model.rotate_points(0.5, offsets, points), reference)


@pytest.mark.parametrize("count", [1, 3, 217])
@pytest.mark.parametrize("epoch", [0., 1.76e9])
def test_selected_native_repairs_preserve_complete_batch_bytes(count: int, epoch: float) -> None:
    rng = np.random.default_rng(1224)
    times = epoch + np.arange(0, 2, .005)
    gyro = GyroSeries(times, rng.normal(0, .8, (len(times), 3)))
    model = _GyroRotationModel(gyro, Rotation.from_rotvec([.2, -.3, .1]).as_matrix(),
                               np.array([.01, -.02, .03]), .003)
    points = rng.normal(0, 10, (4097, 4))
    offsets = rng.uniform(-.03, .1, len(points))
    inputs = model._point_inputs(epoch + .7, offsets, points)
    assert inputs is not None
    expected = model.rotate_points(epoch + .7, offsets, points)
    rows = rng.choice(len(points), count, replace=False).astype(np.int64)
    assert inputs.native_points(rows).tobytes() == expected[rows].tobytes()


def test_odometry_reuses_rotated_points_and_preserves_custom_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = np.arange(0, 1, 0.005)
    gyro = GyroSeries(times, np.tile([0.4, -0.2, 0.1], (len(times), 1)))

    class CountingModel(_GyroRotationModel):
        point_calls = 0

        def rotate_points(self, *args: object) -> np.ndarray:
            self.point_calls += 1
            return super().rotate_points(*args)

    native = CountingModel(gyro, np.eye(3), np.array([0.01, -0.01, 0.01]), 0)
    matrix_calls = []

    def custom(start: float, offsets: np.ndarray) -> np.ndarray:
        matrix_calls.append(start)
        return native(start, offsets)

    def unrelated_method(*args: object) -> None:
        pytest.fail("custom callback attributes must not select the internal fast path")

    custom.rotate_points = unrelated_method
    rng = np.random.default_rng(56)
    points = rng.normal(size=(43, 4))
    offsets = np.linspace(0, 0.08, len(points)).astype(np.float32)
    captured: list[np.ndarray] = []

    def register(source: np.ndarray, *args: object, **kwargs: object) -> odometry.ScanRegistration:
        captured.append(source.copy())
        motion = np.eye(4)
        motion[:3, 3] = np.array([0.02, -0.01, 0.03]) * (len(captured) % 3 + 1)
        return odometry.ScanRegistration(motion, True, 1, len(source), 0, 1)

    monkeypatch.setattr(odometry, "register_point_to_plane", register)
    options = odometry.ScanOdometryOptions(voxel_size_m=0.05, min_range_m=0, deskew_iterations=2)
    for model in (native, custom):
        captured.clear()
        estimator = odometry.IncrementalScanOdometry(options, rotation_model=model)
        estimator.add(points, 0.2, offsets)
        estimator.add(points, 0.3, offsets)
        if model is native:
            saved = [array.copy() for array in captured]
        else:
            assert len(captured) == len(saved) == 3
            for actual, expected in zip(captured, saved, strict=True):
                assert np.allclose(actual, expected, rtol=0, atol=3e-14)
    assert native.point_calls == 1
    assert matrix_calls == [0.3]
