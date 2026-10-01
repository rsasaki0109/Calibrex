from __future__ import annotations

import numpy as np
from tests.unit.test_imu_lidar_rotation import EPOCH, TRUE_ROTATION, synthetic

from calibrex.evaluation.camera_imu_rotation import (
    CAMERA_MIN_WINDOW_ROTATION_DEG,
    CAMERA_OBSERVABLE_ROTATION_STD_DEG,
    CameraImuRunOptions,
)
from calibrex.evaluation.imu_lidar_rotation import (
    ImuLidarRunOptions,
    evaluate_imu_lidar_rotation,
    window_rotation_deg,
)
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.imu_lidar_rotation_solver import RotationOptions


def _static_window(index: int) -> OdometryWindow:
    times = EPOCH + 0.5 + 10.0 * index + np.arange(0.0, 9.9, 0.1)
    return OdometryWindow(f"static{index}", index, times, np.stack([np.eye(4)] * len(times)))


def test_window_rotation_is_the_accumulated_step_angle() -> None:
    _, windows = synthetic(planar=False)

    assert window_rotation_deg(windows[0]) > 10.0
    assert window_rotation_deg(_static_window(0)) == 0.0


def test_static_window_is_excluded_and_does_not_move_the_fit() -> None:
    gyro, windows = synthetic(planar=False, seed=3)
    options = ImuLidarRunOptions(min_window_rotation_deg=1.0)
    baseline = evaluate_imu_lidar_rotation(gyro, windows, options, reference_rotation=TRUE_ROTATION)
    padded = evaluate_imu_lidar_rotation(
        gyro, [*windows, _static_window(9)], options, reference_rotation=TRUE_ROTATION
    )

    assert baseline.static_windows_excluded == 0
    assert padded.static_windows_excluded == 1
    assert padded.train_windows == baseline.train_windows
    assert baseline.result.rotation is not None and padded.result.rotation is not None
    assert np.allclose(baseline.result.rotation, padded.result.rotation, atol=1e-12)
    assert [r.std_reported for r in padded.records] == [r.std_reported for r in baseline.records]


def test_default_evaluation_keeps_every_window() -> None:
    gyro, windows = synthetic(planar=False, seed=3)
    result = evaluate_imu_lidar_rotation(gyro, [*windows, _static_window(9)])

    assert result.static_windows_excluded == 0
    assert result.train_windows + result.holdout_windows == len(windows) + 1


def test_imu_lidar_defaults_are_unchanged() -> None:
    options = ImuLidarRunOptions()

    assert options.min_window_rotation_deg == 0.0
    assert options.solver == RotationOptions()
    assert options.solver.observable_rotation_std_deg == 0.1


def test_camera_imu_uses_its_own_thresholds() -> None:
    evaluation = CameraImuRunOptions().evaluation

    assert evaluation.solver.observable_rotation_std_deg == CAMERA_OBSERVABLE_ROTATION_STD_DEG
    assert CAMERA_OBSERVABLE_ROTATION_STD_DEG > 0.1
    assert evaluation.min_window_rotation_deg == CAMERA_MIN_WINDOW_ROTATION_DEG > 0.0
    assert (
        evaluation.solver.observable_time_offset_std_s
        == RotationOptions().observable_time_offset_std_s
    )


def test_threshold_decides_the_axis_status() -> None:
    gyro, windows = synthetic(planar=False, seed=3)
    tight = evaluate_imu_lidar_rotation(
        gyro, windows, ImuLidarRunOptions(solver=RotationOptions(observable_rotation_std_deg=1e-4))
    )
    loose = evaluate_imu_lidar_rotation(
        gyro, windows, ImuLidarRunOptions(solver=RotationOptions(observable_rotation_std_deg=5.0))
    )

    assert all(r.status == "unobservable" for r in tight.records if r.unit == "deg")
    assert all(r.status == "estimated" for r in loose.records if r.unit == "deg")
