from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.cli.main import main
from calibrex.core.validation import validate_file
from calibrex.data.imu_trajectory import (
    parse_imu_csv,
    parse_tum_trajectory,
    split_trajectory_windows,
)
from calibrex.data.imu_trajectory_synthetic import (
    SyntheticImuTrajectory,
    synthetic_imu_trajectory,
)
from calibrex.data.livox_ros2 import STANDARD_GRAVITY_MPS2
from calibrex.evaluation.imu_trajectory import (
    ImuTrajectoryCalibration,
    calibration_summary,
    run_imu_trajectory_calibration,
)


@functools.cache
def _data() -> SyntheticImuTrajectory:
    return synthetic_imu_trajectory(duration_s=90.0, seed=2)


@functools.cache
def _calibration() -> ImuTrajectoryCalibration:
    data = _data()
    return run_imu_trajectory_calibration(data.imu_csv, data.trajectory_tum)


def test_parsers_accept_headers_comments_whitespace_and_g() -> None:
    imu = parse_imu_csv(
        "t gx gy gz ax ay az\n2 0 0 0 0 0 1\n1, 0.1, 0, 0, 0, 0, 1\n", acceleration_unit="g"
    )
    _, poses = parse_tum_trajectory("# header\n1 0 0 0 0 0 0 2\n2 1 2 3 0 0 0 1\n")

    assert np.allclose(imu.times_s, [1.0, 2.0])
    assert np.allclose(imu.accel_mps2[:, 2], STANDARD_GRAVITY_MPS2)
    assert np.allclose(poses[0, :3, :3], np.eye(3))  # quaternion normalised
    assert np.allclose(poses[1, :3, 3], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="expected 8 columns"):
        parse_tum_trajectory("1 0 0 0 0 0 0 1\n2 0 0 0 0 0 1\n")


def test_windows_never_span_a_tracking_gap() -> None:
    times = np.concatenate([np.arange(0.0, 25.0, 0.1), 40.0 + np.arange(0.0, 12.0, 0.1)])
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    windows = split_trajectory_windows(times, poses)

    assert [window.block for window in windows] == [0, 0, 0, 1, 1]
    assert all(np.max(np.diff(window.times_s)) < 0.5 for window in windows)


def test_synthetic_trajectory_recovers_the_extrinsic_and_passes(tmp_path: Path) -> None:
    data = _data()
    calibration = _calibration()
    rotation = calibration.rotation
    translation = calibration.translation

    assert rotation.policy_status == "pass", rotation.policy_reasons
    assert translation is not None and translation.policy_status == "pass"
    assert rotation.rotation_quat_xyzw is not None
    error = Rotation.from_matrix(data.rotation_lidar_imu).inv() * Rotation.from_quat(
        rotation.rotation_quat_xyzw
    )
    assert np.degrees(error.magnitude()) < 0.05
    assert abs(rotation.time_offset_s - data.time_offset_s) < 0.001
    assert translation.translation_m is not None
    assert np.allclose(translation.translation_m, data.translation_lidar_imu_m, atol=0.004)
    summary = calibration_summary(calibration)
    json.dumps(summary)
    assert np.allclose(np.array(summary["T_lidar_imu"])[:3, 3], translation.translation_m)

    rotation.save(tmp_path / "rotation.yaml")
    translation.save(tmp_path / "translation.yaml")
    assert validate_file(tmp_path / "rotation.yaml").kind == "imu-lidar-rotation"
    assert validate_file(tmp_path / "translation.yaml").kind == "imu-lidar-translation"


def test_cli_writes_both_artifacts(tmp_path: Path) -> None:
    data = _data()
    imu, trajectory = tmp_path / "imu.csv", tmp_path / "trajectory.txt"
    imu.write_text(data.imu_csv, encoding="utf-8")
    trajectory.write_text(data.trajectory_tum, encoding="utf-8")

    exit_code = main(
        [
            "imu-lidar",
            "trajectory",
            "--imu",
            str(imu),
            "--trajectory",
            str(trajectory),
            "--output-rotation",
            str(tmp_path / "rotation.yaml"),
            "--output-translation",
            str(tmp_path / "translation.yaml"),
        ]
    )

    assert exit_code == 0
    assert validate_file(tmp_path / "translation.yaml").valid


def test_too_short_a_trajectory_is_refused() -> None:
    data = synthetic_imu_trajectory(duration_s=20.0)
    with pytest.raises(ValueError, match="at least 3 are needed"):
        run_imu_trajectory_calibration(data.imu_csv, data.trajectory_tum)
