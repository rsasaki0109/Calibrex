"""End-to-end joint LiDAR-IMU calibration of a synthetic rosbag."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.cli.main import main
from calibrex.core.validation import validate_file
from calibrex.data.livox_ros2 import LivoxStreamProfile, iter_livox_points, load_livox_imu
from calibrex.data.synthetic_lidar_imu_bag import (
    SyntheticLidarImuBagSpec,
    write_synthetic_lidar_imu_bag,
)

# A short bag keeps the test near a minute: four map rounds leave the rotation
# within about 0.15 deg here, while a 12 s bag converges to about 0.03 deg.
SPEC = SyntheticLidarImuBagSpec(duration_s=8.0)


def test_synthetic_bag_reads_back_with_point_times_and_clock_offset(tmp_path: Path) -> None:
    spec = SyntheticLidarImuBagSpec(duration_s=1.0, azimuth_steps=36)
    bag = write_synthetic_lidar_imu_bag(tmp_path / "bag", spec)
    profile = LivoxStreamProfile(
        "synthetic", bag.lidar_topic, bag.imu_topic, "offset_time", "offset_s", "mps2"
    )
    scans = list(iter_livox_points(bag.path, profile))
    assert len(scans) == 10
    header, points, offsets = scans[0]
    assert points.shape == (36 * 16, 3)
    assert offsets is not None
    assert float(offsets.min()) == pytest.approx(0.0)
    assert float(offsets.max()) < 0.1
    # Every ray hits a wall of the 12 m x 10 m x 4 m room.
    ranges = np.linalg.norm(points, axis=1)
    assert float(ranges.min()) > 1.0 and float(ranges.max()) < 15.0
    imu = load_livox_imu(bag.path, profile)
    assert imu.times_s.size == 200
    assert imu.times_s[0] - header == pytest.approx(spec.truth.clock_offset_s, abs=1.0e-6)
    # At rest-like motion the specific force is dominated by gravity.
    assert float(np.median(np.linalg.norm(imu.accel_mps2, axis=1))) == pytest.approx(9.81, abs=1.0)


def test_cli_joint_calibration_recovers_the_injected_calibration(tmp_path: Path) -> None:
    bag = write_synthetic_lidar_imu_bag(tmp_path / "bag", SPEC)
    output = tmp_path / "calibration.json"
    exit_code = main(
        [
            "imu-lidar",
            "joint",
            str(bag.path),
            "--lidar-topic",
            bag.lidar_topic,
            "--imu-topic",
            bag.imu_topic,
            "--dataset-family",
            "synthetic",
            "--dataset-license",
            "synthetic",
            "--max-rounds",
            "4",
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    assert validate_file(output, kind="lidar-imu-bag-calibration").valid
    artifact = json.loads(output.read_text(encoding="utf-8"))
    truth = SPEC.truth
    rotation = Rotation.from_quat(artifact["t_lidar_imu_rotation_quat_xyzw"])
    error = rotation * Rotation.from_rotvec(truth.imu_rotation_vector_rad).inv()
    # The angular-rate starting point and the fit both land near the truth.
    assert math.degrees(error.magnitude()) < 0.3
    assert artifact["rotation_difference_from_initial_deg"] < 0.5
    assert np.linalg.norm(
        np.asarray(artifact["t_lidar_imu_translation_m"]) - truth.lever_arm_m
    ) < 0.04
    assert artifact["imu_clock_offset_s"] == pytest.approx(truth.clock_offset_s, abs=1.5e-3)
    assert artifact["policy_status"] in {"pass", "warn"}
    rotation_block = next(block for block in artifact["blocks"] if block["name"] == "imu_rotation")
    assert rotation_block["verdict"] == "estimated"
    # The noise model is re-estimated from the residuals: close to the injected noise.
    assert artifact["noise_sigmas"]["gyro"] == pytest.approx(SPEC.gyro_noise_rad_s, rel=0.3)
    assert artifact["noise_sigmas"]["accel"] == pytest.approx(SPEC.accel_noise_m_s2, rel=0.3)
    assert artifact["holdout_whitened_rms"]["gyro"] < 1.5
    assert artifact["provenance"]["point_time_field"] == "offset_time"
    assert artifact["provenance"]["command"][:3] == ["calibrex", "imu-lidar", "joint"]
