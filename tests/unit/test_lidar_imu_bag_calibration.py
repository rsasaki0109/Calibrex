"""End-to-end joint LiDAR-IMU calibration of a synthetic rosbag."""

from __future__ import annotations

import json
import math
from dataclasses import replace
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


def test_cli_joint_calibration_recovers_the_injected_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrex.evaluation import lidar_imu_bag_calibration as runner

    original = runner.lidar_imu_residuals
    calls: list[tuple[str, ...]] = []

    def observed(*args, **kwargs):
        calls.append(kwargs.get("families", ("plane", "gyro", "accel")))
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "lidar_imu_residuals", observed)
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
    assert (
        np.linalg.norm(np.asarray(artifact["t_lidar_imu_translation_m"]) - truth.lever_arm_m) < 0.04
    )
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
    selected = [family for family in calls if len(family) == 1]
    assert len(selected) == len(artifact["controls"]) + 2
    assert set(selected) == {("gyro",), ("accel",)}


def test_band_limit_removes_vibration_without_delaying_the_motion() -> None:
    from calibrex.data.livox_ros2 import ImuSamples
    from calibrex.evaluation.lidar_imu_bag_calibration import _band_limit

    times = np.arange(0.0, 20.0, 0.01)
    times = np.delete(times, 777)  # one dropped sample
    motion = np.sin(2.0 * np.pi * 0.5 * times)
    vibration = 0.8 * np.sin(2.0 * np.pi * 18.0 * times)
    values = np.stack([motion + vibration, motion, -motion], axis=1)
    band = _band_limit(ImuSamples(times, values, 9.81 + values), cutoff_hz=10.0)

    # Resampled at twice the cutoff on a uniform grid from the first sample.
    assert np.allclose(np.diff(band.times_s), 0.05)
    inside = (band.times_s > 1.0) & (band.times_s < 19.0)
    expected = np.sin(2.0 * np.pi * 0.5 * band.times_s[inside])
    # The 18 Hz vibration is gone and the 0.5 Hz motion keeps its phase; only
    # the interpolated dropped sample leaves a short local error.
    away = np.abs(band.times_s[inside] - times[776]) > 0.5
    error = band.gyro_rps[inside, 0] - expected
    assert np.max(np.abs(error[away])) < 0.01
    assert np.max(np.abs(error)) < 0.1
    assert np.max(np.abs(band.gyro_rps[inside, 1] - expected)) < 1.0e-3
    assert np.max(np.abs(band.accel_mps2[inside, 2] - (9.81 - expected))) < 1.0e-3
    with pytest.raises(ValueError, match="Nyquist"):
        _band_limit(ImuSamples(times, values, values), cutoff_hz=60.0)


def test_synthetic_bag_vibration_reaches_the_gyro_at_its_frequency(tmp_path: Path) -> None:
    calm = SyntheticLidarImuBagSpec(duration_s=2.0, azimuth_steps=36)
    shaking = replace(calm, vibration_hz=18.0, vibration_rate_rad_s=0.25)
    gyro = []
    for name, spec in (("calm", calm), ("shaking", shaking)):
        bag = write_synthetic_lidar_imu_bag(tmp_path / name, spec)
        profile = LivoxStreamProfile(
            "synthetic", bag.lidar_topic, bag.imu_topic, "offset_time", "offset_s", "mps2"
        )
        gyro.append(load_livox_imu(bag.path, profile).gyro_rps)
    # Same seed, same noise: the difference is the vibration of roll and pitch.
    vibration = gyro[1] - gyro[0]
    assert np.allclose(
        np.sqrt(np.mean(vibration[:, :2] ** 2, axis=0)), 0.25 / np.sqrt(2.0), rtol=0.1
    )
    assert np.sqrt(np.mean(vibration[:, 2] ** 2)) < 0.03
    spectrum = np.abs(np.fft.rfft(vibration[:, 0]))
    frequencies = np.fft.rfftfreq(vibration.shape[0], 1.0 / calm.imu_rate_hz)
    assert frequencies[np.argmax(spectrum)] == pytest.approx(18.0, abs=0.5)


def test_imu_lowpass_recovers_the_calibration_through_rig_vibration(tmp_path: Path) -> None:
    # An 18 Hz roll/pitch vibration of 0.25 rad/s, as on a wheeled platform,
    # buries the gyro and accelerometer; low-passing the IMU at 10 Hz (knots
    # widened to 0.3 s) leaves the band the calibration lives in.
    spec = replace(SPEC, vibration_hz=18.0, vibration_rate_rad_s=0.25)
    bag = write_synthetic_lidar_imu_bag(tmp_path / "bag", spec)
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
            "--imu-lowpass-hz",
            "10",
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    assert validate_file(output, kind="lidar-imu-bag-calibration").valid
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["knot_spacing_s"] == pytest.approx(0.3)
    assert artifact["options"]["imu_lowpass_hz"] == 10.0
    truth = spec.truth
    rotation = Rotation.from_quat(artifact["t_lidar_imu_rotation_quat_xyzw"])
    error = rotation * Rotation.from_rotvec(truth.imu_rotation_vector_rad).inv()
    # Four rounds on 8 s leave about 0.27 deg here (a 12 s bag with eight rounds
    # reaches about 0.06 deg); unfiltered, a 12 s vibrating bag reads INCONCLUSIVE.
    assert math.degrees(error.magnitude()) < 0.4
    assert (
        np.linalg.norm(np.asarray(artifact["t_lidar_imu_translation_m"]) - truth.lever_arm_m) < 0.03
    )
    assert artifact["imu_clock_offset_s"] == pytest.approx(truth.clock_offset_s, abs=1.5e-3)
    # The held-out IMU samples are predicted at the noise level, and every
    # signed control is detected.
    assert max(artifact["holdout_whitened_rms"].values()) < 1.5
    assert all(control["detected"] for control in artifact["controls"])
