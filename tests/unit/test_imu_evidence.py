"""Unit tests for LiDAR-IMU rotation evidence metrics."""

from __future__ import annotations

import math

import pytest

from slac.core.geometry import SE3
from slac.data.odometry_track import OdometryPoseSample, OdometryTrack
from slac.evaluation.imu import (
    ImuEvidenceOptions,
    _build_rotation_intervals,
    _evaluate_imu_evidence,
    _gravity_support,
    _ImuSample,
    _known_bad_fraction_grade,
    _rotation_rate_rad_s,
    _split_train_holdout,
)


def _yaw_pose(yaw_rad: float, timestamp_ns: int) -> OdometryPoseSample:
    half = yaw_rad / 2.0
    return OdometryPoseSample(
        timestamp_ns=timestamp_ns,
        pose=SE3((0.0, 0.0, 0.0), (0.0, 0.0, math.sin(half), math.cos(half))),
    )


def _synthetic_track(
    *,
    yaw_rate_rad_s: float,
    pose_count: int = 20,
    dt_ns: int = 100_000_000,
) -> OdometryTrack:
    samples: list[OdometryPoseSample] = []
    for index in range(pose_count):
        yaw = yaw_rate_rad_s * (index * dt_ns / 1_000_000_000)
        samples.append(_yaw_pose(yaw, index * dt_ns))
    return OdometryTrack(samples)


def _synthetic_imu_samples(
    track: OdometryTrack,
    *,
    omega_body: tuple[float, float, float],
) -> list[_ImuSample]:
    samples: list[_ImuSample] = []
    for pose in track.samples:
        for offset in (0, 25_000_000, 50_000_000, 75_000_000):
            timestamp_ns = pose.timestamp_ns + offset
            samples.append(
                _ImuSample(
                    timestamp_ns=timestamp_ns,
                    angular_velocity=omega_body,
                    linear_acceleration=(0.0, 0.0, 9.80665),
                )
            )
    return samples


def test_rotation_rate_analytic_zero_residual() -> None:
    track = _synthetic_track(yaw_rate_rad_s=0.5, pose_count=12)
    omega_body = (0.0, 0.0, 0.5)
    imu_samples = _synthetic_imu_samples(track, omega_body=omega_body)
    intervals = _build_rotation_intervals(
        track,
        imu_samples,
        SE3.identity(),
        options=ImuEvidenceOptions(),
    )
    train, holdout = _split_train_holdout(intervals, block_stride=5)
    assert train
    assert holdout
    for interval in intervals:
        residual = (
            interval.omega_odometry_rad_s[0] - interval.omega_imu_rad_s[0],
            interval.omega_odometry_rad_s[1] - interval.omega_imu_rad_s[1],
            interval.omega_odometry_rad_s[2] - interval.omega_imu_rad_s[2],
        )
        assert math.hypot(*residual) < 1.0e-6


def test_evaluate_imu_evidence_clean_candidate_passes() -> None:
    track = _synthetic_track(yaw_rate_rad_s=0.6, pose_count=25)
    omega_body = (0.0, 0.0, 0.6)
    imu_samples = _synthetic_imu_samples(track, omega_body=omega_body)
    payload = _evaluate_imu_evidence(
        odometry_track=track,
        imu_samples=imu_samples,
        r_root_imu=SE3.identity(),
        options=ImuEvidenceOptions(holdout_block_stride=5),
        imu_sensor_name="os1_imu",
        root_name="velodyne_vlp16",
        imu_topic="/imu",
        odometry_topic="/odom",
    )
    holdout_rmse = payload.metrics["lidar_imu_holdout_rotation_rate_rmse_dps"].holdout
    assert holdout_rmse is not None
    assert holdout_rmse < 0.1
    assert payload.metrics["lidar_imu_known_bad_detectable_fraction"].grade == "pass"


def test_evaluate_imu_evidence_perturbed_candidate_detects_probes() -> None:
    track = _synthetic_track(yaw_rate_rad_s=0.6, pose_count=25)
    omega_body = (0.0, 0.0, 0.6)
    imu_samples = _synthetic_imu_samples(track, omega_body=omega_body)
    wrong = SE3(
        (0.0, 0.0, 0.0),
        (math.sin(math.radians(10.0)), 0.0, 0.0, math.cos(math.radians(10.0))),
    )
    payload = _evaluate_imu_evidence(
        odometry_track=track,
        imu_samples=imu_samples,
        r_root_imu=wrong,
        options=ImuEvidenceOptions(holdout_block_stride=5),
        imu_sensor_name="os1_imu",
        root_name="velodyne_vlp16",
        imu_topic="/imu",
        odometry_topic="/odom",
    )
    holdout_rmse = payload.metrics["lidar_imu_holdout_rotation_rate_rmse_dps"].holdout
    assert holdout_rmse is not None
    assert holdout_rmse > 1.0
    assert payload.metrics["lidar_imu_holdout_rotation_rate_rmse_dps"].grade == "warn"


def test_evaluate_imu_evidence_low_excitation_inconclusive() -> None:
    track = _synthetic_track(yaw_rate_rad_s=0.01, pose_count=25)
    omega_body = (0.0, 0.0, 0.01)
    imu_samples = _synthetic_imu_samples(track, omega_body=omega_body)
    payload = _evaluate_imu_evidence(
        odometry_track=track,
        imu_samples=imu_samples,
        r_root_imu=SE3.identity(),
        options=ImuEvidenceOptions(min_excitation_p95_dps=5.0),
        imu_sensor_name="os1_imu",
        root_name="velodyne_vlp16",
        imu_topic="/imu",
        odometry_topic="/odom",
    )
    excitation = payload.metrics["lidar_imu_odometry_angular_excitation_p95_dps"].value
    assert excitation is not None
    assert excitation < 5.0


def test_gravity_support_near_zero_for_aligned_accel() -> None:
    track = _synthetic_track(yaw_rate_rad_s=0.0, pose_count=5)
    imu_samples = [
        _ImuSample(
            timestamp_ns=pose.timestamp_ns,
            angular_velocity=(0.0, 0.0, 0.0),
            linear_acceleration=(0.0, 0.0, 9.80665),
        )
        for pose in track.samples
    ]
    angle_deg, grade = _gravity_support(
        odometry_track=track,
        imu_samples=imu_samples,
        r_root_imu=SE3.identity(),
        options=ImuEvidenceOptions(gravity_lowpass_window_s=0.1),
    )
    assert angle_deg is not None
    assert angle_deg < 1.0
    assert grade == "pass"


def test_gravity_support_warns_for_tilted_candidate() -> None:
    track = _synthetic_track(yaw_rate_rad_s=0.0, pose_count=5)
    imu_samples = [
        _ImuSample(
            timestamp_ns=pose.timestamp_ns,
            angular_velocity=(0.0, 0.0, 0.0),
            linear_acceleration=(0.0, 0.0, 9.80665),
        )
        for pose in track.samples
    ]
    tilt = SE3(
        (0.0, 0.0, 0.0),
        (math.sin(math.radians(10.0)), 0.0, 0.0, math.cos(math.radians(10.0))),
    )
    angle_deg, grade = _gravity_support(
        odometry_track=track,
        imu_samples=imu_samples,
        r_root_imu=tilt,
        options=ImuEvidenceOptions(gravity_lowpass_window_s=0.1),
    )
    assert angle_deg is not None
    assert angle_deg > 10.0
    assert grade == "warn"


def test_known_bad_fraction_grade_thresholds() -> None:
    assert _known_bad_fraction_grade(0.6) == "pass"
    assert _known_bad_fraction_grade(0.2) == "warn"
    assert _known_bad_fraction_grade(0.0) == "fail"


def test_rotation_rate_rad_s_matches_scalar_yaw_rate() -> None:
    left = _yaw_pose(0.0, 0).pose
    right = _yaw_pose(0.5, 1).pose
    omega = _rotation_rate_rad_s(left, right, 1.0)
    assert omega[2] == pytest.approx(0.5, abs=1.0e-6)
