"""Tests for the synthetic joint continuous-time LiDAR-IMU recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.continuous_time_lidar_imu_joint import (
    ContinuousTimeLidarImuJointArtifact,
)
from calibrex.core.validation import validate_file
from calibrex.evaluation.continuous_time_lidar_imu_joint import (
    run_synthetic_lidar_imu_joint_recovery,
)


@pytest.fixture(scope="module")
def artifact() -> ContinuousTimeLidarImuJointArtifact:
    return run_synthetic_lidar_imu_joint_recovery(seed=20261010)


def test_joint_recovery_passes_with_provenance_and_valid_schema(
    artifact: ContinuousTimeLidarImuJointArtifact, tmp_path: Path
) -> None:
    output = tmp_path / "joint.yaml"
    artifact.save(output)

    assert validate_file(output, kind="continuous-time-lidar-imu-joint").valid
    assert artifact.policy_status == "pass", artifact.policy_reason
    assert artifact.data_source == "synthetic"
    assert artifact.fit_status == "converged"
    assert artifact.provenance.seed == 20261010
    assert artifact.provenance.generator.endswith("continuous_time_lidar_imu_joint")


def test_every_block_is_estimated_jointly_within_budget(
    artifact: ContinuousTimeLidarImuJointArtifact,
) -> None:
    names = {record.name for record in artifact.parameters}
    assert names == {
        "imu_rotation",
        "gyro_bias",
        "lever_arm",
        "imu_clock_offset",
        "accel_bias",
        "gravity",
    }
    for record in artifact.parameters:
        assert record.within_budget, record
        assert record.max_normalized_error < 4.0, record
        assert all(value > 0.0 for value in record.std), record
    rotation = next(record for record in artifact.parameters if record.name == "imu_rotation")
    # The fit starts from an identity mounting rotation about 4 deg from truth.
    assert rotation.initial == [0.0, 0.0, 0.0]
    assert rotation.error_norm < 0.25 * sum(value * value for value in rotation.truth) ** 0.5


def test_signed_controls_cover_every_axis_in_both_directions(
    artifact: ContinuousTimeLidarImuJointArtifact,
) -> None:
    keys = {(record.parameter, record.axis, record.sign) for record in artifact.controls}
    for name in ("imu_rotation", "gyro_bias", "lever_arm", "accel_bias", "gravity"):
        for axis in range(3):
            assert (name, axis, -1) in keys and (name, axis, 1) in keys
    assert ("imu_clock_offset", None, -1) in keys and ("imu_clock_offset", None, 1) in keys
    assert all(record.detected for record in artifact.controls)
    assert all(
        record.delta_chi2 >= artifact.control_delta_chi2_threshold
        for record in artifact.controls
    )


def test_holdout_stays_near_the_truth_calibration(
    artifact: ContinuousTimeLidarImuJointArtifact,
) -> None:
    limit = artifact.holdout_to_oracle_ratio_limit
    assert artifact.holdout_gyro_rmse <= limit * artifact.oracle_holdout_gyro_rmse
    assert artifact.holdout_accel_rmse_m_s2 <= limit * artifact.oracle_holdout_accel_rmse_m_s2
    assert artifact.holdout_gyro_count >= 1
    assert artifact.holdout_accel_count >= 1


def test_separability_reports_every_pair_and_the_bias_gravity_tradeoff(
    artifact: ContinuousTimeLidarImuJointArtifact,
) -> None:
    pairs = {frozenset((record.block_a, record.block_b)) for record in artifact.separability}
    assert len(pairs) == 15
    bias_gravity = next(
        record
        for record in artifact.separability
        if {record.block_a, record.block_b} == {"accel_bias", "gravity"}
    )
    # Accelerometer bias and gravity tilt trade off unless the attitude varies
    # widely; the artifact must say how strongly instead of hiding it.
    assert bias_gravity.max_abs_correlation > 0.5
    assert bias_gravity.entangled == (
        bias_gravity.max_abs_correlation >= artifact.entangled_correlation_threshold
    )
    if bias_gravity.entangled:
        assert "accel_bias/gravity" in artifact.policy_reason


def test_cli_recover_lidar_imu_joint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "joint.json"
    exit_code = main(
        [
            "trajectory",
            "recover-lidar-imu-joint",
            "--output",
            str(output),
            "--seed",
            "20261010",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["policy_status"] == "pass"
    assert payload["controls_detected"] == payload["controls"]
    assert validate_file(output).valid
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["provenance"]["command"][:3] == [
        "calibrex",
        "trajectory",
        "recover-lidar-imu-joint",
    ]
