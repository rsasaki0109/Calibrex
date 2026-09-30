from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calibrex.cli.main import main
from calibrex.core.gnss_imu_lever_arm import GnssImuLeverArmArtifact
from calibrex.core.validation import validate_file
from calibrex.evaluation.gnss_imu_lever_arm import compose_gnss_imu_lever_arm

ASSETS = Path(__file__).resolve().parents[2] / "docs/assets"
GNSS = ASSETS / "rtk_slam_gnss_lidar_stadtgarten_seq2.yaml"
ROTATION = ASSETS / "mid360_imu_lidar_rotation_rtk_slam_seq2.yaml"
TRANSLATION = ASSETS / "mid360_imu_lidar_translation_rtk_slam_seq2.yaml"


def test_composition_matches_the_formula_and_pins_inputs() -> None:
    artifact = compose_gnss_imu_lever_arm(
        GNSS, ROTATION, TRANSLATION, reference_m=[0.023, -0.023, 0.09]
    )

    assert len({item.sha256 for item in artifact.inputs}) == 3
    # Near-identity MID360 rotation: p_imu ~ p_lidar - t_lidar_imu.
    assert np.allclose(artifact.lever_arm_m, [0.0303, -0.0265, 0.0808], atol=2e-3)
    axes = {item.name: item for item in artifact.axes}
    assert axes["z"].status == "unobservable"  # the GNSS-LiDAR z std is 4 cm
    assert artifact.policy_status == "inconclusive"
    assert abs(axes["y"].error_to_reference_m or 1.0) < 0.01


def test_cli_writes_a_valid_artifact(tmp_path: Path) -> None:
    output = tmp_path / "gnss_imu.yaml"
    exit_code = main(
        [
            "gnss-imu", "compose", "--gnss-lidar", str(GNSS),
            "--imu-lidar-rotation", str(ROTATION), "--imu-lidar-translation", str(TRANSLATION),
            "--reference", "0.023", "-0.023", "0.090", "--output", str(output),
        ]
    )  # fmt: skip

    assert exit_code == 0
    assert validate_file(output).kind == "gnss-imu-lever-arm"


def test_pass_requires_every_axis() -> None:
    artifact = compose_gnss_imu_lever_arm(GNSS, ROTATION, TRANSLATION)
    payload = artifact.model_dump(mode="python")
    payload["policy_status"] = "pass"
    with pytest.raises(ValueError, match="x, y, and z"):
        GnssImuLeverArmArtifact.model_validate(payload)


def test_axes_inherit_unobservable_inputs_even_with_small_std() -> None:
    # seq1: the IMU-LiDAR translation's x and the GNSS-LiDAR z are unobservable.
    artifact = compose_gnss_imu_lever_arm(
        ASSETS / "rtk_slam_gnss_lidar_stadtgarten_seq1.yaml",
        ASSETS / "mid360_imu_lidar_rotation_rtk_slam_stadtgarten_seq1.yaml",
        ASSETS / "mid360_imu_lidar_translation_rtk_slam_stadtgarten_seq1.yaml",
    )
    axes = {item.name: item for item in artifact.axes}

    assert axes["x"].std_m <= artifact.observable_std_m
    assert axes["x"].status == "unobservable"
    assert axes["z"].status == "unobservable"
    assert artifact.policy_status == "inconclusive"
