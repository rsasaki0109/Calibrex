from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.core.validation import validate_file
from calibrex.importers.li_init import import_li_init_result, parse_li_init_result

_SECTION = """{title}
Rotation LiDAR to IMU (degree)     = {rpy}
Translation LiDAR to IMU (meter)   = -0.011000 -0.023000 0.044000
Time Lag IMU to LiDAR (second)     = {lag}
Bias of Gyroscope  (rad/s)         = 0.015000 -0.002000 0.001700
Bias of Accelerometer (meters/s^2) = 0.010000 0.020000 -0.030000
Gravity in World Frame(meters/s^2) = 0.000000 0.000000 -9.805000

Homogeneous Transformation Matrix from LiDAR to IMU:
{matrix}


"""


def _section(title: str, rotation: np.ndarray, lag: float) -> str:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [-0.011, -0.023, 0.044]
    rows = "\n".join("  ".join(f"{value: .6f}" for value in row) for row in transform)
    rpy = " ".join(f"{value:.6f}" for value in Rotation.from_matrix(rotation).as_euler("xyz", True))
    return _SECTION.format(title=title, rpy=rpy, lag=f"{lag:.6f}", matrix=rows)


def test_refinement_section_wins_and_conventions_are_converted(tmp_path: Path) -> None:
    initial = Rotation.from_euler("xyz", [1.0, 0.0, 0.0], degrees=True).as_matrix()
    refined = Rotation.from_euler("xyz", [0.1, -0.2, 0.3], degrees=True).as_matrix()
    path = tmp_path / "Initialization_result.txt"
    path.write_text(
        _section("Initialization result:", initial, 0.020)
        + _section("Refinement result:", refined, 0.012),
        encoding="utf-8",
    )

    parsed = parse_li_init_result(path.read_text(encoding="utf-8"))
    artifact = import_li_init_result(
        path, container_digest="sha256:" + "a" * 64, source_commit="66b157a"
    )

    assert parsed["stage"] == "Refinement result"
    assert parsed["time_lag_imu_to_lidar_s"] == pytest.approx(0.012)
    assert artifact.status == "success"
    lidar_imu = artifact.parsed_outputs.transforms["T_lidar_imu"]
    rotation = Rotation.from_quat(lidar_imu.rotation_quat_xyzw).as_matrix()
    assert np.allclose(rotation, refined.T, atol=1e-6)
    assert np.allclose(lidar_imu.translation_m, -refined.T @ np.array([-0.011, -0.023, 0.044]))
    assert artifact.parsed_outputs.time_offsets_seconds["imu_minus_lidar_s"] == pytest.approx(0.012)
    assert artifact.tool.license_boundary == "container"
    assert artifact.execution.mode == "container"


def test_initial_values_are_used_with_a_warning_without_refinement(tmp_path: Path) -> None:
    path = tmp_path / "Initialization_result.txt"
    path.write_text(_section("Initialization result:", np.eye(3), 0.0), encoding="utf-8")

    artifact = import_li_init_result(path)

    assert artifact.status == "success"
    assert any("no refinement section" in warning for warning in artifact.warnings)
    assert artifact.tool.license_boundary == "imported"


def test_empty_result_file_is_invalid_output_and_still_schema_valid(tmp_path: Path) -> None:
    path = tmp_path / "Initialization_result.txt"
    path.write_text("", encoding="utf-8")

    artifact = import_li_init_result(path)
    saved = tmp_path / "li_init_run.yaml"
    artifact.save(saved)

    assert artifact.status == "invalid_output"
    assert validate_file(saved).kind == "external-run"


def test_gpl_tool_cannot_be_declared_in_process(tmp_path: Path) -> None:
    from calibrex.core.external_run import ExternalToolIdentity

    with pytest.raises(ValueError):
        ExternalToolIdentity(
            name="li_init", license_spdx="GPL-2.0-only", license_boundary="in_process"
        )
