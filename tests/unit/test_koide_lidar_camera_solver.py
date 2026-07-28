import sys
from pathlib import Path

import pytest

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.provenance import sha256_path
from calibrex.data.base import StreamSummary
from calibrex.data.inspect import DatasetInspection
from calibrex.solvers.koide_lidar_camera_solver import KoideLidarCameraSolver


def _executable_adapter_fixture(
    tmp_path: Path,
    *,
    command: list[str],
    result_path: Path,
    timeout_sec: float = 5.0,
) -> tuple[CalibrationConfig, DatasetInspection]:
    config = CalibrationConfig.model_validate(
        {
            "schema_version": "slac.config/v0.1",
            "dataset": {"type": "filesystem", "path": str(tmp_path)},
            "sensors": {
                "camera0": {"type": "camera"},
                "lidar0": {"type": "lidar"},
            },
            "frames": {
                "base": {"root": True},
                "camera0": {"parent": "base"},
                "lidar0": {"parent": "base"},
            },
            "pipeline": {
                "factors": {
                    "koide_lidar_camera": {
                        "enabled": True,
                        "options": {
                            "command": command,
                            "execute": True,
                            "result_path": str(result_path),
                            "timeout_sec": timeout_sec,
                        },
                    }
                }
            },
        }
    )
    inspection = DatasetInspection(
        dataset_type="filesystem",
        path=str(tmp_path),
        exists=True,
        streams=[
            StreamSummary("camera_left_color", "image", 1, sensor="camera0"),
            StreamSummary("velodyne_points", "pointcloud", 1, sensor="lidar0"),
        ],
    )
    return config, inspection


def test_koide_lidar_camera_adapter_loads_precomputed_transform(tmp_path: Path) -> None:
    external_result = tmp_path / "koide_result.yaml"
    external_result.write_text(
        """
transforms:
  T_camera0_lidar0:
    convention: T_parent_child
    parent: camera0
    child: lidar0
    translation_m: [1.0, 2.0, 3.0]
    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
""".strip(),
        encoding="utf-8",
    )
    config = CalibrationConfig.model_validate(
        {
            "schema_version": "slac.config/v0.1",
            "dataset": {"type": "filesystem", "path": str(tmp_path)},
            "sensors": {
                "camera0": {"type": "camera"},
                "lidar0": {"type": "lidar"},
            },
            "frames": {
                "base": {"root": True},
                "camera0": {"parent": "base"},
                "lidar0": {"parent": "base"},
            },
            "pipeline": {
                "factors": {
                    "koide_lidar_camera": {
                        "enabled": True,
                        "options": {
                            "result_path": str(external_result),
                            "tool_name": "direct_visual_lidar_calibration",
                            "tool_version": "test-1.0",
                            "source_repository": "https://example.test/koide-toolbox",
                            "source_commit": "0123456789abcdef",
                            "license_spdx": "BSD-3-Clause",
                            "training_isolation_declared": True,
                            "training_isolation_evidence": "fixture output fitted elsewhere",
                        },
                    }
                }
            },
        }
    )
    inspection = DatasetInspection(
        dataset_type="filesystem",
        path=str(tmp_path),
        exists=True,
        streams=[
            StreamSummary("camera_left_color", "image", 1, sensor="camera0"),
            StreamSummary("velodyne_points", "pointcloud", 1, sensor="lidar0"),
        ],
    )

    result = KoideLidarCameraSolver().solve(
        config,
        FrameGraph.from_config(config),
        inspection,
    )

    assert result.backend == "koide_lidar_camera"
    assert result.status == "result_loaded"
    assert result.transforms["T_camera0_lidar0"].translation_m == (1.0, 2.0, 3.0)
    assert result.metrics["koide_lidar_camera_input_ready"].grade == "pass"
    assert result.metrics["koide_lidar_camera_result_available"].grade == "pass"
    assert result.metrics["koide_lidar_camera_provenance_complete"].grade == "pass"
    identity = result.provenance["koide_lidar_camera_tool_identity"]
    assert identity["result_sha256"] == sha256_path(external_result)
    assert identity["result_size_bytes"] == external_result.stat().st_size
    assert identity["missing_required_fields"] == []
    assert identity["complete"] is True
    assert result.provenance["license_boundary"].startswith("external subprocess")
    assert result.external_run is not None
    assert result.external_run.schema_version == "slac.external_calibration_run/v0.1"
    assert result.external_run.status == "success"
    assert result.external_run.tool.name == "direct_visual_lidar_calibration"
    assert result.external_run.tool.license_boundary == "imported"
    assert result.external_run.parsed_outputs.external_metrics_comparable is False
    assert result.external_run.parsed_outputs.transforms[
        "T_camera0_lidar0"
    ].translation_m == [1.0, 2.0, 3.0]
    assert any(
        artifact.sha256 == sha256_path(external_result)
        for artifact in result.external_run.artifacts
        if artifact.role == "output"
    )


def test_koide_lidar_camera_adapter_reports_missing_boundary(tmp_path: Path) -> None:
    config = CalibrationConfig.model_validate(
        {
            "schema_version": "slac.config/v0.1",
            "dataset": {"type": "filesystem", "path": str(tmp_path)},
            "sensors": {
                "camera0": {"type": "camera"},
                "lidar0": {"type": "lidar"},
            },
            "frames": {
                "base": {"root": True},
                "camera0": {"parent": "base"},
                "lidar0": {"parent": "base"},
            },
            "pipeline": {"factors": {"koide_lidar_camera": {"enabled": True}}},
        }
    )
    inspection = DatasetInspection(
        dataset_type="filesystem",
        path=str(tmp_path),
        exists=True,
        streams=[],
    )

    result = KoideLidarCameraSolver().solve(
        config,
        FrameGraph.from_config(config),
        inspection,
    )

    assert result.status == "not_executed"
    assert result.metrics["koide_lidar_camera_adapter_available"].grade == "warn"
    assert result.metrics["koide_lidar_camera_input_ready"].grade == "fail"
    assert result.metrics["koide_lidar_camera_provenance_complete"].grade == "warn"
    assert (
        "result_sha256"
        in result.provenance["koide_lidar_camera_tool_identity"]["missing_required_fields"]
    )
    assert result.warnings


def test_koide_lidar_camera_adapter_executes_external_command(tmp_path: Path) -> None:
    external_result = tmp_path / "generated_result.yaml"
    script = tmp_path / "write_result.py"
    script.write_text(
        """
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    "transforms:\\n"
    "  T_camera0_lidar0:\\n"
    "    parent: camera0\\n"
    "    child: lidar0\\n"
    "    translation_m: [7.0, 8.0, 9.0]\\n"
    "    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]\\n",
    encoding="utf-8",
)
print("wrote", sys.argv[1])
""".strip(),
        encoding="utf-8",
    )
    config = CalibrationConfig.model_validate(
        {
            "schema_version": "slac.config/v0.1",
            "dataset": {"type": "filesystem", "path": str(tmp_path)},
            "sensors": {
                "camera0": {"type": "camera"},
                "lidar0": {"type": "lidar"},
            },
            "frames": {
                "base": {"root": True},
                "camera0": {"parent": "base"},
                "lidar0": {"parent": "base"},
            },
            "pipeline": {
                "factors": {
                    "koide_lidar_camera": {
                        "enabled": True,
                        "options": {
                            "command": [sys.executable, str(script), "{result_path}"],
                            "execute": True,
                            "result_path": str(external_result),
                            "timeout_sec": 5.0,
                        },
                    }
                }
            },
        }
    )
    inspection = DatasetInspection(
        dataset_type="filesystem",
        path=str(tmp_path),
        exists=True,
        streams=[
            StreamSummary("camera_left_color", "image", 1, sensor="camera0"),
            StreamSummary("velodyne_points", "pointcloud", 1, sensor="lidar0"),
        ],
    )

    result = KoideLidarCameraSolver().solve(
        config,
        FrameGraph.from_config(config),
        inspection,
    )

    assert result.status == "result_loaded"
    assert result.transforms["T_camera0_lidar0"].translation_m == (7.0, 8.0, 9.0)
    assert result.metrics["koide_lidar_camera_execution_success"].grade == "pass"
    assert result.metrics["koide_lidar_camera_provenance_complete"].grade == "warn"
    assert result.provenance["koide_lidar_camera_execution"]["attempted"] is True
    assert result.provenance["koide_lidar_camera_execution"]["returncode"] == 0
    assert result.external_run is not None
    assert result.external_run.execution.mode == "subprocess"
    assert result.external_run.execution.stdout_sha256 is not None
    assert result.external_run.execution.duration_seconds is not None
    identity = result.provenance["koide_lidar_camera_tool_identity"]
    assert identity["result_sha256"] == sha256_path(external_result)
    assert "result_sha256" not in identity["missing_required_fields"]
    assert "tool_version" in identity["missing_required_fields"]


@pytest.mark.parametrize(
    ("command", "timeout_sec", "expected_status"),
    [
        (["calibrex-command-that-does-not-exist"], 5.0, "unavailable"),
        ([sys.executable, "-c", "raise SystemExit(7)"], 5.0, "failed"),
        (
            [sys.executable, "-c", "import time; time.sleep(1)"],
            0.1,
            "timeout",
        ),
    ],
)
def test_koide_external_run_materializes_execution_failures(
    tmp_path: Path,
    command: list[str],
    timeout_sec: float,
    expected_status: str,
) -> None:
    config, inspection = _executable_adapter_fixture(
        tmp_path,
        command=command,
        result_path=tmp_path / "missing-result.yaml",
        timeout_sec=timeout_sec,
    )

    result = KoideLidarCameraSolver().solve(
        config,
        FrameGraph.from_config(config),
        inspection,
    )

    assert result.external_run is not None
    assert result.external_run.status == expected_status
    assert result.metrics["koide_lidar_camera_execution_success"].grade == "fail"
    assert result.external_run.warnings


def test_koide_external_run_materializes_malformed_output(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("transforms: [not, a, mapping]", encoding="utf-8")
    config, inspection = _executable_adapter_fixture(
        tmp_path,
        command=[sys.executable, "-c", "raise SystemExit(0)"],
        result_path=malformed,
    )

    result = KoideLidarCameraSolver().solve(
        config,
        FrameGraph.from_config(config),
        inspection,
    )

    assert result.external_run is not None
    assert result.external_run.status == "invalid_output"
    assert result.transforms == {}
    assert any("readable transform" in warning for warning in result.external_run.warnings)
