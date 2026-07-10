import sys
from pathlib import Path

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.data.base import StreamSummary
from calibrex.data.inspect import DatasetInspection
from calibrex.solvers.koide_lidar_camera_solver import KoideLidarCameraSolver


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
                        "options": {"result_path": str(external_result)},
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
    assert result.provenance["license_boundary"].startswith("external subprocess")


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
    assert result.provenance["koide_lidar_camera_execution"]["attempted"] is True
    assert result.provenance["koide_lidar_camera_execution"]["returncode"] == 0
