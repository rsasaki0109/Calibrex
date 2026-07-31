from pathlib import Path

from calibrex.core.config import CalibrationConfig
from calibrex.core.external_run import load_external_run
from calibrex.core.frames import FrameGraph
from calibrex.core.provenance import sha256_path
from calibrex.data.base import StreamSummary
from calibrex.data.inspect import DatasetInspection
from calibrex.solvers.unicalib_lidar_camera_solver import UniCalibLidarCameraSolver


def _inspection(tmp_path: Path) -> DatasetInspection:
    return DatasetInspection(
        dataset_type="filesystem",
        path=str(tmp_path),
        exists=True,
        streams=[
            StreamSummary("camera_left", "image", 2, sensor="camera0"),
            StreamSummary("velodyne", "pointcloud", 2, sensor="lidar0"),
        ],
    )


def _config(tmp_path: Path, options: dict[str, object]) -> CalibrationConfig:
    return CalibrationConfig.model_validate(
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
                    "unicalib_lidar_camera": {
                        "enabled": True,
                        "options": options,
                    }
                }
            },
        }
    )


def _result(path: Path) -> None:
    path.write_text(
        """
transforms:
  T_camera0_lidar0:
    parent: camera0
    child: lidar0
    translation_m: [0.1, 0.2, 0.3]
    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
""".strip(),
        encoding="utf-8",
    )


def test_unicalib_import_emits_generic_artifact(tmp_path: Path) -> None:
    result_path = tmp_path / "unicalib.yaml"
    input_path = tmp_path / "frames.txt"
    artifact_path = tmp_path / "external_run.yaml"
    _result(result_path)
    input_path.write_text("frame-000000\n", encoding="utf-8")
    config = _config(
        tmp_path,
        {
            "result_path": str(result_path),
            "input_paths": [str(input_path)],
            "external_run_output": str(artifact_path),
            "tool_version": "wacv-2026",
            "source_commit": "0123456789abcdef",
            "training_isolation_declared": True,
            "training_isolation_evidence": "held-out fixture",
            "external_metrics": {"reported_rotation_error_deg": 0.044},
        },
    )

    result = UniCalibLidarCameraSolver().solve(
        config, FrameGraph.from_config(config), _inspection(tmp_path)
    )
    artifact = load_external_run(artifact_path)

    assert result.status == "result_loaded"
    assert result.transforms["T_camera0_lidar0"].translation_m == (0.1, 0.2, 0.3)
    assert result.metrics["unicalib_provenance_complete"].grade == "pass"
    assert artifact.status == "success"
    assert artifact.tool.license_spdx == "MIT"
    assert artifact.tool.source_repository == "https://github.com/han-15/UniCalib"
    assert artifact.artifacts[0].sha256 == sha256_path(input_path)
    assert artifact.parsed_outputs.external_metrics_comparable is False
    assert artifact.parsed_outputs.external_metrics[
        "reported_rotation_error_deg"
    ] == 0.044
    assert artifact.transform_provenance("T_camera0_lidar0").producer == (
        "external_tool"
    )


def test_unicalib_digest_mismatch_fails_artifact(tmp_path: Path) -> None:
    result_path = tmp_path / "unicalib.yaml"
    _result(result_path)
    config = _config(
        tmp_path,
        {
            "result_path": str(result_path),
            "expected_result_sha256": "0" * 64,
            "tool_version": "wacv-2026",
            "source_commit": "0123456789abcdef",
        },
    )

    result = UniCalibLidarCameraSolver().solve(
        config, FrameGraph.from_config(config), _inspection(tmp_path)
    )
    artifact = result.provenance["external_calibration_run"]

    assert artifact["status"] == "digest_mismatch"
    assert "UniCalib result digest mismatch" in artifact["warnings"]
    assert result.metrics["unicalib_provenance_complete"].grade == "warn"


def test_unicalib_malformed_result_is_not_loaded(tmp_path: Path) -> None:
    result_path = tmp_path / "unicalib.yaml"
    result_path.write_text("transforms: []\n", encoding="utf-8")
    config = _config(tmp_path, {"result_path": str(result_path)})

    result = UniCalibLidarCameraSolver().solve(
        config, FrameGraph.from_config(config), _inspection(tmp_path)
    )

    assert result.status == "not_executed"
    assert result.metrics["unicalib_result_available"].grade == "warn"
    assert any("malformed UniCalib result" in warning for warning in result.warnings)
