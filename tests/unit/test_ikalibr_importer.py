"""Unit tests for the iKalibr result importer.

Tests cover:
- Successful YAML parse (cereal value0/value1/value2/value3 format).
- Successful YAML parse (x/y/z/w format, e.g. after JSON conversion).
- Successful YAML parse (4-element list format).
- digest_mismatch status when expected_source_sha256 does not match.
- invalid_output status for malformed YAML.
- unavailable status for a missing file.
- Time offset extraction.
- No ROS or iKalibr dependency required.
"""

from __future__ import annotations

import math
from pathlib import Path

from calibrex.core.provenance import sha256_path
from calibrex.importers.ikalibr import import_ikalibr_result

_LIDAR_TOPIC = "/lidar_front"
_CAMERA_TOPIC = "/camera_left/image_raw"
_IMU_TOPIC = "/imu/data"


def _cereal_so3(x: float, y: float, z: float, w: float) -> dict[str, float]:
    """Return a cereal YAML SO3d dict in value0..value3 format."""
    return {"value0": x, "value1": y, "value2": z, "value3": w}


def _cereal_vec3(x: float, y: float, z: float) -> dict[str, float]:
    return {"value0": x, "value1": y, "value2": z}


def _write_result(path: Path) -> None:
    """Write a minimal well-formed iKalibr YAML result file."""
    import yaml

    so3_identity = _cereal_so3(0.0, 0.0, 0.0, 1.0)
    so3_lidar = _cereal_so3(0.0, 0.0, math.sin(math.radians(1.5)), math.cos(math.radians(1.5)))
    pos_lidar = _cereal_vec3(0.27, -0.04, 0.08)
    pos_camera = _cereal_vec3(0.10, 0.02, -0.01)

    payload = {
        "CalibParam": {
            "EXTRI": {
                "SO3_LkToBr": {_LIDAR_TOPIC: so3_lidar},
                "POS_LkInBr": {_LIDAR_TOPIC: pos_lidar},
                "SO3_CmToBr": {_CAMERA_TOPIC: so3_identity},
                "POS_CmInBr": {_CAMERA_TOPIC: pos_camera},
                "SO3_RjToBr": {},
                "POS_RjInBr": {},
                "SO3_BiToBr": {},
                "POS_BiInBr": {},
                "SO3_DnToBr": {},
                "POS_DnInBr": {},
                "SO3_EsToBr": {},
                "POS_EsInBr": {},
            },
            "TEMPORAL": {
                "TO_LkToBr": {_LIDAR_TOPIC: 0.0023},
                "TO_CmToBr": {_CAMERA_TOPIC: -0.0015},
                "TO_RjToBr": {},
                "TO_BiToBr": {},
                "TO_DnToBr": {},
                "TO_EsToBr": {},
                "RS_READOUT": {},
            },
            "INTRI": {},
            "GRAVITY": {"value0": 0.0, "value1": 0.0, "value2": -9.81},
        }
    }
    path.write_text(yaml.dump(payload), encoding="utf-8")


def _write_result_xyzw_format(path: Path) -> None:
    """Write a result file using x/y/z/w key format (e.g. after JSON→YAML conversion)."""
    import yaml

    payload = {
        "CalibParam": {
            "EXTRI": {
                "SO3_LkToBr": {
                    _LIDAR_TOPIC: {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                },
                "POS_LkInBr": {
                    _LIDAR_TOPIC: {"x": 0.1, "y": 0.2, "z": 0.3}
                },
                "SO3_CmToBr": {},
                "POS_CmInBr": {},
                "SO3_RjToBr": {},
                "POS_RjInBr": {},
                "SO3_BiToBr": {},
                "POS_BiInBr": {},
                "SO3_DnToBr": {},
                "POS_DnInBr": {},
                "SO3_EsToBr": {},
                "POS_EsInBr": {},
            },
            "TEMPORAL": {"TO_LkToBr": {}, "TO_CmToBr": {}, "TO_RjToBr": {},
                         "TO_BiToBr": {}, "TO_DnToBr": {}, "TO_EsToBr": {}, "RS_READOUT": {}},
            "INTRI": {},
        }
    }
    path.write_text(yaml.dump(payload), encoding="utf-8")


def _write_result_list_format(path: Path) -> None:
    """Write a result file using list format for quaternion/vector."""
    import yaml

    payload = {
        "CalibParam": {
            "EXTRI": {
                "SO3_LkToBr": {_LIDAR_TOPIC: [0.0, 0.0, 0.0, 1.0]},
                "POS_LkInBr": {_LIDAR_TOPIC: [0.1, 0.2, 0.3]},
                "SO3_CmToBr": {},
                "POS_CmInBr": {},
                "SO3_RjToBr": {},
                "POS_RjInBr": {},
                "SO3_BiToBr": {},
                "POS_BiInBr": {},
                "SO3_DnToBr": {},
                "POS_DnInBr": {},
                "SO3_EsToBr": {},
                "POS_EsInBr": {},
            },
            "TEMPORAL": {"TO_LkToBr": {}, "TO_CmToBr": {}, "TO_RjToBr": {},
                         "TO_BiToBr": {}, "TO_DnToBr": {}, "TO_EsToBr": {}, "RS_READOUT": {}},
            "INTRI": {},
        }
    }
    path.write_text(yaml.dump(payload), encoding="utf-8")


def test_import_ikalibr_result_extracts_lidar_camera_transforms(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "ikalibr-prog-result.yaml"
    bag_path = tmp_path / "recording.bag"
    _write_result(result_path)
    bag_path.write_bytes(b"fixture")

    artifact = import_ikalibr_result(
        result_path,
        input_artifacts=(bag_path,),
        tool_version="2.1.0",
        source_commit="a" * 40,
        training_isolation_declared=True,
        training_isolation_evidence="data split declared before import",
    )

    assert artifact.status == "success"
    assert artifact.tool.name == "ikalibr"
    assert artifact.tool.license_spdx == "BSD-3-Clause"
    assert artifact.tool.license_boundary == "imported"
    assert artifact.execution.mode == "imported"

    transforms = artifact.parsed_outputs.transforms
    lidar_id = _LIDAR_TOPIC.lstrip("/")
    lidar_key = f"T_Br_Lk_{lidar_id}"
    assert lidar_key in transforms
    t = transforms[lidar_key]
    assert t.parent == "Br"
    assert t.child == f"Lk_{lidar_id}"
    assert len(t.translation_m) == 3
    assert abs(t.translation_m[0] - 0.27) < 1e-9
    assert abs(t.translation_m[1] - (-0.04)) < 1e-9
    assert abs(t.translation_m[2] - 0.08) < 1e-9
    quat = t.rotation_quat_xyzw
    assert abs(sum(q * q for q in quat) - 1.0) < 1e-6

    camera_id = _CAMERA_TOPIC.lstrip("/").replace("/", "_")
    camera_key = f"T_Br_Cm_{camera_id}"
    assert camera_key in transforms

    time_offsets = artifact.parsed_outputs.time_offsets_seconds
    to_lidar_key = f"TO_Lk_{lidar_id}_to_Br_s"
    assert to_lidar_key in time_offsets
    assert abs(time_offsets[to_lidar_key] - 0.0023) < 1e-9

    to_camera_key = f"TO_Cm_{camera_id}_to_Br_s"
    assert to_camera_key in time_offsets
    assert abs(time_offsets[to_camera_key] - (-0.0015)) < 1e-9

    assert {item.role for item in artifact.artifacts} == {"input", "output"}
    assert {item.sha256 for item in artifact.artifacts if item.role == "input"} == {
        sha256_path(bag_path)
    }
    assert artifact.parsed_outputs.external_metrics_comparable is False

    assert "T_Br_Xk" in artifact.frame_convention
    assert "SO3_XkToBr" in artifact.frame_convention
    assert "t_Br = t_Xk + TO_XkToBr" in artifact.time_convention


def test_import_ikalibr_result_xyzw_format(tmp_path: Path) -> None:
    result_path = tmp_path / "result.yaml"
    _write_result_xyzw_format(result_path)

    artifact = import_ikalibr_result(result_path)

    assert artifact.status == "success"
    lidar_id = _LIDAR_TOPIC.lstrip("/")
    lidar_key = f"T_Br_Lk_{lidar_id}"
    assert lidar_key in artifact.parsed_outputs.transforms
    t = artifact.parsed_outputs.transforms[lidar_key]
    assert t.translation_m == [0.1, 0.2, 0.3]


def test_import_ikalibr_result_list_format(tmp_path: Path) -> None:
    result_path = tmp_path / "result.yaml"
    _write_result_list_format(result_path)

    artifact = import_ikalibr_result(result_path)

    assert artifact.status == "success"
    lidar_id = _LIDAR_TOPIC.lstrip("/")
    assert f"T_Br_Lk_{lidar_id}" in artifact.parsed_outputs.transforms


def test_import_ikalibr_result_digest_mismatch(tmp_path: Path) -> None:
    result_path = tmp_path / "result.yaml"
    _write_result(result_path)

    artifact = import_ikalibr_result(result_path, expected_source_sha256="0" * 64)

    assert artifact.status == "digest_mismatch"
    assert artifact.parsed_outputs.transforms == {}
    assert "SHA-256 mismatch" in " ".join(artifact.warnings)


def test_import_ikalibr_result_malformed_yaml(tmp_path: Path) -> None:
    result_path = tmp_path / "broken.yaml"
    result_path.write_text("CalibParam: [", encoding="utf-8")

    artifact = import_ikalibr_result(result_path)

    assert artifact.status == "invalid_output"
    assert "could not be parsed" in " ".join(artifact.warnings)


def test_import_ikalibr_result_missing_calib_param_key(tmp_path: Path) -> None:
    result_path = tmp_path / "no_key.yaml"
    result_path.write_text("SomeOtherKey: {}", encoding="utf-8")

    artifact = import_ikalibr_result(result_path)

    assert artifact.status == "invalid_output"
    assert "CalibParam" in " ".join(artifact.warnings)


def test_import_ikalibr_result_unavailable_file(tmp_path: Path) -> None:
    artifact = import_ikalibr_result(tmp_path / "missing.yaml")

    assert artifact.status == "unavailable"
    assert artifact.provenance.source_artifact_sha256 is None


def test_import_ikalibr_result_warns_on_missing_tool_identity(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.yaml"
    _write_result(result_path)

    artifact = import_ikalibr_result(result_path)

    warnings = artifact.warnings
    assert any("version" in w.lower() for w in warnings)
    assert any("commit" in w.lower() for w in warnings)
