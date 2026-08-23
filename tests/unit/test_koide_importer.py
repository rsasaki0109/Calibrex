"""Contract tests for the official Koide native result adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from calibrex.core.provenance import sha256_path
from calibrex.importers.koide import (
    import_koide_result,
    parse_koide_calib_payload,
)


def _native_payload() -> dict[str, object]:
    return {
        "results": {
            "T_lidar_camera": [
                1.2,
                -0.4,
                2.3,
                0.1,
                -0.2,
                0.3,
                0.9,
            ]
        }
    }


def test_native_koide_pose_and_inverse_round_trip() -> None:
    parsed = parse_koide_calib_payload(
        _native_payload(),
        lidar_frame="velodyne_front",
        camera_frame="camera_left",
    )

    forward = parsed.transforms["T_velodyne_front_camera_left"]
    inverse = parsed.transforms["T_camera_left_velodyne_front"]
    point_camera = (0.31, -1.17, 4.2)
    point_lidar = forward.transform_point(point_camera)
    recovered = inverse.transform_point(point_lidar)

    assert all(
        abs(left - right) <= 1.0e-9
        for left, right in zip(point_camera, recovered, strict=True)
    )
    identity = forward.compose(inverse)
    assert all(abs(value) <= 1.0e-9 for value in identity.translation_m)
    assert all(
        abs(left - right) <= 1.0e-9
        for left, right in zip(
            identity.rotation_quat_xyzw,
            (0.0, 0.0, 0.0, 1.0),
            strict=True,
        )
    )
    assert parsed.native_transform_name == "T_lidar_camera"
    assert "camera-frame points" in parsed.declared_convention


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "results"),
        ({"results": {}}, "T_lidar_camera"),
        ({"results": {"T_lidar_camera": [0.0] * 6}}, "exactly 7"),
        (
            {"results": {"T_lidar_camera": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "bad"]}},
            "finite number",
        ),
        (
            {"results": {"T_lidar_camera": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.nan]}},
            "finite number",
        ),
        (
            {"results": {"T_lidar_camera": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}},
            "non-zero",
        ),
    ],
)
def test_native_koide_malformed_payloads_are_explicitly_rejected(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_koide_calib_payload(
            payload,
            lidar_frame="lidar0",
            camera_frame="camera0",
        )


def test_native_koide_requires_explicit_frames() -> None:
    with pytest.raises(ValueError, match="non-empty explicit frame"):
        parse_koide_calib_payload(
            _native_payload(),
            lidar_frame="",
            camera_frame="camera0",
        )


def test_import_koide_result_records_native_provenance(tmp_path: Path) -> None:
    source = tmp_path / "calib.json"
    source.write_text(json.dumps(_native_payload()), encoding="utf-8")
    fitting_input = tmp_path / "capture.mcap"
    fitting_input.write_bytes(b"fixture capture")

    artifact = import_koide_result(
        source,
        lidar_frame="lidar_front",
        camera_frame="camera_left",
        input_artifacts=(fitting_input,),
        tool_version="0.1-fixture",
        source_commit="a" * 40,
        training_isolation_declared=True,
        training_isolation_evidence="fixture split",
    )

    assert artifact.status == "success"
    assert artifact.execution.mode == "imported"
    assert artifact.tool.license_boundary == "imported"
    assert set(artifact.parsed_outputs.transforms) == {
        "T_lidar_front_camera_left",
        "T_camera_left_lidar_front",
    }
    assert artifact.parsed_outputs.transforms["T_lidar_front_camera_left"].parent == (
        "lidar_front"
    )
    assert artifact.parsed_outputs.transforms["T_lidar_front_camera_left"].child == (
        "camera_left"
    )
    metrics = artifact.parsed_outputs.external_metrics
    assert metrics["koide_native_format"] == "direct_visual_lidar_calibration.calib.json"
    assert metrics["koide_native_source_path"] == str(source)
    assert metrics["koide_native_source_sha256"] == sha256_path(source)
    assert "T_lidar_camera" in str(metrics["koide_native_declared_convention"])
    assert artifact.provenance.source_artifact == str(source)
    assert artifact.provenance.source_artifact_sha256 == sha256_path(source)


def test_import_koide_result_materializes_invalid_output(tmp_path: Path) -> None:
    source = tmp_path / "calib.json"
    source.write_text(
        json.dumps({"results": {"T_lidar_camera": [0.0] * 6}}),
        encoding="utf-8",
    )

    artifact = import_koide_result(
        source,
        lidar_frame="lidar0",
        camera_frame="camera0",
    )

    assert artifact.status == "invalid_output"
    assert artifact.provenance.source_artifact_sha256 == sha256_path(source)
    assert any("exactly 7" in warning for warning in artifact.warnings)
