"""Autoware adapter contract tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.io import read_mapping
from calibrex.core.result import TransformResult
from calibrex.export.autoware import (
    AutowareExportConfig,
    AutowareExportError,
    build_autoware_export,
    write_autoware_export,
)


def _transform(
    parent: str = "base",
    child: str = "lidar",
    translation: list[float] | None = None,
    quaternion: list[float] | None = None,
) -> TransformResult:
    return TransformResult(
        parent=parent,
        child=child,
        translation_m=translation or [1.0, 2.0, 3.0],
        rotation_quat_xyzw=quaternion or [0.0, 0.0, 0.0, 1.0],
    )


def test_inverse_rotates_negative_translation_and_records_direction() -> None:
    # 90 degrees around z: inverse(T_lidar_base) has -R^T t = [-2, 1, -3].
    artifact = build_autoware_export(
        {
            "T_lidar_base": _transform(
                "lidar",
                "base",
                quaternion=[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
            )
        },
        config=AutowareExportConfig(
            base_frame="base",
            invert_transforms=["T_lidar_base"],
        ),
    )
    record = artifact.transforms[0]
    assert record.inverse_applied is True
    assert record.parent_frame == "base"
    assert record.child_frame == "lidar"
    assert record.translation_m == pytest.approx([-2.0, 1.0, -3.0])


def test_quaternion_order_and_rpy_are_explicit() -> None:
    half = math.sqrt(0.5)
    artifact = build_autoware_export(
        {"T_base_camera": _transform(child="camera", quaternion=[0.0, 0.0, half, half])}
    )
    sensor = artifact.calibration["base"]["camera"]
    assert artifact.quaternion_order == "xyzw"
    assert sensor.yaw == pytest.approx(math.pi / 2.0)
    assert sensor.roll == pytest.approx(0.0)
    assert sensor.pitch == pytest.approx(0.0)


def test_ambiguous_parent_and_nonfinite_transform_are_rejected() -> None:
    with pytest.raises(AutowareExportError, match="ambiguous"):
        build_autoware_export(
            {
                "a": _transform("base_a", "sensor_a"),
                "b": _transform("base_b", "sensor_b"),
            }
        )
    with pytest.raises(AutowareExportError, match="non-finite"):
        build_autoware_export(
            {"a": _transform(translation=[float("nan"), 0.0, 0.0])}
        )
    with pytest.raises(AutowareExportError, match="non-rigid"):
        build_autoware_export(
            {"a": _transform(quaternion=[0.0, 0.0, 0.0, 2.0])}
        )


def test_export_is_deterministic_and_manifest_has_file_digests(tmp_path: Path) -> None:
    first = build_autoware_export({"T_base_lidar": _transform()})
    second = build_autoware_export({"T_base_lidar": _transform()})
    assert first.artifact_sha256 == second.artifact_sha256
    first_paths = write_autoware_export(
        first,
        tmp_path / "one.yaml",
        static_tf_output=tmp_path / "one.static_tf.launch.py",
        manifest_output=tmp_path / "one.manifest.yaml",
    )
    second_paths = write_autoware_export(
        second,
        tmp_path / "two.yaml",
        static_tf_output=tmp_path / "two.static_tf.launch.py",
        manifest_output=tmp_path / "two.manifest.yaml",
    )
    assert first_paths["calibration"].read_bytes() == second_paths["calibration"].read_bytes()
    assert first_paths["static_tf"].read_bytes() == second_paths["static_tf"].read_bytes()
    manifest = read_mapping(first_paths["manifest"])
    assert manifest["calibration_sha256"]
    assert manifest["static_tf_sha256"]
    assert manifest["manifest_sha256"]
    assert manifest["provenance"]["tool_version"]
    assert manifest["export_config"]["base_frame"] == "base"
    assert json.loads(json.dumps(manifest))


def test_cli_writes_and_validates_autoware_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "sensor_kit_calibration.yaml"
    assert (
        main(
            [
                "export",
                "examples/precomputed/result.yaml",
                "--format",
                "autoware",
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_version"] == "slac.autoware_export/v0.1"
    assert main(["validate", summary["manifest"], "--kind", "autoware-export"]) == 0
