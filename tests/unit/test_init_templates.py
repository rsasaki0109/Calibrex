"""Unit tests for sensor template init."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.init_templates import (
    list_sensor_template_names,
    resolve_sensor_template_name,
    write_sensor_template,
)


def test_list_sensor_template_names_includes_rosbag_templates() -> None:
    names = list_sensor_template_names()
    assert "velodyne_vlp16_pair_rosbag2" in names
    assert "velodyne_vlp16_pair_rosbag1" in names


def test_write_sensor_template_copies_config(tmp_path: Path) -> None:
    output = tmp_path / "my_calib" / "config.yaml"
    written = write_sensor_template("velodyne_vlp16_pair_rosbag2", output)

    assert written == output
    assert output.is_file()
    text = output.read_text(encoding="utf-8")
    assert "native_lidar_point_to_plane" in text
    assert "rosbag2" in text


def test_resolve_sensor_template_name_supports_aliases() -> None:
    assert (
        resolve_sensor_template_name("velodyne-camera")
        == "velodyne_vlp16_pair_rosbag2"
    )


def test_init_list_templates_cli(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["init", "--list-templates"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "velodyne_vlp16_pair_rosbag2" in output


def test_init_template_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "config.yaml"
    exit_code = main(
        [
            "init",
            "--template",
            "velodyne-lidar-pair-rosbag2",
            "--output",
            str(output),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert output.is_file()
