"""Camera-LiDAR edge alignment on KITTI 0005 with an emulated unsynchronized camera (opt-in).

Needs the converted bag (``calibrex convert kitti-raw 2011_09_26_drive_0005_sync --camera
image_02 --output data/public/kitti_raw/bags/kitti_0005``) or ``CALIBREX_KITTI_0005_BAG``.
Every scan is paired with the next image (114 ms later); the gates are the check's own
verdict rule, none is tuned here.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

BAG = Path(os.environ.get("CALIBREX_KITTI_0005_BAG", "data/public/kitti_raw/bags/kitti_0005"))
TOOL = Path("tools/camera_lidar_motion_stress.py")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (BAG / "metadata.yaml").is_file(),
        reason="convert KITTI raw drive 0005 or set CALIBREX_KITTI_0005_BAG",
    ),
]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("camera_lidar_motion_stress", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, rotation: str) -> dict[str, dict[str, object]]:
    output = tmp_path / f"{rotation}.json"
    argv = [str(BAG), "--image-offset", "1", "--rotation-deg", rotation, "--output", str(output)]
    assert _tool().main(argv) == 0
    runs: dict[str, dict[str, object]] = json.loads(output.read_text(encoding="utf-8"))
    return runs


def test_moved_scans_judge_every_rotation_axis_and_pass_the_reference(tmp_path: Path) -> None:
    moved = _run(tmp_path, "0,0,0")["moved with /oxts/odometry"]
    assert moved["verdict"] == "pass"
    assert set(moved["axes"]) == {"roll", "pitch", "yaw"}  # type: ignore[arg-type]
    motion = moved["motion_compensation"]
    assert isinstance(motion, dict) and motion["median_rotation_deg"] > 0.5


def test_moved_scans_flag_a_three_degree_pitch_error(tmp_path: Path) -> None:
    moved = _run(tmp_path, "0,3,0")["moved with /oxts/odometry"]
    assert moved["verdict"] == "fail"
    pitch = moved["axes"]["pitch"]  # type: ignore[index]
    assert abs(pitch["error"]) == pytest.approx(3.0, abs=0.5)
