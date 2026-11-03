"""The KITTI board camera-LiDAR pre-registration agrees with the frozen harness it names."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from tools import kitti_board_camera_lidar_evaluate as harness

ROOT = Path(__file__).resolve().parents[2]
PLAN = yaml.safe_load(
    (ROOT / "docs/benchmarks/kitti_board_camera_lidar_preregistration.yaml").read_text(
        encoding="utf-8"
    )
)


def test_every_recording_is_pinned_as_the_harness_fetches_it() -> None:
    recordings = PLAN["recordings"]["evaluation"]
    # YAML 1.1 reads an unquoted 2011_09_26 as an integer: every key must stay a string
    assert set(recordings) == set(harness.CALIBRATION_DRIVES)
    for day, pin in recordings.items():
        drive, size, etag = harness.CALIBRATION_DRIVES[day]
        assert pin["drive"] == drive and pin["size_bytes"] == size and pin["etag"] == etag
        assert pin["archive"] == f"{day}_drive_{drive}_extract.zip"
        assert re.fullmatch(r"[0-9a-f]{32}(-\d+)?", pin["etag"])
    assert PLAN["recordings"]["calibration_archive_sha256"] == harness.CALIBRATION_SHA256


def test_the_gates_are_the_harness_constants() -> None:
    accuracy = PLAN["claims"]["kitti-board-camera-lidar-accuracy"]["gates"]
    assert accuracy["max_rotation_error_deg"] == harness.MAX_ROTATION_ERROR_DEG == 0.5
    assert accuracy["max_position_error_m"] == harness.MAX_POSITION_ERROR_M == 0.20
    assert accuracy["days_required"] == len(harness.CALIBRATION_DRIVES)
    honesty = PLAN["claims"]["kitti-board-camera-lidar-check-honesty"]["gates"]
    assert honesty["sigma_k"] == harness.SIGMA_K
    assert honesty["rotation_floor_deg"] == harness.ROTATION_FLOOR_DEG
    assert honesty["translation_floor_m"] == harness.TRANSLATION_FLOOR_M
    assert honesty["fail_factor"] == harness.FAIL_FACTOR
    assert honesty["min_converged_days"] == harness.MIN_CONVERGED_DAYS


def test_the_method_is_frozen_with_its_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", str(PLAN["frozen_code_commit"]))
    method = PLAN["method"]
    assert "0.1 m/s" in method["still_interval"] and harness.STILL_SPEED_MPS == 0.1
    assert "4 x 3" in method["frame"] and harness.MIN_PATTERN == (4, 3)
    assert "30 boards" in method["frame"] and harness.MAX_BOARDS == 30
    assert "1.0 px" in method["square_size"] and harness.MAX_STEREO_ERROR_PX == 1.0
    assert "5 ms" in method["square_size"] and harness.MAX_STEREO_GAP_S == 0.005
    assert "20 Velodyne scans" in method["lidar"] and harness.MAX_SCANS == 20
    assert "calib_velo_to_cam" in method["never_an_input"]
