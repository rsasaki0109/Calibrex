"""The v2 pre-registration agrees with the frozen driver, scorer and builder it names."""

import re
from pathlib import Path

import yaml
from tests.unit.test_score_hilti_camera_camera import _load

ROOT = Path(__file__).resolve().parents[2]
PLAN = yaml.safe_load(
    (ROOT / "docs/benchmarks/kitti_camera_lidar_v2_preregistration.yaml").read_text(
        encoding="utf-8"
    )
)
scorer = _load("score_kitti_camera_lidar_v2")
builder = _load("build_kitti_camera_lidar_v2_audit")
driver = _load("kitti_camera_lidar_v2_evaluate")


def test_the_claim_is_the_builders() -> None:
    drives = len(PLAN["recordings"]["evaluation"])
    claim = PLAN["claims"]["kitti-camera-lidar-check-v2"]["text"]
    assert claim == builder.CLAIM.format(drives=drives)
    assert PLAN["claims"]["kitti-camera-lidar-check-v2"]["protocol_id"] == builder.PROTOCOL_ID


def test_every_drive_and_day_is_pinned_with_values_the_driver_reads() -> None:
    recordings = PLAN["recordings"]
    drives = recordings["evaluation"]
    assert len(drives) == len(set(drives)) == 18
    # YAML 1.1 reads an unquoted 2011_09_28 as an integer: every key must stay a string
    assert set(recordings["archives"]) == set(drives)
    for name in drives:
        date, _ = driver.split_drive(name)
        assert date in builder.DAYS
        pin = recordings["archives"][name]
        assert isinstance(pin["size_bytes"], int) and pin["size_bytes"] > 0
        assert isinstance(pin["etag"], str) and re.fullmatch(r"[0-9a-f]{32}(-\d+)?", pin["etag"])
    pins = recordings["calibration_archive_sha256"]
    assert set(pins) == {driver.split_drive(name)[0] for name in drives} == set(builder.DAYS)
    for value in pins.values():
        assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    assert "2011_09_30_0018" not in drives  # a development drive


def test_the_recipe_and_thresholds_are_the_frozen_tools() -> None:
    recipe = PLAN["run_recipe"]
    assert recipe["injections"] == list(scorer.INJECTIONS)
    assert recipe["velodyne_time"] == scorer.VELODYNE_TIME
    assert recipe["max_frames"] == scorer.MAX_FRAMES
    rules = {item["id"]: item["rule"] for item in PLAN["scoring"]["requirements"]}
    assert list(rules) == list(scorer.REQUIREMENTS)
    assert f"<= {scorer.AXIS_ERROR_MAX_DEG:g} deg" in rules["judged-axis-accuracy"]
    assert f"<= {scorer.TURNED_AXIS_ERROR_MAX_DEG:g} deg" in rules["turned-judged-axis-accuracy"]
    assert f"at most {scorer.MISSED_3DEG_MAX:g} " in rules["turn-3deg-flagged"]
    assert f"at most {scorer.MISSED_1DEG_MAX:g} " in rules["turn-1deg-flagged"]
    assert f"at least {scorer.SCORED_DRIVES_MIN} of the 18" in rules["scored-drives-coverage"]
    assert scorer.JUDGED_FRACTION_MIN == 0.5 and "half" in rules["judged-axis-coverage"]
    assert re.fullmatch(r"[0-9a-f]{40}", PLAN["frozen_code_commit"])
