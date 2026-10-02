"""Tests for the "check story" README GIF generator and its committed check summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import check_story_gif as story  # noqa: E402
import check_story_run as story_run  # noqa: E402

ACT1_NAME = "act1_deployed_wrong.json"
ACT3_NAME = "act3_recheck_vendor.json"
KITTI_DRIVE = Path.home() / "data/public/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync"


def test_story_summaries_tell_fail_then_pass() -> None:
    act1 = story.load_json(story.ACT1)
    act3 = story.load_json(story.ACT3)
    for act in (act1, act3):
        assert act["schema"] == story_run.STORY_SCHEMA
        assert "--tf" in act["command"]
    assert act1["yaw_injected_deg"] == 3.0
    assert act3["yaw_injected_deg"] == 0.0  # act 3 is a real check at the vendor tf
    rec1, rec3 = story.lidar_vehicle(act1), story.lidar_vehicle(act3)
    assert rec1["status"] == "fail"
    assert rec1["axes"]["yaw"]["status"] == "fail"
    assert rec3["status"] == "pass"
    assert set(rec3["axes"]) == {"pitch", "yaw"}  # partial coverage: roll is not observable
    assert rec3["unchecked"] == ["roll"]
    assert abs(rec3["axes"]["yaw"]["candidate_error_deg"]) < rec3["axes"]["yaw"]["tolerance_deg"]
    assert abs(rec1["axes"]["yaw"]["candidate_error_deg"]) > 3.0


def test_run_manifest_binds_the_summaries() -> None:
    run = story.load_json(story.RUN_MANIFEST)
    assert "non-commercial" in run["dataset"]
    assert set(run["summary_sha256"]) == {ACT1_NAME, ACT3_NAME}
    for name, digest in run["summary_sha256"].items():
        assert digest == story_run.sha256_file(story.STORY_DIR / name)
    assert "estimate_rotation_quat_xyzw" not in run  # no calibrex estimate is deployed


def test_act3_tf_is_the_vendor_tf() -> None:
    act1 = story.load_json(story.ACT1)
    act3 = story.load_json(story.ACT3)
    vendor = story.vendor_tf(act1)
    # act 3 deploys the vendor calibration, not a calibrex estimate: tf equal to the vendor tf
    assert np.allclose(story.velo_tf(act3), vendor, atol=1e-6)
    # ... and act 1 is the vendor tf turned by the injected yaw
    delta = abs(story.yaw_between_deg(story.velo_tf(act1), story.velo_tf(act3)))
    assert abs(delta - act1["yaw_injected_deg"]) < 1e-3


def test_slerp_endpoints_and_monotonic_yaw() -> None:
    act1 = story.load_json(story.ACT1)
    act3 = story.load_json(story.ACT3)
    t1, t3 = story.velo_tf(act1), story.velo_tf(act3)
    assert np.allclose(story.slerp_tf(t1, t3, 0.0), t1)
    assert np.allclose(story.slerp_tf(t1, t3, 1.0), t3, atol=1e-9)
    offsets = [
        abs(story.yaw_between_deg(story.slerp_tf(t1, t3, t), t3)) for t in np.linspace(0, 1, 6)
    ]
    assert offsets == sorted(offsets, reverse=True)
    rotation = story.slerp_tf(t1, t3, 0.37)[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)


def test_provenance_names_the_gif_and_licence() -> None:
    provenance = story.load_json(story.PROVENANCE)
    assert provenance["output"] == story.GIF.as_posix()
    assert provenance["generator"] == "tools/check_story_gif.py"
    assert "non-commercial" in provenance["dataset_license"]
    run = story.load_json(story.RUN_MANIFEST)
    for name in (ACT1_NAME, ACT3_NAME):
        assert provenance["summaries_sha256"][name] == run["summary_sha256"][name]
    assert "vendor" in provenance["story"]["act3_deployed_tf"]
    assert story.GIF.name in json.dumps(provenance)


def test_story_gif_renders_from_kitti() -> None:
    pytest.importorskip("PIL")
    if not KITTI_DRIVE.exists():
        pytest.skip("KITTI raw drive 0005 is not available")
    frames, durations, previews = story.build(KITTI_DRIVE)
    assert len(frames) == len(durations) == 2 + story.ACT2_FRAMES
    assert frames[0].size == (960, 540)
    assert set(previews) == {"act1", "act2", "act3", "inset"}
    assert sum(durations) / 1000 < 12
