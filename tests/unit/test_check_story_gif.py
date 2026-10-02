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

KITTI_DRIVE = Path.home() / "data/public/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync"


def test_story_summaries_tell_fail_then_pass() -> None:
    act1 = story.load_json(story.ACT1)
    act3 = story.load_json(story.ACT3)
    for act in (act1, act3):
        assert act["schema"] == story_run.STORY_SCHEMA
        assert "--tf" in act["command"]
    rec1, rec3 = story.lidar_vehicle(act1), story.lidar_vehicle(act3)
    assert rec1["status"] == "fail"
    assert rec1["axes"]["yaw"]["status"] == "fail"
    assert rec3["status"] == "pass"
    assert set(rec3["axes"]) == {"pitch", "yaw"}  # partial coverage: roll is not observable
    assert rec3["unchecked"] == ["roll"]
    assert abs(rec3["axes"]["yaw"]["candidate_error_deg"]) < 0.05
    assert abs(rec1["axes"]["yaw"]["candidate_error_deg"]) > 3.0


def test_run_manifest_binds_the_summaries() -> None:
    run = story.load_json(story.RUN_MANIFEST)
    assert "non-commercial" in run["dataset"]
    for name, digest in run["summary_sha256"].items():
        assert digest == story_run.sha256_file(story.STORY_DIR / name)
    quat = np.array(run["estimate_rotation_quat_xyzw"])
    assert abs(np.linalg.norm(quat) - 1.0) < 1e-9


def test_act3_tf_is_the_estimate_and_vendor_is_close() -> None:
    act1 = story.load_json(story.ACT1)
    act3 = story.load_json(story.ACT3)
    assert abs(story.vendor_to_estimate_yaw_deg(act1) - 0.31) < 0.02
    # act 3 deployed the estimate: its tf differs from the act-1 tf by about the injected yaw
    delta = abs(story.yaw_between_deg(story.velo_tf(act1), story.velo_tf(act3)))
    assert abs(delta - (act1["yaw_injected_deg"] + story.vendor_to_estimate_yaw_deg(act1))) < 0.2


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
    for name in ("act1_deployed_wrong.json", "act3_recheck_estimate.json"):
        assert provenance["summaries_sha256"][name] == run["summary_sha256"][name]
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
