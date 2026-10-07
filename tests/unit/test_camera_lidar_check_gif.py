"""Tests for the camera-LiDAR check README GIF generator and its committed check summaries."""

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

import camera_lidar_check_gif as tool  # noqa: E402

KITTI_DRIVE = Path.home() / "data/public/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync"


def test_summaries_tell_pass_then_fail_on_the_turned_axis() -> None:
    reference = tool.load_summary("reference")
    perturbed = tool.load_summary("perturbed")
    for summary in (reference, perturbed):
        assert summary["schema"] == tool.RUN_SCHEMA
        assert "--tf" in summary["command"]
        assert not any(item.startswith("/") for item in summary["command"])
        assert set(summary["unchecked"]) >= {"x", "y", "z"}  # translation is never judged
    assert reference["perturbation_deg"] == 0.0
    assert reference["status"] == "pass"
    assert perturbed["perturbation_deg"] == tool.PERTURB_DEG
    assert perturbed["status"] == "fail"
    assert perturbed["axes"]["pitch"]["status"] == "fail"
    assert perturbed["axes"]["pitch"]["candidate_error_deg"] == pytest.approx(3.0, abs=0.2)
    assert (
        reference["axes"]["pitch"]["candidate_error_deg"]
        < reference["axes"]["pitch"]["tolerance_deg"]
    )


def test_rotation_about_camera_y_matches_the_perturbation_sign() -> None:
    rotation = tool.rotation_about("y", 3.0)
    assert np.allclose(rotation @ rotation.T, np.eye(3))
    assert np.degrees(np.arccos((np.trace(rotation) - 1) / 2)) == pytest.approx(3.0)
    assert rotation[0, 2] > 0  # +y rotation turns z toward +x


def test_provenance_is_digest_bound() -> None:
    provenance = json.loads(tool.PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["schema"] == tool.PROVENANCE_SCHEMA
    assert "no drive 0027-0059" in provenance["drive"]
    for name, digest in provenance["run_summary_sha256"].items():
        assert digest == tool.sweep.sha256_file(tool.OUT_DIR / name)


@pytest.mark.skipif(not KITTI_DRIVE.is_dir(), reason="KITTI drive 0005 not on disk")
def test_overlay_moves_points_when_the_camera_is_turned() -> None:
    pytest.importorskip("PIL")
    scene = tool.Scene(KITTI_DRIVE, tool.IMAGE_INDEX)
    reference = np.asarray(scene.overlay(0.0))
    turned = np.asarray(scene.overlay(tool.PERTURB_DEG))
    assert reference.shape == turned.shape
    assert not np.array_equal(reference, turned)
