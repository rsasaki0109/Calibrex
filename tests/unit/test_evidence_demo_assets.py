"""Measured demo assets retain original observations, splits and numerical meaning."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from build_evidence_demos import SOURCE, board, gnss, handeye, radar, vehicle  # noqa: E402
from fit_real_fisheye_models import ASSETS, camera, observations  # noqa: E402


def test_measured_demo_original_and_generated_hashes() -> None:
    samples = ROOT / "docs/app/samples"
    index = json.loads((samples / "evidence_demo_index.json").read_text())
    for path, value in index["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value
    for path, value in index["assets"].items():
        assert hashlib.sha256((samples / path).read_bytes()).hexdigest() == value
    receipt = json.loads((SOURCE / "comma/receipt.json").read_text())
    for path, value in receipt.items():
        assert hashlib.sha256((SOURCE / "comma" / path).read_bytes()).hexdigest() == value["sha256"]
        assert "4c7f1a6e1957745beadc1def0e7225f559b09a2a" in value["url"]


def test_recorded_camera_orientation_matches_upstream_inverse_convention() -> None:
    source = SOURCE / "comma"
    q = np.load(source / "global_pose_frame_orientations.npy")
    velocity = np.load(source / "global_pose_frame_velocities.npy")
    # Upstream utils/camera.py names this ecef_from_device_rot and transposes it.
    w, x, y, z = q[0]
    forward_ecef = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
    saved = json.loads((ROOT / "docs/app/samples/evidence_vehicle.json").read_text())
    assert saved["body_velocity"][0][0] == pytest.approx(velocity[0] @ forward_ecef)
    assert abs(saved["body_velocity"][0][1]) < 0.2
    geometry = gnss()["lever_geometry"]
    assert np.linalg.det(geometry["body_to_enu"][0]) == pytest.approx(1)


@pytest.mark.parametrize(
    "code,make",
    [
        ("comparison", board),
        ("handeye", handeye),
        ("gnss", gnss),
        ("vehicle", vehicle),
        ("radar", radar),
    ],
)
def test_measured_demo_is_reproducible_from_sources(
    code: str, make: Callable[[], dict[str, Any]]
) -> None:
    saved = json.loads((ROOT / f"docs/app/samples/evidence_{code}.json").read_text())
    generated = make()
    assert saved["capture_kind"] == "real"
    assert len(saved["views"]) == len(generated["views"])
    for actual, expected in zip(saved["views"], generated["views"], strict=True):
        assert actual["id"] == expected["id"]
        for a, b in zip(actual["metrics"], expected["metrics"], strict=True):
            if isinstance(a["value"], (int, float)):
                assert a["value"] == pytest.approx(b["value"], abs=1e-8)
            else:
                assert a["value"] == b["value"]
        if actual["kind"] != "lines":
            for a, b in zip(actual["clouds"], expected["clouds"], strict=True):
                np.testing.assert_allclose(a["points"], b["points"], atol=1e-7, rtol=0)
        else:
            np.testing.assert_allclose(actual["chart"]["x"], expected["chart"]["x"])
            for a, b in zip(actual["chart"]["series"], expected["chart"]["series"], strict=True):
                np.testing.assert_allclose(a["values"], b["values"], atol=1e-7, rtol=0)


@pytest.mark.parametrize("model", ["mei", "ds"])
def test_real_fisheye_calibration_matches_measured_corner_receipts(model: str) -> None:
    data = json.loads((ASSETS / f"calibration_{model}.json").read_text())
    objects, left, right = observations()
    train, hold = data["train_capture_ids"], data["holdout_capture_ids"]
    assert not set(train) & set(hold)
    assert sorted(train + hold) == list(range(34))
    assert data["solver_success"]
    assert data["holdout_right_reprojection_rmse_px"] < 1
    tf = np.array(data["right_from_left"])
    left_camera = camera(model, np.array(data["parameters_left"]))
    right_camera = camera(model, np.array(data["parameters_right"]))
    residuals, hold_errors = [], []
    for i in range(34):
        pose = np.array(data["board_poses_left"][str(i)])
        xyz = objects[i] @ Rotation.from_rotvec(pose[:3]).as_matrix().T + pose[3:]
        u, v, valid = left_camera.project(xyz)
        assert valid.all()
        left_error = np.c_[u, v] - left[i]
        u, v, valid = right_camera.project(xyz @ tf[:3, :3].T + tf[:3, 3])
        assert valid.all()
        right_error = np.c_[u, v] - right[i]
        if i in train:
            residuals.extend(left_error.ravel())
            residuals.extend(right_error.ravel())
        else:
            hold_errors.extend(np.linalg.norm(right_error, axis=1))
    assert np.sqrt(np.mean(np.square(residuals)) * 2) == pytest.approx(
        data["train_reprojection_rmse_px"],
        abs=1e-8,
    )
    assert np.sqrt(np.mean(np.square(hold_errors))) == pytest.approx(
        data["holdout_right_reprojection_rmse_px"],
        abs=1e-8,
    )
