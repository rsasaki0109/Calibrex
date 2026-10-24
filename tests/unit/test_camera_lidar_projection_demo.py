"""Real KITTI projection assets preserve source identity and camera conventions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from build_camera_lidar_demo import ASSETS, build_frame  # noqa: E402


def test_projection_demo_provenance() -> None:
    index = json.loads((ROOT / "docs/app/samples/camera_lidar_demo.json").read_text())
    assert index["capture_kind"] == "real"
    assert index["license"] == "CC BY-NC-SA 3.0"
    for path, digest in index["provenance"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    receipt = json.loads((ASSETS / "download-receipt.json").read_text())
    for name, original in receipt.items():
        assert original["url"].startswith("https://s3.eu-central-1.amazonaws.com/avg-kitti/")
        if not name.endswith(".bin"):
            actual = hashlib.sha256((ASSETS / "source" / name).read_bytes()).hexdigest()
            assert actual == original["sha256"]


@pytest.mark.parametrize("frame", [60, 90])
def test_camera2_projection_includes_rectification_and_camera_baseline(frame: int) -> None:
    samples = ROOT / "docs/app/samples"
    index = json.loads((samples / "camera_lidar_demo.json").read_text())
    entry = index["frames"][str(frame)]
    path = samples / entry["url"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    image_digest = hashlib.sha256((ASSETS / f"frame_{frame:04d}.jpg").read_bytes()).hexdigest()
    assert image_digest == entry["image_sha256"]
    saved = json.loads(path.read_text())
    generated = build_frame(frame)
    np.testing.assert_allclose(saved["points"], generated["points"], atol=1e-10)
    assert Image.open(ASSETS / "source" / f"{frame:010d}.png").size == (1242, 375)
    subset = np.load(ASSETS / "source" / f"points_{frame:04d}.npz")
    points = subset["velodyne"][:, :3]
    assert 1000 < len(points) <= 6000
    assert np.all(np.diff(subset["source_indices"]) > 0)
    # Compute the original KITTI homogeneous P_rect_02 * R_rect_00 * T_cam0_velo
    # independently of the generator's camera-2 coordinate decomposition.
    values = {}
    for name in ["calib_cam_to_cam.txt", "calib_velo_to_cam.txt"]:
        for line in (ASSETS / "source" / name).read_text().splitlines():
            key, _, rest = line.partition(":")
            try:
                values[key] = np.array([float(v) for v in rest.split()])
            except ValueError:
                continue
    cam0 = points @ values["R"].reshape(3, 3).T + values["T"]
    rectified = cam0 @ values["R_rect_00"].reshape(3, 3).T
    pixel = np.c_[rectified, np.ones(len(points))] @ values["P_rect_02"].reshape(3, 4).T
    expected = pixel[:, :2] / pixel[:, 2, None]
    fx, fy, cx, cy = saved["intrinsics"]
    xyz = np.array(saved["points"])
    actual = xyz[:, :2] / xyz[:, 2, None] * [fx, fy] + [cx, cy]
    np.testing.assert_allclose(actual, expected, atol=0.001, rtol=0)
