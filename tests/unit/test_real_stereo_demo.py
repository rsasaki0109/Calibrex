"""Real photographed stereo receipts stay byte-bound and image-derived."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from build_real_stereo_demo import ASSETS, build_frame  # noqa: E402


def test_real_stereo_provenance_and_calibration_source() -> None:
    index = json.loads((ROOT / "docs/app/samples/fisheye_real_demo.json").read_text())
    assert index["capture_kind"] == "real"
    for path, digest in index["provenance"]["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    sources = json.loads((ASSETS / "sources.json").read_text())
    for path, receipt in sources["files"].items():
        assert hashlib.sha256((ASSETS / path).read_bytes()).hexdigest() == receipt["sha256"]
        assert "raw.githubusercontent.com/opencv/" in receipt["url"]
    calibration = json.loads((ASSETS / "calibration.json").read_text())
    upstream = (ASSETS / "source/test_fisheye.cpp.txt").read_text()
    # Reference numbers must occur in the pinned upstream stereo-calibration test.
    section = upstream.split("TEST_F(fisheyeTest, stereoCalibrate)\n")[1].split(
        "TEST_F(fisheyeTest, stereoCalibrateFixIntrinsic)"
    )[0]
    for camera in (calibration["left"], calibration["right"]):
        for value in camera["intrinsics"] + camera["distortion"]:
            assert str(value) in section
    for row in calibration["right_from_left"][:3]:
        for value in row:
            assert str(value) in section


@pytest.mark.parametrize("frame", ["000", "014", "027"])
def test_real_stereo_matches_are_reproducible_from_photographs(frame: str) -> None:
    samples = ROOT / "docs/app/samples"
    index = json.loads((samples / "fisheye_real_demo.json").read_text())
    receipt = index["frames"][frame]
    path = samples / receipt["url"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    saved = json.loads(path.read_text())
    generated = build_frame(frame)
    assert saved["capture_kind"] == "real"
    assert saved["depth_truth"] is None
    assert "median_relative_depth_error" not in saved
    assert saved["matches"] > 100
    assert saved["image_size"] == [640, 400]
    assert saved["matches"] == generated["matches"]
    for key, size in [("raw", (1280, 400)), ("rectified", (1280, 400)), ("depth", (640, 400))]:
        image = Image.open(io.BytesIO(base64.b64decode(saved[key].split(",", 1)[1])))
        assert image.size == size
        assert image.mode == "RGB"
        np.testing.assert_array_equal(
            np.asarray(image),
            np.asarray(Image.open(io.BytesIO(base64.b64decode(generated[key].split(",", 1)[1])))),
        )
    for point, actual in zip(saved["correspondences"], generated["correspondences"], strict=True):
        assert point["left"] == pytest.approx(actual["left"])
        assert point["right"] == pytest.approx(actual["right"])
        assert point["depth_m"] == pytest.approx(actual["depth_m"])
        assert point["left"][0] - point["right"][0] == pytest.approx(point["disparity_px"])
        assert point["depth_m"] == pytest.approx(
            saved["focal_px"] * saved["baseline_m"] / point["disparity_px"]
        )
