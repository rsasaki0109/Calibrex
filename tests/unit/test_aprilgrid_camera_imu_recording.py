"""Grid geometry and camera parsing of the AprilGrid recording tool (no OpenCV needed)."""

from __future__ import annotations

import numpy as np
import pytest
from tools.aprilgrid_camera_imu_recording import camera_model, grid_corners


def test_kalibr_aprilgrid_corners_follow_tag_size_and_spacing() -> None:
    corners = grid_corners(
        {
            "target_type": "aprilgrid",
            "tagCols": 5,
            "tagRows": 4,
            "tagSize": 0.075,
            "tagSpacing": 0.2,
        }
    )

    assert sorted(corners) == list(range(20))
    pitch = 0.075 * 1.2
    # Tag 6 is column 1, row 1; its four corners span one tag size.
    assert corners[6].min(axis=0)[:2] == pytest.approx([pitch, pitch])
    assert np.ptp(corners[6], axis=0) == pytest.approx([0.075, 0.075, 0.0])
    assert corners[19].max(axis=0)[:2] == pytest.approx([4 * pitch + 0.075, 3 * pitch + 0.075])
    with pytest.raises(ValueError, match="aprilgrid"):
        grid_corners({"target_type": "checkerboard"})


def test_camera_model_reads_kalibr_pinhole_intrinsics() -> None:
    matrix, distortion, model = camera_model(
        {
            "cam0": {
                "camera_model": "pinhole",
                "intrinsics": [278.7, 278.5, 319.8, 242.0],
                "distortion_model": "equidistant",
                "distortion_coeffs": [-0.01, 0.02, -0.01, 0.002],
            }
        },
        "cam0",
    )

    assert matrix[0, 0] == 278.7 and matrix[1, 2] == 242.0
    assert model == "equidistant" and distortion.shape == (4,)
    with pytest.raises(ValueError, match="pinhole"):
        camera_model({"cam0": {"camera_model": "omni", "intrinsics": [1, 1, 1, 1, 1]}}, "cam0")
