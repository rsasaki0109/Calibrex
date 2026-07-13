"""Public TUM RGB-D frontend evidence for the multi-capture joint series."""

from pathlib import Path

import pytest

from calibrex.data.tum_rgbd import (
    associate_depth_groundtruth,
    read_groundtruth,
    read_image_index,
    read_tum_depth_png,
    sample_tum_depth_points,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TUM_ROOT = _REPO_ROOT / "data" / "public" / "rgbd_dataset_freiburg1_xyz"

requires_tum_xyz = pytest.mark.skipif(
    not (_TUM_ROOT / "depth.txt").exists()
    or not (_TUM_ROOT / "groundtruth.txt").exists(),
    reason=(
        "TUM fr1/xyz is not downloaded; run "
        "tools/download_public_dataset.py tum_rgbd_freiburg1_xyz"
    ),
)


@requires_tum_xyz
def test_public_tum_depth_and_groundtruth_frontend() -> None:
    depth = read_image_index(_TUM_ROOT / "depth.txt")
    trajectory = read_groundtruth(_TUM_ROOT / "groundtruth.txt")
    paired = associate_depth_groundtruth(depth, trajectory)

    assert len(depth) == 798
    assert len(trajectory) == 3000
    assert len(paired) == 796
    assert max(entry.absolute_time_delta_sec for entry in paired) < 0.011
    image = read_tum_depth_png(_TUM_ROOT / paired[0].depth.path)
    assert (image.width, image.height) == (640, 480)
    points = sample_tum_depth_points(image, max_points=1000)
    assert len(points) >= 500
    assert min(point[2] for point in points) >= 0.2
    assert max(point[2] for point in points) <= 5.0
