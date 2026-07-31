from __future__ import annotations

import pytest

from calibrex.data.kitti_camera_lidar_problem import uniform_kitti_frame_ids


def test_uniform_frame_ids_freeze_borer_kitti_selection() -> None:
    frame_ids = uniform_kitti_frame_ids(2762, 25)

    assert len(frame_ids) == 25
    assert frame_ids[:3] == ["0000000000", "0000000115", "0000000230"]
    assert frame_ids[12] == "0000001381"
    assert frame_ids[-1] == "0000002761"
    assert len(set(frame_ids)) == len(frame_ids)


def test_uniform_frame_ids_reject_oversampling() -> None:
    with pytest.raises(ValueError, match="cannot select"):
        uniform_kitti_frame_ids(2, 3)
