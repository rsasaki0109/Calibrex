"""Failed odometry edges cannot silently become hand-eye observations."""

import numpy as np
import pytest

from calibrex.evaluation.ins_lidar_hand_eye import InsLidarRunOptions, build_sensor_motions


def test_an_interior_failed_edge_excludes_every_crossing_but_keeps_later_motions():
    poses = [np.eye(4) for _ in range(9)]
    for index, pose in enumerate(poses):
        pose[0, 3] = float(index) + (100.0 if index >= 3 else 0.0)
    options = InsLidarRunOptions(step_frames=(1, 2, 4), block_duration_s=100.0)
    times = [float(index) for index in range(9)]
    all_motions = build_sensor_motions(times, poses, options)
    valid = np.ones(8, dtype=bool)
    valid[2] = False  # Edge 2 -> 3, interior to the 0 -> 4 motion.
    filtered = build_sensor_motions(times, poses, options, registration_valid=valid)
    expected = [m for m in all_motions if not (m.start_s <= 2 < m.end_s)]
    assert [m.motion_id for m in filtered] == [m.motion_id for m in expected]
    assert len(filtered) < len(all_motions)
    later = next(m for m in filtered if m.start_s == 4 and m.end_s == 8)
    assert later.motion[0, 3] == pytest.approx(4.0)


def test_all_valid_mask_preserves_every_motion_and_block_boundary():
    times = [float(index) for index in range(21)]
    poses = [np.eye(4) for _ in times]
    options = InsLidarRunOptions(step_frames=(1, 2, 4), block_duration_s=5.0)
    original = build_sensor_motions(times, poses, options, id_prefix="drive/", block_offset=100)
    masked = build_sensor_motions(
        times, poses, options, id_prefix="drive/", block_offset=100,
        registration_valid=[True] * 20,
    )
    assert len(original) == len(masked) > 0
    for before, after in zip(original, masked, strict=True):
        assert before.motion_id == after.motion_id
        assert before.block == after.block
        np.testing.assert_array_equal(before.motion, after.motion)
        assert int(after.start_s // 5) == int(after.end_s // 5)
    assert build_sensor_motions(times, poses, options, registration_valid=[False] * 20) == []


@pytest.mark.parametrize("mask", [[True], [True, False, True], [True, 1], [True, np.nan]])
def test_invalid_masks_fail_instead_of_silently_accepting_observations(mask):
    with pytest.raises(ValueError, match="registration_valid"):
        build_sensor_motions([0.0, 1.0, 2.0], [np.eye(4)] * 3, InsLidarRunOptions(),
                             registration_valid=mask)


def test_empty_and_single_frame_tracks_have_no_edges():
    options = InsLidarRunOptions()
    assert build_sensor_motions([], [], options, registration_valid=[]) == []
    assert build_sensor_motions([0.0], [np.eye(4)], options, registration_valid=[]) == []
    with pytest.raises(ValueError, match="equal length"):
        build_sensor_motions([0.0], [], options)


def test_registration_filter_remains_an_opt_in_development_policy():
    assert InsLidarRunOptions().exclude_failed_registrations is False
