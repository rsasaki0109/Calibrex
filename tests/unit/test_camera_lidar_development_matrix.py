"""Failed, invalid and threshold-boundary results stay failures."""

import numpy as np
from tools.run_camera_lidar_development_matrix import pose_metrics


def test_completion_required_even_with_perfect_pose():
    assert pose_metrics(np.eye(4), np.eye(4), True)["recovered"]
    assert not pose_metrics(np.eye(4), np.eye(4), False)["recovered"]


def test_strict_translation_threshold():
    pose = np.eye(4)
    pose[0, 3] = 0.20
    assert not pose_metrics(pose, np.eye(4), True)["recovered"]


def test_invalid_outputs_do_not_get_projected_to_valid_rotations():
    for pose in [None, np.full((4, 4), np.nan), np.zeros((4, 4))]:
        result = pose_metrics(pose, np.eye(4), True)
        assert not result["recovered"]
        assert not result["valid_output"]
