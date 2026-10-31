"""Attribution must preserve excluded support and match exact estimator scores."""
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.test_kitti_camera_lidar_orientation import CAMERA, _frame
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.evaluation import camera_lidar_edge as edge

diag = _load('kitti_camera_lidar_match_diagnostics')


def test_attribution_keeps_outside_weight_and_matches_pose_dependent_bilinear_score() -> None:
    frame = replace(_frame(), points=np.array([[0, 2, 5], [100, 0, 5]], dtype=np.float32),
                    weights=np.array([1, 3], dtype=np.float32))
    objective = edge.EdgeObjective([frame], CAMERA, np.eye(4))
    _, scores = diag.contributions(objective, np.zeros(6))
    assert scores.tolist() == [0.25, 0.0]
    varied = replace(frame, edge=np.arange(16, dtype=np.float32).reshape(4, 4))
    objective = edge.EdgeObjective([varied], CAMERA, np.eye(4))
    for theta in [np.zeros(6), np.array([3., 4, 5, 0.1, 0.2, 0])]:
        _, scores = diag.contributions(objective, theta)
        assert scores.sum() == pytest.approx(objective.score(theta), abs=1e-12)
        assert scores[1] == 0
