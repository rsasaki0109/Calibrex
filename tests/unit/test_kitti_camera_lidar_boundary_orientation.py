"""Physical tangent geometry and neutral fallback for the research objective."""

import numpy as np
import pytest
from tests.unit.test_kitti_camera_lidar_orientation import CAMERA, _feature, _frame
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.evaluation import camera_lidar_edge as edge

boundary = _load("kitti_camera_lidar_boundary_orientation")


def test_line_tangents_rotate_with_geometry_and_sparse_points_stay_unknown() -> None:
    direction = np.array([1, 2, 3]) / np.sqrt(14)
    points = np.arange(12)[:, None] * 0.05 * direction + [2, 3, 4]
    tangents = boundary.boundary_tangents(points)
    np.testing.assert_allclose(np.abs(tangents @ direction), 1, atol=1e-7)
    assert not boundary.boundary_tangents(points[:4]).any()
    assert not boundary.boundary_tangents(points * 100).any()
    assert not boundary.boundary_tangents(np.ones((8, 3))).any()
    cube = np.array([[x, y, z] for x in [-0.1, 0.1] for y in [-0.1, 0.1] for z in [-0.1, 0.1]])
    assert not boundary.boundary_tangents(cube).any()


@pytest.mark.parametrize("sign", [-1, 1])
def test_tangent_must_be_perpendicular_to_image_gradient(sign: int) -> None:
    tangent = np.array([[sign, 0, 0]], dtype=np.float32)
    aligned = boundary.BoundaryObjective([_frame()], CAMERA, np.eye(4), [_feature(0, 1)], [tangent])
    wrong = boundary.BoundaryObjective([_frame()], CAMERA, np.eye(4), [_feature(1, 0)], [tangent])
    assert aligned.score(np.zeros(6)) == pytest.approx(1)
    assert wrong.score(np.zeros(6)) == pytest.approx(0)


def test_unknown_tangent_is_exactly_neutral_under_pose_changes() -> None:
    scalar = edge.EdgeObjective([_frame()], CAMERA, np.eye(4))
    objective = boundary.BoundaryObjective(
        [_frame()], CAMERA, np.eye(4), [_feature(1, 0)], [np.zeros((1, 3), dtype=np.float32)]
    )
    for theta in [np.zeros(6), np.array([1, 2, 3, 0.1, 0, 0])]:
        assert objective.score(theta) == pytest.approx(scalar.score(theta), abs=1e-12)
