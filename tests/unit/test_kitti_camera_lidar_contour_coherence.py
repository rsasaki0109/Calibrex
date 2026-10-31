"""Graph coherence penalizes contour switching without renormalizing support."""
import numpy as np
from tests.unit.test_kitti_camera_lidar_orientation import CAMERA, _frame
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.evaluation import camera_lidar_edge as edge

coherence = _load('kitti_camera_lidar_contour_coherence')


def test_mutual_graph_keeps_nearby_points_and_isolates_distant_points() -> None:
    points = np.array([[0., 0, 1], [0, 0.1, 1], [0, 0.2, 1], [10, 10, 10]])
    links = coherence.neighbor_links(points)
    assert links.tolist() == [[0, 1], [0, 2], [1, 2]]
    np.testing.assert_allclose(coherence.compatibility(np.array([1, 1, 2, 0]), links),
                               [0.5, 0.5, 0, 1])
    np.testing.assert_allclose(coherence.compatibility(np.array([0, 0, 0, 0]), links),
                               [0, 0, 0, 1])


def test_connected_contour_identity_and_distance_limit() -> None:
    response = np.zeros((20, 30))
    response[2:5, 2] = 1
    response[15:18, 25] = 1
    labels, count = coherence.contour_map(response)
    assert count == 2
    assert labels[3, 2] == labels[3, 3] != 0
    assert labels[3, 2] != labels[16, 25] != 0
    assert labels[10, 15] == 0


def test_isolated_point_keeps_exact_original_reward() -> None:
    objective = edge.EdgeObjective([_frame()], CAMERA, np.eye(4))
    old, new, _ = coherence.score_frame(objective, np.zeros(6), np.ones((4, 4), dtype=int),
                                       np.empty((0, 2), dtype=int))
    assert old == new == 1
