"""Neighbor geometry rejects collapse, reversal and distortion while preserving rigid offsets."""
import numpy as np
from tests.unit.test_score_hilti_camera_camera import _load

geometry = _load('kitti_camera_lidar_match_geometry')


def test_common_offset_is_neutral_but_collapsed_or_reversed_matches_are_penalized() -> None:
    points = np.array([[0., 0], [0, 8], [20, 20]])
    labels = np.array([1, 1, 0])
    links = np.array([[0, 1]])
    np.testing.assert_allclose(geometry.geometry_factors(points, points + 3, labels, links), 1)
    collapsed = geometry.geometry_factors(points, np.zeros_like(points), labels, links)
    reversed_score = geometry.geometry_factors(points, -points, labels, links)
    assert 0 < reversed_score[0] < collapsed[0] < 0.1
    assert collapsed[2] == reversed_score[2] == 1


def test_unknown_and_distinct_components_do_not_receive_credit() -> None:
    points = np.array([[0., 0], [0, 2]])
    links = np.array([[0, 1]])
    for labels in [np.array([1, 2]), np.array([0, 0])]:
        np.testing.assert_array_equal(geometry.geometry_factors(points, points, labels, links), 0)


def test_nearest_coordinate_map_points_to_ridge() -> None:
    response = np.zeros((12, 12))
    response[2:10, 5] = 1
    labels, _, coordinates = geometry.geometry_map(response)
    assert labels[4, 6] > 0
    np.testing.assert_array_equal(coordinates[4, 6], [5, 4])
