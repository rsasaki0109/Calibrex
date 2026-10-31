"""Bounded geometric distance reward must preserve image geometry and unknown support."""
import numpy as np
import pytest
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.evaluation import camera_lidar_edge as edge

nearest = _load('kitti_camera_lidar_nearest_edge')
probe = _load('kitti_camera_lidar_profile_probe')


def test_vertical_ridge_distance_polarity_and_truncation() -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[:, 16:] = 255
    options = edge.EdgeAlignmentOptions(image_max_width=32)
    response, factor, support = nearest.prepare_nearest_edge(image, options)
    opposite, _, _ = nearest.prepare_nearest_edge(255 - image, options)
    assert factor == 1 and support['ridge_pixels'] > 0
    np.testing.assert_allclose(response, opposite)
    assert response[16, 15] == response[16, 16] == 1
    assert response[16, 18] == pytest.approx(np.exp(-0.5))
    assert response[16, 23] == 0
    assert not response.flags.writeable


def test_downsampling_matches_scalar_geometry_and_flat_input_fails() -> None:
    image = np.zeros((33, 65), dtype=np.uint8)
    image[:, 32:] = 255
    options = edge.EdgeAlignmentOptions(image_max_width=32)
    response, factor, _ = nearest.prepare_nearest_edge(image, options)
    scalar, scale = edge.image_edge_response(
        image, max_width=32, gamma=options.spill_gamma, radius=options.spill_radius_px)
    assert factor == scale == 3 and response.shape == scalar.shape
    with pytest.raises(ValueError, match='no image edges'):
        nearest.prepare_nearest_edge(np.zeros_like(image), options)


def test_objective_ablation_cannot_be_combined(tmp_path) -> None:
    with pytest.raises(ValueError, match='one objective'):
        probe.profile_snapshot(tmp_path, None, tmp_path / 'output',
                               training_only_start=True, nearest_edge=True, boundary_tangent=True)
