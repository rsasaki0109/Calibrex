"""Metric planes constrain translation; a single plane cannot constrain all axes."""

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.solvers.stereo_translation_refiner import (
    StereoSurfaceObservation,
    refine_stereo_translation,
)


def observations(normals, rotation, translation):
    height, width = 48, 64
    v, u = np.indices((height, width))
    rays = np.stack([(u - 32) / 80, (v - 24) / 80, np.ones_like(u)], axis=-1)
    result = []
    for i, normal in enumerate(normals):
        depth = 8 / (rays @ np.array(normal))
        camera = (rays * depth[..., None])[4:-4:2, 4:-4:2].reshape(-1, 3)
        points = (camera - translation) @ rotation
        result.append(
            StereoSurfaceObservation(str(i), depth, points, (80.0, 80.0, 32.0, 24.0), 0.5)
        )
    return result


def test_recovers_translation_on_independent_planes_and_preserves_rotation():
    rotation = Rotation.from_euler("xyz", [3, -2, 4], degrees=True).as_matrix()
    true_translation = np.array([0.1, -0.05, 0.2])
    data = observations([[0, 0, 1], [0.7, 0, 1], [0, 0.7, 1]], rotation, true_translation)
    initial = np.eye(4)
    initial[:3, :3] = rotation
    initial[:3, 3] = true_translation + np.array([0.15, -0.1, 0.2])
    result = refine_stereo_translation(data, initial)
    assert result.accepted, result.as_dict()
    output = np.array(result.output_T_camera_lidar)
    np.testing.assert_allclose(output[:3, 3], true_translation, atol=1e-5)
    np.testing.assert_array_equal(output[:3, :3], rotation)


def test_single_plane_rejects_unobservable_axes_without_changing_input():
    initial = np.eye(4)
    initial[:3, 3] = [0.15, -0.1, 0.2]
    data = observations([[0, 0, 1]] * 3, np.eye(3), np.zeros(3))
    result = refine_stereo_translation(data, initial)
    assert result.status == "rank_deficient"
    assert not result.accepted
    np.testing.assert_array_equal(result.output_T_camera_lidar, initial)


def test_missing_depth_cannot_be_accepted_as_a_perfect_fit():
    data = observations([[0, 0, 1], [0.7, 0, 1], [0, 0.7, 1]], np.eye(3), np.zeros(3))
    data[1].depth_m[:] = np.nan
    result = refine_stereo_translation(data, np.eye(4))
    assert result.status == "insufficient_support"
    assert not result.accepted


def test_bound_hit_keeps_initializer_instead_of_publishing_clipped_solution():
    initial = np.eye(4)
    initial[:3, 3] = [0.8, 0.8, 0.8]
    data = observations([[0, 0, 1], [0.7, 0, 1], [0, 0.7, 1]], np.eye(3), np.zeros(3))
    result = refine_stereo_translation(data, initial)
    assert result.status == "at_bound"
    assert not result.accepted
    np.testing.assert_array_equal(result.output_T_camera_lidar, initial)
