"""Known plane geometry tests pose recovery and observable-axis rejection."""

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.solvers.stereo_disparity_pose import solve_disparity_pose
from calibrex.solvers.stereo_translation_refiner import StereoSurfaceObservation


def scene(normals):
    v, u = np.indices((80, 100))
    rays = np.stack([(u - 50) / 100, (v - 40) / 100, np.ones_like(u)], axis=-1)
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler("xyz", [2, -3, 5], degrees=True).as_matrix()
    truth[:3, 3] = [0.1, -0.05, 0.2]
    obs = []
    for i, n in enumerate(normals):
        depth = 8 / (rays @ np.array(n))
        camera = (rays * depth[..., None])[12:-12:2, 12:-12:2].reshape(-1, 3)
        points = (camera - truth[:3, 3]) @ truth[:3, :3]
        obs.append(StereoSurfaceObservation(str(i), depth, points, (100, 100, 50, 40), 0.5))
    return obs, truth


def test_recovers_all_six_axes_from_independent_planes():
    obs, truth = scene([[0, 0, 1], [0.7, 0, 1], [0, 0.7, 1]])
    initial = truth.copy()
    initial[:3, :3] = (
        Rotation.from_euler("xyz", [1, -1, 0.5], degrees=True).as_matrix() @ truth[:3, :3]
    )
    initial[:3, 3] += [0.1, -0.05, 0.05]
    fit = solve_disparity_pose(obs, initial)
    assert fit.accepted, fit.as_dict()
    np.testing.assert_allclose(fit.output_T_camera_lidar, truth, atol=1e-5)


def test_single_plane_rejects_unobservable_pose_with_exact_fallback():
    obs, truth = scene([[0, 0, 1]])
    initial = truth.copy()
    initial[2, 3] += 0.1
    fit = solve_disparity_pose(obs, initial)
    assert not fit.accepted
    assert fit.status == "rank_deficient"
    np.testing.assert_array_equal(fit.output_T_camera_lidar, initial)


def test_invalid_depth_is_not_a_zero_cost_solution():
    obs, truth = scene([[0, 0, 1]])
    obs[0].depth_m[:] = np.nan
    fit = solve_disparity_pose(obs, truth)
    assert fit.status == "insufficient_support"
    np.testing.assert_array_equal(fit.output_T_camera_lidar, truth)
