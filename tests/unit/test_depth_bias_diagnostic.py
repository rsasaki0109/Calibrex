from dataclasses import replace

import numpy as np
from tools.evaluate_depth_bias import local_ambiguity
from tools.joint_metric_pose import depth_residual

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
)


def plane():
    camera = DepthToDepthCameraModel(width=64, height=48, fx=50.0, fy=50.0, cx=31.5, cy=23.5)
    v, u = np.meshgrid(np.arange(8, 40, 2), np.arange(8, 56, 2), indexing="ij")
    points = np.stack([(u - 31.5) * 0.2, (v - 23.5) * 0.2, np.full_like(u, 10.0)], -1).reshape(
        -1, 3
    )
    return DepthToDepthObservation("plane", np.full((48, 64), 10.0), points, camera)


def test_frontoparallel_plane_exposes_bias_translation_ambiguity():
    info = local_ambiguity([plane()], SE3.identity())
    assert info["available"] and info["near_confounded"]
    assert info["bias_unexplained_fraction"] < 1e-8
    assert info["smallest_to_largest_singular_ratio"] < 1e-8


def test_depth_bias_sign_without_modifying_points():
    obs = plane()
    changed = replace(obs, depth_map=obs.depth_map + 0.25)
    residual, valid = depth_residual(changed, SE3.identity())
    assert valid.all()
    np.testing.assert_allclose(residual, -0.25)
    np.testing.assert_array_equal(obs.depth_map, 10.0)
    np.testing.assert_array_equal(changed.lidar_points, obs.lidar_points)
