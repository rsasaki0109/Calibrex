from dataclasses import replace

import numpy as np
from tools.edge_depth_sampling import depth_residual
from tools.joint_metric_pose import depth_residual as bilinear
from tools.known_geometry_joint import CAMERA

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation


def observation(depth):
    u = np.array([100.25, 100.75])
    v = np.array([80.25, 80.75])
    z = np.array([5.0, 10.0])
    xyz = np.column_stack([(u - CAMERA.cx) / CAMERA.fx * z, (v - CAMERA.cy) / CAMERA.fy * z, z])
    return DepthToDepthObservation("edge", depth, xyz, CAMERA, "metric_z")


def test_edge_does_not_blend_foreground_and_background():
    depth = np.full((192, 320), 5.0)
    depth[:, 101:] = 10.0
    obs = observation(depth)
    r, valid = depth_residual(obs, SE3.identity())
    assert valid.all()
    np.testing.assert_allclose(r, 0, atol=1e-12)
    assert np.max(np.abs(bilinear(obs, SE3.identity())[0])) > 1
    shifted = replace(obs, depth_map=depth + 0.25)
    np.testing.assert_allclose(depth_residual(shifted, SE3.identity())[0], -0.25, atol=1e-12)


def test_smooth_patch_and_metric_range_match_bilinear():
    v, u = np.indices((192, 320))
    depth = 5 + 0.01 * u + 0.02 * v
    obs = observation(depth)
    expected = bilinear(obs, SE3.identity())
    np.testing.assert_allclose(depth_residual(obs, SE3.identity())[0], expected[0])
    ranges = depth * np.sqrt(
        1 + ((u - CAMERA.cx) / CAMERA.fx) ** 2 + ((v - CAMERA.cy) / CAMERA.fy) ** 2
    )
    ranged = replace(obs, depth_map=ranges, depth_scale_convention="metric_range")
    np.testing.assert_allclose(depth_residual(ranged, SE3.identity())[0], expected[0])


def test_invalid_neighbors_keep_penalty_and_support():
    depth = np.full((192, 320), 5.0)
    depth[80, 100] = np.nan
    r, valid = depth_residual(observation(depth), SE3.identity())
    assert not valid.any()
    np.testing.assert_array_equal(r, 1.0)
