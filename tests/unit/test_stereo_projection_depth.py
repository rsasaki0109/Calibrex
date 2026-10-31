"""Known 3D projections expose the non-horizontal-baseline depth approximation."""

import numpy as np
import pytest
from tools.kitti_aligned_stereo_bundle import depth_from_disparity


@pytest.mark.parametrize("delta", [[-0.54, 0, 0], [-0.54, 0.004, -0.002]])
def test_recovers_left_optical_z_with_nonzero_camera_origins(delta):
    k = np.array([[700.0, 0, 600], [0, 700, 180], [0, 0, 1]])
    left = np.column_stack((k, k @ np.array([0.06, -0.001, 0.005])))
    right = left.copy()
    right[:, 3] += k @ delta
    _, u = np.indices((3, 1200))
    z = 5 + u / 20
    x = (u - k[0, 2]) * z / k[0, 0]
    ur = k[0, 0] * (x + delta[0]) / (z + delta[2]) + k[0, 2]
    recovered = depth_from_disparity(u - ur, left, right)
    np.testing.assert_allclose(recovered, z, atol=1e-10)


def test_invalid_disparity_stays_invalid():
    k = np.eye(3)
    left = np.column_stack((k, np.zeros(3)))
    right = np.column_stack((k, [-1, 0, 0]))
    result = depth_from_disparity(np.array([[0.0, -1, np.nan, np.inf]]), left, right)
    assert np.isnan(result).all()
