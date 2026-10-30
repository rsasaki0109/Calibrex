import numpy as np
from tools.coherent_surface import estimate
from tools.known_geometry_joint import CAMERA


def test_affine_disparity_recovers_same_plane_across_support_sizes():
    v, u = np.indices((192, 320))
    disparity = 12 + 0.01 * (u - CAMERA.cx) + 0.02 * (v - CAMERA.cy)
    truth = np.array([0.01 * CAMERA.fx, 0.02 * CAMERA.fy, 12.0])
    truth = truth / np.linalg.norm(truth)
    for window in [5, 11, 21]:
        result = estimate(120 / disparity, window)
        np.testing.assert_allclose(
            result["normal"], np.broadcast_to(truth, result["normal"].shape), atol=1e-11
        )
        np.testing.assert_allclose(result["depth"], 120 / disparity[10:-10, 10:-10], atol=1e-11)
        assert result["rms"].max() < 1e-5


def test_larger_support_reduces_front_plane_bound_and_missing_patches_fail():
    depth = np.full((60, 60), 5.0)
    assert np.median(estimate(depth, 21)["radius"]) < np.median(estimate(depth, 11)["radius"])
    assert not estimate(np.full((60, 60), np.nan), 21)["accepted"].any()
