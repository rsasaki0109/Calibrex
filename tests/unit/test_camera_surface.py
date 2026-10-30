import numpy as np
import pytest
from tools.camera_surface import StereoGeometry, estimate
from tools.camera_surface_prediction import gated_estimate
from tools.surface_prediction_gate import gated_estimate as old


def test_analytic_camera_parity_and_changed_stereo_scale():
    geometry = StereoGeometry(240, 240, 159.5, 95.5, 120)
    depth = 5 + np.random.default_rng(1).normal(0, 0.01, (60, 60))
    for key, value in old(depth).items():
        np.testing.assert_array_equal(gated_estimate(depth, geometry)[key], value)
    custom = StereoGeometry(700, 710, 30, 29, 390)
    v, u = np.indices((60, 60))
    disparity = 40 + 0.02 * (u - 30) + 0.03 * (v - 29)
    result = estimate(390 / disparity, 21, custom)
    np.testing.assert_allclose(result["depth"], 390 / disparity[10:-10, 10:-10], atol=1e-10)
    normal = np.array([14, 21.3, 40])
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(
        result["normal"], np.broadcast_to(normal, result["normal"].shape), atol=1e-10
    )


def test_invalid_camera_geometry_is_rejected():
    with pytest.raises(ValueError):
        StereoGeometry(0, 240, 0, 0, 120)
