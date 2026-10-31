from dataclasses import replace

import numpy as np
from tools.evaluate_pose_bias import planes
from tools.joint_disparity_bias import residuals
from tools.known_geometry_joint import reference

from calibrex.core.geometry import SE3


def test_disparity_bias_has_original_metric_sign_and_zero_at_correct_plane():
    batch = [replace(o, depth_map=o.depth_map + .25) for o in planes()]
    raw, objective, valid = residuals(batch, reference(), .25)
    assert valid.all()
    np.testing.assert_allclose(raw, 0, atol=1e-12)
    np.testing.assert_allclose(objective, 0, atol=1e-12)
    assert np.all(residuals(batch, reference(), 0)[1] < 0)


def test_invalid_points_keep_fixed_penalty():
    bad = SE3((0, 0, -100), reference().rotation_quat_xyzw)
    raw, objective, valid = residuals(planes(), bad, 0)
    assert not valid.any()
    np.testing.assert_array_equal(raw, 1)
    np.testing.assert_array_equal(objective, 1)
