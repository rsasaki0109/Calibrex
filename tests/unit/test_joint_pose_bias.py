from dataclasses import replace

import numpy as np
from tools.evaluate_pose_bias import planes
from tools.joint_pose_bias import residuals, sensitivity
from tools.known_geometry_joint import reference


def test_bias_correction_has_right_sign_and_keeps_invalid_penalty():
    batch = [replace(o, depth_map=o.depth_map + 0.25) for o in planes()]
    raw, _, valid = residuals(batch, reference(), 0.25)
    np.testing.assert_allclose(raw[valid], 0.0, atol=1e-12)
    missing = [replace(o, depth_map=np.full_like(o.depth_map, np.nan)) for o in batch]
    raw, weighted, valid = residuals(missing, reference(), 0.4)
    assert not valid.any()
    np.testing.assert_array_equal(raw, 1.0)
    np.testing.assert_array_equal(weighted, 1.0)


def test_plane_bias_ambiguity_survives_robust_weighting():
    diagnostic = sensitivity(planes(), reference(), 0.0)
    for kind in ["raw", "robust"]:
        assert diagnostic[kind]["near_degenerate"]
        assert diagnostic[kind]["bias_unexplained_fraction"] < 1e-6
