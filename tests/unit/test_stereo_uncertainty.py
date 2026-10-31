import numpy as np
from tools.evaluate_pose_bias import planes
from tools.evaluate_stereo_uncertainty import noisy_depth
from tools.joint_uncertain_pose_bias import depth_scale, residuals
from tools.known_geometry_joint import reference


def test_scale_has_quadratic_range_dependence_floor_and_fixed_invalid_penalty():
    scale = depth_scale(np.array([2.0, 10.0, 20.0, 100.0]), np.array([True, True, True, False]))
    np.testing.assert_allclose(scale, [1.0, 0.24, 0.06, 1.0])


def test_noise_is_deterministic_in_disparity_and_depth_error_grows_with_range():
    depth = np.full((100, 100), 10.0)
    near = noisy_depth(depth, 730, 0.5)
    far = noisy_depth(depth * 2, 730, 0.5)
    np.testing.assert_array_equal(near, noisy_depth(depth, 730, 0.5))
    np.testing.assert_allclose(120 / near - 12, 120 / far - 6, atol=1e-14)
    assert np.std(far - 20) > 3.5 * np.std(near - 10)


def test_clean_plane_zero_residual_and_bias_units_are_preserved():
    raw, saturated, valid = residuals(planes(), reference(), 0.0)
    assert valid.all()
    np.testing.assert_allclose(raw, 0.0, atol=1e-12)
    np.testing.assert_allclose(saturated, 0.0, atol=1e-12)
    shifted = residuals(planes(), reference(), -0.25)
    np.testing.assert_allclose(shifted[0], -0.25, atol=1e-12)
    assert np.all(np.abs(shifted[1]) < 0.25)
