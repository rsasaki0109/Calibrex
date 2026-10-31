import numpy as np
from tools.diagnose_analytic_stopping import bias_gradient, soft_cost
from tools.joint_occlusion_pose import occlusion_residual


def test_exact_bias_gradient_matches_finite_difference():
    raw = np.array([-0.4, -0.02, 0.03, 0.25, 2.0, 1.0])
    valid = np.array([True] * 5 + [False])
    h = 1e-6
    costs = [soft_cost(occlusion_residual(raw + sign * h * valid, valid)) for sign in [1, -1]]
    numeric = (costs[0] - costs[1]) / (2 * h)
    exact = bias_gradient(raw, occlusion_residual(raw, valid), valid)
    np.testing.assert_allclose(exact, numeric, rtol=1e-8, atol=1e-10)


def test_cost_matches_soft_l1_and_zero_is_stable():
    values = np.array([0.0, 1e-12, 0.1, 2.0])
    actual = soft_cost(values)
    expected = np.sum(0.01 * (np.sqrt(1 + (values / 0.1) ** 2) - 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    assert soft_cost(np.array([1e-12])) > 0
