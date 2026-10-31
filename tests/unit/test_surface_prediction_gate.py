import numpy as np
from scipy.stats import chi2
from tools.coherent_surface import estimate
from tools.surface_prediction_gate import gated_estimate, predictive_statistic


def test_exact_plane_prediction_and_curvature_discrimination():
    v, u = np.indices((60, 60))
    d = 20 + 0.01 * u + 0.02 * v
    np.testing.assert_allclose(predictive_statistic(120 / d), 0, atol=1e-6)
    curved = 20 + 0.02 * (u - 30) ** 2
    assert predictive_statistic(120 / curved)[20, 20] > chi2.ppf(0.99, 220)
    assert not gated_estimate(np.full((60, 60), np.nan))["accepted"].any()


def test_prediction_gate_only_removes_points_and_preserves_estimates():
    depth = np.full((60, 60), 5.0)
    base = estimate(depth, 21)
    gated = gated_estimate(depth)
    assert np.all(~gated["accepted"] | base["accepted"])
    for k in ["depth", "normal", "radius"]:
        np.testing.assert_array_equal(base[k], gated[k])


def test_gaussian_predictive_coverage_is_near_declared_acceptance():
    from tools.surface_prediction_gate import plane_validation

    assert 0.98 <= plane_validation()["acceptance_fraction"] <= 0.997
