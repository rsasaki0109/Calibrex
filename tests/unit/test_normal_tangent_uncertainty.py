import numpy as np
from scipy.stats import chi2
from tools.known_geometry_joint import CAMERA
from tools.normal_tangent_uncertainty import cone_radius, normal_cone


def test_front_plane_matches_analytic_tangent_variance():
    coefficients = np.array([[0.0, 0.0, 120.0]])
    result = cone_radius(coefficients, np.array([CAMERA.cx]), np.array([CAMERA.cy]))
    expected = np.degrees(np.sqrt(chi2.ppf(0.95, 2) * (0.25 / 50) * CAMERA.fx**2 / 120**2))
    np.testing.assert_allclose(result, [expected])


def test_invalid_patch_has_explicit_uninformative_radius():
    np.testing.assert_array_equal(normal_cone(np.full((12, 12), np.nan)), 90.0)
