import numpy as np
from tools.known_geometry_joint import CAMERA
from tools.normal_uncertainty import cone_radius, normal_cone


def test_coefficient_confidence_covers_exact_plane_monte_carlo():
    rng = np.random.default_rng(20261030)
    truth = np.array([0.3, -0.1, 24.0])
    samples = truth + rng.normal(size=(20000, 3)) * np.sqrt([0.25 / 50, 0.25 / 50, 0.25 / 25])
    radius = cone_radius(samples, np.full(20000, CAMERA.cx), np.full(20000, CAMERA.cy))
    mapping = np.diag([CAMERA.fx, CAMERA.fy, 1.0])
    normals = samples @ mapping.T
    normal = truth @ mapping.T
    cosine = np.abs(normals @ normal) / (np.linalg.norm(normals, axis=1) * np.linalg.norm(normal))
    angle = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
    assert (angle <= radius).mean() >= 0.95


def test_far_front_plane_cannot_certify_precise_normal_and_invalid_is_explicit():
    assert np.all(normal_cone(np.full((12, 12), 20.0)) == 90)
    assert np.all(normal_cone(np.full((12, 12), np.nan)) == 90)
