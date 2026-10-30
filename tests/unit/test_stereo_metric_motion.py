import numpy as np
from tools.stereo_metric_motion import metric_pnp, sample_points


def test_pnp_preserves_target_from_source_metric_direction():
    rng = np.random.default_rng(1)
    points = rng.uniform([-2, -2, 5], [2, 2, 15], (100, 3)).astype(float)
    t = np.array([0.1, -0.02, 0.03])
    target = points + t
    k = np.array([[700.0, 0, 300], [0, 700, 200], [0, 0, 1.0]])
    pixels = target[:, :2] / target[:, 2, None] * 700 + [300, 200]
    result = metric_pnp(points, pixels, k)
    assert result["status"] == "estimated"
    np.testing.assert_allclose(np.array(result["transform"])[:3, 3], t, atol=1e-5)


def test_missing_depth_and_insufficient_points_are_explicit():
    depth = np.full((5, 5), np.nan)
    k = np.eye(3)
    xyz, valid = sample_points(depth, np.array([[2.0, 2.0], [-1.0, 0.0]]), k)
    assert len(xyz) == 0 and not valid.any()
    assert metric_pnp(xyz, np.empty((0, 2)), k)["transform"] is None
