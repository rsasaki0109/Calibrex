import numpy as np
from tools.diagnose_stereo_geometry import geometry_stats


def test_identical_planes_have_zero_distance_and_sign_invariant_normal_agreement():
    x, y = np.meshgrid(np.arange(8) * 0.2, np.arange(8) * 0.2)
    xyz = np.column_stack([x.ravel(), y.ravel(), np.ones(64)])
    result = geometry_stats(xyz, xyz, np.ones(64, dtype=bool))
    assert result["source_populations"][0]["within_fraction"]["0.2"] == 1
    assert result["normal_pair_count"] > 0
    np.testing.assert_allclose(result["normal_angle_quantiles_deg"], 0, atol=1e-5)


def test_fov_denominator_keeps_missing_overlap_explicit():
    x, y = np.meshgrid(np.arange(8) * 0.2, np.arange(8) * 0.2)
    xyz = np.column_stack([x.ravel(), y.ravel(), np.ones(64)])
    result = geometry_stats(xyz, xyz + 10, np.zeros(64, dtype=bool))
    assert result["source_populations"][1]["count"] == 0
    assert result["source_populations"][1]["distance_quantiles_m"] is None
    assert result["normal_pair_count"] == 0
