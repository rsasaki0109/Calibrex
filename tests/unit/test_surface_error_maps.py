import numpy as np
from tools.map_surface_errors import boundary_band, summarize


def test_boundary_band_detects_depth_jump_without_marking_constant_interior():
    flat = np.full((60, 60), 5.0)
    assert not boundary_band(flat, 120).any()
    flat[:, 30:] = 20.0
    band = boundary_band(flat, 120)
    assert band[:, 20].all() and not band[:, 0].any()


def test_region_summary_keeps_distance_and_normal_denominators_separate():
    mask = np.ones((2, 2), dtype=bool)
    supported = np.array([[True, True], [False, False]])
    d = np.array([[0.2, 0.4], [2.0, 2.0]])
    a = np.array([[0.0, 30.0], [np.nan, np.nan]])
    result = summarize(mask, supported, d, a)
    assert result["selected"] == 4 and result["normal_pairs"] == 2
    assert result["distance_failure_fraction"] == 0.5 and result["normal_failure_fraction"] == 0.5
