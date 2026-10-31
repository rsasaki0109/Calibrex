"""Validate identity tracking against known planar projection changes."""

import numpy as np
from tools.diagnose_stereo_translation import identified_correspondences, overlap, robust_cost

from calibrex.solvers.stereo_translation_refiner import (
    StereoSurfaceObservation,
    StereoTranslationOptions,
    _correspondences,
    _surface,
)


def test_pixel_shift_changes_pair_identity_with_same_lidar_points() -> None:
    depth = np.full((40, 40), 10.0)
    v, u = np.mgrid[5:30, 5:30]
    points = np.column_stack([(u.ravel() - 20) / 10, (v.ravel() - 20) / 10, np.full(u.size, 10.0)])
    obs = StereoSurfaceObservation("plane", depth, points, (100, 100, 20, 20), 0.5)
    options = StereoTranslationOptions(max_points_per_frame=100)
    surface = _surface(obs, options)
    rotation = np.eye(3)
    a, b, all_ids, selected = identified_correspondences(surface, rotation, np.zeros(3), options)
    original = _correspondences(surface, rotation, np.zeros(3), options)
    np.testing.assert_array_equal(a, original[0])
    np.testing.assert_array_equal(b, original[1])
    assert len(all_ids) == 625
    assert len(selected) == 100
    assert overlap(selected, all_ids) == 100
    shifted = identified_correspondences(surface, rotation, np.array([0.1, 0, 0]), options)
    assert overlap(all_ids, shifted[2]) == 0
    # A whole-pixel shift preserves point IDs, but changes the matched pixel.
    np.testing.assert_array_equal(all_ids // 1600, shifted[2] // 1600)
    assert robust_cost(a, b, np.zeros(3)) == 0
    assert robust_cost(a, b, np.array([0, 0, 0.1])) > 0
