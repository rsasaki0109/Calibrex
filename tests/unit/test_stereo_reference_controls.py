"""Check that diagnostic residuals use camera optical z with the documented sign."""

import numpy as np
from tools.diagnose_stereo_reference_controls import surface_check

from calibrex.solvers.stereo_translation_refiner import (
    StereoSurfaceObservation,
    StereoTranslationOptions,
)


def test_reference_plane_residuals_have_metric_scale_and_lidar_minus_stereo_sign():
    v, u = np.mgrid[5:25, 5:25]
    points = np.column_stack(((u.ravel() - 15) / 10, (v.ravel() - 15) / 10, np.full(u.size, 10.0)))
    obs = StereoSurfaceObservation(
        "plane", np.full((30, 30), 10.0), points, (100, 100, 15, 15), 0.5
    )
    pose = np.eye(4)
    pose[2, 3] = 0.2
    check = surface_check(obs, pose, StereoTranslationOptions(sampling="point_priority"))
    assert check.selected_count == 400
    np.testing.assert_allclose(check.signed_z_median_m, 0.2)
    np.testing.assert_allclose(check.absolute_z_p90_m, 0.2)
    np.testing.assert_allclose(check.absolute_plane_median_m, 0.2)
    pose[2, 3] = -0.2
    np.testing.assert_allclose(
        surface_check(obs, pose, StereoTranslationOptions()).signed_z_median_m, -0.2
    )
