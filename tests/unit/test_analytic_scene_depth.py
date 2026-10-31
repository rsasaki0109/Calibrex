import numpy as np
from tools.analytic_scene_depth import AnalyticObservation, depth_residual
from tools.known_geometry_joint import reference
from tools.occlusion_geometry import observations

from calibrex.core.geometry import SE3


def test_oracle_sample_matches_continuous_visible_hits_and_bias_sign():
    origins = [(0.0, 0.0, 0.0), (1.0, 0.2, 0.4), (-0.8, -0.3, 0.7)]
    for source, origin in zip(observations("original", "exact_z_visible"), origins, strict=True):
        obs = AnalyticObservation(
            source.frame_id,
            source.depth_map + 0.25,
            source.lidar_points,
            source.camera,
            scene="original",
            camera_origin=origin,
            measurement_offset_m=0.25,
        )
        residual, valid = depth_residual(obs, reference())
        assert valid.all()
        np.testing.assert_allclose(residual, -0.25, atol=1e-7)


def test_all_returns_keep_camera_occlusions_and_invalid_penalty():
    source = observations("original", "exact_z_all")[0]
    obs = AnalyticObservation(source.frame_id, source.depth_map, source.lidar_points, source.camera)
    residual, valid = depth_residual(obs, reference())
    assert valid.all()
    assert (residual > 0.2).any()
    _, bad = depth_residual(obs, SE3((0.0, 0.0, -100.0), (0.0, 0.0, 0.0, 1.0)))
    assert not bad.any()
    penalty, _ = depth_residual(obs, SE3((0.0, 0.0, -100.0), (0.0, 0.0, 0.0, 1.0)))
    np.testing.assert_array_equal(penalty, 1.0)
