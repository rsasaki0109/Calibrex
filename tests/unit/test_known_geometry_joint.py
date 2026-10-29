import numpy as np
from scipy.spatial.transform import Rotation
from tools.known_geometry_joint import observations, raycast, reference


def test_analytic_intersections_hit_surface_and_scale_rays():
    origin = np.zeros(3)
    rays = np.array([[-3.0, -1.0, 9.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    distance = raycast(origin, rays)
    np.testing.assert_allclose(distance[0], 1 - 1.8 / np.linalg.norm(rays[0]), atol=1e-12)
    np.testing.assert_allclose(raycast(origin, rays * 2), distance / 2, atol=1e-12)
    assert np.isfinite(distance).all() and (distance > 0).all()


def test_visible_control_and_sensor_transform_are_consistent():
    visible, full = observations("exact_z_visible"), observations("exact_z_all")
    ref = reference()
    rotation = Rotation.from_quat(ref.rotation_quat_xyzw).as_matrix()
    origins = [np.zeros(3), np.array([1.0, 0.2, 0.4]), np.array([-0.8, -0.3, 0.7])]
    for a, b, origin in zip(visible, full, origins, strict=True):
        assert 64 < len(a.lidar_points) < len(b.lidar_points)
        camera_points = a.lidar_points @ rotation.T + ref.translation_m
        np.testing.assert_allclose(raycast(origin, camera_points), 1.0, atol=1e-7)
        np.testing.assert_array_equal(a.depth_map, b.depth_map)


def test_measurement_perturbations_do_not_change_lidar():
    exact = observations("exact_z_visible")
    for condition in ["biased_z_visible", "holes_z_visible", "exact_range_visible"]:
        batch = observations(condition)
        for a, b in zip(exact, batch, strict=True):
            np.testing.assert_array_equal(a.lidar_points, b.lidar_points)
            if condition == "biased_z_visible":
                np.testing.assert_allclose(b.depth_map - a.depth_map, 0.25)
            elif condition == "holes_z_visible":
                assert np.isnan(b.depth_map).sum() == 9000
                np.testing.assert_array_equal(b.depth_map[:50], a.depth_map[:50])
            else:
                assert (b.depth_map >= a.depth_map).all()
