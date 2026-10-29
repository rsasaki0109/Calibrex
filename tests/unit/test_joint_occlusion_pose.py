import numpy as np
from tools.joint_occlusion_pose import occlusion_residual
from tools.known_geometry_joint import observations as original
from tools.occlusion_geometry import SCENES, observations


def test_one_sided_penalty_preserves_front_and_unsupported_points():
    residual = np.array([-2.0, -0.1, 0.0, 0.001, 0.2, 2.0, 1.0])
    valid = np.array([True] * 6 + [False])
    mapped = occlusion_residual(residual, valid)
    np.testing.assert_array_equal(mapped[:3], residual[:3])
    assert mapped[-1] == 1.0
    assert 0 < mapped[3] < mapped[4] < mapped[5] < 0.2
    assert abs(mapped[3] - 0.001) < 1e-7


def test_original_scene_reproduced_and_new_layouts_have_occlusion():
    for condition in ["exact_z_visible", "exact_z_all"]:
        for a, b in zip(original(condition), observations("original", condition), strict=True):
            np.testing.assert_array_equal(a.depth_map, b.depth_map)
            np.testing.assert_array_equal(a.lidar_points, b.lidar_points)
    for scene in SCENES:
        visible, full = observations(scene, "exact_z_visible"), observations(scene, "exact_z_all")
        for a, b in zip(visible, full, strict=True):
            assert 64 < len(a.lidar_points) < len(b.lidar_points)
            np.testing.assert_array_equal(a.depth_map, b.depth_map)
