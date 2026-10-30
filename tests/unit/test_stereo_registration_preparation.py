import numpy as np
from tools.prepare_stereo_registration import backproject, voxel_first


def test_backprojection_preserves_camera_rays_and_rejects_missing_depth():
    depth = np.array([[2.0, np.nan], [0.0, 4.0]])
    xyz = backproject(depth, [2.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(xyz, [[0, 0, 2], [2, 2, 4]])


def test_voxel_selection_preserves_first_point_without_pose_transform():
    points = np.array([[0.01, 0.02, 1], [0.02, 0.03, 1], [1.0, 0, 1], [np.nan, 0, 0], [90, 0, 0]])
    np.testing.assert_array_equal(voxel_first(points), points[[0, 2]])
