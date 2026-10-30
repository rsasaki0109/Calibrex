import numpy as np
from tools.evaluate_real_metric_transfer import as_pose, select_points
from tools.joint_multistart_control import matrix


def test_point_selection_preserves_small_cloud_and_uniform_file_order():
    small = np.arange(30).reshape(10, 3)
    np.testing.assert_array_equal(select_points(small), small)
    large = np.arange(30000).reshape(10000, 3)
    picked = select_points(large)
    assert len(picked) == 4096
    np.testing.assert_array_equal(picked[[0, -1]], large[[0, -1]])
    assert len(np.unique(picked[:, 0])) == 4096
    np.testing.assert_array_equal(select_points(large + 7), picked + 7)


def test_pose_conversion_preserves_reference_frame():
    m = np.eye(4)
    m[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    m[:3, 3] = [0.1, -0.2, 0.3]
    np.testing.assert_allclose(matrix(as_pose(m.tolist())), m, atol=1e-15)
