import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tools.diagnose_camera_motion_proxy import camera_motion, interpolate


def test_large_timestamp_interpolation_and_no_extrapolation():
    start = 1700000000000000000
    times = np.array([start, start + 100000000], dtype=np.int64)
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[1, 0, 3] = 2
    result = interpolate(times, poses, start + 50000000)
    assert result[0, 3] == 1
    np.testing.assert_array_equal(interpolate(times, poses, start), poses[0])
    with pytest.raises(ValueError):
        interpolate(times, poses, start - 1)


def test_camera_relative_direction_and_lever_arm_rotation():
    a = np.eye(4)
    b = np.eye(4)
    b[0, 3] = 0.1
    np.testing.assert_allclose(camera_motion(a, b, np.eye(4))[:3, 3], [-0.1, 0, 0])
    b = np.eye(4)
    b[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    ci = np.eye(4)
    ci[0, 3] = 1
    np.testing.assert_allclose(camera_motion(a, b, ci)[:3, 3], [1, 1, 0], atol=1e-15)
