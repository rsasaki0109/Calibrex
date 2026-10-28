"""Motion compensation must map static world points into the reference LiDAR."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tools.kitti_aligned_stereo_bundle import align_points


def test_rotating_translating_rig_with_lidar_lever_arm():
    times = np.array([0.0, 0.25, 0.75, 1.0])
    rotations = Rotation.from_euler("z", (times * 20)[:, None], degrees=True)
    positions = np.column_stack([times * 3, times * 0, times * 0])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_euler("y", 5, degrees=True).as_matrix()
    extrinsic[:3, 3] = [0.3, -0.2, 0.1]
    world = np.array([[10.0, 3, 2]] * len(times))
    imu = rotations.inv().apply(world - positions)
    points = (imu - extrinsic[:3, 3]) @ extrinsic[:3, :3]
    got = align_points(
        points,
        times,
        0.5,
        times[[0, -1]],
        positions[[0, -1]],
        rotations.as_quat()[[0, -1]],
        extrinsic,
    )
    expected_imu = Rotation.from_euler("z", 10, degrees=True).inv().apply(world - [1.5, 0, 0])
    expected = (expected_imu - extrinsic[:3, 3]) @ extrinsic[:3, :3]
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("query", [-0.01, 1.01])
def test_extrapolation_is_rejected(query):
    with pytest.raises(ValueError, match="coverage"):
        align_points(
            np.ones((1, 3)),
            np.array([query]),
            0.5,
            np.array([0.0, 1.0]),
            np.zeros((2, 3)),
            np.array([[0.0, 0, 0, 1]] * 2),
            np.eye(4),
        )
