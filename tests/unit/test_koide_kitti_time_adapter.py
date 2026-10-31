"""The external time convention must preserve acquisition times and geometry."""

import numpy as np
import pytest
from tools.koide_kitti_time_adapter import rebase_kitti_cloud

from calibrex.data.ros_cdr import decode_ros2_pointcloud2
from calibrex.data.ros_cdr_writer import encode_pointcloud2


def payload(times):
    points = np.array([[i, i + 1, i + 2, i + 3, t] for i, t in enumerate(times)], dtype="<f4")
    return encode_pointcloud2(
        frame_id="velo",
        timestamp_ns=1_000_000_000,
        fields=tuple((n, 4 * i, 7, 1) for i, n in enumerate(("x", "y", "z", "intensity", "time"))),
        point_step=20,
        data=points.tobytes(),
    )


@pytest.mark.parametrize("times", [[-0.049, 0.003, -0.02, 0.051], [0.02, 0.01, 0.03]])
def test_preserves_absolute_time_and_point_identity(times):
    data = payload(times)
    original = decode_ros2_pointcloud2("/points", 0, data, point_time_field="time")
    encoded, report = rebase_kitti_cloud("/points", 0, data)
    result = decode_ros2_pointcloud2("/points", 0, encoded, point_time_field="time")
    indices = result.xyz[:, 0].astype(int)
    np.testing.assert_array_equal(result.xyz, original.xyz[indices])
    np.testing.assert_array_equal(result.intensity, original.intensity[indices])
    np.testing.assert_allclose(
        (result.timestamp_ns - original.timestamp_ns) * 1e-9 + result.point_time_offsets_s,
        original.point_time_offsets_s[indices],
        atol=1e-15,
        rtol=0,
    )
    assert 0 <= result.point_time_offsets_s[0] < 1e-9
    assert np.all(np.diff(result.point_time_offsets_s) >= 0)
    assert report["max_absolute_time_change_s"] < 1e-15


@pytest.mark.parametrize("times", [[0, float("nan")], [0, 0], [0, 2]])
def test_rejects_unusable_scan_times(times):
    with pytest.raises(ValueError):
        rebase_kitti_cloud("/points", 0, payload(times))
