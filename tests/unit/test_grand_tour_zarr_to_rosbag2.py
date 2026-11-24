from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.data.livox_ros2 import LivoxStreamProfile, iter_livox_points
from calibrex.data.ros_cdr_writer import encode_pointcloud2
from calibrex.data.rosbag2_writer import Rosbag2Writer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.grand_tour_zarr_to_rosbag2 import (
    FIELDS,
    POINT_DTYPE,
    pack_scan,
    transform_matrix,
)


def test_transform_matrix_reads_a_grand_tour_attribute() -> None:
    attribute = {
        "rotation": {"w": -0.004419949690193552, "x": -0.7093301791430758,
                     "y": 0.7048516039326548, "z": 0.003921407292290494},
        "translation": {"x": -0.0446, "y": 0.3022, "z": -0.0125},
    }  # fmt: skip
    matrix = transform_matrix(attribute)
    expected = Rotation.from_quat([-0.70933, 0.70485, 0.00392, -0.00442]).as_matrix()
    assert np.allclose(matrix[:3, :3], expected, atol=1e-4)
    assert np.allclose(matrix[:3, 3], [-0.0446, 0.3022, -0.0125])


def test_pack_scan_drops_origin_and_nonfinite_points() -> None:
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], [4.0, 5.0, 6.0]])
    times = np.array([10.0, 10.1, 10.2, np.nan])
    packed = np.frombuffer(pack_scan(points, times), dtype=POINT_DTYPE)
    assert len(packed) == 1
    assert packed["time"][0] == 10.0 and packed["x"][0] == pytest.approx(1.0)


@pytest.mark.parametrize(("encoding", "scale"), [("absolute_s", 1.0), ("absolute_ns", 1.0e9)])
def test_converted_scans_decode_with_absolute_point_times(
    tmp_path: Path, encoding: str, scale: float
) -> None:
    stamp = 1730565871.0
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    times = (stamp + np.array([0.0, 0.05, 0.099])) * scale
    with Rosbag2Writer(tmp_path / "bag") as writer:
        writer.add_topic("/livox_points", "sensor_msgs/msg/PointCloud2")
        stamp_ns = round(stamp * 1e9)
        writer.write(
            "/livox_points", stamp_ns,
            encode_pointcloud2(frame_id="livox_lidar", timestamp_ns=stamp_ns, fields=FIELDS,
                               point_step=POINT_DTYPE.itemsize, data=pack_scan(points, times)),
        )  # fmt: skip
    profile = LivoxStreamProfile("lidar", "/livox_points", "", "time", encoding, "mps2")  # type: ignore[arg-type]
    ((time_s, xyz, offsets),) = list(iter_livox_points(tmp_path / "bag", profile))
    assert time_s == pytest.approx(stamp)
    assert np.allclose(xyz, points)
    assert offsets is not None and np.allclose(offsets, [0.0, 0.05, 0.099], atol=1e-6)
    assert json.dumps(FIELDS)  # plain data, recorded in conversion.json
