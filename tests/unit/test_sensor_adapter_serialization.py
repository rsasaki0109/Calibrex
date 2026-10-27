"""Sensor adapters must honor declared serialization before decoding CDR bytes."""
import sqlite3
import struct

import numpy as np
import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data import livox_ros2, ros2_camera_imu, rtk_slam
from calibrex.data.ros_cdr_writer import encode_image, encode_imu, encode_pointcloud2
from calibrex.data.rosbag2_writer import Rosbag2Writer


def make_bag(tmp_path, encoding):
    bag = tmp_path / 'sensors'
    with Rosbag2Writer(bag) as writer:
        writer.add_topic('/imu', 'sensor_msgs/msg/Imu')
        writer.add_topic('/image', 'sensor_msgs/msg/Image')
        writer.add_topic('/points', 'sensor_msgs/msg/PointCloud2')
        for stamp in (1_000_000_000, 2_000_000_000):
            writer.write('/imu', stamp, encode_imu(
                frame_id='imu', timestamp_ns=stamp, angular_velocity=(1.0, 2.0, 3.0),
                linear_acceleration=(0.0, 0.0, 9.8)))
            writer.write('/image', stamp, encode_image(
                frame_id='camera', timestamp_ns=stamp, height=1, width=2,
                encoding='mono8', step=2, data=b'\x12\x34'))
            writer.write('/points', stamp, encode_pointcloud2(
                frame_id='lidar', timestamp_ns=stamp,
                fields=[('x', 0, 7, 1), ('y', 4, 7, 1), ('z', 8, 7, 1)],
                point_step=12, data=struct.pack('<fff', 1.0, 2.0, 3.0)))
    with sqlite3.connect(next(bag.glob('*.db3'))) as db:
        db.execute('UPDATE topics SET serialization_format = ?', (encoding,))
    return bag


@pytest.mark.parametrize('adapter', ['camera_imu', 'image', 'livox_imu', 'livox_points', 'rtk'])
@pytest.mark.parametrize('encoding', ['cdr', 'json', ''])
def test_sensor_adapters_validate_serialization(tmp_path, adapter, encoding):
    bag = make_bag(tmp_path, encoding)
    profile = livox_ros2.LivoxStreamProfile(
        'test', '/points', '/imu', None, 'offset_s', 'mps2')

    def read():
        if adapter == 'camera_imu':
            return ros2_camera_imu.load_ros2_imu(bag, '/imu').gyro_rps
        if adapter == 'livox_imu':
            return livox_ros2.load_livox_imu(bag, profile).gyro_rps
        if adapter == 'image':
            images = ros2_camera_imu.iter_gray_images(bag, '/image')
            return np.asarray([pixels for _, pixels in images])
        if adapter == 'livox_points':
            return np.asarray([xyz for _, xyz, _ in livox_ros2.iter_livox_points(bag, profile)])
        return np.asarray([xyz for _, xyz, _ in rtk_slam.iter_livox_scans(
            bag, '/points', point_time_field=None)])

    if encoding != 'cdr':
        with pytest.raises(DatasetError, match='unsupported serialization format'):
            read()
    else:
        result = read()
        expected = ([[18, 52]],) * 2 if adapter == 'image' else (
            [[1.0, 2.0, 3.0]] * 2 if adapter.endswith('imu') else [[[1.0, 2.0, 3.0]]] * 2)
        np.testing.assert_array_equal(result, expected)
