"""CDR-only bag transforms must reject incompatible inputs before overwriting."""
import sqlite3

import numpy as np
import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr_writer import encode_imu
from calibrex.data.rosbag2 import iter_messages, list_rosbag2_connections
from calibrex.data.rosbag2_imu_rotate import rotate_imu_in_bag
from calibrex.data.rosbag2_time_shift import shift_stamps_in_bag
from calibrex.data.rosbag2_velodyne import convert_velodyne_bag
from calibrex.data.rosbag2_writer import Rosbag2Writer
from calibrex.data.velodyne_packets import VELODYNE_SCAN_TYPE


def source_bag(tmp_path, topic, encoding):
    source = tmp_path / 'source'
    payload = encode_imu(frame_id='imu', timestamp_ns=10,
                         angular_velocity=(1.0, 2.0, 3.0),
                         linear_acceleration=(0.0, 0.0, 9.8))
    with Rosbag2Writer(source) as writer:
        writer.add_topic('/imu', 'sensor_msgs/msg/Imu')
        writer.add_topic('/packets', VELODYNE_SCAN_TYPE)
        writer.add_topic('/other', 'custom/msg/Other')
        writer.write('/imu', 10, payload)
        writer.write('/packets', 10, b'packet bytes')
        writer.write('/other', 10, b'other bytes')
    with sqlite3.connect(next(source.glob('*.db3'))) as db:
        db.execute('UPDATE topics SET serialization_format = ? WHERE name = ?', (encoding, topic))
    return source


@pytest.mark.parametrize('operation', ['shift', 'rotate', 'velodyne'])
@pytest.mark.parametrize('topic', ['/imu', '/packets', '/other'])
@pytest.mark.parametrize('existing', [False, True])
def test_transforms_reject_non_cdr_without_touching_destination(
    tmp_path, operation, topic, existing,
):
    source = source_bag(tmp_path, topic, 'json')
    destination = tmp_path / 'output'
    before = {'old.db3': b'original bag', 'metadata.yaml': b'original metadata'}
    if existing:
        destination.mkdir()
        for name, content in before.items():
            (destination / name).write_bytes(content)
    with pytest.raises(DatasetError, match='unsupported serialization format') as error:
        if operation == 'shift':
            shift_stamps_in_bag(source, destination, 0.1, topics=['/imu'], overwrite=True)
        elif operation == 'rotate':
            rotate_imu_in_bag(source, destination, np.eye(3), overwrite=True)
        else:
            convert_velodyne_bag(source, destination, overwrite=True)
    assert topic in str(error.value)
    if existing:
        assert {p.name: p.read_bytes() for p in destination.iterdir()} == before
    else:
        assert not destination.exists()


def test_velodyne_conversion_allows_explicitly_dropped_non_cdr_topics(tmp_path):
    source = source_bag(tmp_path, '/packets', 'json')
    destination = tmp_path / 'output'
    result = convert_velodyne_bag(source, destination, drop_topics=['/packets'])
    assert result.messages_dropped == 1
    assert result.messages_copied == 2
    assert {c.topic for c, _ in list_rosbag2_connections(destination)} == {'/imu', '/other'}
    original = {c.topic: payload for c, _, payload in iter_messages(source)}
    assert all(payload == original[c.topic] for c, _, payload in iter_messages(destination))
