"""Check inputs must reject incompatible serialization before decoding sensor bytes."""
import sqlite3

import pytest

from calibrex.check import estimators, roles, tf_sources, vehicle_inputs
from calibrex.core.exceptions import DatasetError
from calibrex.data.navsatfix_track import read_navsatfix_track
from calibrex.data.rosbag2 import list_rosbag2_connections
from calibrex.data.rosbag2_writer import Rosbag2Writer


def make_bag(tmp_path, message_type, topic='/sensor'):
    bag = tmp_path / 'input'
    with Rosbag2Writer(bag) as writer:
        writer.add_topic(topic, message_type)
        writer.write(topic, 1, b'not CDR')
    with sqlite3.connect(next(bag.glob('*.db3'))) as db:
        db.execute("UPDATE topics SET serialization_format = 'json'")
    return bag


@pytest.mark.parametrize('reader,message_type', [
    (read_navsatfix_track, 'sensor_msgs/msg/NavSatFix'),
    (vehicle_inputs.read_ins_track, 'nav_msgs/msg/Odometry'),
    (vehicle_inputs.read_lidar_odometry, 'sensor_msgs/msg/PointCloud2'),
    (estimators.detect_point_time, 'sensor_msgs/msg/PointCloud2'),
    (estimators.camera_model_from_camera_info, 'sensor_msgs/msg/CameraInfo'),
])
def test_diagnostic_readers_reject_non_cdr(tmp_path, reader, message_type):
    bag = make_bag(tmp_path, message_type)
    with pytest.raises(DatasetError, match="unsupported serialization format 'json'"):
        reader(bag, '/sensor')


@pytest.mark.parametrize('message_type', [
    'nav_msgs/msg/Odometry', 'geometry_msgs/msg/TwistStamped',
    'geometry_msgs/msg/TwistWithCovarianceStamped',
])
def test_vehicle_twist_readers_reject_non_cdr(tmp_path, message_type):
    bag = make_bag(tmp_path, message_type)
    with pytest.raises(DatasetError, match='unsupported serialization format'):
        vehicle_inputs.read_twist_track(bag, '/sensor', message_type)


def test_tf_source_rejects_non_cdr(tmp_path):
    bag = make_bag(tmp_path, 'tf2_msgs/msg/TFMessage', '/tf_static')
    with pytest.raises(DatasetError, match='unsupported serialization format'):
        tf_sources.load_bag_tf_static(bag)


def test_role_probes_mark_non_cdr_cloud_unreadable(tmp_path):
    bag = make_bag(tmp_path, 'sensor_msgs/msg/PointCloud2')
    records = roles.classify_topics(list_rosbag2_connections(bag))
    assert roles.read_header_frames(bag, records) == {'/sensor': None}
    flagged = roles.flag_unreadable_pointclouds(bag, records)
    assert 'unsupported serialization format' in flagged[0].ignored_reason
