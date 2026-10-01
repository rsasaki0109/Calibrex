"""Round trips of the ROS-free rosbag2 writer through the existing reader and decoders."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import decode_ros2_header_frame_id
from calibrex.data.ros_cdr_writer import (
    POINT_FIELD_FLOAT32,
    CdrWriter,
    encode_imu,
    encode_navsatfix,
    encode_odometry,
    encode_pointcloud2,
    encode_tf_message,
    encode_twist_stamped,
)
from calibrex.data.rosbag2 import (
    IMU_TYPE,
    NAVSATFIX_TYPE,
    ODOMETRY_TYPE,
    POINTCLOUD2_TYPE,
    TF_MESSAGE_TYPE,
    TWIST_COVARIANCE_STAMPED_TYPE,
    TWIST_STAMPED_TYPE,
    decode_rosbag2_message,
    iter_topic_messages,
    list_rosbag2_connections,
    resolve_storage,
)
from calibrex.data.rosbag2_writer import LATCHED_QOS_PROFILE, Rosbag2Writer

STAMP_NS = 1_317_042_272_335_337_762  # a 2011 epoch stamp: > 2**53 ns, must not lose bits


def _bag(path: Path) -> Rosbag2Writer:
    return Rosbag2Writer(path, custom_data={"generator": "test"})


def _first(path: Path, topic: str, message_type: str):  # type: ignore[no-untyped-def]
    for _conn, stamp, payload in iter_topic_messages(path, topic, limit=1):
        return decode_rosbag2_message(topic, message_type, stamp, payload)
    raise AssertionError(f"no message on {topic}")


def test_pointcloud2_roundtrip_keeps_xyz_and_intensity(tmp_path: Path) -> None:
    points = np.array(
        [[1.5, -2.25, 0.125, 0.5], [10.0, 3.0, -1.0, 0.25], [0.0, 0.0, 0.0, 1.0]], dtype="<f4"
    )
    fields = (
        ("x", 0, POINT_FIELD_FLOAT32, 1),
        ("y", 4, POINT_FIELD_FLOAT32, 1),
        ("z", 8, POINT_FIELD_FLOAT32, 1),
        ("intensity", 12, POINT_FIELD_FLOAT32, 1),
    )
    with _bag(tmp_path / "bag") as writer:
        writer.add_topic("/points", POINTCLOUD2_TYPE)
        writer.write(
            "/points",
            STAMP_NS,
            encode_pointcloud2(
                frame_id="velo_link",
                timestamp_ns=STAMP_NS,
                fields=fields,
                point_step=16,
                data=points.tobytes(),
            ),
        )
    message = _first(tmp_path / "bag", "/points", POINTCLOUD2_TYPE)
    assert message.frame_id == "velo_link"
    assert message.timestamp_ns == STAMP_NS
    assert message.xyz.dtype == np.float64
    np.testing.assert_array_equal(message.xyz, points[:, :3].astype(np.float64))
    assert message.intensity is not None
    np.testing.assert_array_equal(message.intensity, points[:, 3].astype(np.float64))
    assert {field.name for field in message.fields} == {"x", "y", "z", "intensity"}


def test_pointcloud2_rejects_ragged_data() -> None:
    with pytest.raises(ValueError, match="multiple of point_step"):
        encode_pointcloud2(
            frame_id="f", timestamp_ns=0, fields=(), point_step=16, data=b"\x00" * 10
        )


def test_imu_roundtrip(tmp_path: Path) -> None:
    with _bag(tmp_path / "bag") as writer:
        writer.add_topic("/imu", IMU_TYPE)
        writer.write(
            "/imu",
            STAMP_NS,
            encode_imu(
                frame_id="imu_link",
                timestamp_ns=STAMP_NS,
                angular_velocity=(0.1, -0.2, 0.3),
                linear_acceleration=(0.01, 0.02, 9.81),
            ),
        )
    message = _first(tmp_path / "bag", "/imu", IMU_TYPE)
    assert (message.frame_id, message.timestamp_ns) == ("imu_link", STAMP_NS)
    assert message.angular_velocity == (0.1, -0.2, 0.3)
    assert message.linear_acceleration == (0.01, 0.02, 9.81)
    assert message.orientation_covariance[0] == -1.0  # no orientation estimate


def test_navsatfix_roundtrip(tmp_path: Path) -> None:
    with _bag(tmp_path / "bag") as writer:
        writer.add_topic("/fix", NAVSATFIX_TYPE)
        writer.write(
            "/fix",
            STAMP_NS,
            encode_navsatfix(
                frame_id="imu_link",
                secs=STAMP_NS // 10**9,
                nsecs=STAMP_NS % 10**9,
                latitude=49.0095,
                longitude=8.3954,
                altitude=112.5,
                status=0,
                service=1,
            ),
        )
    message = _first(tmp_path / "bag", "/fix", NAVSATFIX_TYPE)
    assert message.frame_id == "imu_link"
    assert message.timestamp_ns == STAMP_NS
    assert (message.latitude_deg, message.longitude_deg, message.altitude_m) == (
        49.0095,
        8.3954,
        112.5,
    )
    assert (message.status, message.service) == (0, 1)


def test_odometry_roundtrip_pose_in_frame_twist_in_child(tmp_path: Path) -> None:
    quaternion = (0.0, 0.0, float(np.sin(0.3)), float(np.cos(0.3)))
    with _bag(tmp_path / "bag") as writer:
        writer.add_topic("/odom", ODOMETRY_TYPE)
        writer.write(
            "/odom",
            STAMP_NS,
            encode_odometry(
                frame_id="odom",
                child_frame_id="imu_link",
                timestamp_ns=STAMP_NS,
                position=(1.0, -2.0, 0.5),
                orientation_xyzw=quaternion,
                linear_velocity=(5.0, 0.01, -0.02),
                angular_velocity=(0.001, 0.002, 0.2),
            ),
        )
    message = _first(tmp_path / "bag", "/odom", ODOMETRY_TYPE)
    assert (message.frame_id, message.child_frame_id) == ("odom", "imu_link")
    assert message.position == (1.0, -2.0, 0.5)
    assert message.orientation_xyzw == quaternion
    assert message.linear_velocity == (5.0, 0.01, -0.02)
    assert message.angular_velocity == (0.001, 0.002, 0.2)


@pytest.mark.parametrize(
    ("message_type", "covariance"),
    [(TWIST_STAMPED_TYPE, None), (TWIST_COVARIANCE_STAMPED_TYPE, tuple(range(36)))],
)
def test_twist_roundtrip(
    tmp_path: Path, message_type: str, covariance: tuple[float, ...] | None
) -> None:
    with _bag(tmp_path / "bag") as writer:
        writer.add_topic("/twist", message_type)
        writer.write(
            "/twist",
            STAMP_NS,
            encode_twist_stamped(
                frame_id="base_link",
                timestamp_ns=STAMP_NS,
                linear=(4.0, 0.0, 0.1),
                angular=(0.0, 0.01, -0.3),
                covariance=covariance,
            ),
        )
    message = _first(tmp_path / "bag", "/twist", message_type)
    assert message.frame_id == "base_link"
    assert message.timestamp_ns == STAMP_NS
    assert message.linear_velocity == (4.0, 0.0, 0.1)
    assert message.angular_velocity == (0.0, 0.01, -0.3)
    assert message.covariance == (covariance or ())


def test_tf_static_roundtrip_is_transient_local(tmp_path: Path) -> None:
    transforms = [
        ("base_link", "imu_link", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ("imu_link", "velo_link", (-0.81, 0.32, -0.8), (0.5, -0.5, 0.5, 0.5)),
    ]
    with _bag(tmp_path / "bag") as writer:
        writer.add_topic("/tf_static", TF_MESSAGE_TYPE, latched=True)
        writer.add_topic("/other", IMU_TYPE)
        writer.write("/tf_static", 5, encode_tf_message(transforms))
    message = _first(tmp_path / "bag", "/tf_static", TF_MESSAGE_TYPE)
    assert [(t.frame_id, t.child_frame_id) for t in message.transforms] == [
        ("base_link", "imu_link"),
        ("imu_link", "velo_link"),
    ]
    assert message.transforms[1].translation_m == (-0.81, 0.32, -0.8)
    assert message.transforms[1].rotation_xyzw == (0.5, -0.5, 0.5, 0.5)

    connections = {conn.topic: conn for conn, _count in list_rosbag2_connections(tmp_path / "bag")}
    assert connections["/tf_static"].offered_qos_profiles == LATCHED_QOS_PROFILE
    assert "durability: 1" in LATCHED_QOS_PROFILE  # transient local
    assert not connections["/other"].offered_qos_profiles
    info = yaml.safe_load((tmp_path / "bag" / "metadata.yaml").read_text())[
        "rosbag2_bagfile_information"
    ]
    qos = {
        item["topic_metadata"]["name"]: item["topic_metadata"]["offered_qos_profiles"]
        for item in info["topics_with_message_count"]
    }
    assert qos["/tf_static"] == LATCHED_QOS_PROFILE


def test_metadata_counts_times_and_custom_data(tmp_path: Path) -> None:
    with _bag(tmp_path / "bag") as writer:
        writer.add_topic("/a", IMU_TYPE)
        writer.add_topic("/b", TWIST_STAMPED_TYPE)
        for index in range(3):
            writer.write("/a", 100 + index, b"\x00\x01\x00\x00")
        writer.write("/b", 50, b"\x00\x01\x00\x00")
    info = yaml.safe_load((tmp_path / "bag" / "metadata.yaml").read_text())[
        "rosbag2_bagfile_information"
    ]
    assert info["storage_identifier"] == "sqlite3"
    assert info["message_count"] == 4
    assert info["starting_time"]["nanoseconds_since_epoch"] == 50
    assert info["duration"]["nanoseconds"] == 52
    assert info["custom_data"] == {"generator": "test"}
    counts = {
        item["topic_metadata"]["name"]: item["message_count"]
        for item in info["topics_with_message_count"]
    }
    assert counts == {"/a": 3, "/b": 1}
    storage, storage_id = resolve_storage(tmp_path / "bag")
    assert (storage.name, storage_id) == ("bag_0.db3", "sqlite3")
    assert [conn.topic for conn, _count in list_rosbag2_connections(tmp_path / "bag")] == [
        "/a",
        "/b",
    ]


def test_messages_come_back_in_timestamp_order(tmp_path: Path) -> None:
    with _bag(tmp_path / "bag") as writer:
        writer.add_topic("/a", IMU_TYPE)
        for stamp in (30, 10, 20):
            writer.write("/a", stamp, struct.pack("<I", stamp))
    stamps = [stamp for _c, stamp, _d in iter_topic_messages(tmp_path / "bag", "/a")]
    assert stamps == [10, 20, 30]
    database = sqlite3.connect(tmp_path / "bag" / "bag_0.db3")
    try:
        assert database.execute("SELECT count(*) FROM messages").fetchone() == (3,)
    finally:
        database.close()


def test_writer_guards(tmp_path: Path) -> None:
    writer = _bag(tmp_path / "bag")
    writer.add_topic("/a", IMU_TYPE)
    with pytest.raises(DatasetError, match="not declared"):
        writer.write("/missing", 1, b"")
    with pytest.raises(DatasetError, match="already declared"):
        writer.add_topic("/a", ODOMETRY_TYPE)
    writer.close()
    writer.close()  # idempotent
    with pytest.raises(DatasetError, match="closed"):
        writer.write("/a", 1, b"")
    with pytest.raises(DatasetError, match="not empty"):
        _bag(tmp_path / "bag")
    replaced = Rosbag2Writer(tmp_path / "bag", overwrite=True)
    replaced.close()


def test_big_endian_header_only_helper_and_frame_id() -> None:
    writer = CdrWriter(little_endian=False)
    writer.write_int32(1)
    writer.write_uint32(2)
    writer.write_string("map")
    assert decode_ros2_header_frame_id(writer.finish()) == "map"
