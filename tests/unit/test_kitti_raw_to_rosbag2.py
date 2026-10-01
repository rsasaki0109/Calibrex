"""``calibrex convert kitti-raw`` on a tiny KITTI-shaped drive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.check.tf_sources import load_bag_tf_static
from calibrex.cli.main import main
from calibrex.core.exceptions import DatasetError
from calibrex.data.kitti_ins_lidar import load_kitti_ins_lidar_drive
from calibrex.data.kitti_raw_to_rosbag2 import (
    CONVERSION_FILENAME,
    FIX_TOPIC,
    IMU_TOPIC,
    LIDAR_TOPIC,
    ODOMETRY_TOPIC,
    TF_STATIC_TOPIC,
    TWIST_TOPIC,
    convert_kitti_raw_to_rosbag2,
)
from calibrex.data.rosbag2 import (
    decode_rosbag2_message,
    iter_topic_messages,
    list_rosbag2_connections,
)
from calibrex.evaluation.vehicle_frame import oxts_motions

ROLL, PITCH, YAW = 0.02, -0.03, 0.4
FRAMES = 6
R_VELO_IMU = Rotation.from_euler("xyz", [0.5, -1.0, 90.0], degrees=True).as_matrix()
T_VELO_IMU = np.array([-0.81, 0.32, -0.8])


def write_drive(
    root: Path, *, drop_velodyne: tuple[int, ...] = (), number: int = 1, minute: int = 0
) -> Path:
    """A 6-packet drive: moving north-east, rolled and pitched, with level-frame velocities."""

    day = root / "2011_01_01"
    drive = day / f"2011_01_01_drive_{number:04d}_sync"
    (drive / "oxts" / "data").mkdir(parents=True)
    (drive / "velodyne_points" / "data").mkdir(parents=True)
    rotation = " ".join(f"{value:.12f}" for value in R_VELO_IMU.reshape(-1))
    translation = " ".join(f"{value:.12f}" for value in T_VELO_IMU)
    (day / "calib_imu_to_velo.txt").write_text(
        f"calib_time: fake\nR: {rotation}\nT: {translation}\n", encoding="utf-8"
    )
    stamps = [f"2011-01-01 12:{minute:02d}:00.{index}00000000" for index in range(FRAMES)]
    (drive / "oxts" / "timestamps.txt").write_text("\n".join(stamps) + "\n", encoding="utf-8")
    for index in range(FRAMES):
        values = [0.0] * 30
        values[0:6] = [
            49.0 + index * 1e-5,
            8.4 + index * 2e-5,
            110.0 + 0.1 * index,
            ROLL,
            PITCH,
            YAW,
        ]
        values[8:11] = [5.0 + 0.1 * index, 0.05, -0.02]  # vf, vl, vu (level frame)
        values[11:14] = [0.1, 0.2, 9.8]  # ax ay az
        values[17:20] = [0.01, 0.02, 0.03]  # wx wy wz
        values[20:23] = [0.001, -0.002, 0.2 + 0.01 * index]  # wf wl wu (level frame)
        values[23] = 0.5  # position accuracy
        (drive / "oxts" / "data" / f"{index:010d}.txt").write_text(
            " ".join(repr(value) for value in values), encoding="utf-8"
        )
    velodyne_stamps = [
        "" if index in drop_velodyne else f"2011-01-01 12:{minute:02d}:00.{index}50000000"
        for index in range(FRAMES)
    ]
    (drive / "velodyne_points" / "timestamps.txt").write_text(
        "\n".join(velodyne_stamps) + "\n", encoding="utf-8"
    )
    rng = np.random.default_rng(7)
    for index in range(FRAMES):
        if index in drop_velodyne:
            continue
        points = rng.uniform(-20.0, 20.0, (40, 4)).astype(np.float32)
        points[:, 3] = rng.uniform(0.0, 1.0, 40).astype(np.float32)
        points.tofile(drive / "velodyne_points" / "data" / f"{index:010d}.bin")
    return drive


def decode(bag: Path, topic: str, message_type: str) -> list:  # type: ignore[type-arg]
    return [
        decode_rosbag2_message(topic, message_type, stamp, payload)
        for _conn, stamp, payload in iter_topic_messages(bag, topic)
    ]


@pytest.fixture
def converted(tmp_path: Path) -> tuple[Path, Path]:
    drive = write_drive(tmp_path)
    bag = tmp_path / "bag"
    convert_kitti_raw_to_rosbag2(drive, bag, command=["calibrex", "convert", "kitti-raw"])
    return drive, bag


def test_topics_types_counts_and_frames(converted: tuple[Path, Path]) -> None:
    _drive, bag = converted
    listing = {
        conn.topic: (conn.message_type, count) for conn, count in list_rosbag2_connections(bag)
    }
    assert listing == {
        LIDAR_TOPIC: ("sensor_msgs/msg/PointCloud2", FRAMES),
        IMU_TOPIC: ("sensor_msgs/msg/Imu", FRAMES),
        FIX_TOPIC: ("sensor_msgs/msg/NavSatFix", FRAMES),
        ODOMETRY_TOPIC: ("nav_msgs/msg/Odometry", FRAMES),
        TWIST_TOPIC: ("geometry_msgs/msg/TwistStamped", FRAMES),
        TF_STATIC_TOPIC: ("tf2_msgs/msg/TFMessage", 1),
    }
    connections = {conn.topic: conn for conn, _ in list_rosbag2_connections(bag)}
    assert "durability: 1" in (connections[TF_STATIC_TOPIC].offered_qos_profiles or "")

    clouds = decode(bag, LIDAR_TOPIC, "sensor_msgs/msg/PointCloud2")
    assert {c.frame_id for c in clouds} == {"velo_link"}
    assert {field.name for field in clouds[0].fields} == {"x", "y", "z", "intensity"}  # no time
    assert clouds[0].point_time_offsets_s is None
    imus = decode(bag, IMU_TOPIC, "sensor_msgs/msg/Imu")
    assert {m.frame_id for m in imus} == {"imu_link"}
    assert imus[0].angular_velocity == (0.01, 0.02, 0.03)
    assert imus[0].linear_acceleration == (0.1, 0.2, 9.8)
    fixes = decode(bag, FIX_TOPIC, "sensor_msgs/msg/NavSatFix")
    assert {m.frame_id for m in fixes} == {"imu_link"}  # the GNSS fix is at the OXTS unit
    assert fixes[2].latitude_deg == pytest.approx(49.00002)
    assert fixes[0].position_covariance[0] == pytest.approx(0.25)
    twists = decode(bag, TWIST_TOPIC, "geometry_msgs/msg/TwistStamped")
    assert {m.frame_id for m in twists} == {"base_link"}


def test_point_clouds_roundtrip_exactly(converted: tuple[Path, Path]) -> None:
    drive, bag = converted
    clouds = decode(bag, LIDAR_TOPIC, "sensor_msgs/msg/PointCloud2")
    for index, cloud in enumerate(clouds):
        raw = np.fromfile(drive / "velodyne_points" / "data" / f"{index:010d}.bin", "<f4").reshape(
            -1, 4
        )
        np.testing.assert_array_equal(cloud.xyz, raw[:, :3].astype(np.float64))
        np.testing.assert_array_equal(cloud.intensity, raw[:, 3].astype(np.float64))


def test_tf_static_is_the_inverse_of_calib_imu_to_velo(converted: tuple[Path, Path]) -> None:
    drive, bag = converted
    source = load_bag_tf_static(bag)
    assert source is not None
    edges = {(e.parent, e.child): e.transform for e in source.edges}
    assert set(edges) == {("base_link", "imu_link"), ("imu_link", "velo_link")}
    assert edges[("base_link", "imu_link")].translation_m == (0.0, 0.0, 0.0)  # base_link = imu_link
    imu_velo = edges[("imu_link", "velo_link")]
    expected = load_kitti_ins_lidar_drive(drive).vendor_t_imu_lidar
    np.testing.assert_allclose(imu_velo.translation_m, expected[:3, 3], atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_quat(imu_velo.rotation_quat_xyzw).as_matrix(), expected[:3, :3], atol=1e-12
    )
    # and T_velo_imu maps a point as the KITTI file says: x_velo = R x_imu + T
    np.testing.assert_allclose(expected[:3, :3].T, R_VELO_IMU, atol=1e-9)


def test_odometry_pose_is_the_kitti_ins_trajectory_and_twist_is_body_frame(
    converted: tuple[Path, Path],
) -> None:
    drive, bag = converted
    trajectory = load_kitti_ins_lidar_drive(drive).trajectory
    odometry = decode(bag, ODOMETRY_TOPIC, "nav_msgs/msg/Odometry")
    assert {(m.frame_id, m.child_frame_id) for m in odometry} == {("odom", "imu_link")}
    for message, time_s, pose in zip(odometry, trajectory.times_s, trajectory.poses, strict=True):
        assert message.timestamp_ns * 1e-9 == time_s
        np.testing.assert_array_equal(message.position, pose[:3, 3])
        np.testing.assert_array_equal(
            Rotation.from_quat(message.orientation_xyzw).as_matrix(), pose[:3, :3]
        )
    # the twist is the OXTS level-frame velocity rotated into the body frame, as oxts_motions does
    motions = oxts_motions(drive, block_duration_s=1.0e9)
    for message, motion in zip(odometry, motions, strict=True):
        np.testing.assert_array_equal(message.linear_velocity, motion.velocity_mps)
        np.testing.assert_array_equal(message.angular_velocity, motion.angular_rate_rps)
    level = np.array([5.0, 0.05, -0.02])
    body = Rotation.from_euler("YX", [PITCH, ROLL]).as_matrix().T @ level
    np.testing.assert_allclose(odometry[0].linear_velocity, body)
    assert not np.allclose(odometry[0].linear_velocity, level)  # a real change of frame
    twists = decode(bag, TWIST_TOPIC, "geometry_msgs/msg/TwistStamped")
    assert [t.linear_velocity for t in twists] == [m.linear_velocity for m in odometry]
    assert [t.angular_velocity for t in twists] == [m.angular_velocity for m in odometry]


def test_dropped_velodyne_frames_are_skipped_by_index(tmp_path: Path) -> None:
    drive = write_drive(tmp_path, drop_velodyne=(2,))
    result = convert_kitti_raw_to_rosbag2(drive, tmp_path / "bag")
    assert result.message_counts[LIDAR_TOPIC] == FRAMES - 1
    stamps = [stamp for _c, stamp, _d in iter_topic_messages(tmp_path / "bag", LIDAR_TOPIC)]
    offsets = [(stamp - stamps[0]) * 1e-9 for stamp in stamps]
    assert offsets == pytest.approx([0.0, 0.1, 0.3, 0.4, 0.5])
    assert result.provenance["source"]["drives"][0]["missing_velodyne_indices"] == [2]


def test_provenance_sidecar_and_metadata(converted: tuple[Path, Path]) -> None:
    drive, bag = converted
    record = json.loads((bag / CONVERSION_FILENAME).read_text())
    assert record["schema_version"] == "calibrex.rosbag_conversion/v0.1"
    assert record["command"] == ["calibrex", "convert", "kitti-raw"]
    assert record["generator"].startswith("calibrex convert kitti-raw")
    (listed,) = record["source"]["drives"]
    assert listed["drive"] == drive.name
    digest = load_kitti_ins_lidar_drive(drive).input_sha256
    assert listed["input_sha256"] == digest
    assert record["source"]["input_sha256"] == hashlib.sha256(digest.encode("ascii")).hexdigest()
    assert len(record["source"]["calib_imu_to_velo_sha256"]) == 64
    assert "OXTS/IMU frame" in record["frames"]["base_link"]
    assert "no per-point time" in record["topics"][LIDAR_TOPIC]
    assert record["message_counts"][LIDAR_TOPIC] == FRAMES


def test_calibration_directory_override_and_missing_calibration(tmp_path: Path) -> None:
    drive = write_drive(tmp_path)
    calib = tmp_path / "elsewhere"
    calib.mkdir()
    (drive.parent / "calib_imu_to_velo.txt").rename(calib / "calib_imu_to_velo.txt")
    with pytest.raises(DatasetError, match=r"calib_imu_to_velo\.txt"):
        convert_kitti_raw_to_rosbag2(drive, tmp_path / "missing")
    convert_kitti_raw_to_rosbag2(drive, tmp_path / "ok", calibration_dir=calib)
    assert load_bag_tf_static(tmp_path / "ok") is not None


def test_not_a_drive_and_overwrite_guard(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not a KITTI raw drive"):
        convert_kitti_raw_to_rosbag2(tmp_path, tmp_path / "bag")
    drive = write_drive(tmp_path)
    convert_kitti_raw_to_rosbag2(drive, tmp_path / "bag")
    with pytest.raises(DatasetError, match="not empty"):
        convert_kitti_raw_to_rosbag2(drive, tmp_path / "bag")
    convert_kitti_raw_to_rosbag2(drive, tmp_path / "bag", overwrite=True)


def test_cli_convert_kitti_raw(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    drive = write_drive(tmp_path)
    bag = tmp_path / "cli_bag"

    code = main(["convert", "kitti-raw", str(drive), "--output", str(bag), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["message_counts"][ODOMETRY_TOPIC] == FRAMES
    recorded = json.loads((bag / CONVERSION_FILENAME).read_text())
    assert recorded["command"][:3] == ["calibrex", "convert", "kitti-raw"]
    assert main(["convert", "kitti-raw", str(tmp_path), "--output", str(tmp_path / "x")]) == 2


def test_several_drives_of_one_calibration_share_a_bag_with_a_gap(tmp_path: Path) -> None:
    later = write_drive(tmp_path, number=2, minute=5)
    earlier = write_drive(tmp_path, number=1, minute=0)
    bag = tmp_path / "bag"

    result = convert_kitti_raw_to_rosbag2([later, earlier], bag)  # given out of order

    assert result.message_counts[LIDAR_TOPIC] == 2 * FRAMES
    assert result.message_counts[ODOMETRY_TOPIC] == 2 * FRAMES
    assert [d["drive"] for d in result.provenance["source"]["drives"]] == [
        earlier.name,
        later.name,
    ]
    stamps = [stamp for _c, stamp, _d in iter_topic_messages(bag, ODOMETRY_TOPIC)]
    gaps = np.diff(stamps) * 1e-9
    assert np.sum(gaps > 60.0) == 1  # the five minutes between the drives
    assert load_bag_tf_static(bag) is not None  # one /tf_static for the shared calibration
    assert [c for c, _ in list_rosbag2_connections(bag) if c.topic == TF_STATIC_TOPIC]
    info = (bag / "metadata.yaml").read_text()
    assert earlier.name in info and later.name in info
