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


# ------------------------------------------------------------------------- the camera


def _png(rows: np.ndarray) -> bytes:
    """A minimal 8-bit grayscale PNG (filter 0 rows, one IDAT)."""

    import struct
    import zlib

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    height, width = rows.shape
    raw = b"".join(b"\x00" + rows[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


R_VELO_CAM = Rotation.from_euler("xyz", [-90.0, 0.3, -90.0], degrees=True).as_matrix()
T_VELO_CAM = np.array([-0.004, -0.076, -0.272])
R_RECT = Rotation.from_euler("xyz", [0.2, -0.3, 0.1], degrees=True).as_matrix()
P_RECT_02 = np.array([[721.5, 0.0, 609.6, 44.86], [0.0, 721.5, 172.9, 0.2], [0.0, 0.0, 1.0, 0.003]])
IMAGE_SHAPE = (8, 12)


def add_camera(drive: Path) -> np.ndarray:
    """``image_02`` and the camera calibration files; returns the written pixel rows."""

    day = drive.parent
    (day / "calib_velo_to_cam.txt").write_text(
        "calib_time: fake\n"
        f"R: {' '.join(f'{v:.12e}' for v in R_VELO_CAM.reshape(-1))}\n"
        f"T: {' '.join(f'{v:.12e}' for v in T_VELO_CAM)}\n",
        encoding="utf-8",
    )
    (day / "calib_cam_to_cam.txt").write_text(
        "calib_time: fake\n"
        f"R_rect_00: {' '.join(f'{v:.12e}' for v in R_RECT.reshape(-1))}\n"
        f"S_rect_02: {IMAGE_SHAPE[1]}.0 {IMAGE_SHAPE[0]}.0\n"
        f"P_rect_02: {' '.join(f'{v:.12e}' for v in P_RECT_02.reshape(-1))}\n",
        encoding="utf-8",
    )
    (drive / "image_02" / "data").mkdir(parents=True)
    (drive / "image_02" / "timestamps.txt").write_text(
        "\n".join(f"2011-01-01 12:00:00.{index}20000000" for index in range(FRAMES)) + "\n",
        encoding="utf-8",
    )
    rows = np.arange(IMAGE_SHAPE[0] * IMAGE_SHAPE[1], dtype=np.uint8).reshape(IMAGE_SHAPE)
    for index in range(FRAMES):
        (drive / "image_02" / "data" / f"{index:010d}.png").write_bytes(_png(rows + index))
    return rows


def test_camera_is_opt_in(converted: tuple[Path, Path]) -> None:
    _drive, bag = converted
    topics = {conn.topic for conn, _ in list_rosbag2_connections(bag)}
    assert "/camera/image_raw" not in topics and "/camera/camera_info" not in topics


def test_camera_images_info_and_tf_follow_the_vendor_calibration(tmp_path: Path) -> None:
    drive = write_drive(tmp_path)
    rows = add_camera(drive)
    bag = tmp_path / "bag"

    result = convert_kitti_raw_to_rosbag2(drive, bag, camera="image_02")

    assert result.message_counts["/camera/image_raw"] == FRAMES
    assert result.message_counts["/camera/camera_info"] == FRAMES
    listing = {c.topic: c.message_type for c, _ in list_rosbag2_connections(bag)}
    assert listing["/camera/image_raw"] == "sensor_msgs/msg/Image"
    assert listing["/camera/camera_info"] == "sensor_msgs/msg/CameraInfo"
    images = decode(bag, "/camera/image_raw", "sensor_msgs/msg/Image")
    assert {(m.frame_id, m.encoding, m.width, m.height) for m in images} == {
        ("cam2_optical", "mono8", IMAGE_SHAPE[1], IMAGE_SHAPE[0])
    }
    from calibrex.data import ros_cdr

    payloads = [
        (stamp, payload) for _c, stamp, payload in iter_topic_messages(bag, "/camera/image_raw")
    ]
    full = ros_cdr.decode_ros2_image("/camera/image_raw", *payloads[3])
    assert full.data is not None
    np.testing.assert_array_equal(
        np.frombuffer(bytes(full.data), dtype=np.uint8).reshape(IMAGE_SHAPE), rows + 3
    )
    # stamped with the camera's own timestamps.txt (0.1 s per frame)
    assert (images[1].timestamp_ns - images[0].timestamp_ns) == 100_000_000
    info = decode(bag, "/camera/camera_info", "sensor_msgs/msg/CameraInfo")[0]
    assert info.frame_id == "cam2_optical" and (info.width, info.height) == (12, 8)
    assert info.k[0] == pytest.approx(721.5) and info.k[2] == pytest.approx(609.6)
    assert not any(info.d)  # rectified images: no distortion

    # tf: T_imu_cam2 = T_imu_velo * inv(T_cam2_velo), T_cam2_velo = [I | K^-1 P[:,3]] R_rect [R|T]
    source = load_bag_tf_static(bag)
    assert source is not None
    edges = {(e.parent, e.child): e.transform for e in source.edges}
    assert set(edges) == {
        ("base_link", "imu_link"),
        ("imu_link", "velo_link"),
        ("imu_link", "cam2_optical"),
    }
    velo_cam0 = np.eye(4)
    velo_cam0[:3, :3], velo_cam0[:3, 3] = R_VELO_CAM, T_VELO_CAM
    rect = np.eye(4)
    rect[:3, :3] = R_RECT
    shift = np.eye(4)
    shift[:3, 3] = np.linalg.solve(P_RECT_02[:, :3], P_RECT_02[:, 3])
    t_cam_velo = shift @ rect @ velo_cam0
    vendor = load_kitti_ins_lidar_drive(drive).vendor_t_imu_lidar
    expected = vendor @ np.linalg.inv(t_cam_velo)
    got = edges[("imu_link", "cam2_optical")]
    np.testing.assert_allclose(got.translation_m, expected[:3, 3], atol=1e-9)
    np.testing.assert_allclose(
        Rotation.from_quat(got.rotation_quat_xyzw).as_matrix(), expected[:3, :3], atol=1e-9
    )
    # the camera <- velodyne transform the check composes through the tree is the vendor one
    composed = np.linalg.inv(expected) @ vendor
    np.testing.assert_allclose(composed, t_cam_velo, atol=1e-9)

    record = json.loads((bag / CONVERSION_FILENAME).read_text())
    assert record["camera"]["frame"] == "cam2_optical"
    assert len(record["camera"]["calib_cam_to_cam_sha256"]) == 64
    assert "/camera/image_raw" in record["topics"]


def test_camera_errors_name_the_missing_files(tmp_path: Path) -> None:
    drive = write_drive(tmp_path)
    with pytest.raises(DatasetError, match=r"image_02/data"):
        convert_kitti_raw_to_rosbag2(drive, tmp_path / "a", camera="image_02")
    add_camera(drive)
    (drive.parent / "calib_cam_to_cam.txt").unlink()
    with pytest.raises(DatasetError, match=r"calib_cam_to_cam\.txt"):
        convert_kitti_raw_to_rosbag2(drive, tmp_path / "b", camera="image_02")


def test_cli_convert_with_camera(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    drive = write_drive(tmp_path)
    add_camera(drive)
    bag = tmp_path / "b"
    code = main(
        ["convert", "kitti-raw", str(drive), "--camera", "image_02", "--output", str(bag), "--json"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["message_counts"]["/camera/image_raw"] == FRAMES
