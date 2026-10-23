"""Synthetic KITTI-360 split-layout round trips; no external sensor data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

from calibrex.check.tf_sources import load_bag_tf_static
from calibrex.cli.main import main
from calibrex.core.exceptions import DatasetError
from calibrex.data import ros_cdr
from calibrex.data.kitti import read_timestamps
from calibrex.data.kitti360_to_rosbag2 import ODOMETRY_TOPIC, convert_kitti360_to_rosbag2
from calibrex.data.kitti_raw_to_rosbag2 import (
    CAMERA_INFO_TOPIC,
    CONVERSION_FILENAME,
    IMAGE_TOPIC,
    LIDAR_TOPIC,
    TF_STATIC_TOPIC,
)
from calibrex.data.ros2_camera_lidar import load_camera_lidar_frames
from calibrex.data.rosbag2 import (
    decode_rosbag2_message,
    iter_topic_messages,
    list_rosbag2_connections,
)

SEQUENCE = "2013_05_28_drive_0000_sync"


def _oxts_fixture(root: Path) -> Path:
    folder = root / "data_poses" / SEQUENCE / "oxts"
    (folder / "data").mkdir(parents=True)
    stamps = []
    for index in range(6):
        values = [49.0, 8.4 + index * 1e-6, 110.0, 0.0, 0.0, 0.1, *([0.0] * 24)]
        (folder / "data" / f"{index:010d}.txt").write_text(
            " ".join(map(str, values)) + "\n", encoding="utf-8",
        )
        stamps.append(f"2013-05-28 00:00:00.{index * 100000000 + 2000000:09d}")
    (folder / "timestamps.txt").write_text("\n".join(stamps) + "\n", encoding="utf-8")
    return folder
CAM0 = np.eye(4)
CAM0[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.1]).as_matrix()
CAM0[:3, 3] = [0.7, -0.1, 0.2]
CAM1 = CAM0.copy()
CAM1[:3, 3] += CAM0[:3, :3] @ np.array([0.6, 0, 0])
VELO_CAM0 = np.eye(4)
VELO_CAM0[:3, :3] = Rotation.from_euler("xyz", [-0.1, 0.4, -0.2]).as_matrix()
VELO_CAM0[:3, 3] = [0.3, -0.5, 0.4]
POINTS = np.array([[4, 0, 1, 0.2], [0, 4, 1, 0.4], [0, -4, 1, 0.6]], dtype="<f4")


def _row(matrix: np.ndarray) -> str:
    return " ".join(str(float(v)) for v in matrix.reshape(-1))


def _fixture(root: Path, sequence: str = SEQUENCE) -> None:
    calibration = root / "calibration"
    calibration.mkdir(parents=True)
    (calibration / "calib_cam_to_velo.txt").write_text(_row(VELO_CAM0[:3]))
    (calibration / "calib_cam_to_pose.txt").write_text(
        f"image_00: {_row(CAM0[:3])}\nimage_01: {_row(CAM1[:3])}\n"
    )
    perspective = []
    for number in range(2):
        name = f"image_{number:02d}"
        rotation = Rotation.from_euler("z", (number + 1) * 0.1).as_matrix()
        projection = np.array([[8, 0, 2, -4.8 * number], [0, 9, 1, 0], [0, 0, 1, 0]])
        perspective.extend(
            [
                f"P_rect_{number:02d}: {_row(projection)}",
                f"R_rect_{number:02d}: {_row(rotation)}",
                f"S_rect_{number:02d}: 6 4",
            ]
        )
        folder = root / "data_2d_raw" / sequence / name
        (folder / "data_rect").mkdir(parents=True)
        stamps = [f"2013-05-28 00:00:00.{i * 100000000 + number * 6000000:09d}" for i in range(6)]
        (folder / "timestamps.txt").write_text("\n".join(stamps) + "\n")
        for i in range(6):
            Image.fromarray(np.full((4, 6), i * 20, dtype=np.uint8)).save(
                folder / "data_rect" / f"{i:010d}.png"
            )
    (calibration / "perspective.txt").write_text("\n".join(perspective) + "\n")
    lidar = root / "data_3d_raw" / sequence / "velodyne_points"
    (lidar / "data").mkdir(parents=True)
    (lidar / "timestamps.txt").write_text(
        "\n".join(f"2013-05-28 00:00:00.{i * 100000000 + 1000000:09d}" for i in range(6)) + "\n"
    )
    for i in (0, 1, 3, 4, 5):
        POINTS.tofile(lidar / "data" / f"{i:010d}.bin")
    poses = root / "data_poses" / sequence
    poses.mkdir(parents=True)
    rows = []
    for i in (0, 2, 5):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler("z", 0.3 + 0.01 * i).as_matrix()
        matrix[0, 3] = i * 0.2
        rows.append(f"{i} {_row(matrix[:3])}")
    (poses / "poses.txt").write_text("\n".join(rows) + "\n")


def _messages(bag: Path, topic: str, kind: str) -> list:
    return [
        decode_rosbag2_message(topic, kind, stamp, payload)
        for _conn, stamp, payload in iter_topic_messages(bag, topic)
    ]


def test_raw_oxts_uses_own_clock_and_hashes_the_origin_outside_the_window(tmp_path: Path) -> None:
    root, bag = tmp_path / "dataset", tmp_path / "bag"
    _fixture(root)
    oxts = _oxts_fixture(root)
    optimized = root / "data_poses" / SEQUENCE / "poses.txt"
    optimized.write_text("must not be read for raw OXTS\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="expected frame ID"):
        convert_kitti360_to_rosbag2(root, tmp_path / "published", sequence=SEQUENCE)
    result = convert_kitti360_to_rosbag2(
        root, bag, sequence=SEQUENCE, motion_source="oxts", start_frame=1, max_frames=3,
    )
    rows = _messages(bag, ODOMETRY_TOPIC, "nav_msgs/msg/Odometry")
    stamps = read_timestamps(oxts / "timestamps.txt")
    assert [r.timestamp_ns for r in rows] == [stamps[i].timestamp_ns for i in (1, 2, 3)]
    assert all(r.frame_id == "kitti360_oxts_world" for r in rows)
    assert rows[0].position[0] > 0  # Origin is packet 0, not selected packet 1.
    info = _messages(bag, CAMERA_INFO_TOPIC, "sensor_msgs/msg/CameraInfo")
    assert info[0].timestamp_ns != rows[0].timestamp_ns
    assert result.provenance["motion"]["mode"] == "oxts"
    assert result.provenance["motion"]["origin_packet"].endswith("0000000000.txt")
    files = result.provenance["source"]["files_sha256"]
    assert str(optimized) not in files
    assert str(oxts / "data/0000000000.txt") in files
    assert str(oxts / "data/0000000003.txt") in files
    assert str(oxts / "data/0000000005.txt") not in files


def test_raw_oxts_never_falls_back_to_available_published_poses(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _fixture(root)
    with pytest.raises(DatasetError, match="oxts"):
        convert_kitti360_to_rosbag2(root, tmp_path / "bag", sequence=SEQUENCE, motion_source="oxts")
    assert not (tmp_path / "bag").exists()


def test_cli_records_the_raw_motion_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root, bag = tmp_path / "dataset", tmp_path / "bag"
    _fixture(root)
    _oxts_fixture(root)
    assert main([
        "convert", "kitti-360", str(root), "--sequence", SEQUENCE,
        "--motion-source", "oxts", "--output", str(bag), "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["bag"] == str(bag.resolve())
    provenance = json.loads((bag / CONVERSION_FILENAME).read_text())
    assert provenance["motion"]["mode"] == "oxts"
    assert "--motion-source" in provenance["command"]


@pytest.mark.parametrize("camera", ["image_00", "image_01"])
def test_split_layout_roundtrip_and_camera_frame_chain(tmp_path: Path, camera: str) -> None:
    root, bag = tmp_path / "dataset", tmp_path / "bag"
    _fixture(root)
    result = convert_kitti360_to_rosbag2(root, bag, sequence=SEQUENCE, camera=camera)
    listing = {c.topic: (c.message_type, count) for c, count in list_rosbag2_connections(bag)}
    assert listing == {
        IMAGE_TOPIC: ("sensor_msgs/msg/Image", 6),
        CAMERA_INFO_TOPIC: ("sensor_msgs/msg/CameraInfo", 6),
        LIDAR_TOPIC: ("sensor_msgs/msg/PointCloud2", 5),
        ODOMETRY_TOPIC: ("nav_msgs/msg/Odometry", 3),
        TF_STATIC_TOPIC: ("tf2_msgs/msg/TFMessage", 1),
    }
    source = load_bag_tf_static(bag)
    assert source is not None
    edges = {(edge.parent, edge.child): edge.transform for edge in source.edges}
    number = int(camera[-2:])
    transform = edges[("imu_link", f"cam{number}_optical")]
    point_rect = np.array([1.0, 0.0, 0.0])
    point_pose = Rotation.from_quat(transform.rotation_quat_xyzw).apply(point_rect) + np.asarray(
        transform.translation_m
    )
    angle = -(number + 1) * 0.1
    point_unrect = np.array([np.cos(angle), np.sin(angle), 0])
    physical_camera = CAM0 if number == 0 else CAM1
    expected_pose = physical_camera[:3, :3] @ point_unrect + physical_camera[:3, 3]
    np.testing.assert_allclose(point_pose, expected_pose, atol=1e-12)
    velo = edges[("imu_link", "velo_link")]
    point_velo = VELO_CAM0[:3, :3] @ point_unrect + VELO_CAM0[:3, 3]
    body_from_velo = Rotation.from_quat(velo.rotation_quat_xyzw).apply(point_velo)
    body_from_velo += np.asarray(velo.translation_m)
    np.testing.assert_allclose(body_from_velo, CAM0[:3, :3] @ point_unrect + CAM0[:3, 3])
    infos = _messages(bag, CAMERA_INFO_TOPIC, "sensor_msgs/msg/CameraInfo")
    assert infos[0].frame_id == f"cam{number}_optical"
    assert infos[0].width == 6 and infos[0].height == 4
    np.testing.assert_array_equal(np.asarray(infos[0].p).reshape(3, 4)[:, 3], [0, 0, 0])
    images = [
        ros_cdr.decode_ros2_image(IMAGE_TOPIC, stamp, payload, include_data=True)
        for _conn, stamp, payload in iter_topic_messages(bag, IMAGE_TOPIC)
    ]
    for i, image in enumerate(images):
        assert image.data == bytes([i * 20]) * 24
        assert image.timestamp_ns == infos[i].timestamp_ns
    tf_connection = next(c for c, _n in list_rosbag2_connections(bag) if c.topic == TF_STATIC_TOPIC)
    assert "durability: 1" in tf_connection.offered_qos_profiles
    assert result.provenance["selection"]["image_without_scan_ids"] == [2]
    metadata = yaml.safe_load((bag / "metadata.yaml").read_text())
    assert isinstance(
        metadata["rosbag2_bagfile_information"]["custom_data"]["source_sequence"], str
    )


def test_point_times_sparse_pose_ids_and_derived_body_twist(tmp_path: Path) -> None:
    root, bag = tmp_path / "dataset", tmp_path / "bag"
    _fixture(root)
    convert_kitti360_to_rosbag2(root, bag, sequence=SEQUENCE, camera="image_01")
    clouds = [
        ros_cdr.decode_ros2_pointcloud2(LIDAR_TOPIC, stamp, payload, point_time_field="time")
        for _conn, stamp, payload in iter_topic_messages(bag, LIDAR_TOPIC)
    ]
    np.testing.assert_array_equal(clouds[0].xyz, POINTS[:, :3])
    np.testing.assert_allclose(clouds[0].intensity, POINTS[:, 3])
    np.testing.assert_allclose(clouds[0].point_time_offsets_s, [0, -0.025, 0.025])
    odometry = _messages(bag, ODOMETRY_TOPIC, "nav_msgs/msg/Odometry")
    stamps = read_timestamps(root / "data_2d_raw" / SEQUENCE / "image_00" / "timestamps.txt")
    assert [row.timestamp_ns for row in odometry] == [stamps[i].timestamp_ns for i in (0, 2, 5)]
    for row, index in zip(odometry, (0, 2, 5), strict=True):
        assert (row.frame_id, row.child_frame_id) == ("kitti360_world", "imu_link")
        np.testing.assert_allclose(row.position, [index * 0.2, 0, 0])
        expected_velocity = Rotation.from_euler("z", -(0.3 + 0.01 * index)).apply([2, 0, 0])
        np.testing.assert_allclose(row.linear_velocity, expected_velocity)
        np.testing.assert_allclose(row.angular_velocity, [0, 0, 0.1], atol=1e-10)
    frames = load_camera_lidar_frames(
        bag,
        IMAGE_TOPIC,
        LIDAR_TOPIC,
        spacing_s=0.09,
        max_frames=6,
        max_gap_s=0.01,
        point_time=("time", "offset_s"),
    )
    assert len(frames) == 5
    assert frames[0].point_time_offsets_s is not None


def test_window_selection_rigid_mode_and_provenance(tmp_path: Path) -> None:
    sequence = "2013_05_28_drive_0002_sync"
    root, bag = tmp_path / "dataset", tmp_path / "bag"
    _fixture(root, sequence)
    result = convert_kitti360_to_rosbag2(
        root,
        bag,
        sequence=sequence,
        start_frame=2,
        max_frames=3,
        velodyne_time="none",
        command=["fixture"],
    )
    record = json.loads(result.provenance_path.read_text())
    assert record["source"]["sequence"] == sequence
    assert record["command"] == ["fixture"]
    assert record["selection"]["image_ids"] == [2, 3, 4]
    assert record["selection"]["velodyne_ids"] == [3, 4]
    assert record["selection"]["pose_ids"] == [2]
    assert record["motion"]["twist_interval_s"] == [0.0]
    for path, digest in record["source"]["files_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert not any("0000000005.bin" in path for path in record["source"]["files_sha256"])
    cloud = _messages(bag, LIDAR_TOPIC, "sensor_msgs/msg/PointCloud2")[0]
    assert cloud.point_time_offsets_s is None
    assert {f.name for f in cloud.fields} == {"x", "y", "z", "intensity"}


def test_cli_and_source_digest_changes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, bag = tmp_path / "dataset", tmp_path / "bag"
    _fixture(root)
    argv = [
        "convert",
        "kitti-360",
        str(root),
        "--sequence",
        SEQUENCE,
        "--output",
        str(bag),
        "--json",
    ]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["message_counts"][LIDAR_TOPIC] == 5
    old = json.loads((bag / CONVERSION_FILENAME).read_text())["source"]["input_sha256"]
    path = root / "data_3d_raw" / SEQUENCE / "velodyne_points" / "data" / "0000000000.bin"
    points = POINTS.copy()
    points[0, 0] += 1
    points.tofile(path)
    assert main([*argv, "--overwrite"]) == 0
    new = json.loads((bag / CONVERSION_FILENAME).read_text())["source"]["input_sha256"]
    assert old != new


@pytest.mark.parametrize(
    "options",
    [
        {"camera": "image_02"},
        {"scan_period_s": 0},
        {"scan_period_s": float("nan")},
        {"start_frame": -1},
        {"max_frames": 0},
        {"sequence": "../0000"},
    ],
)
def test_invalid_options_create_no_bag(tmp_path: Path, options: dict) -> None:
    root, bag = tmp_path / "dataset", tmp_path / "bag"
    _fixture(root)
    arguments = {"sequence": SEQUENCE, **options}
    with pytest.raises(DatasetError):
        convert_kitti360_to_rosbag2(root, bag, **arguments)
    assert not bag.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "timestamp",
        "pose_duplicate",
        "pose_rotation",
        "scan",
        "image",
        "nonfinite_scan",
        "repeated_stamp",
        "missing_calibration",
        "missing_pose_timestamp",
    ],
)
def test_corrupt_inputs_do_not_replace_previous_bag(tmp_path: Path, damage: str) -> None:
    root, bag = tmp_path / "dataset", tmp_path / "bag"
    _fixture(root)
    convert_kitti360_to_rosbag2(root, bag, sequence=SEQUENCE)
    before = {p.name: p.read_bytes() for p in bag.iterdir()}
    images = root / "data_2d_raw" / SEQUENCE / "image_00"
    poses = root / "data_poses" / SEQUENCE / "poses.txt"
    if damage == "timestamp":
        (images / "timestamps.txt").write_text(
            "\n" + "\n".join(f"2013-05-28 00:00:00.{i}00000000" for i in range(1, 6)) + "\n"
        )
    elif damage == "pose_duplicate":
        poses.write_text(poses.read_text() + poses.read_text().splitlines()[-1] + "\n")
    elif damage == "pose_rotation":
        poses.write_text("0 " + " ".join(["0"] * 12) + "\n")
    elif damage == "scan":
        scan = root / "data_3d_raw" / SEQUENCE / "velodyne_points" / "data" / "0000000000.bin"
        scan.write_bytes(b"bad")
    elif damage == "nonfinite_scan":
        scan = root / "data_3d_raw" / SEQUENCE / "velodyne_points" / "data" / "0000000000.bin"
        points = POINTS.copy()
        points[0, 0] = np.nan
        points.tofile(scan)
    elif damage == "repeated_stamp":
        path = images / "timestamps.txt"
        rows = path.read_text().splitlines()
        rows[1] = rows[0]
        path.write_text("\n".join(rows) + "\n")
    elif damage == "missing_calibration":
        (root / "calibration" / "calib_cam_to_pose.txt").unlink()
    elif damage == "missing_pose_timestamp":
        poses.write_text(poses.read_text() + "6 " + _row(np.eye(4)[:3]) + "\n")
    else:
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(
            images / "data_rect" / "0000000000.png"
        )
    with pytest.raises(DatasetError):
        convert_kitti360_to_rosbag2(root, bag, sequence=SEQUENCE, overwrite=True)
    assert before == {p.name: p.read_bytes() for p in bag.iterdir()}
    assert not list(tmp_path.glob(".bag-*"))


def test_refuse_dataset_output_and_existing_nonbag(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _fixture(root)
    with pytest.raises(DatasetError, match="overlap"):
        convert_kitti360_to_rosbag2(root, root, sequence=SEQUENCE, overwrite=True)
    out = tmp_path / "not_a_bag"
    out.mkdir()
    (out / "keep.txt").write_text("retain")
    with pytest.raises(DatasetError, match=r"metadata\.yaml"):
        convert_kitti360_to_rosbag2(root, out, sequence=SEQUENCE, overwrite=True)
    assert (out / "keep.txt").read_text() == "retain"
