"""Synthetic Argoverse 2 log round trips; no Argoverse 2 data is read."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pa = pytest.importorskip("pyarrow")
feather = pytest.importorskip("pyarrow.feather")
Image = pytest.importorskip("PIL.Image")

from calibrex.check.tf_sources import load_bag_tf_static  # noqa: E402
from calibrex.cli.main import main  # noqa: E402
from calibrex.core.exceptions import DatasetError  # noqa: E402
from calibrex.data import av2_to_rosbag2 as av2  # noqa: E402
from calibrex.data import ros_cdr  # noqa: E402
from calibrex.data.kitti_raw_to_rosbag2 import (  # noqa: E402
    CAMERA_INFO_TOPIC,
    IMAGE_TOPIC,
    LIDAR_TOPIC,
)
from calibrex.data.rosbag2 import iter_topic_messages, list_rosbag2_connections  # noqa: E402

T0 = 315_969_904_359_876_000
SPEED = 8.0  # m/s along the ego x axis


def _pose(rotation: Rotation, translation: list[float]) -> dict[str, float]:
    x, y, z, w = rotation.as_quat()
    return {
        "qw": w,
        "qx": x,
        "qy": y,
        "qz": z,
        "tx_m": translation[0],
        "ty_m": translation[1],
        "tz_m": translation[2],
    }


CAMERA = _pose(Rotation.from_euler("xyz", [-90.0, 0.0, -90.0], degrees=True), [1.6, 0.0, 1.4])
UP = _pose(Rotation.from_euler("z", -0.5, degrees=True), [1.35, 0.0, 1.64])
DOWN = _pose(Rotation.from_euler("x", 180.0, degrees=True), [1.35, 0.0, 1.52])


def _write(path: Path, columns: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(pa.table(columns), str(path))


def _log(root: Path, *, sweeps: int = 4, images: int = 8) -> Path:
    log = root / "val" / "0000aaaa-log"
    rows = {"ring_front_center": CAMERA, "up_lidar": UP, "down_lidar": DOWN}
    _write(
        log / "calibration" / "egovehicle_SE3_sensor.feather",
        {"sensor_name": list(rows), **{k: [r[k] for r in rows.values()] for k in CAMERA}},
    )
    _write(
        log / "calibration" / "intrinsics.feather",
        {
            "sensor_name": ["ring_front_center"],
            "fx_px": [10.0],
            "fy_px": [10.0],
            "cx_px": [3.0],
            "cy_px": [4.0],
            "k1": [-0.2],
            "k2": [0.05],
            "k3": [0.01],
            "height_px": pa.array([8], pa.uint16()),
            "width_px": pa.array([6], pa.uint16()),
        },
    )
    stamps = [T0 - 200_000_000 + 5_000_000 * i for i in range(160)]
    _write(
        log / "city_SE3_egovehicle.feather",
        {
            "timestamp_ns": pa.array(stamps, pa.int64()),
            "qw": [1.0] * len(stamps),
            "qx": [0.0] * len(stamps),
            "qy": [0.0] * len(stamps),
            "qz": [0.0] * len(stamps),
            "tx_m": [100.0 + SPEED * (s - stamps[0]) * 1e-9 for s in stamps],
            "ty_m": [50.0] * len(stamps),
            "tz_m": [10.0] * len(stamps),
        },
    )
    for i in range(sweeps):
        lasers = np.repeat(np.arange(64, dtype=np.uint8), 20)
        points = np.column_stack(
            (np.linspace(5, 30, lasers.size), np.linspace(-4, 4, lasers.size), lasers * 0.05)
        ).astype(np.float16)
        _write(
            log / "sensors" / "lidar" / f"{T0 + 100_000_000 * i}.feather",
            {
                "x": pa.array(points[:, 0]),
                "y": pa.array(points[:, 1]),
                "z": pa.array(points[:, 2]),
                "intensity": pa.array(np.full(lasers.size, 7, np.uint8)),
                "laser_number": pa.array(lasers),
                "offset_ns": pa.array(np.linspace(0, 99_000_000, lasers.size).astype(np.int32)),
            },
        )
    folder = log / "sensors" / "cameras" / "ring_front_center"
    folder.mkdir(parents=True)
    for i in range(images):
        Image.fromarray(np.full((8, 6, 3), 30 * i, dtype=np.uint8)).save(
            folder / f"{T0 + 40_000_000 + 50_000_000 * i}.jpg"
        )
    return log


def _matrix(transform) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(transform.rotation_quat_xyzw).as_matrix()
    value[:3, 3] = transform.translation_m
    return value


def _ego(row: dict[str, float]) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]]).as_matrix()
    value[:3, 3] = [row["tx_m"], row["ty_m"], row["tz_m"]]
    return value


@pytest.mark.parametrize("lidar", ["up_lidar", "down_lidar"])
def test_round_trip_keeps_one_lidar_in_its_sensor_frame_and_the_tf_chain(
    tmp_path: Path, lidar: str
) -> None:
    log = _log(tmp_path)
    bag = tmp_path / "bag"
    result = av2.convert_av2_to_rosbag2(log, bag, lidar=lidar, point_time="offset_ns")
    listing = {c.topic: count for c, count in list_rosbag2_connections(bag)}
    assert listing[IMAGE_TOPIC] == listing[CAMERA_INFO_TOPIC] == 8
    assert listing[LIDAR_TOPIC] == 4 and listing[av2.ODOMETRY_TOPIC] > 0
    edges = {(e.parent, e.child): e.transform for e in load_bag_tf_static(bag).edges}
    chain = np.linalg.inv(_matrix(edges[("imu_link", "ring_front_center_optical")])) @ _matrix(
        edges[("imu_link", "velo_link")]
    )
    np.testing.assert_allclose(chain, result.provenance["camera"]["t_camera_velodyne"], atol=1e-9)
    np.testing.assert_allclose(
        result.provenance["camera"]["t_pose_velodyne"],
        _ego(UP if lidar == "up_lidar" else DOWN),
        atol=1e-12,
    )
    (_c, stamp, payload), *_ = list(iter_topic_messages(bag, LIDAR_TOPIC, limit=1))
    cloud = ros_cdr.decode_ros2_pointcloud2(LIDAR_TOPIC, stamp, payload, point_time_field="time")
    points = np.column_stack((cloud.xyz, cloud.point_time_offsets_s))
    assert stamp == T0 and points.shape[0] == 32 * 20 and cloud.frame_id == "velo_link"
    back = (
        points[:, :3].astype(np.float64) @ _ego(UP if lidar == "up_lidar" else DOWN)[:3, :3].T
        + _ego(UP if lidar == "up_lidar" else DOWN)[:3, 3]
    )
    lasers = np.repeat(np.arange(64), 20)
    mine = lasers < 32 if lidar == "up_lidar" else lasers >= 32
    expected = np.column_stack(
        (np.linspace(5, 30, lasers.size), np.linspace(-4, 4, lasers.size), lasers * 0.05)
    ).astype(np.float16)[mine]
    np.testing.assert_allclose(back, expected.astype(np.float64), atol=1e-4)
    times = np.linspace(0, 99_000_000, lasers.size).astype(np.int32)[mine] * 1e-9
    np.testing.assert_allclose(points[:, 3], times, atol=1e-7)


def test_images_are_undistorted_pinhole_and_sweeps_are_snapshots_by_default(
    tmp_path: Path,
) -> None:
    bag = tmp_path / "bag"
    result = av2.convert_av2_to_rosbag2(_log(tmp_path), bag)
    assert result.provenance["camera"]["source_k1_k2_k3"] == [-0.2, 0.05, 0.01]
    assert result.provenance["velodyne_time"]["mode"] == "sweep"
    (_c, stamp, payload), *_ = list(iter_topic_messages(bag, LIDAR_TOPIC, limit=1))
    cloud = ros_cdr.decode_ros2_pointcloud2(LIDAR_TOPIC, stamp, payload)
    assert [field.name for field in cloud.fields] == ["x", "y", "z", "intensity"]
    (_c, stamp, payload), *_ = list(iter_topic_messages(bag, CAMERA_INFO_TOPIC, limit=1))
    info = ros_cdr.decode_ros2_camera_info(CAMERA_INFO_TOPIC, stamp, payload)
    assert info.distortion_model == "plumb_bob"
    np.testing.assert_allclose(info.d, [0.0] * 5)
    assert (info.width, info.height) == (6, 8)
    (_c, stamp, payload), *_ = list(iter_topic_messages(bag, IMAGE_TOPIC, limit=1))
    image = ros_cdr.decode_ros2_image(IMAGE_TOPIC, stamp, payload, include_data=True)
    assert image.encoding == "mono8" and len(image.data) == 48


def test_jpeg_format_keeps_the_source_files_and_reads_as_their_gray_image(
    tmp_path: Path,
) -> None:
    from calibrex.data.ros2_camera_imu import iter_gray_images

    log, bag = _log(tmp_path), tmp_path / "bag"
    result = av2.convert_av2_to_rosbag2(log, bag, image_format="jpeg")
    topic = av2.COMPRESSED_IMAGE_TOPIC
    assert result.provenance["camera"]["image_topic"] == topic
    assert result.message_counts[topic] == 8 and IMAGE_TOPIC not in result.message_counts
    types = {c.topic: c.message_type for c, _count in list_rosbag2_connections(bag)}
    assert types[topic] == "sensor_msgs/msg/CompressedImage" and IMAGE_TOPIC not in types
    files = sorted((log / "sensors" / "cameras" / "ring_front_center").glob("*.jpg"))
    (_c, stamp, payload), *_ = list(iter_topic_messages(bag, topic, limit=1))
    message = ros_cdr.decode_ros2_compressed_image(topic, stamp, payload)
    assert message.data == files[0].read_bytes() and "jpeg" in message.format
    _time, gray = next(iter(iter_gray_images(bag, topic)))
    with Image.open(files[0]) as source:
        expected = np.asarray(source.convert("L"), dtype=np.int64)
    assert gray.shape == (8, 6)
    assert np.abs(gray.astype(np.int64) - expected).max() <= 1  # luma rounding only


def test_odometry_twist_is_the_body_velocity_and_the_window_selects(tmp_path: Path) -> None:
    log = _log(tmp_path)
    bag = tmp_path / "bag"
    result = av2.convert_av2_to_rosbag2(log, bag, start_s=0.1, max_seconds=0.2)
    assert result.message_counts[LIDAR_TOPIC] == 2
    assert result.message_counts[IMAGE_TOPIC] == 4
    rows = [
        ros_cdr.decode_ros2_odometry(av2.ODOMETRY_TOPIC, stamp, payload)
        for _c, stamp, payload in iter_topic_messages(bag, av2.ODOMETRY_TOPIC)
    ]
    middle = rows[len(rows) // 2]
    np.testing.assert_allclose(middle.linear_velocity, (SPEED, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(middle.angular_velocity, (0.0, 0.0, 0.0), atol=1e-9)
    provenance = json.loads(result.provenance_path.read_text())
    assert provenance["source"]["split"] == "val"
    assert provenance["lidar"]["laser_numbers"] == [0, 31]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"camera": "fisheye"}, "camera must be"),
        ({"lidar": "both"}, "lidar must be"),
        ({"point_time": "azimuth"}, "point_time must be"),
        ({"image_format": "png"}, "image_format must be"),
        ({"start_s": -1.0}, "start_s"),
        ({"max_seconds": 0.0}, "max_seconds"),
        ({"start_s": 50.0}, "no sweep or no image"),
    ],
)
def test_invalid_requests_write_nothing(tmp_path: Path, options: dict, message: str) -> None:
    bag = tmp_path / "bag"
    with pytest.raises(DatasetError, match=message):
        av2.convert_av2_to_rosbag2(_log(tmp_path), bag, **options)
    assert not bag.exists()


def test_output_must_not_overlap_or_replace_without_overwrite(tmp_path: Path) -> None:
    log = _log(tmp_path)
    with pytest.raises(DatasetError, match="overlap"):
        av2.convert_av2_to_rosbag2(log, log / "bag")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "file").write_text("x")
    with pytest.raises(DatasetError, match="already exists"):
        av2.convert_av2_to_rosbag2(log, occupied)


def test_missing_pyarrow_is_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(name: str):
        raise ImportError(name)

    monkeypatch.setattr(av2.importlib, "import_module", refuse)
    with pytest.raises(DatasetError, match=r"calibrex\[av2\]"):
        av2.convert_av2_to_rosbag2(_log(tmp_path), tmp_path / "bag")


def test_cli_converts_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bag = tmp_path / "bag"
    code = main(
        [
            "convert",
            "av2",
            str(_log(tmp_path)),
            "--lidar",
            "down_lidar",
            "--output",
            str(bag),
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["message_counts"][LIDAR_TOPIC] == 4
    assert "convert" in json.loads((bag / "calibrex_conversion.json").read_text())["command"]
