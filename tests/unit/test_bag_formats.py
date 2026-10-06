"""ROS 1 ``.bag`` input through the common bag reader, plus format auto-detection."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.unit.check_fixtures import default_tf, write_check_bag
from tests.unit.ros1_bag_fixtures import Ros1BagWriter, serialize

from calibrex.check.runner import build_calibration_check
from calibrex.cli.main import main
from calibrex.core.exceptions import DatasetError
from calibrex.data import ros1_cdr
from calibrex.data.bag_formats import detect_bag_format, is_ros1_bag
from calibrex.data.ros1_cdr import Ros1ToCdrTranscoder, ros2_message_type
from calibrex.data.ros_cdr import decode_ros2_imu, decode_ros2_livox_custommsg
from calibrex.data.ros_cdr_writer import (
    encode_camera_info,
    encode_image,
    encode_imu,
    encode_navsatfix,
    encode_odometry,
    encode_pointcloud2,
    encode_tf_message,
    encode_twist_stamped,
)
from calibrex.data.rosbag2 import (
    iter_messages,
    iter_topic_messages,
    list_rosbag2_connections,
    resolve_storage,
)

HAS_LZ4 = importlib.util.find_spec("lz4") is not None
SEC, NSEC = 12, 345
STAMP_NS = SEC * 1_000_000_000 + NSEC


def _header(frame_id: str) -> dict[str, Any]:
    return {"seq": 99, "stamp": (SEC, NSEC), "frame_id": frame_id}


def _transcode(ros1_type: str, values: dict[str, Any]) -> bytes:
    from tests.unit.ros1_bag_fixtures import message_definition

    return Ros1ToCdrTranscoder(ros1_type, message_definition(ros1_type)).transcode(
        serialize(ros1_type, values)
    )


# -- decoder round trips: ROS 1 -> CDR is byte-identical to the CDR encoders ------------


def test_imu_roundtrip() -> None:
    cdr = _transcode(
        "sensor_msgs/Imu",
        {
            "header": _header("imu_link"),
            "orientation": {"w": 1.0},
            "orientation_covariance": [-1.0] + [0.0] * 8,
            "angular_velocity": {"x": 0.1, "y": 0.2, "z": 0.3},
            "linear_acceleration": {"x": 1.0, "y": 2.0, "z": 9.8},
        },
    )
    assert cdr == encode_imu(
        frame_id="imu_link",
        timestamp_ns=STAMP_NS,
        angular_velocity=(0.1, 0.2, 0.3),
        linear_acceleration=(1.0, 2.0, 9.8),
    )
    message = decode_ros2_imu("/imu", 0, cdr)
    assert message.frame_id == "imu_link"


def test_tf_message_roundtrip() -> None:
    transforms = [
        ("base_link", "lidar", (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)),
        ("base_link", "imu", (0.0, 0.5, 0.0), (0.0, 0.0, 1.0, 0.0)),
    ]
    values = {
        "transforms": [
            {
                "header": {**_header(parent), "stamp": (5, 7)},
                "child_frame_id": child,
                "transform": {
                    "translation": dict(zip("xyz", t, strict=True)),
                    "rotation": dict(zip("xyzw", q, strict=True)),
                },
            }
            for parent, child, t, q in transforms
        ]
    }
    assert _transcode("tf2_msgs/TFMessage", values) == encode_tf_message(transforms)


def test_navsatfix_roundtrip() -> None:
    cdr = _transcode(
        "sensor_msgs/NavSatFix",
        {
            "header": {**_header("gps"), "stamp": (9, 3)},
            "status": {"status": 0, "service": 1},
            "latitude": 35.5,
            "longitude": 139.25,
            "altitude": 42.0,
            "position_covariance": [1.0, 0, 0, 0, 2.0, 0, 0, 0, 3.0],
            "position_covariance_type": 2,
        },
    )
    assert cdr == encode_navsatfix()


def test_odometry_and_twist_roundtrip() -> None:
    cdr = _transcode(
        "nav_msgs/Odometry",
        {
            "header": _header("odom"),
            "child_frame_id": "base",
            "pose": {"pose": {"position": {"x": 1.0}, "orientation": {"w": 1.0}}},
            "twist": {"twist": {"linear": {"x": 0.5}, "angular": {"z": 0.25}}},
        },
    )
    assert cdr == encode_odometry(
        frame_id="odom",
        child_frame_id="base",
        timestamp_ns=STAMP_NS,
        position=(1.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        linear_velocity=(0.5, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.25),
    )
    twist = _transcode(
        "geometry_msgs/TwistStamped",
        {"header": _header("v"), "twist": {"linear": {"x": 3.0}, "angular": {"y": 1.0}}},
    )
    assert twist == encode_twist_stamped(
        frame_id="v", timestamp_ns=STAMP_NS, linear=(3.0, 0.0, 0.0), angular=(0.0, 1.0, 0.0)
    )


def test_pointcloud2_roundtrip() -> None:
    data = np.arange(3, dtype="<f4").tobytes()
    fields = [("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1)]
    cdr4 = _transcode(
        "sensor_msgs/PointCloud2",
        {
            "header": _header("lidar"),
            "height": 1,
            "width": 1,
            "fields": [
                {"name": n, "offset": o, "datatype": d, "count": c} for n, o, d, c in fields
            ],
            "point_step": 12,
            "row_step": 12,
            "data": data,
            "is_dense": 1,
        },
    )
    assert cdr4 == encode_pointcloud2(
        frame_id="lidar", timestamp_ns=STAMP_NS, fields=fields, point_step=12, data=data
    )


def test_camera_info_roundtrip() -> None:
    k = [500.0, 0, 320, 0, 500.0, 240, 0, 0, 1.0]
    info = _transcode(
        "sensor_msgs/CameraInfo",
        {
            "header": _header("cam"),
            "height": 480,
            "width": 640,
            "distortion_model": "plumb_bob",
            "D": [0.1, 0.01],
            "K": k,
            "R": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
            "P": [500.0, 0, 320, 0, 0, 500.0, 240, 0, 0, 0, 1.0, 0],
        },
    )
    assert info == encode_camera_info(
        frame_id="cam", timestamp_ns=STAMP_NS, height=480, width=640, k=k, d=[0.1, 0.01]
    )


def test_image_roundtrip_exact() -> None:
    pixels = bytes(range(24))
    cdr = _transcode(
        "sensor_msgs/Image",
        {
            "header": _header("cam"),
            "height": 2,
            "width": 4,
            "encoding": "mono8",
            "step": 12,
            "data": pixels,
        },
    )
    assert cdr == encode_image(
        frame_id="cam",
        timestamp_ns=STAMP_NS,
        height=2,
        width=4,
        encoding="mono8",
        step=12,
        data=pixels,
    )


def _livox_values(count: int) -> dict[str, Any]:
    return {
        "header": _header("livox"),
        "timebase": 1_000_000,
        "point_num": count,
        "lidar_id": 3,
        "points": [
            {
                "offset_time": 100 * index,
                "x": float(index),
                "y": 0.5 * index,
                "z": -float(index),
                "reflectivity": index % 256,
                "tag": 1,
                "line": index % 4,
            }
            for index in range(count)
        ],
    }


@pytest.mark.parametrize("count", [0, 1, 5, 16, 257])
def test_livox_custommsg_roundtrip_and_vectorized_path_matches_scalar(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = _transcode("livox_ros_driver/CustomMsg", _livox_values(count))
    monkeypatch.setattr(ros1_cdr, "_VECTORIZE_MIN_ELEMENTS", 10**9)
    scalar = _transcode("livox_ros_driver/CustomMsg", _livox_values(count))
    assert fast == scalar
    message = decode_ros2_livox_custommsg("/livox", 0, fast)
    assert len(message.xyz) == count
    if count:
        assert message.xyz[count - 1].tolist() == [count - 1.0, 0.5 * (count - 1), -(count - 1.0)]
        assert message.lidar_id == 3


def test_truncated_and_trailing_payloads_are_rejected() -> None:
    from tests.unit.ros1_bag_fixtures import message_definition

    transcoder = Ros1ToCdrTranscoder("sensor_msgs/Imu", message_definition("sensor_msgs/Imu"))
    good = serialize("sensor_msgs/Imu", {"header": _header("x")})
    transcoder.transcode(good)
    with pytest.raises(DatasetError):
        transcoder.transcode(good[:-3])
    with pytest.raises(DatasetError):
        transcoder.transcode(good + b"\x00")


def test_type_names() -> None:
    assert ros2_message_type("sensor_msgs/Imu") == "sensor_msgs/msg/Imu"
    assert ros2_message_type("tf/tfMessage") == "tf2_msgs/msg/TFMessage"
    assert ros2_message_type("sensor_msgs/msg/Imu") == "sensor_msgs/msg/Imu"


# -- the ROS 1 backend --------------------------------------------------------------


def _imu(frame: str, stamp_ns: int, ax: float) -> bytes:
    return serialize(
        "sensor_msgs/Imu",
        {
            "header": {
                "stamp": (stamp_ns // 1_000_000_000, stamp_ns % 1_000_000_000),
                "frame_id": frame,
            },
            "linear_acceleration": {"x": ax},
        },
    )


def _write_imu_bag(path: Path, **options: Any) -> list[tuple[str, int, float]]:
    expected: list[tuple[str, int, float]] = []
    with Ros1BagWriter(path, **options) as writer:
        writer.add_connection("/imu", "sensor_msgs/Imu")
        writer.add_connection("/other", "sensor_msgs/Imu")
        for index in range(40):
            stamp = 1_000_000_000 + index * 10_000_000
            topic = "/imu" if index % 4 else "/other"
            writer.write(topic, stamp, _imu(topic[1:], stamp, float(index)))
            expected.append((topic, stamp, float(index)))
    return expected


def _read_back(path: Path, **kwargs: Any) -> list[tuple[str, int, float]]:
    result = []
    for connection, stamp, payload in iter_messages(path, **kwargs):
        assert connection.message_type == "sensor_msgs/msg/Imu"
        message = decode_ros2_imu(connection.topic, stamp, payload)
        result.append((connection.topic, stamp, message.linear_acceleration[0]))
    return result


@pytest.mark.parametrize(
    "compression",
    ["none", "bz2", pytest.param("lz4", marks=pytest.mark.skipif(not HAS_LZ4, reason="no lz4"))],
)
def test_ros1_backend_reads_every_chunk_compression(tmp_path: Path, compression: str) -> None:
    bag = tmp_path / f"{compression}.bag"
    expected = _write_imu_bag(bag, compression=compression, chunk_messages=7)
    assert _read_back(bag) == expected
    only = _read_back(bag, topics={"/imu"})
    assert only == [item for item in expected if item[0] == "/imu"]
    connections = {c.topic: (c.message_type, n) for c, n in list_rosbag2_connections(bag)}
    assert connections == {
        "/imu": ("sensor_msgs/msg/Imu", 30),
        "/other": ("sensor_msgs/msg/Imu", 10),
    }


def test_ros1_unindexed_bag_falls_back_to_a_scan(tmp_path: Path) -> None:
    bag = tmp_path / "unindexed.bag"
    expected = _write_imu_bag(bag, indexed=False, chunk_messages=9)
    assert _read_back(bag) == expected
    assert {c.topic: n for c, n in list_rosbag2_connections(bag)} == {"/imu": 30, "/other": 10}


def test_ros1_topic_query_skips_chunks_without_the_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bag = tmp_path / "skip.bag"
    with Ros1BagWriter(bag, chunk_messages=5) as writer:
        writer.add_connection("/a", "sensor_msgs/Imu")
        writer.add_connection("/b", "sensor_msgs/Imu")
        for index in range(10):
            writer.write("/a", index * 1000, _imu("a", index * 1000, 0.0))
        for index in range(10, 15):
            writer.write("/b", index * 1000, _imu("b", index * 1000, 0.0))
    from calibrex.data import rosbag1

    calls: list[str] = []
    original = rosbag1._decompress_chunk

    def counting(compression: str, size: int, data: bytes) -> bytes:
        calls.append(compression)
        return original(compression, size, data)

    monkeypatch.setattr(rosbag1, "_decompress_chunk", counting)
    messages = list(iter_topic_messages(bag, "/b"))
    assert len(messages) == 5 and len(calls) == 1
    first = list(iter_topic_messages(bag, "/a", limit=2))
    assert [stamp for _c, stamp, _p in first] == [0, 1000]


def test_ros1_messages_come_out_in_time_order_within_a_chunk(tmp_path: Path) -> None:
    bag = tmp_path / "order.bag"
    with Ros1BagWriter(bag) as writer:
        writer.add_connection("/imu", "sensor_msgs/Imu")
        for stamp in (3_000, 1_000, 2_000):
            writer.write("/imu", stamp, _imu("i", stamp, 0.0))
    assert [s for _c, s, _p in iter_messages(bag)] == [1_000, 2_000, 3_000]


def test_ros1_unreadable_definition_is_skipped_unless_requested(tmp_path: Path) -> None:
    bag = tmp_path / "stub.bag"
    with Ros1BagWriter(bag) as writer:
        writer.add_connection("/imu", "sensor_msgs/Imu")
        writer.add_connection("/weird", "vendor/Thing", definition="# no definition")
        writer.write("/imu", 1, _imu("i", 1, 0.0))
        writer.write("/weird", 2, b"\x01\x02")
    assert [c.topic for c, _s, _p in iter_messages(bag)] == ["/imu"]
    with pytest.raises(DatasetError, match="/weird"):
        list(iter_messages(bag, topics={"/weird"}))


# -- detection ---------------------------------------------------------------------


def test_format_detection(tmp_path: Path) -> None:
    from tests.unit.test_rosbag2 import _write_mcap_fixture, _write_sqlite_bag

    ros1 = tmp_path / "recording.bag"
    _write_imu_bag(ros1)
    odd = tmp_path / "recording.dat"
    odd.write_bytes(ros1.read_bytes())
    mcap = _write_mcap_fixture(tmp_path / "x.mcap")
    db3 = tmp_path / "x.db3"
    _write_sqlite_bag(db3, topics=[("/a", "std_msgs/msg/String")], messages=[])
    assert resolve_storage(ros1)[1] == "rosbag1"
    assert resolve_storage(odd)[1] == "rosbag1"
    assert detect_bag_format(ros1) == "rosbag1"
    assert detect_bag_format(odd) == "rosbag1"
    assert detect_bag_format(mcap) == "mcap"
    assert detect_bag_format(db3) == "sqlite3"
    assert is_ros1_bag(ros1) and not is_ros1_bag(mcap)
    (tmp_path / "junk.bin").write_bytes(b"not a bag")
    with pytest.raises(DatasetError):
        resolve_storage(tmp_path / "junk.bin")


# -- calibrex check on each format ------------------------------------------------------


def _write_ros1_check_bag(path: Path, **options: Any) -> Path:
    sensors = {
        "/imu": ("sensor_msgs/Imu", "imu_link"),
        "/lidar_front/points": ("sensor_msgs/PointCloud2", "lidar_front"),
        "/lidar_rear/points": ("sensor_msgs/PointCloud2", "lidar_rear"),
        "/camera/image_raw": ("sensor_msgs/Image", "camera_optical"),
        "/gnss/fix": ("sensor_msgs/NavSatFix", "gnss_link"),
    }
    with Ros1BagWriter(path, **options) as writer:
        writer.add_connection("/tf_static", "tf2_msgs/TFMessage")
        writer.write(
            "/tf_static",
            1_000,
            serialize(
                "tf2_msgs/TFMessage",
                {
                    "transforms": [
                        {
                            "header": {"frame_id": parent},
                            "child_frame_id": child,
                            "transform": {
                                "translation": dict(zip("xyz", t, strict=True)),
                                "rotation": dict(zip("xyzw", q, strict=True)),
                            },
                        }
                        for parent, child, t, q in default_tf()
                    ]
                },
            ),
        )
        for index, (topic, (ros1_type, frame)) in enumerate(sensors.items()):
            writer.add_connection(topic, ros1_type)
            writer.write(
                topic, 2_000 + index, serialize(ros1_type, {"header": {"frame_id": frame}})
            )
    return path


def _plan_view(bag: Path) -> dict[str, Any]:
    artifact = build_calibration_check(bag).model_dump(mode="json", exclude_none=True)
    for topic in artifact["topics"]:
        topic.pop("message_count", None)  # a ROS 1 index knows counts; a bare db3 does not
    return {key: artifact[key] for key in ("topics", "pairs", "frame_tree", "summary")} | {
        "storage": artifact["bag"]["storage_identifier"]
    }


def test_check_plan_of_a_ros1_bag_matches_the_rosbag2_plan(tmp_path: Path) -> None:
    sqlite_dir = tmp_path / "db3"
    sqlite_dir.mkdir()
    reference = _plan_view(write_check_bag(sqlite_dir / "ref.db3", tf_messages=[default_tf()]))
    ros1 = _plan_view(_write_ros1_check_bag(tmp_path / "rig.bag", compression="bz2"))
    assert ros1.pop("storage") == "rosbag1"
    assert reference.pop("storage") == "sqlite3"
    assert ros1 == reference
    assert ros1["summary"]["runnable_count"] > 0


def test_check_plan_of_an_lz4_mcap_matches_the_rosbag2_plan(tmp_path: Path) -> None:
    pytest.importorskip("lz4.block")
    from tests.unit.test_rosbag2 import _write_mcap_bag

    sqlite_dir = tmp_path / "db3"
    sqlite_dir.mkdir()
    db3 = write_check_bag(sqlite_dir / "ref.db3", tf_messages=[default_tf()])
    messages = [
        (c.topic_id, c.topic_id, c.topic, c.message_type, stamp, bytes(data))
        for c, stamp, data in iter_messages(db3)
    ]
    mcap = tmp_path / "rig.mcap"
    _write_mcap_bag(mcap, messages=messages, chunked=True, compression="lz4")
    reference = _plan_view(db3)
    plan = _plan_view(mcap)
    assert plan.pop("storage") == "mcap"
    reference.pop("storage")
    assert plan == reference


def test_cli_check_plan_accepts_a_bag_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bag = _write_ros1_check_bag(tmp_path / "rig.bag")
    out = tmp_path / "plan.json"
    assert main(["check", str(bag), "--plan", "--output", str(out)]) == 0
    text = capsys.readouterr().out
    assert "imu" in text and out.is_file()


def test_header_stamps_survive_the_transcoding(tmp_path: Path) -> None:
    bag = tmp_path / "stamp.bag"
    with Ros1BagWriter(bag) as writer:
        writer.add_connection("/imu", "sensor_msgs/Imu")
        writer.write("/imu", 5, _imu("frame", 12_000_000_345, 1.0))
    ((connection, stamp, payload),) = itertools.islice(iter_messages(bag), 1)
    message = decode_ros2_imu(connection.topic, stamp, payload)
    assert message.frame_id == "frame" and message.timestamp_ns == 12_000_000_345


# -- MCAP chunk compression as real writers emit it ----------------------------------------


def _write_library_mcap(path: Path, compression: str) -> list[tuple[int, bytes]]:
    writer_module = pytest.importorskip("mcap.writer")
    pytest.importorskip("lz4.frame")
    pytest.importorskip("zstandard")
    kind = {
        "lz4": writer_module.CompressionType.LZ4,
        "zstd": writer_module.CompressionType.ZSTD,
        "none": writer_module.CompressionType.NONE,
    }[compression]
    expected = []
    with path.open("wb") as stream:
        writer = writer_module.Writer(stream, compression=kind, chunk_size=2048)
        writer.start()
        schema = writer.register_schema("sensor_msgs/msg/Imu", "ros2msg", b"")
        channel = writer.register_channel("/imu", "cdr", schema)
        for index in range(60):
            payload = encode_imu(
                frame_id="imu",
                timestamp_ns=index,
                angular_velocity=(0.0, 0.0, 0.0),
                linear_acceleration=(float(index), 0.0, 0.0),
            )
            writer.add_message(channel, 1_000 + index, payload, 1_000 + index)
            expected.append((1_000 + index, payload))
        writer.finish()
    return expected


@pytest.mark.parametrize("compression", ["none", "lz4", "zstd"])
def test_mcap_chunks_written_by_the_mcap_library_are_read(tmp_path: Path, compression: str) -> None:
    path = tmp_path / f"{compression}.mcap"
    expected = _write_library_mcap(path, compression)
    got = [(stamp, bytes(payload)) for _c, stamp, payload in iter_messages(path)]
    assert got == expected
    assert detect_bag_format(path) == "mcap"


def test_file_compressed_mcap_directory_is_streamed(tmp_path: Path) -> None:
    zstandard = pytest.importorskip("zstandard")
    expected = _write_library_mcap(tmp_path / "plain.mcap", "none")
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "bag.mcap.zstd").write_bytes(
        zstandard.ZstdCompressor().compress((tmp_path / "plain.mcap").read_bytes())
    )
    (bag / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  version: 9\n  storage_identifier: mcap\n"
        "  compression_format: zstd\n  compression_mode: file\n"
        "  relative_file_paths:\n  - bag.mcap.zstd\n  topics_with_message_count: []\n",
        encoding="utf-8",
    )
    got = [(stamp, bytes(payload)) for _c, stamp, payload in iter_messages(bag)]
    assert got == expected
    assert [c.topic for c, _n in list_rosbag2_connections(bag)] == ["/imu"]


def test_estimator_input_digests_accept_a_bare_bag_file(tmp_path: Path) -> None:
    from calibrex.data.livox_ros2 import bag_input_digest
    from calibrex.data.rtk_slam import rtk_slam_input_digest

    bag = tmp_path / "recording.bag"
    _write_imu_bag(bag)
    digest, scope = bag_input_digest([bag])
    assert len(digest) == 64 and "metadata.yaml" in scope
    rtk = tmp_path / "rtk.txt"
    calib = tmp_path / "calib.yaml"
    rtk.write_text("0 0 0\n", encoding="utf-8")
    calib.write_text("a: 1\n", encoding="utf-8")
    rtk_digest, _scope = rtk_slam_input_digest(bag, rtk, calib)
    assert len(rtk_digest) == 64
    bag.write_bytes(bag.read_bytes()[:-1])  # a different file digests differently
    assert bag_input_digest([bag])[0] != digest


# -- global time order across overlapping chunks ---------------------------------------------


def test_ros1_chunks_with_overlapping_time_ranges_merge_in_global_order(tmp_path: Path) -> None:
    bag = tmp_path / "overlap.bag"
    stamps = [1, 5, 9, 3, 7, 11, 4, 8, 12, 2, 6, 10]
    with Ros1BagWriter(bag, chunk_messages=3) as writer:
        writer.add_connection("/imu", "sensor_msgs/Imu")
        for stamp in stamps:
            writer.write("/imu", stamp * 1000, _imu("i", stamp * 1000, float(stamp)))
    got = [s for _c, s, _p in iter_messages(bag)]
    assert got == sorted(s * 1000 for s in stamps)
    only = [s for _c, s, _p in iter_messages(bag, topics={"/imu"})]
    assert only == got


def test_mcap_chunks_with_overlapping_time_ranges_merge_in_global_order(tmp_path: Path) -> None:
    import struct

    from tests.unit.test_rosbag2 import (
        MCAP_MAGIC,
        _mcap_channel,
        _mcap_chunk,
        _mcap_message,
        _mcap_record,
        _mcap_schema,
        _mcap_string,
    )

    preamble = _mcap_schema(1, "sensor_msgs/msg/Imu") + _mcap_channel(1, 1, "/imu")
    groups = [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]  # chunk starts rise, ranges overlap
    body = b""
    for number, group in enumerate(groups):
        records = (preamble if number == 0 else b"") + b"".join(
            _mcap_message(1, stamp * 1000, b"\x00\x01\x00\x00") for stamp in group
        )
        body += _mcap_chunk(records, start_time=group[0] * 1000, end_time=group[-1] * 1000)
    header = _mcap_record(0x01, _mcap_string("rosbag2") + _mcap_string("slac-test"))
    footer = _mcap_record(0x02, struct.pack("<QQI", 0, 0, 0))
    path = tmp_path / "overlap.mcap"
    path.write_bytes(
        MCAP_MAGIC + header + body + _mcap_record(0x0F, struct.pack("<I", 0)) + footer + MCAP_MAGIC
    )
    got = [s for _c, s, _p in iter_messages(path)]
    assert got == [stamp * 1000 for stamp in range(1, 13)]
