"""Livox ``CustomMsg`` scans in the readers the check's LiDAR pairs share.

A ``CustomMsg`` scan must read exactly like a ``PointCloud2`` carrying the same points
and a float64 offset-seconds time field: same header times, points and per-point
offsets (``offset_time`` nanoseconds after the header stamp), with empty returns at the
origin dropped.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibrex.check.estimators import CheckSkipError, _require_lidar_topic, detect_point_time
from calibrex.data.livox_ros2 import (
    LIVOX_POINT_TIME_FIELD,
    LivoxStreamProfile,
    decode_lidar_scan,
    iter_livox_points,
)
from calibrex.data.ros_cdr_writer import (
    POINT_FIELD_FLOAT32,
    POINT_FIELD_FLOAT64,
    encode_livox_custommsg,
    encode_pointcloud2,
)
from calibrex.data.rosbag2_writer import Rosbag2Writer

CUSTOM = "livox_ros_driver2/msg/CustomMsg"
CLOUD = "sensor_msgs/msg/PointCloud2"
BASE_NS = 1_700_000_000 * 10**9
FIELDS = (
    ("x", 0, POINT_FIELD_FLOAT32, 1),
    ("y", 4, POINT_FIELD_FLOAT32, 1),
    ("z", 8, POINT_FIELD_FLOAT32, 1),
    ("time", 16, POINT_FIELD_FLOAT64, 1),
)


def _scan(index: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(index)
    xyz = rng.uniform(-20.0, 20.0, (200, 3)).astype(np.float32)
    offsets_ns = np.sort(rng.integers(0, 100_000_000, 200)).astype(np.uint32)
    return xyz, offsets_ns


def _cloud_payload(stamp: int, xyz: np.ndarray, offsets_ns: np.ndarray) -> bytes:
    records = np.zeros(len(xyz), dtype=[("xyz", "<f4", 3), ("pad", "<u4"), ("time", "<f8")])
    records["xyz"] = xyz
    records["time"] = offsets_ns.astype(np.float64) * 1.0e-9
    return encode_pointcloud2(
        frame_id="livox", timestamp_ns=stamp, fields=FIELDS, point_step=24, data=records.tobytes()
    )


def _custom_payload(stamp: int, xyz: np.ndarray, offsets_ns: np.ndarray) -> bytes:
    points = [
        (int(t), float(x), float(y), float(z), 10, 0, i % 6)
        for i, (t, (x, y, z)) in enumerate(zip(offsets_ns, xyz, strict=True))
    ]
    return encode_livox_custommsg(frame_id="livox", timestamp_ns=stamp, points=points)


def _bag(path: Path, message_type: str) -> Path:
    with Rosbag2Writer(path) as writer:
        writer.add_topic("/livox/lidar", message_type)
        for index in range(3):
            stamp = BASE_NS + index * 100_000_000
            xyz, offsets = _scan(index)
            encode = _custom_payload if message_type == CUSTOM else _cloud_payload
            writer.write("/livox/lidar", stamp, encode(stamp, xyz, offsets))
    return path


def test_a_custommsg_scan_drops_empty_returns_and_reads_offset_time() -> None:
    payload = encode_livox_custommsg(
        frame_id="livox",
        timestamp_ns=12 * 10**9,
        points=[
            (0, 1.0, 2.0, 3.0, 10, 0, 1),
            (5_000, 0.0, 0.0, 0.0, 0, 0, 2),  # no return
            (100_000, -4.0, 5.5, 6.0, 20, 0, 3),
        ],
    )
    scan = decode_lidar_scan(CUSTOM, "/livox/lidar", 1, payload, point_time_field="offset_time")
    assert (scan.timestamp_ns, scan.frame_id) == (12 * 10**9, "livox")
    np.testing.assert_array_equal(scan.xyz, [[1.0, 2.0, 3.0], [-4.0, 5.5, 6.0]])
    assert scan.offsets_s is not None
    np.testing.assert_allclose(scan.offsets_s, [0.0, 1.0e-4])
    rigid = decode_lidar_scan(CUSTOM, "/livox/lidar", 1, payload)
    assert rigid.offsets_s is None and len(rigid.xyz) == 2


def test_custommsg_scans_read_like_the_same_pointcloud2(tmp_path: Path) -> None:
    custom = _bag(tmp_path / "custom", CUSTOM)
    cloud = _bag(tmp_path / "cloud", CLOUD)
    by_custom = list(
        iter_livox_points(
            custom,
            LivoxStreamProfile("c", "/livox/lidar", "", LIVOX_POINT_TIME_FIELD, "offset_s", "mps2"),
        )
    )
    by_cloud = list(
        iter_livox_points(
            cloud, LivoxStreamProfile("p", "/livox/lidar", "", "time", "offset_s", "mps2")
        )
    )
    assert len(by_custom) == len(by_cloud) == 3
    for (t_a, xyz_a, off_a), (t_b, xyz_b, off_b) in zip(by_custom, by_cloud, strict=True):
        assert t_a == t_b
        np.testing.assert_array_equal(xyz_a, xyz_b)
        np.testing.assert_array_equal(off_a, off_b)


def test_point_time_of_a_custommsg_topic_is_its_offset_time(tmp_path: Path) -> None:
    assert detect_point_time(_bag(tmp_path / "custom", CUSTOM), "/livox/lidar") == (
        "offset_time",
        "offset_s",
    )
    assert detect_point_time(_bag(tmp_path / "cloud", CLOUD), "/livox/lidar") == (
        "time",
        "offset_s",
    )


def test_a_pair_reads_pointcloud2_first_then_custommsg() -> None:
    def pick(types: dict[str, str]) -> str:
        return _require_lidar_topic(SimpleNamespace(topic_types=types), sorted(types))  # type: ignore[arg-type]

    assert pick({"/a/custom": CUSTOM, "/b/points": CLOUD}) == "/b/points"
    assert pick({"/a/custom": "livox_ros_driver/msg/CustomMsg"}) == "/a/custom"
    with pytest.raises(CheckSkipError) as skipped:
        pick({"/a/scan": "velodyne_msgs/msg/VelodyneScan"})
    assert skipped.value.code == "unsupported_sensor"


def test_a_duration_cap_projects_the_scan_total_for_the_progress_eta(tmp_path: Path) -> None:
    """The first pass of a capped run has no message count to go by; the scan rate gives one."""

    from calibrex.core.progress import ProgressEvent, progress_sink

    bag = tmp_path / "long"
    with Rosbag2Writer(bag) as writer:
        writer.add_topic("/livox/lidar", CLOUD)
        for index in range(40):  # 4 s at 10 Hz
            stamp = BASE_NS + index * 100_000_000
            writer.write("/livox/lidar", stamp, _cloud_payload(stamp, *_scan(index)))
    profile = LivoxStreamProfile("p", "/livox/lidar", "", "time", "offset_s", "mps2")

    def ticks(max_seconds: float | None) -> tuple[int, list[ProgressEvent]]:
        events: list[ProgressEvent] = []
        with progress_sink(events.append):
            scans = list(iter_livox_points(bag, profile, max_seconds=max_seconds))
        return len(scans), events

    read, capped = ticks(2.0)
    assert [event.done for event in capped] == list(range(1, read + 1))
    assert all(event.total is None for event in capped[:11])  # under a second of data
    totals = [event.total for event in capped[11:]]
    assert totals and all(total is not None and abs(total - read) <= 2 for total in totals)
    assert capped[-1].total == read
    assert all(event.total is None for event in ticks(None)[1])  # the runner knows that total
