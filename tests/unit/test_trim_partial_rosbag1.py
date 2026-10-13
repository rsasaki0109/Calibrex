"""tools/trim_partial_rosbag1.py: a ROS 1 bag cut mid-record reads as a shorter recording."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from tests.unit.ros1_bag_fixtures import Ros1BagWriter, serialize

from calibrex.core.exceptions import DatasetError
from calibrex.data.rosbag1 import iter_messages

TOOL = Path(__file__).resolve().parents[2] / "tools" / "trim_partial_rosbag1.py"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("trim_partial_rosbag1", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bag(path: Path, messages: int = 12) -> Path:
    writer = Ros1BagWriter(path, chunk_messages=3)
    writer.add_connection("/imu", "sensor_msgs/Imu")
    for index in range(messages):
        stamp = 1_000_000_000 + index * 10_000_000
        header = {"seq": index, "stamp": (stamp // 10**9, stamp % 10**9), "frame_id": "imu"}
        writer.write("/imu", stamp, serialize("sensor_msgs/Imu", {"header": header}))
    writer.close()
    return path


def test_a_bag_cut_inside_a_chunk_reads_its_complete_chunks(tmp_path: Path) -> None:
    bag = _bag(tmp_path / "full.bag")
    full = [stamp for _, stamp, _ in iter_messages(bag)]
    assert len(full) == 12
    data = bag.read_bytes()
    cut = tmp_path / "head.bag"
    cut.write_bytes(data[: len(data) // 2])  # a range download ending mid-record
    with pytest.raises(DatasetError, match="unexpected end"):
        list(iter_messages(cut))

    assert _tool().main([str(cut)]) == 0
    head = [stamp for _, stamp, _ in iter_messages(cut)]
    assert 0 < len(head) < len(full) and len(head) % 3 == 0  # whole chunks of three
    assert head == full[: len(head)]


def test_a_complete_bag_keeps_every_message(tmp_path: Path) -> None:
    bag = _bag(tmp_path / "full.bag")
    before = [stamp for _, stamp, _ in iter_messages(bag)]
    end, records = _tool().last_complete_record_end(bag)
    assert end == bag.stat().st_size and records > 0
    assert _tool().main([str(bag)]) == 0
    assert [stamp for _, stamp, _ in iter_messages(bag)] == before


def test_a_file_that_is_not_a_bag_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other.bag"
    other.write_bytes(b"not a bag at all")
    with pytest.raises(ValueError, match="not a ROS 1 bag"):
        _tool().last_complete_record_end(other)
