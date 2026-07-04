"""CLI validation tests for the rosbag1→rosbag2 conversion tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "tools" / "rosbag1_to_rosbag2_online_pair.py"


def _load_tool_module():
    import sys
    from unittest.mock import MagicMock

    if "rosbags" not in sys.modules:
        rosbags = MagicMock()
        rosbags.highlevel = MagicMock()
        rosbags.rosbag2 = MagicMock()
        rosbags.typesys = MagicMock()
        sys.modules["rosbags"] = rosbags
        sys.modules["rosbags.highlevel"] = rosbags.highlevel
        sys.modules["rosbags.rosbag2"] = rosbags.rosbag2
        sys.modules["rosbags.typesys"] = rosbags.typesys

    spec = importlib.util.spec_from_file_location("rosbag1_to_rosbag2_online_pair", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool_module():
    return _load_tool_module()


def test_parse_args_rejects_native_deskew_with_two_pass(tool_module) -> None:
    with pytest.raises(SystemExit):
        tool_module._parse_args(
            [
                "--src",
                "in.bag",
                "--dst",
                "out",
                "--topic",
                "/velodyne_points",
                "--odom-source",
                "kiss-icp-two-pass",
                "--kiss-icp-topic",
                "/velodyne_points",
                "--kiss-icp-native-deskew",
            ]
        )


def test_parse_args_rejects_native_deskew_without_kiss_icp(tool_module) -> None:
    with pytest.raises(SystemExit):
        tool_module._parse_args(
            [
                "--src",
                "in.bag",
                "--dst",
                "out",
                "--topic",
                "/velodyne_points",
                "--odom-source",
                "pose-topic",
                "--pose-topic",
                "/pose",
                "--kiss-icp-native-deskew",
            ]
        )


def test_pointcloud2_field_error_lists_available_fields(tool_module) -> None:
    class _Field:
        def __init__(self, name: str, offset: int, datatype: int, count: int) -> None:
            self.name = name
            self.offset = offset
            self.datatype = datatype
            self.count = count

    class _Message:
        fields = (
            _Field("x", 0, 7, 1),
            _Field("y", 4, 7, 1),
            _Field("z", 8, 7, 1),
        )
        is_bigendian = False
        point_step = 16
        height = 1
        width = 1
        data = (b"\x00" * 16)

    with pytest.raises(ValueError, match="missing field 'time'"):
        tool_module._pointcloud2_field(_Message(), "time")


try:
    import kiss_icp  # noqa: F401

    has_kiss_icp = True
except ModuleNotFoundError:
    has_kiss_icp = False

requires_kiss_icp = pytest.mark.skipif(not has_kiss_icp, reason="kiss-icp not installed")


@requires_kiss_icp
def test_create_kiss_icp_sets_deskew_flag(tool_module) -> None:
    kiss, version = tool_module._create_kiss_icp(30.0, native_deskew=True)
    assert version
    assert kiss.config.data.deskew is True
    kiss_raw, _ = tool_module._create_kiss_icp(30.0, native_deskew=False)
    assert kiss_raw.config.data.deskew is False
