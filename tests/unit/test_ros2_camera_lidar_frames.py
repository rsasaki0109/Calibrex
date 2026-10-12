"""Frame selection of ``load_camera_lidar_frames`` on a small synthetic rosbag2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calibrex.data.ros2_camera_lidar import load_camera_lidar_frames
from calibrex.data.ros_cdr_writer import POINT_FIELD_FLOAT32, encode_image, encode_pointcloud2
from calibrex.data.rosbag2_writer import Rosbag2Writer

FIELDS = (
    ("x", 0, POINT_FIELD_FLOAT32, 1),
    ("y", 4, POINT_FIELD_FLOAT32, 1),
    ("z", 8, POINT_FIELD_FLOAT32, 1),
)


def _write(path: Path, *, moving: bool) -> Path:
    """Six seconds of 10 Hz scans, each with an image 5 ms later (changing when moving)."""

    rng = np.random.default_rng(3)
    points = np.asarray([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0], [1.0, 1.0, 6.0]], dtype=np.float32)
    with Rosbag2Writer(path) as writer:
        writer.add_topic("/image", "sensor_msgs/msg/Image")
        writer.add_topic("/points", "sensor_msgs/msg/PointCloud2")
        for index in range(60):
            stamp = 1_000_000_000 + index * 100_000_000
            writer.write(
                "/points",
                stamp,
                encode_pointcloud2(
                    frame_id="lidar",
                    timestamp_ns=stamp,
                    fields=FIELDS,
                    point_step=12,
                    data=points.tobytes(),
                ),
            )
            writer.write(
                "/image",
                stamp + 5_000_000,
                encode_image(
                    frame_id="camera",
                    timestamp_ns=stamp + 5_000_000,
                    height=4,
                    width=4,
                    encoding="mono8",
                    step=4,
                    data=rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
                    if moving
                    else bytes(16),
                ),
            )
    return path


@pytest.fixture(scope="module")
def bag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write(tmp_path_factory.mktemp("frames") / "bag", moving=True)


def _times(frames: list) -> list[float]:
    return [round(frame.time_s - 1.0, 3) for frame in frames]


def test_regular_spacing_without_the_short_recording_option(bag: Path) -> None:
    frames = load_camera_lidar_frames(bag, "/image", "/points", spacing_s=1.0, max_frames=48)

    assert _times(frames) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_a_short_recording_spreads_the_frame_budget(bag: Path) -> None:
    report: dict[str, object] = {}
    frames = load_camera_lidar_frames(
        bag,
        "/image",
        "/points",
        spacing_s=1.0,
        max_frames=48,
        short_recording_spacing_s=0.25,
        report=report,
    )

    # 6 s is too short for 48 scans 1 s apart: scans at least 0.25 s apart are used.
    assert _times(frames) == [round(0.3 * index, 3) for index in range(20)]
    assert all(frame.gap_s == pytest.approx(0.005) for frame in frames)
    assert report["selection"] == "short_recording"


def test_a_short_static_recording_keeps_the_regular_frames(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    static = _write(tmp_path_factory.mktemp("frames") / "static", moving=False)
    report: dict[str, object] = {}

    frames = load_camera_lidar_frames(
        static,
        "/image",
        "/points",
        spacing_s=1.0,
        max_frames=48,
        short_recording_spacing_s=0.25,
        report=report,
    )

    # The same view again and again is not more evidence.
    assert _times(frames) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert report == {"selection": "regular_static_view", "median_view_change": 0.0}


def test_a_long_enough_recording_keeps_the_regular_frames(bag: Path) -> None:
    regular = load_camera_lidar_frames(bag, "/image", "/points", spacing_s=1.0, max_frames=5)
    with_option = load_camera_lidar_frames(
        bag, "/image", "/points", spacing_s=1.0, max_frames=5, short_recording_spacing_s=0.25
    )

    assert _times(with_option) == _times(regular) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_the_dense_set_stays_bounded_and_spans_the_recording(bag: Path) -> None:
    frames = load_camera_lidar_frames(
        bag, "/image", "/points", spacing_s=10.0, max_frames=4, short_recording_spacing_s=0.1
    )

    times = _times(frames)
    assert len(times) == 4
    assert times[0] == 0.0 and times[-1] >= 4.5
    assert times == sorted(times)
