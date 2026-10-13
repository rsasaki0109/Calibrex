"""Camera intrinsics from a CameraInfo topic, for raw and rectified image streams."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrex.check.estimators import camera_model_from_camera_info, is_rectified_image_topic
from calibrex.data.ros_cdr_writer import encode_camera_info
from calibrex.data.rosbag2_writer import Rosbag2Writer

K = (1127.3, 0.0, 1222.6, 0.0, 1126.7, 994.3, 0.0, 0.0, 1.0)
D = (-0.27, 0.08, 0.0002, 0.0003, -0.011)
# the rectified camera image_proc publishes: its own focal length and centre
P = (1050.0, 0.0, 1230.0, 0.0, 0.0, 1050.0, 1000.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _bag(path: Path) -> Path:
    with Rosbag2Writer(path) as writer:
        writer.add_topic("/cam/camera_info", "sensor_msgs/msg/CameraInfo")
        writer.write(
            "/cam/camera_info",
            1,
            encode_camera_info(
                frame_id="cam", timestamp_ns=1, height=2048, width=2448, k=K, d=D, p=P
            ),
        )
    return path


def test_a_raw_image_uses_k_and_its_distortion(tmp_path: Path) -> None:
    model = camera_model_from_camera_info(_bag(tmp_path / "bag"), "/cam/camera_info")
    assert (model.fx, model.fy, model.cx, model.cy) == (K[0], K[4], K[2], K[5])
    assert model.distortion_model == "radtan"
    assert model.distortion == pytest.approx(D)


def test_a_rectified_image_uses_p_without_distortion(tmp_path: Path) -> None:
    model = camera_model_from_camera_info(
        _bag(tmp_path / "bag"), "/cam/camera_info", rectified=True
    )
    assert (model.fx, model.fy, model.cx, model.cy) == (P[0], P[5], P[2], P[6])
    assert model.distortion_model == "none"
    assert model.distortion == ()


@pytest.mark.parametrize(
    ("topic", "rectified"),
    [
        ("/sensor/camera/front/image_rect_color", True),
        ("/cam/image_rect", True),
        ("/cam/image_rect/", True),
        ("/cam/image_raw", False),
        ("/cam/image_color", False),
        ("/rect/image_raw", False),
    ],
)
def test_rectified_streams_are_named_image_rect(topic: str, rectified: bool) -> None:
    assert is_rectified_image_topic(topic) is rectified
