"""Self-contained ROS 1/ROS 2 Image and CameraInfo adapter tests."""

from __future__ import annotations

import math
import sqlite3
import struct
from pathlib import Path

import pytest

from calibrex.core.capture_manifest import (
    SensorIdentity,
    SourceInventory,
    StreamInventory,
    _evaluate_capture,
    _infer_bindings,
    inspect_capture,
)
from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import decode_ros2_camera_info, decode_ros2_image
from calibrex.data.ros_messages import (
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
    CameraInfoMessage,
    RegionOfInterest,
)
from calibrex.data.rosbag1 import decode_camera_info as decode_ros1_camera_info
from calibrex.data.rosbag1 import decode_image as decode_ros1_image
from calibrex.data.rosbag2 import summarize_rosbag2


def _ros1_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _ros1_header(*, frame_id: str, secs: int = 3, nsecs: int = 4) -> bytes:
    return struct.pack("<III", 9, secs, nsecs) + _ros1_string(frame_id)


def _ros1_image(
    *,
    width: int = 2,
    height: int = 2,
    encoding: str = "mono8",
    step: int = 2,
    payload: bytes = b"abcd",
    frame_id: str = "camera",
    is_bigendian: int = 0,
) -> bytes:
    return (
        _ros1_header(frame_id=frame_id)
        + struct.pack("<II", height, width)
        + _ros1_string(encoding)
        + struct.pack("<BI", is_bigendian, step)
        + struct.pack("<I", len(payload))
        + payload
    )


def _ros1_camera_info(
    *,
    width: int = 640,
    height: int = 480,
    frame_id: str = "camera",
    k: tuple[float, ...] | None = None,
    p: tuple[float, ...] | None = None,
    d: tuple[float, ...] = (-0.1, 0.01),
    roi: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    distortion_model: str = "plumb_bob",
) -> bytes:
    k_values = k or (500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0)
    p_values = p or (
        500.0,
        0.0,
        320.0,
        0.0,
        0.0,
        500.0,
        240.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    return (
        _ros1_header(frame_id=frame_id)
        + struct.pack("<II", height, width)
        + _ros1_string(distortion_model)
        + struct.pack("<I", len(d))
        + struct.pack(f"<{len(d)}d", *d)
        + struct.pack("<9d", *k_values)
        + struct.pack("<9d", *([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
        + struct.pack("<12d", *p_values)
        + struct.pack("<II", 1, 1)
        + struct.pack("<IIII B", *roi)
    )


class _CdrWriter:
    def __init__(self, *, little_endian: bool = True) -> None:
        self._endian = "<" if little_endian else ">"
        self._buf = bytearray([0, 1 if little_endian else 0, 0, 0])

    def align(self, alignment: int) -> None:
        relative = len(self._buf) - 4
        self._buf.extend(b"\x00" * ((-relative) % alignment))

    def _pack(self, fmt: str, value: object) -> None:
        self.align(struct.calcsize(fmt))
        self._buf.extend(struct.pack(f"{self._endian}{fmt}", value))

    def int32(self, value: int) -> None:
        self._pack("i", value)

    def uint8(self, value: int) -> None:
        self._pack("B", value)

    def uint32(self, value: int) -> None:
        self._pack("I", value)

    def float64(self, value: float) -> None:
        self.align(8)
        self._buf.extend(struct.pack(f"{self._endian}d", value))

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8") + b"\x00"
        self.uint32(len(encoded))
        self._buf.extend(encoded)

    def bytes(self, value: bytes) -> None:
        self.uint32(len(value))
        self._buf.extend(value)

    def floats(self, values: tuple[float, ...]) -> None:
        for value in values:
            self.float64(value)

    def finish(self) -> bytes:
        return bytes(self._buf)


def _ros2_image(
    *,
    width: int = 2,
    height: int = 2,
    encoding: str = "mono8",
    step: int = 2,
    payload: bytes = b"abcd",
    frame_id: str = "camera",
    is_bigendian: int = 0,
    little_endian: bool = True,
) -> bytes:
    writer = _CdrWriter(little_endian=little_endian)
    writer.int32(3)
    writer.uint32(4)
    writer.string(frame_id)
    writer.uint32(height)
    writer.uint32(width)
    writer.string(encoding)
    writer.uint8(is_bigendian)
    writer.uint32(step)
    writer.bytes(payload)
    return writer.finish()


def _ros2_camera_info(
    *,
    width: int = 640,
    height: int = 480,
    frame_id: str = "camera",
    k: tuple[float, ...] | None = None,
    p: tuple[float, ...] | None = None,
    d: tuple[float, ...] = (-0.1, 0.01),
    roi: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    distortion_model: str = "plumb_bob",
    little_endian: bool = True,
) -> bytes:
    k_values = k or (500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0)
    p_values = p or (
        500.0,
        0.0,
        320.0,
        0.0,
        0.0,
        500.0,
        240.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    writer = _CdrWriter(little_endian=little_endian)
    writer.int32(3)
    writer.uint32(4)
    writer.string(frame_id)
    writer.uint32(height)
    writer.uint32(width)
    writer.string(distortion_model)
    writer.uint32(len(d))
    writer.floats(d)
    writer.floats(k_values)
    writer.floats((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    writer.floats(p_values)
    writer.uint32(1)
    writer.uint32(1)
    x_offset, y_offset, roi_height, roi_width, do_rectify = roi
    writer.uint32(x_offset)
    writer.uint32(y_offset)
    writer.uint32(roi_height)
    writer.uint32(roi_width)
    writer.uint8(do_rectify)
    return writer.finish()


@pytest.mark.parametrize(
    ("decoder", "payload"),
    [
        (decode_ros1_image, _ros1_image()),
        (decode_ros2_image, _ros2_image()),
    ],
)
def test_image_decoders_validate_both_serializations_without_inventory_copy(
    decoder: object,
    payload: bytes,
) -> None:
    message = decoder("/camera/image", 99, payload, include_data=False)  # type: ignore[operator]
    assert message.frame_id == "camera"
    assert message.width == 2
    assert message.height == 2
    assert message.data_length == 4
    assert message.data is None


def test_ros2_image_big_endian_and_ros1_image_payload_are_decoded() -> None:
    ros2 = decode_ros2_image("/camera/image", 99, _ros2_image(little_endian=False))
    ros1 = decode_ros1_image("/camera/image", 99, _ros1_image(), include_data=True)
    assert ros2.header_stamp_ns == 3_000_000_004
    assert ros1.header_stamp_ns == 3_000_000_004
    assert ros1.data == b"abcd"


@pytest.mark.parametrize("decoder", [decode_ros1_image, decode_ros2_image])
def test_image_decoders_reject_truncation_trailing_and_invalid_layout(
    decoder: object,
) -> None:
    valid = _ros1_image() if decoder is decode_ros1_image else _ros2_image()
    with pytest.raises(DatasetError):
        decoder("/camera/image", 0, valid[:-1])  # type: ignore[operator]
    with pytest.raises(DatasetError, match="trailing"):
        decoder("/camera/image", 0, valid + b"x")  # type: ignore[operator]
    base = _ros1_image if decoder is decode_ros1_image else _ros2_image
    with pytest.raises(DatasetError, match="encoding"):
        decoder("/camera/image", 0, base(encoding="unsupported", step=2, payload=b"ab"))  # type: ignore[operator]
    with pytest.raises(DatasetError, match="step"):
        decoder("/camera/image", 0, base(step=1, payload=b"ab"))  # type: ignore[operator]
    with pytest.raises(DatasetError, match="data length"):
        decoder("/camera/image", 0, base(payload=b"abc"))  # type: ignore[operator]


@pytest.mark.parametrize("decoder", [decode_ros1_image, decode_ros2_image])
def test_image_decoders_reject_oversized_dimensions(decoder: object) -> None:
    base = _ros1_image if decoder is decode_ros1_image else _ros2_image
    with pytest.raises(DatasetError, match="height"):
        decoder(
            "/camera/image",
            0,
            base(height=MAX_IMAGE_HEIGHT + 1, width=1, step=1, payload=b"a"),  # type: ignore[operator]
        )
    with pytest.raises(DatasetError, match="width"):
        decoder(
            "/camera/image",
            0,
            base(width=MAX_IMAGE_WIDTH + 1, height=1, step=1, payload=b"a"),  # type: ignore[operator]
        )


@pytest.mark.parametrize(
    ("decoder", "payload"),
    [
        (decode_ros1_camera_info, _ros1_camera_info()),
        (decode_ros2_camera_info, _ros2_camera_info(little_endian=False)),
    ],
)
def test_camera_info_decoders_preserve_matrices_and_frame(
    decoder: object,
    payload: bytes,
) -> None:
    message = decoder("/camera/camera_info", 99, payload)  # type: ignore[operator]
    assert isinstance(message, CameraInfoMessage)
    assert message.frame_id == "camera"
    assert message.is_calibrated is True
    assert message.roi == RegionOfInterest(0, 0, 0, 0, False)
    assert message.k[0] == pytest.approx(500.0)


@pytest.mark.parametrize("decoder", [decode_ros1_camera_info, decode_ros2_camera_info])
def test_camera_info_zero_intrinsics_are_decoded_but_not_calibrated(decoder: object) -> None:
    base = _ros1_camera_info if decoder is decode_ros1_camera_info else _ros2_camera_info
    zeros_k = (0.0,) * 9
    zeros_p = (0.0,) * 12
    message = decoder(  # type: ignore[operator]
        "/camera/camera_info",
        0,
        base(k=zeros_k, p=zeros_p, distortion_model=""),
    )
    assert isinstance(message, CameraInfoMessage)
    assert message.k[0] == 0.0
    assert message.is_calibrated is False
    assert message.calibration_status == "uncalibrated"


@pytest.mark.parametrize("decoder", [decode_ros1_camera_info, decode_ros2_camera_info])
def test_camera_info_rejects_nonfinite_roi_and_oversized_values(decoder: object) -> None:
    base = _ros1_camera_info if decoder is decode_ros1_camera_info else _ros2_camera_info
    bad_k = (math.nan,) + (0.0,) * 8
    with pytest.raises(DatasetError, match="non-finite"):
        decoder("/camera/camera_info", 0, base(k=bad_k))  # type: ignore[operator]
    with pytest.raises(DatasetError, match="ROI"):
        decoder("/camera/camera_info", 0, base(roi=(639, 0, 0, 2, 0)))  # type: ignore[operator]
    with pytest.raises(DatasetError, match="dimensions"):
        decoder(
            "/camera/camera_info",
            0,
            base(width=MAX_IMAGE_WIDTH + 1),  # type: ignore[operator]
        )
    with pytest.raises(DatasetError, match="trailing"):
        decoder("/camera/camera_info", 0, base() + b"x")  # type: ignore[operator]


def test_manifest_readiness_records_camera_calibration_and_frame_mismatch() -> None:
    image = StreamInventory(
        stream_id="/camera/image",
        kind="image",
        topic="/camera/image",
        frame_id="camera_frame",
        decode_status="supported",
        capabilities=["image"],
        clock_domain="record_time_ns",
        message_count=1,
    )
    camera_info = StreamInventory(
        stream_id="/camera/camera_info",
        kind="camera_info",
        topic="/camera/camera_info",
        frame_id="camera_info_frame",
        decode_status="supported",
        capabilities=["camera_info"],
        calibration_status="uncalibrated",
        clock_domain="record_time_ns",
        message_count=1,
    )
    streams = [image, camera_info]
    clocks, frames, _generic = _infer_bindings(streams)
    checks, _actions, status, _summary = _evaluate_capture(
        source=SourceInventory(
            path="capture.db3",
            format="rosbag2",
            exists=True,
            sha256="a" * 64,
            storage_identifier="sqlite3",
            storage_status="known",
            index_status="known",
            crc_status="known",
        ),
        streams=streams,
        sensors=[],
        capture_id="capture",
        session_id="session",
        vehicle_id="vehicle",
        sensor_kit_id="kit",
        clock_bindings=clocks,
        frame_bindings=frames,
        readiness_profile="strict",
        required_streams=(),
        optional_streams=(),
        required_stream_kinds=(),
    )
    by_id = {check.check_id: check for check in checks}
    assert by_id["camera_info.calibration"].status == "blocked"
    assert by_id["bindings.cross_stream"].status == "warn"
    assert status == "blocked"


def test_rosbag2_capture_manifest_decodes_image_and_camera_info(tmp_path: Path) -> None:
    bag = tmp_path / "capture.db3"
    connection = sqlite3.connect(bag)
    try:
        connection.executescript(
            """
            CREATE TABLE topics(
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
              type TEXT NOT NULL,
              serialization_format TEXT NOT NULL,
              offered_qos_profiles TEXT NOT NULL
            );
            CREATE TABLE messages(
              id INTEGER PRIMARY KEY,
              topic_id INTEGER NOT NULL,
              timestamp INTEGER NOT NULL,
              data BLOB NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO topics VALUES (?, ?, ?, ?, ?)",
            [
                (1, "/camera/image", "sensor_msgs/msg/Image", "cdr", ""),
                (2, "/camera/camera_info", "sensor_msgs/msg/CameraInfo", "cdr", ""),
            ],
        )
        connection.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?)",
            [
                (1, 1, 100, _ros2_image(frame_id="camera_frame")),
                (
                    2,
                    2,
                    100,
                    _ros2_camera_info(
                        frame_id="camera_info_frame",
                        k=(0.0,) * 9,
                        p=(0.0,) * 12,
                    ),
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    sensor = SensorIdentity(
        sensor_id="camera",
        type="camera",
        serial="camera-001",
        model="fixture-camera",
        firmware="1.0",
        mount_id="mount-camera",
        frame_id="camera_frame",
    )
    artifact = inspect_capture(
        bag,
        source_format="rosbag2",
        capture_id="capture",
        session_id="session",
        vehicle_id="vehicle",
        sensor_kit_id="kit",
        sensors=[sensor],
    )
    streams = {stream.topic: stream for stream in artifact.streams}
    assert streams["/camera/image"].decode_status == "supported"
    assert streams["/camera/image"].kind == "image"
    assert streams["/camera/camera_info"].decode_status == "supported"
    assert streams["/camera/camera_info"].calibration_status == "uncalibrated"
    checks = {check.check_id: check for check in artifact.checks}
    assert checks["camera_info.calibration"].status == "blocked"
    assert checks["bindings.cross_stream"].status == "warn"
    summary = summarize_rosbag2(bag)
    summary_streams = {stream.topic: stream for stream in summary.streams}
    assert summary_streams["/camera/image"].sample_frame_id == "camera_frame"
    assert summary_streams["/camera/camera_info"].sample_frame_id == "camera_info_frame"
