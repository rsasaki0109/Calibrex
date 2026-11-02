"""``sensor_msgs/CompressedImage`` decoding for the camera pairs of ``calibrex check``.

A lossless PNG of a raw image must decode to the same gray image as the raw message, so
a camera pair reads a compressed bag exactly as it reads the raw one; a JPEG decodes to
the raw image up to its compression error. The topic and intrinsics helpers follow
image_transport's naming (``<raw topic>/compressed``).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from calibrex.check.estimators import (
    COMPRESSED_IMAGE_TYPE,
    IMAGE_TYPE,
    is_rectified_image_topic,
    select_image_topic,
)
from calibrex.check.frame_tree import FrameHints, StaticFrameTree
from calibrex.check.roles import classify_topics, image_transport_base, map_topics_to_frames
from calibrex.core.exceptions import DatasetError
from calibrex.data import compressed_image, ros_cdr
from calibrex.data.ros2_camera_imu import (
    decode_gray_message,
    image_header,
    iter_gray_images,
    to_gray,
)
from calibrex.data.ros_cdr_writer import encode_compressed_image, encode_image
from calibrex.data.rosbag2_writer import Rosbag2Writer

Image = pytest.importorskip("PIL.Image")

HEIGHT, WIDTH = 24, 32


def _color() -> np.ndarray:
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    return np.stack(
        [(rows * 9) % 256, (cols * 7) % 256, ((rows + cols) * 5) % 256], axis=-1
    ).astype(np.uint8)


def _encode(array: np.ndarray, kind: str, **options: object) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format=kind, **options)
    return buffer.getvalue()


# --------------------------------------------------------------------------------- CDR


def test_cdr_round_trip_keeps_header_format_and_bytes() -> None:
    payload = encode_compressed_image(
        frame_id="camera", timestamp_ns=12_000_000_345, format="png", data=b"\x89PNGabc"
    )
    message = ros_cdr.decode_ros2_compressed_image("/c/image_raw/compressed", 1, payload)
    assert (message.timestamp_ns, message.frame_id, message.format) == (
        12_000_000_345,
        "camera",
        "png",
    )
    assert message.data == b"\x89PNGabc" and message.data_length == 7
    header = ros_cdr.decode_ros2_compressed_image("/t", 1, payload, include_data=False)
    assert header.data is None and header.data_length == 7


def test_a_zero_header_stamp_falls_back_to_the_bag_time() -> None:
    payload = encode_compressed_image(frame_id="c", timestamp_ns=0, format="jpeg", data=b"x")
    assert ros_cdr.decode_ros2_compressed_image("/t", 77, payload).timestamp_ns == 77


def test_an_empty_or_padded_payload_is_rejected() -> None:
    empty = encode_compressed_image(frame_id="c", timestamp_ns=1, format="jpeg", data=b"")
    with pytest.raises(DatasetError, match="no data"):
        ros_cdr.decode_ros2_compressed_image("/t", 1, empty)
    good = encode_compressed_image(frame_id="c", timestamp_ns=1, format="jpeg", data=b"x")
    with pytest.raises(DatasetError, match="trailing"):
        ros_cdr.decode_ros2_compressed_image("/t", 1, good + b"\x00" * 8)


# ----------------------------------------------------------------------------- pixels


@pytest.mark.parametrize("encoding", ["mono8", "rgb8", "bgr8", "rgba8"])
def test_a_lossless_png_decodes_to_the_raw_gray_image(encoding: str) -> None:
    color = _color()
    if encoding == "mono8":
        raw = color[:, :, 0].copy()
        png = _encode(raw, "PNG")
    elif encoding == "rgba8":
        raw = np.dstack([color, np.full((HEIGHT, WIDTH), 200, np.uint8)])
        png = _encode(raw, "PNG")
    else:
        raw = color[:, :, ::-1].copy() if encoding == "bgr8" else color
        png = _encode(color, "PNG")  # image_transport writes a correctly ordered file
    channels = 1 if raw.ndim == 2 else raw.shape[2]
    expected = to_gray(
        raw.tobytes(), height=HEIGHT, width=WIDTH, step=WIDTH * channels, encoding=encoding
    )
    gray = compressed_image.decode_compressed_gray(png, f"{encoding}; png compressed {encoding}")
    np.testing.assert_array_equal(gray, expected)


def test_a_16_bit_png_keeps_the_high_byte_like_mono16() -> None:
    values = ((np.arange(HEIGHT * WIDTH).reshape(HEIGHT, WIDTH) * 61) % 65536).astype(np.uint16)
    buffer = BytesIO()
    Image.fromarray(values.astype(np.uint16)).save(buffer, format="PNG")
    expected = to_gray(
        values.astype("<u2").tobytes(),
        height=HEIGHT,
        width=WIDTH,
        step=WIDTH * 2,
        encoding="mono16",
    )
    gray = compressed_image.decode_compressed_gray(buffer.getvalue(), "mono16; png compressed")
    np.testing.assert_array_equal(gray, expected)


def test_a_jpeg_decodes_to_the_raw_image_up_to_its_compression_error() -> None:
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    smooth = np.stack([rows * 4, cols * 3, rows + cols], axis=-1).astype(np.uint8)
    gray = compressed_image.decode_compressed_gray(_encode(smooth, "JPEG", quality=95), "jpeg")
    expected = to_gray(
        smooth.tobytes(), height=HEIGHT, width=WIDTH, step=WIDTH * 3, encoding="rgb8"
    )
    assert gray.shape == (HEIGHT, WIDTH) and gray.dtype == np.uint8
    assert np.abs(gray.astype(int) - expected.astype(int)).mean() < 3.0


def test_depth_maps_and_garbage_are_refused() -> None:
    with pytest.raises(ValueError, match="depth"):
        compressed_image.decode_compressed_gray(b"x", "16UC1; compressedDepth png")
    with pytest.raises(ValueError, match="cannot decode"):
        compressed_image.decode_compressed_gray(b"not an image", "jpeg")


def test_without_a_decoder_the_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compressed_image, "_pillow", lambda: None)
    monkeypatch.setattr(compressed_image, "_opencv", lambda: None)
    assert compressed_image.compressed_image_decoder() is None
    with pytest.raises(ImportError, match=r"calibrex\[images\]"):
        compressed_image.decode_compressed_gray(_encode(_color(), "PNG"), "png")


def test_opencv_decodes_like_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    png = _encode(_color(), "PNG")
    expected = compressed_image.decode_compressed_gray(png, "png")
    monkeypatch.setattr(compressed_image, "_pillow", lambda: None)
    assert (compressed_image.compressed_image_decoder() or "").startswith("opencv")
    np.testing.assert_array_equal(compressed_image.decode_compressed_gray(png, "png"), expected)


# ------------------------------------------------------------------------- bag reading


def _bag(path: Path, compressed: bool) -> Path:
    topic = "/cam/image_raw/compressed" if compressed else "/cam/image_raw"
    message_type = COMPRESSED_IMAGE_TYPE if compressed else IMAGE_TYPE
    color = _color()
    with Rosbag2Writer(path) as writer:
        writer.add_topic(topic, message_type)
        for index in range(3):
            stamp = 1_000_000_000 * (index + 1)
            frame = np.roll(color, index, axis=1)
            payload = (
                encode_compressed_image(
                    frame_id="cam",
                    timestamp_ns=stamp,
                    format="rgb8; png compressed bgr8",
                    data=_encode(frame, "PNG"),
                )
                if compressed
                else encode_image(
                    frame_id="cam",
                    timestamp_ns=stamp,
                    height=HEIGHT,
                    width=WIDTH,
                    encoding="rgb8",
                    step=WIDTH * 3,
                    data=frame.tobytes(),
                )
            )
            writer.write(topic, stamp, payload)
    return path


def test_a_compressed_bag_reads_like_the_raw_bag(tmp_path: Path) -> None:
    raw = list(iter_gray_images(_bag(tmp_path / "raw", False), "/cam/image_raw"))
    packed = list(iter_gray_images(_bag(tmp_path / "png", True), "/cam/image_raw/compressed"))
    assert [time for time, _ in raw] == [time for time, _ in packed] == [1.0, 2.0, 3.0]
    for (_, left), (_, right) in zip(raw, packed, strict=True):
        np.testing.assert_array_equal(left, right)


def test_header_and_pixels_dispatch_on_the_message_type() -> None:
    payload = encode_compressed_image(
        frame_id="cam", timestamp_ns=5, format="png", data=_encode(_color(), "PNG")
    )
    assert image_header(COMPRESSED_IMAGE_TYPE, "/t", 1, payload) == (5, "cam")
    stamp, frame, gray = decode_gray_message(COMPRESSED_IMAGE_TYPE, "/t", 1, payload)
    assert (stamp, frame, gray.shape) == (5, "cam", (HEIGHT, WIDTH))
    with pytest.raises(ValueError, match="not an image topic"):
        decode_gray_message("sensor_msgs/msg/Imu", "/t", 1, payload)


# ------------------------------------------------------------------------------ topics


def test_a_raw_topic_is_preferred_and_depth_is_never_chosen() -> None:
    types = {
        "/cam/image_raw/compressed": COMPRESSED_IMAGE_TYPE,
        "/cam/image_raw": IMAGE_TYPE,
        "/depth/image_raw/compressedDepth": COMPRESSED_IMAGE_TYPE,
    }
    assert select_image_topic(sorted(types), types) == "/cam/image_raw"
    only_compressed = {k: v for k, v in types.items() if k != "/cam/image_raw"}
    assert select_image_topic(sorted(only_compressed), only_compressed) == (
        "/cam/image_raw/compressed"
    )
    depth = {"/depth/image_raw/compressedDepth": COMPRESSED_IMAGE_TYPE}
    assert select_image_topic(sorted(depth), depth) is None


def test_compressed_topics_keep_their_raw_topic_meaning() -> None:
    assert image_transport_base("/cam/image_raw/compressed") == "/cam/image_raw"
    assert image_transport_base("/cam/image_raw") == "/cam/image_raw"
    assert is_rectified_image_topic("/cam/image_rect_color/compressed")
    assert not is_rectified_image_topic("/cam/image_raw/compressed")


def test_a_camchain_topic_maps_its_compressed_twin() -> None:
    from calibrex.data.rosbag2 import Rosbag2Connection

    connection = Rosbag2Connection(
        topic_id=1, topic="/left/image_raw/compressed", message_type=COMPRESSED_IMAGE_TYPE
    )
    (record,) = classify_topics([(connection, 3)])
    tree = StaticFrameTree([])
    hints = FrameHints(topic_frames={"/left/image_raw": "left/image_raw"})
    (mapped,) = map_topics_to_frames([record], {record.topic: "left_optical"}, tree, hints)
    assert (mapped.mapped_frame, mapped.frame_source) == ("left/image_raw", "source_topic_hint")
