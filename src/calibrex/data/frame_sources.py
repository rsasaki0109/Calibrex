"""Grey frames with times from a ROS bag, a video file or a live camera.

Each source yields ``(time in s, grey image)`` so the intrinsics commands read them alike.
The video and camera readers need OpenCV.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

GrayFrame = tuple[float, NDArray[np.uint8]]


def _opencv() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "video and camera input need OpenCV: pip install 'calibrex[opencv]'"
        ) from exc


def bag_frames(
    bag: str | Path,
    topic: str,
    *,
    stride: int = 1,
    max_seconds: float | None = None,
    start_seconds: float = 0.0,
) -> Iterator[GrayFrame]:
    """Read a bag topic within a window relative to its first image stamp."""

    if Path(bag).suffix.lower() == ".bag" and Path(bag).is_file():
        yield from _ros1_bag_frames(bag, topic, stride, max_seconds, start_seconds)
        return

    from calibrex.data.ros2_camera_imu import iter_gray_images

    yield from iter_gray_images(
        bag, topic, stride=stride, max_seconds=max_seconds, start_seconds=start_seconds
    )


def _ros1_bag_frames(
    bag: str | Path,
    topic: str,
    stride: int,
    max_seconds: float | None,
    start_seconds: float,
) -> Iterator[GrayFrame]:
    from calibrex.data.ros2_camera_imu import to_gray
    from calibrex.data.rosbag1 import read_image_messages

    first: int | None = None
    for index, image in enumerate(read_image_messages(bag, topic=topic, include_data=True)):
        first = image.timestamp_ns if first is None else first
        elapsed = (image.timestamp_ns - first) * 1e-9
        if max_seconds is not None and elapsed > start_seconds + max_seconds:
            return
        if elapsed < start_seconds or index % stride:
            continue
        assert image.data is not None
        yield (
            image.timestamp_ns * 1e-9,
            to_gray(
                bytes(image.data),
                height=image.height,
                width=image.width,
                step=image.step,
                encoding=image.encoding,
                is_bigendian=image.is_bigendian,
            ),
        )


def _grey(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    cv2 = _opencv()
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def video_frames(
    path: str | Path, *, stride: int = 1, max_seconds: float | None = None
) -> Iterator[GrayFrame]:
    """Every ``stride``-th frame of a video, timed by its frame rate."""

    cv2 = _opencv()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                return
            time_s = index / fps
            if max_seconds is not None and time_s > max_seconds:
                return
            if index % stride == 0:
                yield time_s, _grey(image)
            index += 1
    finally:
        capture.release()


def camera_frames(
    index: int,
    *,
    seconds: float,
    resolution: tuple[int, int] | None = None,
    stride: int = 1,
    on_frame: Callable[[NDArray[np.uint8], float], bool] | None = None,
) -> Iterator[GrayFrame]:
    """Frames of a live camera for ``seconds``.

    ``on_frame(bgr image, seconds elapsed)`` is called for every frame (to show it);
    returning ``False`` stops early.
    """

    cv2 = _opencv()
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise OSError(f"cannot open camera {index}")
    if resolution is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    start = time.monotonic()
    try:
        count = 0
        while True:
            ok, image = capture.read()
            elapsed = time.monotonic() - start
            if not ok or elapsed > seconds:
                return
            if on_frame is not None and not on_frame(image, elapsed):
                return
            if count % stride == 0:
                yield elapsed, _grey(image)
            count += 1
    finally:
        capture.release()


def split_side_by_side(frames: Iterator[GrayFrame], half: int) -> Iterator[GrayFrame]:
    """The left (``half=0``) or right (``half=1``) half of side-by-side stereo frames."""

    for time_s, image in frames:
        width = image.shape[1] // 2
        yield time_s, image[:, half * width : (half + 1) * width]


def paired_frames(
    left: Iterator[GrayFrame],
    right: Iterator[GrayFrame],
    *,
    max_gap_s: float = 0.01,
    every_s: float = 1.0,
) -> Iterator[tuple[NDArray[np.uint8], NDArray[np.uint8]]]:
    """Simultaneous (within ``max_gap_s``) left and right images, about one per ``every_s``."""

    right_iter = iter(right)
    pending = next(right_iter, None)
    last = -float("inf")
    for time_s, image in left:
        if time_s - last < every_s:
            continue
        while pending is not None and pending[0] < time_s - max_gap_s:
            pending = next(right_iter, None)
        if pending is None:
            return
        if abs(pending[0] - time_s) <= max_gap_s:
            yield image, pending[1]
            last = time_s
