"""Camera images recorded outside the bag: an image folder or a video file.

``calibrex check BAG --camera-source NAME=PATH`` reads a camera that is not in the bag,
while the IMU (and the rest of the rig) is. ``PATH`` is either

* a folder of images (``.png``, ``.jpg``, ``.jpeg``, ``.bmp``, ``.tif``, ``.tiff``). The
  capture time of each image comes from a ``data.csv`` in the folder or its parent
  (EuRoC/ASL: ``#timestamp [ns],filename``), or a ``timestamps.csv`` / ``timestamps.txt``
  with the same two columns, or else the file name itself: an integer stem is
  nanoseconds (``1520530308199447626.png``), a decimal stem is seconds
  (``1520530308.199447.png``);
* a video file (``.mp4``, ``.mov``, ``.avi``, ``.mkv``, ``.webm``): the frame times are the
  video's own presentation times, starting at 0.

Either clock may be unrelated to the IMU's. The offset between them is found from the
data (:mod:`calibrex.evaluation.clock_alignment`) unless ``NAME=PATH@SECONDS`` gives it:
``camera time + SECONDS = IMU time``.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from calibrex.core.exceptions import DatasetError

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
TIMESTAMP_FILES = ("data.csv", "timestamps.csv", "timestamps.txt")
EXTERNAL_TOPIC_PREFIX = "/external/"
IMAGE_TYPE = "sensor_msgs/msg/Image"
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class ExternalCameraSource:
    """One camera outside the bag: ``name`` (a camchain camera or a frame), its path and kind."""

    name: str
    path: Path
    kind: str
    """``images`` or ``video``."""
    clock_offset_s: float | None = None
    """``camera time + clock_offset_s = IMU time`` when given; ``None`` means find it."""

    @property
    def topic(self) -> str:
        """The topic the source stands in for in ``calibrex check``'s topic list."""

        return f"{EXTERNAL_TOPIC_PREFIX}{self.name}"

    def digest(self) -> str:
        """SHA-256 over what identifies the frames: names, sizes and times (not pixels)."""

        h = hashlib.sha256(f"{self.kind}:{self.path.name}".encode())
        if self.kind == "video":
            stat = self.path.stat()
            h.update(f"{stat.st_size}".encode())
            with self.path.open("rb") as stream:
                h.update(stream.read(1 << 20))
        else:
            for time_s, image in folder_frames(self.path):
                h.update(f"{image.name}:{image.stat().st_size}:{time_s:.9f}".encode())
        return h.hexdigest()


def parse_camera_source(text: str) -> ExternalCameraSource:
    """Parse ``NAME=PATH`` or ``NAME=PATH@SECONDS``."""

    name, sep, rest = text.partition("=")
    if not sep or not name or not rest:
        raise ValueError(f"--camera-source {text!r}: expected NAME=PATH or NAME=PATH@SECONDS")
    if not _NAME.match(name):
        raise ValueError(f"--camera-source {text!r}: NAME may use letters, digits, '_', '.', '-'")
    offset: float | None = None
    path_text = rest
    head, at, tail = rest.rpartition("@")
    if at and head:
        try:
            offset = float(tail)
        except ValueError:
            pass
        else:
            path_text = head
    path = Path(path_text).expanduser()
    if path.is_dir():
        kind = "images"
    elif path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        kind = "video"
    elif path.is_file():
        raise ValueError(
            f"--camera-source {text!r}: {path.name} is not a video "
            f"({', '.join(sorted(VIDEO_SUFFIXES))}) or an image folder"
        )
    else:
        raise ValueError(f"--camera-source {text!r}: {path} does not exist")
    if offset is not None and not math.isfinite(offset):
        raise ValueError(f"--camera-source {text!r}: the clock offset must be finite")
    return ExternalCameraSource(name=name, path=path, kind=kind, clock_offset_s=offset)


def _stem_time_s(stem: str) -> float | None:
    if re.fullmatch(r"\d+", stem):
        value = int(stem)
        return value * 1.0e-9 if value > 10**12 else float(value)
    if re.fullmatch(r"\d+\.\d+", stem):
        return float(stem)
    return None


def _timestamp_table(folder: Path) -> dict[str, float] | None:
    for directory in (folder, folder.parent):
        for name in TIMESTAMP_FILES:
            table = directory / name
            if not table.is_file():
                continue
            times: dict[str, float] = {}
            with table.open(newline="", encoding="utf-8") as stream:
                for row in csv.reader(stream):
                    if not row or row[0].lstrip().startswith("#") or len(row) < 2:
                        continue
                    stamp, file = row[0].strip(), row[1].strip()
                    try:
                        value = int(stamp)
                        times[file] = value * 1.0e-9 if value > 10**12 else float(value)
                    except ValueError:
                        try:
                            times[file] = float(stamp)
                        except ValueError:
                            continue
            if times:
                return times
    return None


def folder_frames(folder: Path) -> list[tuple[float, Path]]:
    """Every image of ``folder`` with its capture time (s), in time order."""

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise DatasetError(f"{folder}: no images ({', '.join(sorted(IMAGE_SUFFIXES))})")
    table = _timestamp_table(folder)
    frames: list[tuple[float, Path]] = []
    for image in images:
        time_s = table.get(image.name) if table is not None else _stem_time_s(image.stem)
        if time_s is None:
            raise DatasetError(
                f"{folder}: no capture time for {image.name}; put a data.csv "
                "(timestamp_ns,filename) next to the images or name them by their time"
            )
        frames.append((time_s, image))
    frames.sort(key=lambda item: item[0])
    if len({time for time, _ in frames}) != len(frames):
        raise DatasetError(f"{folder}: two images share a capture time")
    return frames


def _opencv() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:  # pragma: no cover - the opencv extra is installed in CI
        raise DatasetError("external cameras need OpenCV: pip install 'calibrex[opencv]'") from exc


def frame_interval_s(source: ExternalCameraSource) -> float | None:
    """Median time between frames, or ``None`` with fewer than two frames."""

    if source.kind == "video":
        cv2 = _opencv()
        capture = cv2.VideoCapture(str(source.path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        capture.release()
        return 1.0 / fps if fps > 0.0 else None
    times = [time for time, _ in folder_frames(source.path)]
    return float(np.median(np.diff(times))) if len(times) > 1 else None


def iter_external_gray(
    source: ExternalCameraSource,
    *,
    stride: int = 1,
    max_seconds: float | None = None,
    time_shift_s: float = 0.0,
) -> Iterator[tuple[float, NDArray[np.uint8]]]:
    """Yield ``(time in s, gray image)`` for every ``stride``-th frame, times shifted."""

    cv2 = _opencv()
    first: float | None = None
    if source.kind == "video":
        capture = cv2.VideoCapture(str(source.path))
        if not capture.isOpened():
            raise DatasetError(f"{source.path}: OpenCV cannot open this video")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        try:
            index = 0
            while True:
                ok = capture.grab()
                if not ok:
                    return
                if index % stride == 0:
                    position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    time_s = (
                        position_ms * 1.0e-3
                        if position_ms > 0.0 or index == 0 or fps <= 0.0
                        else index / fps
                    )
                    first = time_s if first is None else first
                    if max_seconds is not None and time_s - first > max_seconds:
                        return
                    ok, frame = capture.retrieve()
                    if ok:
                        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        yield time_s + time_shift_s, np.ascontiguousarray(gray, dtype=np.uint8)
                index += 1
        finally:
            capture.release()
        return
    for index, (time_s, image) in enumerate(folder_frames(source.path)):
        if index % stride:
            continue
        first = time_s if first is None else first
        if max_seconds is not None and time_s - first > max_seconds:
            return
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if gray is None:
            raise DatasetError(f"{image}: OpenCV cannot read this image")
        if gray.dtype == np.uint16:  # 16-bit: keep the high byte, as the bag's mono16 decoder
            gray = np.asarray(gray >> 8, dtype=np.uint8)
        elif gray.dtype != np.uint8:
            raise DatasetError(f"{image}: unsupported pixel type {gray.dtype}")
        yield time_s + time_shift_s, gray


def external_sources_by_topic(
    sources: Sequence[ExternalCameraSource],
) -> dict[str, ExternalCameraSource]:
    """The sources keyed by their stand-in topic; a repeated name is an error."""

    by_topic: dict[str, ExternalCameraSource] = {}
    for source in sources:
        if source.topic in by_topic:
            raise ValueError(f"--camera-source {source.name} is given twice")
        by_topic[source.topic] = source
    return by_topic
