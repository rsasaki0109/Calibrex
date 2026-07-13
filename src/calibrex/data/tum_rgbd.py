"""TUM RGB-D dataset reader."""

from __future__ import annotations

import bisect
import math
import struct
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from calibrex.core.geometry import SE3, Vector3
from calibrex.core.time import TimestampNormalizer
from calibrex.data.base import StreamSummary, TimestampedRecord
from calibrex.data.manifest import DatasetManifest, find_manifest, load_manifest


@dataclass(frozen=True)
class TUMImageEntry:
    """One TUM RGB or depth image entry."""

    timestamp_sec: float
    path: str


@dataclass(frozen=True)
class TUMTrajectoryEntry:
    """One TUM ground-truth pose entry."""

    timestamp_sec: float
    translation_m: tuple[float, float, float]
    rotation_quat_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class TUMAssociationEntry:
    """One RGB/depth association entry."""

    rgb_timestamp_sec: float
    rgb_path: str
    depth_timestamp_sec: float
    depth_path: str


@dataclass(frozen=True)
class TUMDepthImage:
    """One decoded 16-bit metric-depth source image in row-major order."""

    width: int
    height: int
    raw_values: tuple[int, ...]


@dataclass(frozen=True)
class TUMDepthIntrinsics:
    """Pinhole parameters used to back-project registered TUM depth pixels."""

    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5
    depth_scale: float = 5000.0


@dataclass(frozen=True)
class TUMDepthPoseEntry:
    """A depth frame paired with the nearest public ground-truth camera pose."""

    depth: TUMImageEntry
    trajectory: TUMTrajectoryEntry
    absolute_time_delta_sec: float

    @property
    def transform_world_camera(self) -> SE3:
        return SE3(self.trajectory.translation_m, self.trajectory.rotation_quat_xyzw)


class TUMRGBDDataset:
    """Reader for the public TUM RGB-D dataset format."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest = _load_optional_manifest(self.path)
        self.normalizer = TimestampNormalizer("sensor_time_sec")

    def streams(self) -> list[StreamSummary]:
        """Return stream counts from actual TUM files when present."""

        rgb_entries = read_image_index(self.path / "rgb.txt")
        depth_entries = read_image_index(self.path / "depth.txt")
        associations = read_associations(self.path / "associations.txt")
        trajectory = read_groundtruth(self.path / "groundtruth.txt")
        if not any([rgb_entries, depth_entries, associations, trajectory]) and self.manifest:
            return _manifest_streams(self.manifest)
        return [
            StreamSummary("rgb", "image", len(rgb_entries), sensor="rgbd0"),
            StreamSummary("depth", "depth_image", len(depth_entries), sensor="rgbd0"),
            StreamSummary("rgbd_associations", "rgbd", len(associations), sensor="rgbd0"),
            StreamSummary("groundtruth", "trajectory", len(trajectory)),
        ]

    def records(self, stream: str) -> Iterable[TimestampedRecord]:
        """Yield normalized TUM records for a stream."""

        if stream == "rgb":
            return _image_records("rgb", self.path, read_image_index(self.path / "rgb.txt"))
        if stream == "depth":
            return _image_records("depth", self.path, read_image_index(self.path / "depth.txt"))
        if stream == "groundtruth":
            return _trajectory_records(read_groundtruth(self.path / "groundtruth.txt"))
        if stream == "rgbd_associations":
            associations = read_associations(self.path / "associations.txt")
            return _association_records(self.path, associations)
        return []


def read_image_index(path: Path) -> list[TUMImageEntry]:
    """Read `rgb.txt` or `depth.txt`."""

    if not path.exists():
        return []
    entries: list[TUMImageEntry] = []
    for fields in _iter_fields(path):
        if len(fields) < 2:
            continue
        entries.append(TUMImageEntry(timestamp_sec=float(fields[0]), path=fields[1]))
    return entries


def read_groundtruth(path: Path) -> list[TUMTrajectoryEntry]:
    """Read TUM `groundtruth.txt` trajectory."""

    if not path.exists():
        return []
    entries: list[TUMTrajectoryEntry] = []
    for fields in _iter_fields(path):
        if len(fields) < 8:
            continue
        entries.append(
            TUMTrajectoryEntry(
                timestamp_sec=float(fields[0]),
                translation_m=(float(fields[1]), float(fields[2]), float(fields[3])),
                rotation_quat_xyzw=(
                    float(fields[4]),
                    float(fields[5]),
                    float(fields[6]),
                    float(fields[7]),
                ),
            )
        )
    return entries


def read_associations(path: Path) -> list[TUMAssociationEntry]:
    """Read TUM RGB/depth `associations.txt`."""

    if not path.exists():
        return []
    entries: list[TUMAssociationEntry] = []
    for fields in _iter_fields(path):
        if len(fields) < 4:
            continue
        entries.append(
            TUMAssociationEntry(
                rgb_timestamp_sec=float(fields[0]),
                rgb_path=fields[1],
                depth_timestamp_sec=float(fields[2]),
                depth_path=fields[3],
            )
        )
    return entries


def associate_rgb_depth(
    rgb_entries: list[TUMImageEntry],
    depth_entries: list[TUMImageEntry],
    max_difference_sec: float = 0.02,
) -> list[TUMAssociationEntry]:
    """Associate RGB and depth frames by nearest timestamp."""

    associations: list[TUMAssociationEntry] = []
    depth_sorted = sorted(depth_entries, key=lambda entry: entry.timestamp_sec)
    cursor = 0
    for rgb in sorted(rgb_entries, key=lambda entry: entry.timestamp_sec):
        if not depth_sorted:
            break
        while cursor + 1 < len(depth_sorted) and abs(
            depth_sorted[cursor + 1].timestamp_sec - rgb.timestamp_sec
        ) <= abs(depth_sorted[cursor].timestamp_sec - rgb.timestamp_sec):
            cursor += 1
        depth = depth_sorted[cursor]
        if abs(depth.timestamp_sec - rgb.timestamp_sec) <= max_difference_sec:
            associations.append(
                TUMAssociationEntry(
                    rgb_timestamp_sec=rgb.timestamp_sec,
                    rgb_path=rgb.path,
                    depth_timestamp_sec=depth.timestamp_sec,
                    depth_path=depth.path,
                )
            )
    return associations


def associate_depth_groundtruth(
    depth_entries: list[TUMImageEntry],
    trajectory: list[TUMTrajectoryEntry],
    max_difference_sec: float = 0.02,
) -> list[TUMDepthPoseEntry]:
    """Pair depth frames with nearest ground-truth poses without reusing time order."""

    if not trajectory:
        return []
    ordered = sorted(trajectory, key=lambda entry: entry.timestamp_sec)
    timestamps = [entry.timestamp_sec for entry in ordered]
    output: list[TUMDepthPoseEntry] = []
    for depth in sorted(depth_entries, key=lambda entry: entry.timestamp_sec):
        insertion = bisect.bisect_left(timestamps, depth.timestamp_sec)
        candidates = [
            index for index in (insertion - 1, insertion) if 0 <= index < len(ordered)
        ]
        closest = min(
            candidates,
            key=lambda index: abs(ordered[index].timestamp_sec - depth.timestamp_sec),
        )
        delta = abs(ordered[closest].timestamp_sec - depth.timestamp_sec)
        if delta <= max_difference_sec:
            output.append(TUMDepthPoseEntry(depth, ordered[closest], delta))
    return output


def read_tum_depth_png(path: str | Path) -> TUMDepthImage:
    """Decode a non-interlaced 16-bit grayscale TUM depth PNG using the stdlib."""

    data = Path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("TUM depth image is not a PNG")
    cursor = 8
    width = height = 0
    compressed = bytearray()
    while cursor + 12 <= len(data):
        length = struct.unpack(">I", data[cursor : cursor + 4])[0]
        kind = data[cursor + 4 : cursor + 8]
        payload = data[cursor + 8 : cursor + 8 + length]
        cursor += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            if (
                bit_depth != 16
                or color_type != 0
                or compression != 0
                or filter_method != 0
                or interlace != 0
            ):
                raise ValueError(
                    "TUM depth PNG must be non-interlaced 16-bit grayscale"
                )
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if width <= 0 or height <= 0 or not compressed:
        raise ValueError("TUM depth PNG is missing IHDR or IDAT data")
    stride = width * 2
    raw = zlib.decompress(bytes(compressed))
    if len(raw) != height * (stride + 1):
        raise ValueError("TUM depth PNG decompressed size does not match IHDR")
    previous = bytearray(stride)
    decoded = bytearray()
    offset = 0
    for _row in range(height):
        filter_type = raw[offset]
        scanline = bytearray(raw[offset + 1 : offset + 1 + stride])
        _unfilter_png_scanline(scanline, previous, filter_type, bytes_per_pixel=2)
        decoded.extend(scanline)
        previous = scanline
        offset += stride + 1
    values = tuple(
        struct.unpack(">H", decoded[index : index + 2])[0]
        for index in range(0, len(decoded), 2)
    )
    return TUMDepthImage(width, height, values)


def sample_tum_depth_points(
    image: TUMDepthImage,
    intrinsics: TUMDepthIntrinsics | None = None,
    *,
    max_points: int = 2000,
    min_depth_m: float = 0.2,
    max_depth_m: float = 5.0,
) -> list[Vector3]:
    """Back-project a deterministic grid sample of valid registered depth pixels."""

    camera = intrinsics or TUMDepthIntrinsics()
    if camera.fx <= 0.0 or camera.fy <= 0.0 or camera.depth_scale <= 0.0:
        raise ValueError("TUM depth intrinsics and scale must be positive")
    if max_points < 1 or min_depth_m < 0.0 or max_depth_m <= min_depth_m:
        raise ValueError("invalid TUM depth sampling limits")
    grid_step = max(1, math.ceil(math.sqrt(image.width * image.height / max_points)))
    points: list[Vector3] = []
    for v in range(0, image.height, grid_step):
        row = v * image.width
        for u in range(0, image.width, grid_step):
            raw_depth = image.raw_values[row + u]
            if raw_depth == 0:
                continue
            z = raw_depth / camera.depth_scale
            if z < min_depth_m or z > max_depth_m:
                continue
            points.append(
                (
                    (u - camera.cx) * z / camera.fx,
                    (v - camera.cy) * z / camera.fy,
                    z,
                )
            )
    return points[:max_points]


def _unfilter_png_scanline(
    scanline: bytearray,
    previous: bytearray,
    filter_type: int,
    *,
    bytes_per_pixel: int,
) -> None:
    if filter_type not in {0, 1, 2, 3, 4}:
        raise ValueError(f"unsupported PNG filter type {filter_type}")
    for index in range(len(scanline)):
        left = scanline[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        above = previous[index]
        upper_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
        if filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = above
        elif filter_type == 3:
            predictor = (left + above) // 2
        elif filter_type == 4:
            predictor = _paeth(left, above, upper_left)
        else:
            predictor = 0
        scanline[index] = (scanline[index] + predictor) & 0xFF


def _paeth(left: int, above: int, upper_left: int) -> int:
    prediction = left + above - upper_left
    left_distance = abs(prediction - left)
    above_distance = abs(prediction - above)
    upper_left_distance = abs(prediction - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def write_associations(path: Path, associations: list[TUMAssociationEntry]) -> None:
    """Write a TUM-style association file."""

    lines = [
        f"{entry.rgb_timestamp_sec:.6f} {entry.rgb_path} "
        f"{entry.depth_timestamp_sec:.6f} {entry.depth_path}"
        for entry in associations
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _iter_fields(path: Path) -> Iterable[list[str]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield stripped.split()


def _image_records(
    stream: str,
    root: Path,
    entries: list[TUMImageEntry],
) -> list[TimestampedRecord]:
    normalizer = TimestampNormalizer("sensor_time_sec")
    return [
        TimestampedRecord(
            stream=stream,
            timestamp_ns=normalizer.to_nanoseconds(entry.timestamp_sec),
            payload_path=str(root / entry.path),
            metadata={"timestamp_sec": entry.timestamp_sec},
        )
        for entry in entries
    ]


def _trajectory_records(entries: list[TUMTrajectoryEntry]) -> list[TimestampedRecord]:
    normalizer = TimestampNormalizer("sensor_time_sec")
    return [
        TimestampedRecord(
            stream="groundtruth",
            timestamp_ns=normalizer.to_nanoseconds(entry.timestamp_sec),
            metadata={
                "translation_m": list(entry.translation_m),
                "rotation_quat_xyzw": list(entry.rotation_quat_xyzw),
                "timestamp_sec": entry.timestamp_sec,
            },
        )
        for entry in entries
    ]


def _association_records(root: Path, entries: list[TUMAssociationEntry]) -> list[TimestampedRecord]:
    normalizer = TimestampNormalizer("sensor_time_sec")
    return [
        TimestampedRecord(
            stream="rgbd_associations",
            timestamp_ns=normalizer.to_nanoseconds(entry.rgb_timestamp_sec),
            payload_path=str(root / entry.rgb_path),
            metadata={
                "rgb_timestamp_sec": entry.rgb_timestamp_sec,
                "rgb_path": str(root / entry.rgb_path),
                "depth_timestamp_sec": entry.depth_timestamp_sec,
                "depth_path": str(root / entry.depth_path),
            },
        )
        for entry in entries
    ]


def _load_optional_manifest(path: Path) -> DatasetManifest | None:
    if find_manifest(path) is None:
        return None
    return load_manifest(path)


def _manifest_streams(manifest: DatasetManifest) -> list[StreamSummary]:
    return [
        StreamSummary(
            name=name,
            kind=stream.kind,
            message_count=stream.count,
            topic=stream.topic,
            sensor=stream.sensor,
        )
        for name, stream in sorted(manifest.streams.items())
    ]
