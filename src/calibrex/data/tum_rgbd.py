"""TUM RGB-D dataset reader."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

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
