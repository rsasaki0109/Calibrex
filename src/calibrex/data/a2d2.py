"""A2D2 LiDAR NPZ dataset helpers."""

from __future__ import annotations

import ast
import struct
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from calibrex.data.base import StreamSummary, TimestampedRecord

SampleT = TypeVar("SampleT")


@dataclass(frozen=True)
class A2D2PhysicalLidarStats:
    """Point statistics for one physical LiDAR id inside an A2D2 NPZ file."""

    lidar_id: int
    point_count: int
    valid_count: int
    bounds_min_m: tuple[float, float, float] | None = None
    bounds_max_m: tuple[float, float, float] | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "lidar_id": self.lidar_id,
            "point_count": self.point_count,
            "valid_count": self.valid_count,
            "bounds_min_m": list(self.bounds_min_m) if self.bounds_min_m else None,
            "bounds_max_m": list(self.bounds_max_m) if self.bounds_max_m else None,
        }


@dataclass(frozen=True)
class A2D2LidarSampleStats:
    """Point statistics for one A2D2 LiDAR NPZ sample."""

    path: str
    point_count: int
    valid_count: int
    physical_lidars: tuple[A2D2PhysicalLidarStats, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "path": self.path,
            "point_count": self.point_count,
            "valid_count": self.valid_count,
            "physical_lidars": [lidar.as_dict() for lidar in self.physical_lidars],
        }


@dataclass(frozen=True)
class A2D2LidarDatasetStats:
    """Dataset-level statistics for A2D2 LiDAR NPZ files."""

    status: str
    sample_count: int
    sampled_file_count: int
    total_point_count: int
    total_valid_count: int
    physical_lidar_ids: tuple[int, ...]
    samples: tuple[A2D2LidarSampleStats, ...] = ()
    malformed_files: tuple[str, ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "sample_count": self.sample_count,
            "sampled_file_count": self.sampled_file_count,
            "total_point_count": self.total_point_count,
            "total_valid_count": self.total_valid_count,
            "physical_lidar_ids": list(self.physical_lidar_ids),
            "samples": [sample.as_dict() for sample in self.samples],
            "malformed_files": list(self.malformed_files),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _LidarAccumulator:
    lidar_id: int
    point_count: int = 0
    valid_count: int = 0
    bounds_min_m: tuple[float, float, float] | None = None
    bounds_max_m: tuple[float, float, float] | None = None

    def add(self, point: tuple[float, float, float], *, valid: bool) -> _LidarAccumulator:
        point_count = self.point_count + 1
        valid_count = self.valid_count + (1 if valid else 0)
        if not valid:
            return _LidarAccumulator(
                lidar_id=self.lidar_id,
                point_count=point_count,
                valid_count=valid_count,
                bounds_min_m=self.bounds_min_m,
                bounds_max_m=self.bounds_max_m,
            )
        bounds_min = point if self.bounds_min_m is None else _min3(self.bounds_min_m, point)
        bounds_max = point if self.bounds_max_m is None else _max3(self.bounds_max_m, point)
        return _LidarAccumulator(
            lidar_id=self.lidar_id,
            point_count=point_count,
            valid_count=valid_count,
            bounds_min_m=bounds_min,
            bounds_max_m=bounds_max,
        )

    def as_stats(self) -> A2D2PhysicalLidarStats:
        """Return immutable public stats."""

        return A2D2PhysicalLidarStats(
            lidar_id=self.lidar_id,
            point_count=self.point_count,
            valid_count=self.valid_count,
            bounds_min_m=self.bounds_min_m,
            bounds_max_m=self.bounds_max_m,
        )


class A2D2LidarDataset:
    """Reader for A2D2 public LiDAR NPZ files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def streams(self) -> list[StreamSummary]:
        """Return A2D2 LiDAR streams discovered from NPZ files."""

        files = find_a2d2_lidar_npz_files(self.path)
        streams = [
            StreamSummary(
                name="a2d2_lidar_npz",
                kind="pointcloud",
                message_count=len(files),
                sensor="a2d2_multi_lidar",
            )
        ]
        image_count = len(list(self.path.rglob("*.png"))) if self.path.is_dir() else 0
        if image_count:
            streams.append(
                StreamSummary(
                    name="a2d2_camera_png",
                    kind="camera",
                    message_count=image_count,
                    sensor="a2d2_camera",
                )
            )
        return streams

    def records(self, stream: str) -> Iterable[TimestampedRecord]:
        """Yield one normalized record per NPZ file."""

        if stream != "a2d2_lidar_npz":
            return
        for index, path in enumerate(find_a2d2_lidar_npz_files(self.path)):
            yield TimestampedRecord(
                stream=stream,
                timestamp_ns=index,
                payload_path=str(path),
                metadata={"format": "a2d2_npz"},
            )


def find_a2d2_lidar_npz_files(path: str | Path) -> list[Path]:
    """Return sorted A2D2 LiDAR NPZ files from a file or directory path."""

    root = Path(path)
    if root.is_file() and root.suffix == ".npz":
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob("*.npz"))


def summarize_a2d2_lidar_npz(path: str | Path, *, sample_limit: int = 3) -> A2D2LidarDatasetStats:
    """Summarize real A2D2 public LiDAR NPZ files."""

    files = find_a2d2_lidar_npz_files(path)
    if not files:
        return A2D2LidarDatasetStats(
            status="missing",
            sample_count=0,
            sampled_file_count=0,
            total_point_count=0,
            total_valid_count=0,
            physical_lidar_ids=(),
            reason="no A2D2 LiDAR NPZ files found",
        )

    samples: list[A2D2LidarSampleStats] = []
    malformed_files: list[str] = []
    total_points = 0
    total_valid = 0
    lidar_ids: set[int] = set()
    for sample_path in files[:sample_limit]:
        try:
            sample = summarize_a2d2_lidar_sample(sample_path)
        except (KeyError, ValueError, zipfile.BadZipFile, struct.error) as exc:
            malformed_files.append(f"{sample_path}: {exc}")
            continue
        samples.append(sample)
        total_points += sample.point_count
        total_valid += sample.valid_count
        lidar_ids.update(lidar.lidar_id for lidar in sample.physical_lidars)

    return A2D2LidarDatasetStats(
        status="scored" if samples else "malformed",
        sample_count=len(files),
        sampled_file_count=len(samples),
        total_point_count=total_points,
        total_valid_count=total_valid,
        physical_lidar_ids=tuple(sorted(lidar_ids)),
        samples=tuple(samples),
        malformed_files=tuple(malformed_files),
        reason=None if samples else "sampled A2D2 NPZ files could not be parsed",
    )


def read_a2d2_lidar_points(
    path: str | Path,
    *,
    max_points: int | None = None,
) -> list[tuple[float, float, float]]:
    """Return valid ``(x, y, z)`` points from one A2D2 LiDAR NPZ sample.

    Points are expressed in the camera-view frame stored in
    ``pcloud_points.npy``. A2D2's official tutorial states that sensor-fusion
    point clouds are mapped into the corresponding camera view. Invalid
    returns are dropped. When ``max_points`` is set, a deterministic uniform
    stride keeps the point count at or below the requested budget.
    """

    sample_path = Path(path)
    with zipfile.ZipFile(sample_path) as archive:
        points_header, points = read_npy_array(archive, "pcloud_points.npy")
        _valid_header, valid = read_npy_array(archive, "pcloud_attr.valid.npy")

    shape = points_header.get("shape")
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[1] != 3:
        raise ValueError("pcloud_points.npy must have shape Nx3")
    point_count = int(shape[0])
    if len(valid) != point_count:
        raise ValueError("valid array must match point count")

    valid_points: list[tuple[float, float, float]] = []
    for index in range(point_count):
        if not cast(bool, valid[index]):
            continue
        valid_points.append(
            (
                cast(float, points[index * 3]),
                cast(float, points[index * 3 + 1]),
                cast(float, points[index * 3 + 2]),
            )
        )
    return _stride_sample(valid_points, max_points)


def read_a2d2_lidar_points_reflectivity(
    path: str | Path,
    *,
    max_points: int | None = None,
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """Return paired valid camera-view points and calibrated reflectivity.

    Deterministic uniform stride sampling preserves point/reflectivity pairing.
    The reflectivity values are the A2D2 ``pcloud_attr.reflectance`` field and
    are not normalized by this dataset adapter.
    """

    sample_path = Path(path)
    with zipfile.ZipFile(sample_path) as archive:
        points_header, points = read_npy_array(archive, "pcloud_points.npy")
        _valid_header, valid = read_npy_array(archive, "pcloud_attr.valid.npy")
        _reflectivity_header, reflectivity = read_npy_array(
            archive, "pcloud_attr.reflectance.npy"
        )
    shape = points_header.get("shape")
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[1] != 3:
        raise ValueError("pcloud_points.npy must have shape Nx3")
    point_count = int(shape[0])
    if len(valid) != point_count or len(reflectivity) != point_count:
        raise ValueError("valid and reflectivity arrays must match point count")
    paired = [
        (
            (
                cast(float, points[index * 3]),
                cast(float, points[index * 3 + 1]),
                cast(float, points[index * 3 + 2]),
            ),
            float(cast(int | float, reflectivity[index])),
        )
        for index in range(point_count)
        if cast(bool, valid[index])
    ]
    sampled = _stride_sample(paired, max_points)
    return [item[0] for item in sampled], [item[1] for item in sampled]


def _stride_sample(
    points: list[SampleT],
    max_points: int | None,
) -> list[SampleT]:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points
    stride = (len(points) + max_points - 1) // max_points
    return points[::stride]


def summarize_a2d2_lidar_sample(path: str | Path) -> A2D2LidarSampleStats:
    """Summarize one A2D2 LiDAR NPZ sample."""

    sample_path = Path(path)
    with zipfile.ZipFile(sample_path) as archive:
        points_header, points = read_npy_array(archive, "pcloud_points.npy")
        _ids_header, lidar_ids = read_npy_array(archive, "pcloud_attr.lidar_id.npy")
        _valid_header, valid = read_npy_array(archive, "pcloud_attr.valid.npy")

    shape = points_header.get("shape")
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[1] != 3:
        raise ValueError("pcloud_points.npy must have shape Nx3")
    point_count = int(shape[0])
    if len(lidar_ids) != point_count or len(valid) != point_count:
        raise ValueError("lidar_id and valid arrays must match point count")

    accumulators: dict[int, _LidarAccumulator] = {}
    valid_count = 0
    for index, raw_lidar_id in enumerate(lidar_ids):
        lidar_id = cast(int, raw_lidar_id)
        is_valid = cast(bool, valid[index])
        valid_count += 1 if is_valid else 0
        point = (
            cast(float, points[index * 3]),
            cast(float, points[index * 3 + 1]),
            cast(float, points[index * 3 + 2]),
        )
        accumulator = accumulators.get(lidar_id, _LidarAccumulator(lidar_id=lidar_id))
        accumulators[lidar_id] = accumulator.add(point, valid=is_valid)

    physical_lidars = tuple(
        accumulators[lidar_id].as_stats() for lidar_id in sorted(accumulators)
    )
    return A2D2LidarSampleStats(
        path=str(sample_path),
        point_count=point_count,
        valid_count=valid_count,
        physical_lidars=physical_lidars,
    )


def read_npy_array(
    archive: zipfile.ZipFile,
    name: str,
) -> tuple[dict[str, Any], tuple[object, ...]]:
    """Read simple C-order numeric NPY arrays from an NPZ archive."""

    data = archive.read(name)
    if data[:6] != b"\x93NUMPY":
        raise ValueError(f"{name} is not an NPY array")
    offset = 6
    version = data[offset : offset + 2]
    offset += 2
    if version == b"\x01\x00":
        header_length = struct.unpack("<H", data[offset : offset + 2])[0]
        offset += 2
    else:
        header_length = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
    header = ast.literal_eval(data[offset : offset + header_length].decode("latin1").strip())
    offset += header_length
    if not isinstance(header, dict):
        raise ValueError(f"{name} has invalid NPY header")
    if header.get("fortran_order") is not False:
        raise ValueError(f"{name} uses unsupported Fortran order")
    shape = header.get("shape")
    if not isinstance(shape, tuple):
        raise ValueError(f"{name} has invalid shape metadata")
    count = 1
    for dimension in shape:
        count *= int(dimension)
    dtype = header.get("descr")
    type_map = {"<f8": "d", "<i8": "q", "|b1": "?"}
    if dtype not in type_map:
        raise ValueError(f"{name} uses unsupported dtype {dtype}")
    values = struct.unpack_from("<" + type_map[str(dtype)] * count, data, offset)
    return header, values


def _min3(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (min(left[0], right[0]), min(left[1], right[1]), min(left[2], right[2]))


def _max3(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (max(left[0], right[0]), max(left[1], right[1]), max(left[2], right[2]))
