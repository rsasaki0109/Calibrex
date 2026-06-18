"""Livox solid-state LiDAR PCD dataset helpers."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from calibrex.core.geometry import SE3
from calibrex.data.base import StreamSummary, TimestampedRecord

LivoxPoint = tuple[float, float, float, float]


@dataclass(frozen=True)
class LivoxPCDSampleStats:
    """Point statistics for one Livox PCD frame."""

    path: str
    point_count: int
    bounds_min_m: tuple[float, float, float] | None = None
    bounds_max_m: tuple[float, float, float] | None = None
    intensity_min: float | None = None
    intensity_max: float | None = None
    intensity_mean: float | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "path": self.path,
            "point_count": self.point_count,
            "bounds_min_m": list(self.bounds_min_m) if self.bounds_min_m else None,
            "bounds_max_m": list(self.bounds_max_m) if self.bounds_max_m else None,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "intensity_mean": self.intensity_mean,
        }


@dataclass(frozen=True)
class LivoxPCDDatasetStats:
    """Dataset-level statistics for Livox PCD files."""

    status: str
    sample_count: int
    sampled_file_count: int
    sampled_point_count: int
    bounds_min_m: tuple[float, float, float] | None = None
    bounds_max_m: tuple[float, float, float] | None = None
    intensity_min: float | None = None
    intensity_max: float | None = None
    intensity_mean: float | None = None
    pair_target_transform_applied: bool = False
    pair_transform_convention: str | None = None
    pair_voxel_size_m: float | None = None
    pair_source_voxel_count: int | None = None
    pair_target_voxel_count: int | None = None
    pair_shared_voxel_count: int | None = None
    pair_unmatched_source_voxel_count: int | None = None
    pair_unmatched_target_voxel_count: int | None = None
    pair_source_voxel_recall_in_target: float | None = None
    pair_target_voxel_recall_in_source: float | None = None
    pair_shared_voxel_centroid_rmse_m: float | None = None
    samples: tuple[LivoxPCDSampleStats, ...] = ()
    malformed_files: tuple[str, ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "sample_count": self.sample_count,
            "sampled_file_count": self.sampled_file_count,
            "sampled_point_count": self.sampled_point_count,
            "bounds_min_m": list(self.bounds_min_m) if self.bounds_min_m else None,
            "bounds_max_m": list(self.bounds_max_m) if self.bounds_max_m else None,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
            "intensity_mean": self.intensity_mean,
            "pair_target_transform_applied": self.pair_target_transform_applied,
            "pair_transform_convention": self.pair_transform_convention,
            "pair_voxel_size_m": self.pair_voxel_size_m,
            "pair_source_voxel_count": self.pair_source_voxel_count,
            "pair_target_voxel_count": self.pair_target_voxel_count,
            "pair_shared_voxel_count": self.pair_shared_voxel_count,
            "pair_unmatched_source_voxel_count": self.pair_unmatched_source_voxel_count,
            "pair_unmatched_target_voxel_count": self.pair_unmatched_target_voxel_count,
            "pair_source_voxel_recall_in_target": self.pair_source_voxel_recall_in_target,
            "pair_target_voxel_recall_in_source": self.pair_target_voxel_recall_in_source,
            "pair_shared_voxel_centroid_rmse_m": self.pair_shared_voxel_centroid_rmse_m,
            "samples": [sample.as_dict() for sample in self.samples],
            "malformed_files": list(self.malformed_files),
            "reason": self.reason,
        }


@dataclass
class _PointAccumulator:
    count: int = 0
    min_x: float = float("inf")
    min_y: float = float("inf")
    min_z: float = float("inf")
    max_x: float = float("-inf")
    max_y: float = float("-inf")
    max_z: float = float("-inf")
    intensity_min: float = float("inf")
    intensity_max: float = float("-inf")
    intensity_sum: float = 0.0

    def add(self, point: LivoxPoint) -> None:
        """Add one point to the accumulator."""

        x, y, z, intensity = point
        self.count += 1
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.min_z = min(self.min_z, z)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)
        self.max_z = max(self.max_z, z)
        self.intensity_min = min(self.intensity_min, intensity)
        self.intensity_max = max(self.intensity_max, intensity)
        self.intensity_sum += intensity

    @property
    def bounds_min_m(self) -> tuple[float, float, float] | None:
        return (self.min_x, self.min_y, self.min_z) if self.count else None

    @property
    def bounds_max_m(self) -> tuple[float, float, float] | None:
        return (self.max_x, self.max_y, self.max_z) if self.count else None

    @property
    def intensity_mean(self) -> float | None:
        return self.intensity_sum / self.count if self.count else None


@dataclass
class _VoxelCentroid:
    count: int = 0
    x_sum: float = 0.0
    y_sum: float = 0.0
    z_sum: float = 0.0

    def add(self, point: LivoxPoint) -> None:
        x, y, z, _intensity = point
        self.count += 1
        self.x_sum += x
        self.y_sum += y
        self.z_sum += z

    @property
    def centroid(self) -> tuple[float, float, float]:
        if self.count == 0:
            return (0.0, 0.0, 0.0)
        return (self.x_sum / self.count, self.y_sum / self.count, self.z_sum / self.count)


@dataclass(frozen=True)
class _VoxelPairSummary:
    voxel_size_m: float
    source_voxel_count: int
    target_voxel_count: int
    shared_voxel_count: int
    unmatched_source_voxel_count: int
    unmatched_target_voxel_count: int
    source_voxel_recall_in_target: float
    target_voxel_recall_in_source: float
    shared_voxel_centroid_rmse_m: float | None


class LivoxPCDDataset:
    """Reader for Livox solid-state LiDAR PCD files."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def streams(self) -> list[StreamSummary]:
        """Return Livox PCD streams discovered from files."""

        files = find_livox_pcd_files(self.path)
        return [
            StreamSummary(
                name="livox_pcd",
                kind="pointcloud",
                message_count=len(files),
                sensor="livox_solid_state_lidar",
            )
        ]

    def records(self, stream: str) -> Iterable[TimestampedRecord]:
        """Yield one normalized record per PCD file."""

        if stream != "livox_pcd":
            return
        for index, path in enumerate(find_livox_pcd_files(self.path)):
            yield TimestampedRecord(
                stream=stream,
                timestamp_ns=index,
                payload_path=str(path),
                metadata={"format": "pcd", "sensor_family": "livox"},
            )


def find_livox_pcd_files(path: str | Path) -> list[Path]:
    """Return sorted Livox PCD files from a file or directory path."""

    root = Path(path)
    if root.is_file() and root.suffix.lower() == ".pcd":
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob("*.pcd"))


def summarize_livox_pcd(
    path: str | Path,
    *,
    sample_limit: int = 4,
    pair_voxel_size_m: float = 1.0,
    target_transform: SE3 | None = None,
) -> LivoxPCDDatasetStats:
    """Summarize Livox public solid-state LiDAR PCD files."""

    files = find_livox_pcd_files(path)
    if not files:
        return LivoxPCDDatasetStats(
            status="missing",
            sample_count=0,
            sampled_file_count=0,
            sampled_point_count=0,
            reason="no Livox PCD files found",
        )

    samples: list[LivoxPCDSampleStats] = []
    sampled_points_by_file: list[list[LivoxPoint]] = []
    malformed_files: list[str] = []
    total = _PointAccumulator()
    for sample_path in files[:sample_limit]:
        try:
            points = read_livox_binary_pcd(sample_path)
        except (ValueError, struct.error) as exc:
            malformed_files.append(f"{sample_path}: {exc}")
            continue
        sample = _summarize_sample(sample_path, points)
        samples.append(sample)
        sampled_points_by_file.append(points)
        for point in points:
            total.add(point)

    if len(sampled_points_by_file) >= 2:
        target_points = sampled_points_by_file[1]
        if target_transform is not None:
            target_points = _transform_points(target_points, target_transform)
        pair = _summarize_voxel_pair(
            sampled_points_by_file[0],
            target_points,
            voxel_size_m=pair_voxel_size_m,
        )
    else:
        pair = None
    status = "scored" if samples else "malformed"
    return LivoxPCDDatasetStats(
        status=status,
        sample_count=len(files),
        sampled_file_count=len(samples),
        sampled_point_count=total.count,
        bounds_min_m=total.bounds_min_m,
        bounds_max_m=total.bounds_max_m,
        intensity_min=total.intensity_min if total.count else None,
        intensity_max=total.intensity_max if total.count else None,
        intensity_mean=total.intensity_mean,
        pair_target_transform_applied=target_transform is not None,
        pair_transform_convention=(
            "T_source_target maps target PCD points into the source PCD frame"
            if target_transform is not None
            else None
        ),
        pair_voxel_size_m=pair.voxel_size_m if pair is not None else None,
        pair_source_voxel_count=pair.source_voxel_count if pair is not None else None,
        pair_target_voxel_count=pair.target_voxel_count if pair is not None else None,
        pair_shared_voxel_count=pair.shared_voxel_count if pair is not None else None,
        pair_unmatched_source_voxel_count=(
            pair.unmatched_source_voxel_count if pair is not None else None
        ),
        pair_unmatched_target_voxel_count=(
            pair.unmatched_target_voxel_count if pair is not None else None
        ),
        pair_source_voxel_recall_in_target=(
            pair.source_voxel_recall_in_target if pair is not None else None
        ),
        pair_target_voxel_recall_in_source=(
            pair.target_voxel_recall_in_source if pair is not None else None
        ),
        pair_shared_voxel_centroid_rmse_m=(
            pair.shared_voxel_centroid_rmse_m if pair is not None else None
        ),
        samples=tuple(samples),
        malformed_files=tuple(malformed_files),
        reason=None if samples else "sampled Livox PCD files could not be parsed",
    )


def read_livox_binary_pcd(path: str | Path) -> list[LivoxPoint]:
    """Read x/y/z/intensity from a binary float32 PCD file."""

    pcd_path = Path(path)
    data = pcd_path.read_bytes()
    try:
        header_end = data.index(b"\n", data.index(b"DATA binary")) + 1
    except ValueError as exc:
        raise ValueError("only binary PCD files are supported") from exc
    header = data[:header_end].decode("ascii", errors="strict")
    fields, sizes, types, counts, point_count = _parse_pcd_header(header)
    if fields[:4] != ["x", "y", "z", "intensity"]:
        raise ValueError("PCD must store x/y/z/intensity as the first fields")
    if sizes[:4] != [4, 4, 4, 4] or types[:4] != ["F", "F", "F", "F"]:
        raise ValueError("PCD x/y/z/intensity fields must be float32")
    point_step = sum(size * count for size, count in zip(sizes, counts, strict=True))
    available = (len(data) - header_end) // point_step
    point_count = min(point_count or available, available)
    points: list[LivoxPoint] = []
    for index in range(point_count):
        x, y, z, intensity = struct.unpack_from(
            "<ffff",
            data,
            header_end + index * point_step,
        )
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        points.append((float(x), float(y), float(z), float(intensity)))
    return points


def _parse_pcd_header(
    header: str,
) -> tuple[list[str], list[int], list[str], list[int], int]:
    fields: list[str] = []
    sizes: list[int] = []
    types: list[str] = []
    counts: list[int] = []
    point_count = 0
    for line_text in header.splitlines():
        parts = line_text.split()
        if not parts:
            continue
        key = parts[0]
        if key == "FIELDS":
            fields = parts[1:]
        elif key == "SIZE":
            sizes = [int(value) for value in parts[1:]]
        elif key == "TYPE":
            types = parts[1:]
        elif key == "COUNT":
            counts = [int(value) for value in parts[1:]]
        elif key == "POINTS":
            point_count = int(parts[1])
    if not fields or not sizes or not types:
        raise ValueError("PCD header is missing FIELDS, SIZE, or TYPE")
    if not counts:
        counts = [1] * len(fields)
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD header field metadata lengths do not match")
    return fields, sizes, types, counts, point_count


def _summarize_sample(path: Path, points: list[LivoxPoint]) -> LivoxPCDSampleStats:
    accumulator = _PointAccumulator()
    for point in points:
        accumulator.add(point)
    return LivoxPCDSampleStats(
        path=str(path),
        point_count=accumulator.count,
        bounds_min_m=accumulator.bounds_min_m,
        bounds_max_m=accumulator.bounds_max_m,
        intensity_min=accumulator.intensity_min if accumulator.count else None,
        intensity_max=accumulator.intensity_max if accumulator.count else None,
        intensity_mean=accumulator.intensity_mean,
    )


def _summarize_voxel_pair(
    source_points: list[LivoxPoint],
    target_points: list[LivoxPoint],
    *,
    voxel_size_m: float,
) -> _VoxelPairSummary:
    source_voxels = _voxel_centroids(source_points, voxel_size_m)
    target_voxels = _voxel_centroids(target_points, voxel_size_m)
    shared_keys = set(source_voxels) & set(target_voxels)
    squared_distances: list[float] = []
    for key in shared_keys:
        sx, sy, sz = source_voxels[key].centroid
        tx, ty, tz = target_voxels[key].centroid
        squared_distances.append(
            (tx - sx) * (tx - sx)
            + (ty - sy) * (ty - sy)
            + (tz - sz) * (tz - sz)
        )
    rmse = math.sqrt(sum(squared_distances) / len(squared_distances)) if squared_distances else None
    source_voxel_count = len(source_voxels)
    target_voxel_count = len(target_voxels)
    shared_voxel_count = len(shared_keys)
    return _VoxelPairSummary(
        voxel_size_m=voxel_size_m,
        source_voxel_count=source_voxel_count,
        target_voxel_count=target_voxel_count,
        shared_voxel_count=shared_voxel_count,
        unmatched_source_voxel_count=source_voxel_count - shared_voxel_count,
        unmatched_target_voxel_count=target_voxel_count - shared_voxel_count,
        source_voxel_recall_in_target=shared_voxel_count / max(1, source_voxel_count),
        target_voxel_recall_in_source=shared_voxel_count / max(1, target_voxel_count),
        shared_voxel_centroid_rmse_m=rmse,
    )


def _voxel_centroids(
    points: list[LivoxPoint],
    voxel_size_m: float,
) -> dict[tuple[int, int, int], _VoxelCentroid]:
    voxels: dict[tuple[int, int, int], _VoxelCentroid] = {}
    for point in points:
        voxel = voxels.setdefault(_voxel_key(point, voxel_size_m), _VoxelCentroid())
        voxel.add(point)
    return voxels


def _transform_points(points: list[LivoxPoint], transform: SE3) -> list[LivoxPoint]:
    transformed: list[LivoxPoint] = []
    for x, y, z, intensity in points:
        tx, ty, tz = transform.transform_point((x, y, z))
        transformed.append((tx, ty, tz, intensity))
    return transformed


def _voxel_key(point: LivoxPoint, voxel_size_m: float) -> tuple[int, int, int]:
    return (
        math.floor(point[0] / voxel_size_m),
        math.floor(point[1] / voxel_size_m),
        math.floor(point[2] / voxel_size_m),
    )
