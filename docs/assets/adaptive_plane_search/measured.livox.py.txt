"""Livox solid-state LiDAR PCD dataset helpers."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass, fields, replace
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import (
    SE3,
    Vector3,
    normalize_quaternion_xyzw,
    quaternion_conjugate_xyzw,
)
from calibrex.data.base import StreamSummary, TimestampedRecord

LivoxPoint = tuple[float, float, float, float]
_PCD_ARRAY_MIN_POINTS = 256


@dataclass(frozen=True)
class LivoxPointRecord:
    """Livox point with optional per-point normal fields from the PCD record."""

    point: LivoxPoint
    normal_xyz: Vector3 | None = None


@dataclass(frozen=True)
class VoxelPlaneMatch:
    """Stable identity and geometry for one nearest voxel-plane association."""

    voxel_key: tuple[int, int, int]
    centroid: Vector3
    normal: Vector3
    centroid_distance_m: float

    @property
    def target_id(self) -> str:
        """Return a deterministic target identifier independent of float formatting."""

        return f"voxel:{self.voxel_key[0]}:{self.voxel_key[1]}:{self.voxel_key[2]}"


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
    range_min_m: float | None = None
    range_max_m: float | None = None
    fov_azimuth_deg: float | None = None
    fov_elevation_deg: float | None = None

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
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
            "fov_azimuth_deg": self.fov_azimuth_deg,
            "fov_elevation_deg": self.fov_elevation_deg,
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
    range_min_m: float | None = None
    range_max_m: float | None = None
    fov_azimuth_deg: float | None = None
    fov_elevation_deg: float | None = None
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
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
            "fov_azimuth_deg": self.fov_azimuth_deg,
            "fov_elevation_deg": self.fov_elevation_deg,
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

    def add_many(self, points: list[LivoxPoint]) -> None:
        """Accumulate finite arrays with first extrema and input-order additions."""

        if len(points) < 256:
            for point in points:
                self.add(point)
            return
        values = np.asarray(points, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            for point in points:
                self.add(point)
            return
        columns = np.arange(4)
        lower = values[np.argmin(values, axis=0), columns].tolist()
        upper = values[np.argmax(values, axis=0), columns].tolist()
        self.min_x, self.min_y, self.min_z, self.intensity_min = (
            min(previous, value) for previous, value in zip(
                (self.min_x, self.min_y, self.min_z, self.intensity_min), lower, strict=True,
            )
        )
        self.max_x, self.max_y, self.max_z, self.intensity_max = (
            max(previous, value) for previous, value in zip(
                (self.max_x, self.max_y, self.max_z, self.intensity_max), upper, strict=True,
            )
        )
        with np.errstate(over="ignore", invalid="ignore"):
            self.intensity_sum = float(np.add.accumulate(np.concatenate((
                [self.intensity_sum], values[:, 3],
            )))[-1])
        self.count += len(points)

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


@dataclass
class _VoxelPlane:
    count: int = 0
    x_sum: float = 0.0
    y_sum: float = 0.0
    z_sum: float = 0.0
    normal_x_sum: float = 0.0
    normal_y_sum: float = 0.0
    normal_z_sum: float = 0.0
    normal_count: int = 0
    fallback_points: list[Vector3] | None = None

    def add(self, record: LivoxPointRecord) -> None:
        x, y, z, _intensity = record.point
        self.count += 1
        self.x_sum += x
        self.y_sum += y
        self.z_sum += z
        if self.fallback_points is None:
            self.fallback_points = []
        if len(self.fallback_points) < 12:
            self.fallback_points.append((x, y, z))
        if record.normal_xyz is None:
            return
        nx, ny, nz = record.normal_xyz
        self.normal_x_sum += nx
        self.normal_y_sum += ny
        self.normal_z_sum += nz
        self.normal_count += 1

    @property
    def centroid(self) -> Vector3:
        if self.count == 0:
            return (0.0, 0.0, 0.0)
        return (self.x_sum / self.count, self.y_sum / self.count, self.z_sum / self.count)

    @property
    def normal(self) -> Vector3 | None:
        if self.normal_count:
            return _normalized3((self.normal_x_sum, self.normal_y_sum, self.normal_z_sum))
        return _fallback_plane_normal(self.fallback_points or [])


VoxelPlaneMap = dict[tuple[int, int, int], _VoxelPlane]


_DENSE_PLANE_INDEX_MAX_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class _PlaneIndex:
    keys: NDArray[np.int64]
    lower: NDArray[np.int64]
    upper: NDArray[np.int64]
    widths: tuple[int, int, int]
    linear_keys: NDArray[np.int64] | None
    dense_positions: NDArray[np.int64] | None
    centroids: NDArray[np.float64]
    normals: NDArray[np.float64]
    valid_normal: NDArray[np.bool_]
    neighborhoods: _PlaneNeighborhoods | None = None


@dataclass(frozen=True)
class _PlaneNeighborhoods:
    lower: NDArray[np.int64]
    upper: NDArray[np.int64]
    widths: tuple[int, int, int]
    offsets: NDArray[np.int64]
    positions: NDArray[np.int64]


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


@dataclass(frozen=True)
class LivoxPairPointToPlaneStats:
    """Single-pair holdout point-to-plane evidence for a Livox PCD pair."""

    status: str
    split_policy: str
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    independent_holdout: bool
    support_population_id: str
    support_definition: str
    eligible_point_count: int
    considered_point_count: int
    accepted_correspondence_count: int
    support_ratio: float | None
    map_voxel_count: int
    matched_point_count: int
    unmatched_point_count: int
    unmatched_fraction: float | None
    median_abs_m: float | None
    p90_abs_m: float | None
    rmse_m: float | None
    inlier_fraction: float | None
    voxel_size_m: float
    correspondence_gate_m: float
    inlier_threshold_m: float
    plane_normal_source: str
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "status": self.status,
            "split_policy": self.split_policy,
            "train_frame_ids": list(self.train_frame_ids),
            "holdout_frame_ids": list(self.holdout_frame_ids),
            "independent_holdout": self.independent_holdout,
            "support_population_id": self.support_population_id,
            "support_definition": self.support_definition,
            "eligible_point_count": self.eligible_point_count,
            "considered_point_count": self.considered_point_count,
            "accepted_correspondence_count": self.accepted_correspondence_count,
            "support_ratio": self.support_ratio,
            "exclusion_counts": {
                "no_plane_within_gate": self.unmatched_point_count,
            },
            "map_voxel_count": self.map_voxel_count,
            "matched_point_count": self.matched_point_count,
            "unmatched_point_count": self.unmatched_point_count,
            "unmatched_fraction": self.unmatched_fraction,
            "median_abs_m": self.median_abs_m,
            "p90_abs_m": self.p90_abs_m,
            "rmse_m": self.rmse_m,
            "inlier_fraction": self.inlier_fraction,
            "voxel_size_m": self.voxel_size_m,
            "correspondence_gate_m": self.correspondence_gate_m,
            "inlier_threshold_m": self.inlier_threshold_m,
            "plane_normal_source": self.plane_normal_source,
            "reason": self.reason,
        }


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
                metadata={
                    "format": "pcd",
                    "sensor_family": "livox",
                    "timestamp_semantics": "file_order_index_not_physical_capture_time",
                    "point_time_available": False,
                },
            )


def find_livox_pcd_files(path: str | Path) -> list[Path]:
    """Return sorted Livox PCD files from a file or directory path."""

    root = Path(path)
    if root.is_file() and root.suffix.lower() == ".pcd":
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob("*.pcd"))


class LivoxPCDEvaluationContext:
    """Reuse one captured dataset's geometry during a calibration run.

    Create a new context for every calibration/evaluation invocation. File discovery, decoded
    records, invariant sample statistics, source voxels and prepared planes are
    scoped to this object; no process-wide or cross-invocation cache is used.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        self._files = find_livox_pcd_files(path)
        self._records: dict[Path, list[LivoxPointRecord]] = {}
        self._errors: dict[Path, ValueError | struct.error] = {}
        self._points: dict[Path, list[LivoxPoint]] = {}
        self._coordinates: dict[Path, NDArray[np.float64]] = {}
        # Retain only the most recent transform per file, shared by the two
        # evidence checks. Identity comparison preserves signed-zero inputs.
        self._queries: dict[Path, tuple[SE3 | None, NDArray[np.float64]]] = {}
        self._stats: dict[tuple[tuple[Path, ...], float], LivoxPCDDatasetStats] = {}
        self._planes: dict[tuple[Path, float], VoxelPlaneMap] = {}
        self._plane_indices: dict[tuple[Path, float], _PlaneIndex | None] = {}
        self._voxels: dict[tuple[Path, float], dict[tuple[int, int, int], _VoxelCentroid]] = {}
        self._distance_bins: dict[tuple[tuple[float, ...], int], tuple[int, ...]] = {}

    def _validate_path(self, path: str | Path) -> None:
        if Path(path).resolve() != self._path:
            raise ValueError("Livox evaluation context belongs to a different dataset")

    def _fork_queries(self) -> LivoxPCDEvaluationContext:
        # Candidate evaluation has prepared the invariant records, statistics,
        # source maps and coordinate arrays. A probe owns its transform cache.
        fork = copy(self)
        fork._queries = {}
        return fork

    def _read_records(self, path: Path) -> list[LivoxPointRecord]:
        if path in self._errors:
            raise self._errors[path]
        if path not in self._records:
            try:
                self._records[path] = read_livox_binary_pcd_records(path)
            except (ValueError, struct.error) as exc:
                self._errors[path] = exc
                raise
        return self._records[path]

    def _read_points(self, path: Path) -> list[LivoxPoint]:
        if path not in self._points:
            self._points[path] = [record.point for record in self._read_records(path)]
        return self._points[path]

    def read_points(self, path: str | Path) -> list[LivoxPoint]:
        """Return a copy of one captured PCD's finite points within this run."""

        resolved = Path(path).resolve()
        for captured in self._files:
            if captured.resolve() == resolved:
                return list(self._read_points(captured))
        raise ValueError("PCD file is not part of this Livox evaluation context")

    def read_records(self, path: str | Path) -> list[LivoxPointRecord]:
        """Return a copy of captured finite points and normals within this run."""

        resolved = Path(path).resolve()
        for captured in self._files:
            if captured.resolve() == resolved:
                return list(self._read_records(captured))
        raise ValueError("PCD file is not part of this Livox evaluation context")

    def distance_bin_counts(
        self, path: str | Path, edges_m: Iterable[float], *, sample_limit: int = 4,
    ) -> list[int] | None:
        """Count sampled captured points with the scalar closed-last-bin policy."""

        self._validate_path(path)
        if not self._files:
            return None
        edges = tuple(edges_m)
        key = (edges, sample_limit)
        if key not in self._distance_bins:
            counts = [0] * (len(edges) - 1)
            for captured in self._files[:sample_limit]:
                try:
                    points = self._read_points(captured)
                except (OSError, ValueError, struct.error):
                    continue
                for x, y, z, _intensity in points:
                    distance = math.sqrt(x * x + y * y + z * z)
                    for index, (lower, upper) in enumerate(pairwise(edges)):
                        if lower <= distance < upper or (
                            index == len(counts) - 1 and distance == upper
                        ):
                            counts[index] += 1
                            break
            self._distance_bins[key] = tuple(counts)
        return list(self._distance_bins[key])

    def _query_coordinates(self, path: Path, transform: SE3 | None) -> NDArray[np.float64]:
        cached = self._queries.get(path)
        if cached is not None and cached[0] is transform:
            return cached[1]
        if path not in self._coordinates:
            coordinates = np.asarray(self._read_points(path), dtype=np.float64).reshape((-1, 4))
            self._coordinates[path] = coordinates[:, :3].copy()
            self._coordinates[path].setflags(write=False)
        query = _transform_coordinates(self._coordinates[path], transform)
        query.setflags(write=False)
        self._queries[path] = (transform, query)
        return query

    def _plane_map(self, path: Path, voxel_size_m: float) -> VoxelPlaneMap:
        key = (path, voxel_size_m)
        if key not in self._planes:
            self._planes[key] = _voxel_plane_map(self._read_records(path), voxel_size_m)
        return self._planes[key]

    def _plane_index(self, path: Path, voxel_size_m: float) -> _PlaneIndex | None:
        key = (path, voxel_size_m)
        if key not in self._plane_indices:
            self._plane_indices[key] = _prepare_plane_index(self._plane_map(path, voxel_size_m))
        return self._plane_indices[key]

    def _voxel_map(
        self, path: Path, voxel_size_m: float,
    ) -> dict[tuple[int, int, int], _VoxelCentroid]:
        key = (path, voxel_size_m)
        if key not in self._voxels:
            self._voxels[key] = _voxel_centroids(self._read_points(path), voxel_size_m)
        return self._voxels[key]


def summarize_livox_pcd(
    path: str | Path,
    *,
    sample_limit: int = 4,
    pair_voxel_size_m: float = 1.0,
    target_transform: SE3 | None = None,
    context: LivoxPCDEvaluationContext | None = None,
) -> LivoxPCDDatasetStats:
    """Summarize PCD files, optionally reusing one evaluation's captured inputs."""

    if context is None:
        return _summarize_livox_pcd(
            path, sample_limit=sample_limit, pair_voxel_size_m=pair_voxel_size_m,
            target_transform=target_transform,
        )
    context._validate_path(path)
    key = (tuple(context._files[:sample_limit]), pair_voxel_size_m)
    if key not in context._stats:
        context._stats[key] = _summarize_livox_pcd(
            path, sample_limit=sample_limit, pair_voxel_size_m=pair_voxel_size_m,
            target_transform=None, context=context,
        )
    baseline = context._stats[key]
    if target_transform is None or not context._files:
        return baseline
    points = [context._read_points(Path(sample.path)) for sample in baseline.samples]
    pair = (
        _summarize_voxel_pair(
            points[0], (
                context._query_coordinates(Path(baseline.samples[1].path), target_transform)
                if len(points[1]) >= 64 else _transform_points(points[1], target_transform)
            ),
            voxel_size_m=pair_voxel_size_m,
            source_voxels=context._voxel_map(Path(baseline.samples[0].path), pair_voxel_size_m),
        )
        if len(points) >= 2 else None
    )
    pair_fields = (
        {f"pair_{field.name}": getattr(pair, field.name) for field in fields(pair)}
        if pair is not None else {}
    )
    return replace(
        baseline, **pair_fields, pair_target_transform_applied=True,
        pair_transform_convention=(
            "T_source_target maps target PCD points into the source PCD frame"
        ),
    )


def _summarize_livox_pcd(
    path: str | Path,
    *,
    sample_limit: int = 4,
    pair_voxel_size_m: float = 1.0,
    target_transform: SE3 | None = None,
    context: LivoxPCDEvaluationContext | None = None,
) -> LivoxPCDDatasetStats:
    """Summarize Livox public solid-state LiDAR PCD files."""

    files = context._files if context is not None else find_livox_pcd_files(path)
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
    geometry_values: tuple[list[float], list[float], list[float]] = ([], [], [])
    for sample_path in files[:sample_limit]:
        try:
            points = (
                context._read_points(sample_path) if context is not None
                else read_livox_binary_pcd(sample_path)
            )
        except (ValueError, struct.error) as exc:
            malformed_files.append(f"{sample_path}: {exc}")
            continue
        sample_geometry = _geometry_values(points)
        sample = _summarize_sample(sample_path, points, geometry_values=sample_geometry)
        samples.append(sample)
        sampled_points_by_file.append(points)
        total.add_many(points)
        for combined, sampled in zip(geometry_values, sample_geometry, strict=True):
            combined.extend(sampled)

    if len(sampled_points_by_file) >= 2:
        target_points = sampled_points_by_file[1]
        if target_transform is not None:
            target_points = _transform_points(target_points, target_transform)
        pair = _summarize_voxel_pair(
            sampled_points_by_file[0],
            target_points,
            voxel_size_m=pair_voxel_size_m,
            source_voxels=(
                context._voxel_map(Path(samples[0].path), pair_voxel_size_m)
                if context is not None else None
            ),
        )
    else:
        pair = None
    status = "scored" if samples else "malformed"
    range_min_m, range_max_m, fov_azimuth_deg, fov_elevation_deg = _coverage_from_values(
        geometry_values
    )
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
        range_min_m=range_min_m,
        range_max_m=range_max_m,
        fov_azimuth_deg=fov_azimuth_deg,
        fov_elevation_deg=fov_elevation_deg,
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


def summarize_livox_pair_point_to_plane(
    path: str | Path,
    *,
    target_transform: SE3 | None = None,
    voxel_size_m: float = 1.0,
    correspondence_gate_m: float = 1.5,
    inlier_threshold_m: float = 0.25,
    context: LivoxPCDEvaluationContext | None = None,
) -> LivoxPairPointToPlaneStats:
    """Score transformed target PCD points against source-frame voxel planes.

    The public Livox Horizon-Horizon sample contains one source and one target
    PCD frame. This is therefore a single-pair holdout geometry check rather
    than an independent multi-window temporal validation.
    """

    if context is not None:
        context._validate_path(path)
    files = context._files if context is not None else find_livox_pcd_files(path)
    if len(files) < 2:
        train_frame_ids = tuple(_frame_id(file_path) for file_path in files[:1])
        holdout_frame_ids: tuple[str, ...] = ()
        return LivoxPairPointToPlaneStats(
            status="insufficient_support",
            split_policy="single_pair_source_map_target_query",
            train_frame_ids=train_frame_ids,
            holdout_frame_ids=holdout_frame_ids,
            independent_holdout=False,
            support_population_id=_support_population_id(
                train_frame_ids=train_frame_ids,
                holdout_frame_ids=holdout_frame_ids,
                eligible_point_count=0,
                voxel_size_m=voxel_size_m,
                correspondence_gate_m=correspondence_gate_m,
            ),
            support_definition=_support_definition(),
            eligible_point_count=0,
            considered_point_count=0,
            accepted_correspondence_count=0,
            support_ratio=None,
            map_voxel_count=0,
            matched_point_count=0,
            unmatched_point_count=0,
            unmatched_fraction=None,
            median_abs_m=None,
            p90_abs_m=None,
            rmse_m=None,
            inlier_fraction=None,
            voxel_size_m=voxel_size_m,
            correspondence_gate_m=correspondence_gate_m,
            inlier_threshold_m=inlier_threshold_m,
            plane_normal_source="pcd_normal_fields_or_local_fallback",
            reason="at least two Livox PCD files are required",
        )

    try:
        source_records = (
            context._read_records(files[0]) if context is not None
            else read_livox_binary_pcd_records(files[0])
        )
        target_records = (
            context._read_records(files[1]) if context is not None
            else read_livox_binary_pcd_records(files[1])
        )
    except (ValueError, struct.error) as exc:
        train_frame_ids = (_frame_id(files[0]),)
        holdout_frame_ids = (_frame_id(files[1]),)
        return LivoxPairPointToPlaneStats(
            status="malformed",
            split_policy="single_pair_source_map_target_query",
            train_frame_ids=train_frame_ids,
            holdout_frame_ids=holdout_frame_ids,
            independent_holdout=False,
            support_population_id=_support_population_id(
                train_frame_ids=train_frame_ids,
                holdout_frame_ids=holdout_frame_ids,
                eligible_point_count=0,
                voxel_size_m=voxel_size_m,
                correspondence_gate_m=correspondence_gate_m,
            ),
            support_definition=_support_definition(),
            eligible_point_count=0,
            considered_point_count=0,
            accepted_correspondence_count=0,
            support_ratio=None,
            map_voxel_count=0,
            matched_point_count=0,
            unmatched_point_count=0,
            unmatched_fraction=None,
            median_abs_m=None,
            p90_abs_m=None,
            rmse_m=None,
            inlier_fraction=None,
            voxel_size_m=voxel_size_m,
            correspondence_gate_m=correspondence_gate_m,
            inlier_threshold_m=inlier_threshold_m,
            plane_normal_source="pcd_normal_fields_or_local_fallback",
            reason=str(exc),
        )

    plane_map = (
        context._plane_map(files[0], voxel_size_m) if context is not None
        else _voxel_plane_map(source_records, voxel_size_m)
    )
    train_frame_ids = (_frame_id(files[0]),)
    holdout_frame_ids = (_frame_id(files[1]),)
    query_points = (
        context._query_coordinates(files[1], target_transform)
        if context is not None and len(target_records) >= 64 else _transform_query_points(
            [record.point for record in target_records], target_transform,
        )
    )
    eligible_count = len(target_records)
    considered_count = len(query_points)
    residuals, unmatched_count = _point_to_plane_residuals(
        query_points,
        plane_map=plane_map,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
        index=(
            context._plane_index(files[0], voxel_size_m)
            if context is not None and len(query_points) >= 64 else None
        ),
    )
    query_count = len(query_points)
    matched_count = len(residuals)
    unmatched_fraction = unmatched_count / query_count if query_count else None
    support_ratio = matched_count / eligible_count if eligible_count else None
    status = "scored" if residuals else "insufficient_support"
    inlier_count = sum(1 for residual in residuals if residual <= inlier_threshold_m)
    return LivoxPairPointToPlaneStats(
        status=status,
        split_policy="single_pair_source_map_target_query",
        train_frame_ids=train_frame_ids,
        holdout_frame_ids=holdout_frame_ids,
        independent_holdout=False,
        support_population_id=_support_population_id(
            train_frame_ids=train_frame_ids,
            holdout_frame_ids=holdout_frame_ids,
            eligible_point_count=eligible_count,
            voxel_size_m=voxel_size_m,
            correspondence_gate_m=correspondence_gate_m,
        ),
        support_definition=_support_definition(),
        eligible_point_count=eligible_count,
        considered_point_count=considered_count,
        accepted_correspondence_count=matched_count,
        support_ratio=support_ratio,
        map_voxel_count=len(plane_map),
        matched_point_count=matched_count,
        unmatched_point_count=unmatched_count,
        unmatched_fraction=unmatched_fraction,
        median_abs_m=_percentile(residuals, 0.5),
        p90_abs_m=_percentile(residuals, 0.9),
        rmse_m=_rmse(residuals),
        inlier_fraction=inlier_count / matched_count if matched_count else None,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
        inlier_threshold_m=inlier_threshold_m,
        plane_normal_source="pcd_normal_fields_or_local_fallback",
        reason=(
            "single source/target PCD pair; use as limited holdout geometry evidence, "
            "not independent temporal validation"
        ),
    )


def _support_population_id(
    *,
    train_frame_ids: tuple[str, ...],
    holdout_frame_ids: tuple[str, ...],
    eligible_point_count: int,
    voxel_size_m: float,
    correspondence_gate_m: float,
) -> str:
    train = ",".join(train_frame_ids) or "none"
    holdout = ",".join(holdout_frame_ids) or "none"
    return (
        "livox_pair_support:"
        f"train={train}:holdout={holdout}:eligible_points={eligible_point_count}:"
        f"voxel_m={voxel_size_m:g}:gate_m={correspondence_gate_m:g}"
    )


def _support_definition() -> str:
    return (
        "eligible population is every finite target PCD point in the declared "
        "source/target pair; candidate transforms may change accepted "
        "correspondences but not this denominator"
    )


def build_voxel_plane_map(
    records: list[LivoxPointRecord],
    voxel_size_m: float,
) -> VoxelPlaneMap:
    """Return the voxel-plane map used for point-to-plane correspondences.

    Each voxel aggregates a centroid and a plane normal (from per-point PCD
    normal fields when present, otherwise a local geometric fallback). Voxels
    without a resolvable normal are dropped.
    """

    return _voxel_plane_map(records, voxel_size_m)


def nearest_voxel_plane(
    point: Vector3,
    plane_map: VoxelPlaneMap,
    *,
    voxel_size_m: float,
    correspondence_gate_m: float,
) -> _VoxelPlane | None:
    """Return the nearest voxel plane within ``correspondence_gate_m`` or ``None``."""

    return _nearest_plane(
        point,
        plane_map=plane_map,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
    )


def nearest_voxel_plane_match(
    point: Vector3,
    plane_map: dict[tuple[int, int, int], _VoxelPlane],
    *,
    voxel_size_m: float,
    correspondence_gate_m: float,
) -> VoxelPlaneMatch | None:
    """Return nearest plane geometry plus its stable voxel-map identity."""

    return _nearest_plane_match(
        point,
        plane_map=plane_map,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
    )


def nearest_voxel_planes(
    points: NDArray[np.float64], plane_map: VoxelPlaneMap, *,
    voxel_size_m: float, correspondence_gate_m: float,
) -> list[_VoxelPlane | None]:
    """Return ordered exact matches, using packed occupied neighbors when possible."""

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cells = np.floor(points / voxel_size_m)
    index = _prepare_plane_index(plane_map)
    if (
        len(points) < 64 or index is None or index.neighborhoods is None
        or not np.isfinite(cells).all() or np.any(cells < -(2.0**63))
        or np.any(cells >= 2.0**63)
    ):
        return [nearest_voxel_plane(
            (float(p[0]), float(p[1]), float(p[2])), plane_map,
            voxel_size_m=voxel_size_m, correspondence_gate_m=correspondence_gate_m,
        ) for p in points]
    positions = _packed_plane_positions(points, cells.astype(np.int64), index,
                                        correspondence_gate_m)
    ordered = [plane_map[key] for key in sorted(plane_map)]
    return [ordered[position] if position >= 0 else None for position in positions]


def read_livox_binary_pcd(path: str | Path) -> list[LivoxPoint]:
    """Read x/y/z/intensity from a binary float32 PCD file."""

    return [record.point for record in read_livox_binary_pcd_records(path)]


def read_livox_binary_pcd_records(path: str | Path) -> list[LivoxPointRecord]:
    """Read x/y/z/intensity plus optional normal_x/y/z from a binary float32 PCD file."""

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
    point_offsets = _pcd_field_offsets(sizes, counts)
    point_step = sum(size * count for size, count in zip(sizes, counts, strict=True))
    available = (len(data) - header_end) // point_step
    point_count = min(point_count or available, available)
    normal_indexes = _normal_field_indexes(fields, sizes, types, counts)
    if (
        point_count >= _PCD_ARRAY_MIN_POINTS and point_step >= 16
        and (normal_indexes is None or all(
            0 <= point_offsets[index] <= point_step - 4 for index in normal_indexes
        ))
    ):
        return _array_pcd_records(data, header_end, point_step, point_count,
                                  point_offsets, normal_indexes)
    records: list[LivoxPointRecord] = []
    for index in range(point_count):
        base_offset = header_end + index * point_step
        x, y, z, intensity = struct.unpack_from(
            "<ffff",
            data,
            base_offset,
        )
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        normal = _read_point_normal(data, base_offset, point_offsets, normal_indexes)
        records.append(
            LivoxPointRecord(
                point=(float(x), float(y), float(z), float(intensity)),
                normal_xyz=normal,
            )
        )
    return records


def _array_pcd_records(
    data: bytes, header_end: int, point_step: int, point_count: int,
    point_offsets: list[int], normal_indexes: tuple[int, int, int] | None,
) -> list[LivoxPointRecord]:
    # The legacy reader takes the first contiguous 16 bytes even if COUNT
    # describes an unusual first-field layout. Keep that byte interpretation.
    with np.errstate(invalid="ignore"):
        coordinates = np.ndarray(
            (point_count, 4), dtype="<f4", buffer=data, offset=header_end,
            strides=(point_step, 4),
        ).astype(np.float64)
    selected = np.flatnonzero(np.all(np.isfinite(coordinates[:, :3]), axis=1))
    points = coordinates[selected].tolist()
    if normal_indexes is None:
        return [LivoxPointRecord(point=(p[0], p[1], p[2], p[3])) for p in points]
    with np.errstate(invalid="ignore"):
        normals = np.column_stack([
            np.ndarray(
                (point_count,), dtype="<f4", buffer=data,
                offset=header_end + point_offsets[index], strides=(point_step,),
            ) for index in normal_indexes
        ]).astype(np.float64)[selected]
    valid = np.all(np.isfinite(normals), axis=1)
    x, y, z = normals.T
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lengths = np.sqrt(x * x + y * y + z * z)
        valid &= lengths != 0.0
        normalized = (normals / lengths[:, None]).tolist()
    return [
        LivoxPointRecord(
            point=(p[0], p[1], p[2], p[3]),
            normal_xyz=(n[0], n[1], n[2]) if accepted else None,
        ) for p, n, accepted in zip(points, normalized, valid, strict=True)
    ]


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


def _pcd_field_offsets(sizes: list[int], counts: list[int]) -> list[int]:
    offsets: list[int] = []
    offset = 0
    for size, count in zip(sizes, counts, strict=True):
        offsets.append(offset)
        offset += size * count
    return offsets


def _normal_field_indexes(
    fields: list[str],
    sizes: list[int],
    types: list[str],
    counts: list[int],
) -> tuple[int, int, int] | None:
    try:
        indexes = (
            fields.index("normal_x"),
            fields.index("normal_y"),
            fields.index("normal_z"),
        )
    except ValueError:
        return None
    for index in indexes:
        if sizes[index] != 4 or types[index] != "F" or counts[index] != 1:
            return None
    return indexes


def _read_point_normal(
    data: bytes,
    base_offset: int,
    point_offsets: list[int],
    normal_indexes: tuple[int, int, int] | None,
) -> Vector3 | None:
    if normal_indexes is None:
        return None
    nx = struct.unpack_from("<f", data, base_offset + point_offsets[normal_indexes[0]])[0]
    ny = struct.unpack_from("<f", data, base_offset + point_offsets[normal_indexes[1]])[0]
    nz = struct.unpack_from("<f", data, base_offset + point_offsets[normal_indexes[2]])[0]
    if not (math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)):
        return None
    return _normalized3((float(nx), float(ny), float(nz)))


def _summarize_sample(
    path: Path, points: list[LivoxPoint], *,
    geometry_values: tuple[list[float], list[float], list[float]] | None = None,
) -> LivoxPCDSampleStats:
    accumulator = _PointAccumulator()
    accumulator.add_many(points)
    range_min_m, range_max_m, fov_azimuth_deg, fov_elevation_deg = _coverage_from_values(
        _geometry_values(points) if geometry_values is None else geometry_values,
    )
    return LivoxPCDSampleStats(
        path=str(path),
        point_count=accumulator.count,
        bounds_min_m=accumulator.bounds_min_m,
        bounds_max_m=accumulator.bounds_max_m,
        intensity_min=accumulator.intensity_min if accumulator.count else None,
        intensity_max=accumulator.intensity_max if accumulator.count else None,
        intensity_mean=accumulator.intensity_mean,
        range_min_m=range_min_m,
        range_max_m=range_max_m,
        fov_azimuth_deg=fov_azimuth_deg,
        fov_elevation_deg=fov_elevation_deg,
    )


def _geometry_coverage(
    points: list[LivoxPoint],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return range and angular coverage for a sampled solid-state cloud."""

    return _coverage_from_values(_geometry_values(points))


def _geometry_values(
    points: list[LivoxPoint],
) -> tuple[list[float], list[float], list[float]]:
    ranges: list[float] = []
    azimuths_deg: list[float] = []
    elevations_deg: list[float] = []
    for x, y, z, _intensity in points:
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        horizontal_range = math.hypot(x, y)
        ranges.append(math.sqrt(horizontal_range * horizontal_range + z * z))
        azimuths_deg.append(math.degrees(math.atan2(y, x)))
        elevations_deg.append(math.degrees(math.atan2(z, horizontal_range)))
    return ranges, azimuths_deg, elevations_deg


def _coverage_from_values(
    values: tuple[list[float], list[float], list[float]],
) -> tuple[float | None, float | None, float | None, float | None]:
    ranges, azimuths_deg, elevations_deg = values
    if not ranges:
        return None, None, None, None
    return (
        min(ranges),
        max(ranges),
        _circular_span_deg(azimuths_deg),
        max(elevations_deg) - min(elevations_deg),
    )


def _circular_span_deg(angles_deg: list[float]) -> float | None:
    """Return the shortest circular arc containing all azimuth samples."""

    if not angles_deg:
        return None
    if len(angles_deg) == 1:
        return 0.0
    ordered = sorted((angle % 360.0) for angle in angles_deg)
    largest_gap = max(
        right - left
        for left, right in pairwise(ordered)
    )
    largest_gap = max(largest_gap, ordered[0] + 360.0 - ordered[-1])
    return max(0.0, 360.0 - largest_gap)


def _summarize_voxel_pair(
    source_points: list[LivoxPoint],
    target_points: list[LivoxPoint] | NDArray[np.float64],
    *,
    voxel_size_m: float,
    source_voxels: dict[tuple[int, int, int], _VoxelCentroid] | None = None,
) -> _VoxelPairSummary:
    if source_voxels is None:
        source_voxels = _voxel_centroids(source_points, voxel_size_m)
    target_voxels = _voxel_centroids(target_points, voxel_size_m)
    shared_keys = set(source_voxels) & set(target_voxels)
    squared_distances: list[float] = []
    for key in shared_keys:
        sx, sy, sz = source_voxels[key].centroid
        tx, ty, tz = target_voxels[key].centroid
        squared_distances.append(
            (tx - sx) * (tx - sx) + (ty - sy) * (ty - sy) + (tz - sz) * (tz - sz)
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
    points: list[LivoxPoint] | NDArray[np.float64],
    voxel_size_m: float,
) -> dict[tuple[int, int, int], _VoxelCentroid]:
    if len(points) >= 256:
        batched = _array_voxel_centroids(points, voxel_size_m)
        if batched is not None:
            return batched
    if isinstance(points, np.ndarray):
        points = [(float(p[0]), float(p[1]), float(p[2]), 0.0) for p in points]
    voxels: dict[tuple[int, int, int], _VoxelCentroid] = {}
    for point in points:
        key = _voxel_key(point, voxel_size_m)
        voxel = voxels.get(key)
        if voxel is None:
            voxel = _VoxelCentroid()
            voxels[key] = voxel
        voxel.add(point)
    return voxels


def _array_voxel_centroids(
    points: list[LivoxPoint] | NDArray[np.float64], voxel_size_m: float,
) -> dict[tuple[int, int, int], _VoxelCentroid] | None:
    coordinates = np.asarray(points, dtype=np.float64)
    expected_width = 3 if isinstance(points, np.ndarray) else 4
    if coordinates.ndim != 2 or coordinates.shape[1] != expected_width:
        return None
    coordinates = coordinates[:, :3]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cells = np.floor(coordinates / voxel_size_m)
    if (
        not np.isfinite(cells).all()
        or np.any(cells < -(2.0**63) + 2048)
        or np.any(cells >= 2.0**63 - 2048)
    ):
        return None
    cells = cells.astype(np.int64)
    lower, upper = cells.min(axis=0), cells.max(axis=0)
    widths = tuple(int(upper[a]) - int(lower[a]) + 1 for a in range(3))
    if math.prod(widths) > 2**63 - 1:
        return None
    encoded = (
        ((cells[:, 0] - lower[0]) * widths[1] + cells[:, 1] - lower[1]) * widths[2]
        + cells[:, 2] - lower[2]
    )
    _, first, inverse, counts = np.unique(
        encoded, return_index=True, return_inverse=True, return_counts=True,
    )
    sums = np.zeros((len(first), 3), dtype=np.float64)
    # Unbuffered accumulation retains input order for each voxel and axis.
    # Pairwise reductions would change the scalar sums and downstream RMSE.
    with np.errstate(over="ignore", invalid="ignore"):
        np.add.at(sums, inverse, coordinates)
    result: dict[tuple[int, int, int], _VoxelCentroid] = {}
    for position in np.argsort(first):
        row = cells[first[position]]
        key = (int(row[0]), int(row[1]), int(row[2]))
        x, y, z = sums[position].tolist()
        result[key] = _VoxelCentroid(int(counts[position]), x, y, z)
    return result


def _voxel_plane_map(
    records: list[LivoxPointRecord],
    voxel_size_m: float,
) -> dict[tuple[int, int, int], _VoxelPlane]:
    planes: dict[tuple[int, int, int], _VoxelPlane] = {}
    for record in records:
        voxel = planes.setdefault(_voxel_key(record.point, voxel_size_m), _VoxelPlane())
        voxel.add(record)
    return {key: plane for key, plane in planes.items() if plane.normal is not None}


def _point_to_plane_residuals(
    query_points: list[Vector3] | NDArray[np.float64],
    *,
    plane_map: dict[tuple[int, int, int], _VoxelPlane],
    voxel_size_m: float,
    correspondence_gate_m: float,
    index: _PlaneIndex | None = None,
) -> tuple[list[float], int]:
    batched = _batched_plane_residuals(
        query_points, plane_map=plane_map, voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
        index=index,
    )
    if batched is not None:
        return batched
    if isinstance(query_points, np.ndarray):
        query_points = [(float(p[0]), float(p[1]), float(p[2])) for p in query_points]
    return _scalar_plane_residuals(
        query_points, plane_map=plane_map, voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
    )


def _prepare_plane_index(plane_map: VoxelPlaneMap) -> _PlaneIndex | None:
    if not plane_map:
        return None
    ordered = sorted(plane_map)
    try:
        keys = np.asarray(ordered, dtype=np.int64)
    except OverflowError:
        return None
    lower = keys.min(axis=0)
    upper = keys.max(axis=0)
    widths = tuple(int(upper[axis]) - int(lower[axis]) + 1 for axis in range(3))
    # A sparse integer encoding avoids structured-key comparisons. The extent
    # product is checked with Python integers before any int64 arithmetic; huge
    # extents retain structured lookup.
    linear_keys = (
        ((keys[:, 0] - lower[0]) * widths[1] + keys[:, 1] - lower[1]) * widths[2]
        + keys[:, 2] - lower[2]
        if math.prod(widths) <= 2**63 - 1 else None
    )
    dense_positions: NDArray[np.int64] | None = None
    if (
        linear_keys is not None
        and math.prod(widths) <= _DENSE_PLANE_INDEX_MAX_BYTES // keys.dtype.itemsize
    ):
        # The bounded direct index is only used for compact extents. Large and
        # sparse clouds keep the checked integer or structured search instead.
        dense_positions = np.full(math.prod(widths), -1, dtype=np.int64)
        dense_positions[linear_keys] = np.arange(len(keys), dtype=np.int64)
    centroids = np.asarray([plane_map[key].centroid for key in ordered], dtype=np.float64)
    normal_list = [plane_map[key].normal for key in ordered]
    normals = np.asarray(
        [normal if normal is not None else (0.0, 0.0, 0.0) for normal in normal_list],
        dtype=np.float64,
    )
    valid_normal = np.asarray([normal is not None for normal in normal_list])
    neighborhoods = _prepare_plane_neighborhoods(
        keys, _DENSE_PLANE_INDEX_MAX_BYTES - (
            dense_positions.nbytes if dense_positions is not None else 0
        ),
    )
    for array in (
        keys, lower, upper, linear_keys, dense_positions, centroids, normals, valid_normal,
    ):
        if array is not None:
            array.setflags(write=False)
    return _PlaneIndex(
        keys, lower, upper, (widths[0], widths[1], widths[2]), linear_keys,
        dense_positions, centroids, normals, valid_normal,
        neighborhoods,
    )


def _prepare_plane_neighborhoods(
    keys: NDArray[np.int64], budget: int,
) -> _PlaneNeighborhoods | None:
    lower, upper = keys.min(axis=0), keys.max(axis=0)
    if np.any(lower == -(2**63)) or np.any(upper == 2**63 - 1):
        return None
    lower, upper = lower - 1, upper + 1
    widths = tuple(int(upper[a]) - int(lower[a]) + 1 for a in range(3))
    cell_count = math.prod(widths)
    # The CSR rows contain only occupied neighbors, ordered exactly like the
    # scalar 27-offset scan. Total retained lookup storage shares the 8 MiB cap.
    if (cell_count + 1 + len(keys) * 27 + 6) * 8 > budget:
        return None
    offsets = np.asarray(list(_neighbor_keys((0, 0, 0))), dtype=np.int64)
    cells = keys[:, None, :] - offsets
    encoded = (
        ((cells[..., 0] - lower[0]) * widths[1] + cells[..., 1] - lower[1]) * widths[2]
        + cells[..., 2] - lower[2]
    ).reshape(-1)
    order = np.lexsort((np.tile(np.arange(27), len(keys)), encoded))
    positions = np.repeat(np.arange(len(keys), dtype=np.int64), 27)[order]
    counts = np.bincount(encoded, minlength=cell_count)
    pointers = np.empty(cell_count + 1, dtype=np.int64)
    pointers[0] = 0
    np.cumsum(counts, out=pointers[1:])
    for array in (lower, upper, pointers, positions):
        array.setflags(write=False)
    return _PlaneNeighborhoods(
        lower, upper, (widths[0], widths[1], widths[2]), pointers, positions,
    )


def _batched_plane_residuals(
    query_points: list[Vector3] | NDArray[np.float64],
    *,
    plane_map: VoxelPlaneMap,
    voxel_size_m: float,
    correspondence_gate_m: float,
    index: _PlaneIndex | None = None,
) -> tuple[list[float], int] | None:
    # Small inputs avoid array setup; unusual coordinates retain scalar floor,
    # arbitrary-precision voxel keys and the original exception behavior.
    if len(query_points) < 64:
        return None
    coordinates = np.asarray(query_points, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cells = np.floor(coordinates / voxel_size_m)
    if (
        not np.isfinite(cells).all()
        or np.any(cells < -(2.0**63) + 2048)
        or np.any(cells >= 2.0**63 - 2048)
    ):
        return None
    if not plane_map:
        return [], len(query_points)
    if index is None:
        index = _prepare_plane_index(plane_map)
    if index is None:
        return None
    keys, lower, upper, widths = index.keys, index.lower, index.upper, index.widths
    linear_keys = index.linear_keys
    key_type = np.dtype([("x", "i8"), ("y", "i8"), ("z", "i8")])
    searchable = keys.view(key_type).reshape(-1)
    centroids, normals, valid_normal = index.centroids, index.normals, index.valid_normal
    cells = cells.astype(np.int64)
    if index.neighborhoods is not None:
        return _packed_plane_residuals(coordinates, cells, index, correspondence_gate_m)
    # Order matches _neighbor_keys: equal distances retain the first voxel.
    offsets = np.asarray(list(_neighbor_keys((0, 0, 0))), dtype=np.int64)
    residuals: list[float] = []
    # At most 4096 * 27 candidate associations are materialized per chunk.
    for start in range(0, len(coordinates), 4096):
        chunk = coordinates[start : start + 4096]
        neighboring = cells[start : start + 4096, None, :] + offsets
        if linear_keys is None:
            lookup = np.searchsorted(searchable, neighboring.view(key_type).reshape(-1))
        else:
            clipped = np.clip(neighboring.reshape(-1, 3), lower, upper)
            encoded = (
                ((clipped[:, 0] - lower[0]) * widths[1] + clipped[:, 1] - lower[1])
                * widths[2] + clipped[:, 2] - lower[2]
            )
            lookup = (
                np.searchsorted(linear_keys, encoded) if index.dense_positions is None
                else index.dense_positions[encoded]
            )
        bounded = np.clip(lookup, 0, len(keys) - 1)
        present = (lookup >= 0) & (lookup < len(keys)) & np.all(
            keys[bounded] == neighboring.reshape(-1, 3), axis=1,
        )
        positions = bounded.reshape((-1, 27))
        with np.errstate(over="ignore", invalid="ignore"):
            compact = np.count_nonzero(present) * 2 < present.size
            if compact:
                associations = np.flatnonzero(present)
                delta = chunk[associations // 27] - centroids[bounded[present]]
            else:
                delta = chunk[:, None, :] - centroids[positions]
            # Keep the scalar multiply/add/sqrt sequence, including tie rounding.
            measured = np.sqrt(
                delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]
                + delta[..., 2] * delta[..., 2]
            )
        if compact:
            distance = np.full(present.size, math.inf, dtype=np.float64)
            distance[associations] = measured
            distance = distance.reshape((-1, 27))
        else:
            distance = measured
        distance[~present.reshape((-1, 27)) | ~np.isfinite(distance)] = math.inf
        selected = distance.argmin(axis=1)
        row = np.arange(len(chunk))
        selected_index = positions[row, selected]
        best = distance[row, selected]
        # Scalar comparison permits a NaN gate, and rejects an infinite best
        # distance even with an infinite gate. A nearest normal-less voxel is
        # rejected rather than replaced by a farther voxel with a normal.
        accepted = (
            (best < math.inf) & ~(best > correspondence_gate_m) & valid_normal[selected_index]
        )
        differences = chunk[accepted] - centroids[selected_index[accepted]]
        matched_normals = normals[selected_index[accepted]]
        with np.errstate(over="ignore", invalid="ignore"):
            residual = np.abs(
                matched_normals[:, 0] * differences[:, 0]
                + matched_normals[:, 1] * differences[:, 1]
                + matched_normals[:, 2] * differences[:, 2]
            )
        residuals.extend(residual.tolist())
    return residuals, len(query_points) - len(residuals)


def _packed_plane_positions(
    coordinates: NDArray[np.float64], cells: NDArray[np.int64],
    index: _PlaneIndex, gate: float,
) -> NDArray[np.int64]:
    neighborhoods = index.neighborhoods
    assert neighborhoods is not None
    inside = np.all((cells >= neighborhoods.lower) & (cells <= neighborhoods.upper), axis=1)
    rows = np.flatnonzero(inside)
    matches = np.full(len(coordinates), -1, dtype=np.int64)
    widths, lower = neighborhoods.widths, neighborhoods.lower
    for start in range(0, len(rows), 4096):
        row = rows[start : start + 4096]
        chunk_cells = cells[row]
        encoded = (
            ((chunk_cells[:, 0] - lower[0]) * widths[1] + chunk_cells[:, 1] - lower[1])
            * widths[2] + chunk_cells[:, 2] - lower[2]
        )
        begin = neighborhoods.offsets[encoded]
        counts = neighborhoods.offsets[encoded + 1] - begin
        width = int(counts.max())
        if not width:
            continue
        columns = np.arange(width)
        present = columns[None, :] < counts[:, None]
        positions = neighborhoods.positions[
            np.minimum(begin[:, None] + columns, len(neighborhoods.positions) - 1)
        ]
        with np.errstate(over="ignore", invalid="ignore"):
            delta = coordinates[row, None, :] - index.centroids[positions]
            distances = np.sqrt(
                delta[..., 0] * delta[..., 0] + delta[..., 1] * delta[..., 1]
                + delta[..., 2] * delta[..., 2]
            )
        distances[~present | ~np.isfinite(distances)] = math.inf
        selected = distances.argmin(axis=1)
        local_rows = np.arange(len(row))
        position = positions[local_rows, selected]
        best = distances[local_rows, selected]
        accepted = (best < math.inf) & ~(best > gate) & index.valid_normal[position]
        matches[row[accepted]] = position[accepted]
    return matches


def _packed_plane_residuals(
    coordinates: NDArray[np.float64], cells: NDArray[np.int64],
    index: _PlaneIndex, gate: float,
) -> tuple[list[float], int]:
    matches = _packed_plane_positions(coordinates, cells, index, gate)
    rows = np.flatnonzero(matches >= 0)
    residuals: list[float] = []
    for start in range(0, len(rows), 4096):
        row = rows[start:start + 4096]
        position = matches[row]
        difference = coordinates[row] - index.centroids[position]
        normal = index.normals[position]
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.abs(
                normal[:, 0] * difference[:, 0] + normal[:, 1] * difference[:, 1]
                + normal[:, 2] * difference[:, 2]
            )
        residuals.extend(values.tolist())
    return residuals, len(coordinates) - len(residuals)


def _scalar_plane_residuals(
    query_points: list[Vector3],
    *,
    plane_map: VoxelPlaneMap,
    voxel_size_m: float,
    correspondence_gate_m: float,
) -> tuple[list[float], int]:
    residuals: list[float] = []
    unmatched_count = 0
    for point in query_points:
        plane = _nearest_plane(
            point,
            plane_map=plane_map,
            voxel_size_m=voxel_size_m,
            correspondence_gate_m=correspondence_gate_m,
        )
        if plane is None or plane.normal is None:
            unmatched_count += 1
            continue
        residuals.append(abs(_dot3(plane.normal, _sub3(point, plane.centroid))))
    return residuals, unmatched_count


def _nearest_plane(
    point: Vector3,
    *,
    plane_map: dict[tuple[int, int, int], _VoxelPlane],
    voxel_size_m: float,
    correspondence_gate_m: float,
) -> _VoxelPlane | None:
    match = _nearest_plane_match(
        point,
        plane_map=plane_map,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
    )
    return plane_map[match.voxel_key] if match is not None else None


def _nearest_plane_match(
    point: Vector3,
    *,
    plane_map: dict[tuple[int, int, int], _VoxelPlane],
    voxel_size_m: float,
    correspondence_gate_m: float,
) -> VoxelPlaneMatch | None:
    key = _voxel_key((point[0], point[1], point[2], 0.0), voxel_size_m)
    best_plane: _VoxelPlane | None = None
    best_key: tuple[int, int, int] | None = None
    best_distance = float("inf")
    for neighbor_key in _neighbor_keys(key):
        plane = plane_map.get(neighbor_key)
        if plane is None:
            continue
        distance = _norm3(_sub3(point, plane.centroid))
        if distance < best_distance:
            best_distance = distance
            best_plane = plane
            best_key = neighbor_key
    if (
        best_plane is None
        or best_key is None
        or best_plane.normal is None
        or best_distance > correspondence_gate_m
    ):
        return None
    return VoxelPlaneMatch(
        best_key,
        best_plane.centroid,
        best_plane.normal,
        best_distance,
    )


def _neighbor_keys(key: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    x, y, z = key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (x + dx, y + dy, z + dz)


def _transform_points(points: list[LivoxPoint], transform: SE3) -> list[LivoxPoint]:
    return [
        (xyz[0], xyz[1], xyz[2], point[3])
        for xyz, point in zip(_transform_query_points(points, transform), points, strict=True)
    ]


def _transform_query_points(points: list[LivoxPoint], transform: SE3 | None) -> list[Vector3]:
    if transform is None or len(points) < 64:
        return [_transformed_point(point, transform) for point in points]
    xyz = _transform_coordinates(np.asarray(points, dtype=np.float64)[:, :3], transform)
    return list(zip(xyz[:, 0].tolist(), xyz[:, 1].tolist(), xyz[:, 2].tolist(), strict=True))


def _transform_coordinates(
    xyz: NDArray[np.float64], transform: SE3 | None,
) -> NDArray[np.float64]:
    if transform is None:
        return xyz
    # Match rotate_vector_xyzw's two Hamilton products and normalizations.
    # A rotation matrix or dot reduction could change floating-point rounding.
    qx, qy, qz, qw = normalize_quaternion_xyzw(transform.rotation_quat_xyzw)
    cx, cy, cz, cw = quaternion_conjugate_xyzw((qx, qy, qz, qw))
    px, py, pz = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    with np.errstate(over="ignore", invalid="ignore"):
        ax = qw * px + qx * 0.0 + qy * pz - qz * py
        ay = qw * py - qx * pz + qy * 0.0 + qz * px
        az = qw * pz + qx * py - qy * px + qz * 0.0
        aw = qw * 0.0 - qx * px - qy * py - qz * pz
        tx = aw * cx + ax * cw + ay * cz - az * cy + transform.translation_m[0]
        ty = aw * cy - ax * cz + ay * cw + az * cx + transform.translation_m[1]
        tz = aw * cz + ax * cy - ay * cx + az * cw + transform.translation_m[2]
    return np.column_stack((tx, ty, tz))


def _transformed_point(point: LivoxPoint, transform: SE3 | None) -> Vector3:
    if transform is None:
        return (point[0], point[1], point[2])
    return transform.transform_point((point[0], point[1], point[2]))


def _voxel_key(point: LivoxPoint, voxel_size_m: float) -> tuple[int, int, int]:
    return (
        math.floor(point[0] / voxel_size_m),
        math.floor(point[1] / voxel_size_m),
        math.floor(point[2] / voxel_size_m),
    )


def _frame_id(path: Path) -> str:
    return path.stem


def _normalized3(vector: Vector3) -> Vector3 | None:
    norm = _norm3(vector)
    if norm <= 1.0e-12:
        return None
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _fallback_plane_normal(points: list[Vector3]) -> Vector3 | None:
    if len(points) < 3:
        return None
    origin = points[0]
    first = max(points[1:], key=lambda point: _norm3(_sub3(point, origin)))
    second = max(
        points[1:],
        key=lambda point: _norm3(_cross3(_sub3(first, origin), _sub3(point, origin))),
    )
    return _normalized3(_cross3(_sub3(first, origin), _sub3(second, origin)))


def _sub3(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _dot3(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross3(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm3(vector: Vector3) -> float:
    return math.sqrt(_dot3(vector, vector))


def _rmse(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return ordered[index]
