"""Livox solid-state LiDAR PCD dataset helpers."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from slac.core.geometry import SE3, Vector3
from slac.data.base import StreamSummary, TimestampedRecord

LivoxPoint = tuple[float, float, float, float]


@dataclass(frozen=True)
class LivoxPointRecord:
    """Livox point with optional per-point normal fields from the PCD record."""

    point: LivoxPoint
    normal_xyz: Vector3 | None = None


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
            return _normalized3(
                (self.normal_x_sum, self.normal_y_sum, self.normal_z_sum)
            )
        return _fallback_plane_normal(self.fallback_points or [])


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


def summarize_livox_pair_point_to_plane(
    path: str | Path,
    *,
    target_transform: SE3 | None = None,
    voxel_size_m: float = 1.0,
    correspondence_gate_m: float = 1.5,
    inlier_threshold_m: float = 0.25,
) -> LivoxPairPointToPlaneStats:
    """Score transformed target PCD points against source-frame voxel planes.

    The public Livox Horizon-Horizon sample contains one source and one target
    PCD frame. This is therefore a single-pair holdout geometry check rather
    than an independent multi-window temporal validation.
    """

    files = find_livox_pcd_files(path)
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
        source_records = read_livox_binary_pcd_records(files[0])
        target_records = read_livox_binary_pcd_records(files[1])
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

    plane_map = _voxel_plane_map(source_records, voxel_size_m)
    train_frame_ids = (_frame_id(files[0]),)
    holdout_frame_ids = (_frame_id(files[1]),)
    query_points = [
        _transformed_point(record.point, target_transform) for record in target_records
    ]
    eligible_count = len(target_records)
    considered_count = len(query_points)
    residuals, unmatched_count = _point_to_plane_residuals(
        query_points,
        plane_map=plane_map,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
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
) -> dict[tuple[int, int, int], _VoxelPlane]:
    """Return the voxel-plane map used for point-to-plane correspondences.

    Each voxel aggregates a centroid and a plane normal (from per-point PCD
    normal fields when present, otherwise a local geometric fallback). Voxels
    without a resolvable normal are dropped.
    """

    return _voxel_plane_map(records, voxel_size_m)


def nearest_voxel_plane(
    point: Vector3,
    plane_map: dict[tuple[int, int, int], _VoxelPlane],
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
    query_points: list[Vector3],
    *,
    plane_map: dict[tuple[int, int, int], _VoxelPlane],
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
    key = _voxel_key((point[0], point[1], point[2], 0.0), voxel_size_m)
    best_plane: _VoxelPlane | None = None
    best_distance = float("inf")
    for neighbor_key in _neighbor_keys(key):
        plane = plane_map.get(neighbor_key)
        if plane is None:
            continue
        distance = _norm3(_sub3(point, plane.centroid))
        if distance < best_distance:
            best_distance = distance
            best_plane = plane
    if best_plane is None or best_distance > correspondence_gate_m:
        return None
    return best_plane


def _neighbor_keys(key: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    x, y, z = key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (x + dx, y + dy, z + dz)


def _transform_points(points: list[LivoxPoint], transform: SE3) -> list[LivoxPoint]:
    transformed: list[LivoxPoint] = []
    for x, y, z, intensity in points:
        tx, ty, tz = transform.transform_point((x, y, z))
        transformed.append((tx, ty, tz, intensity))
    return transformed


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
        key=lambda point: _norm3(
            _cross3(_sub3(first, origin), _sub3(point, origin))
        ),
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
