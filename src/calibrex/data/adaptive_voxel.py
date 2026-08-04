"""Range-adaptive voxel planes for non-repetitive solid-state LiDAR."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from calibrex.core.geometry import Vector3
from calibrex.data.livox import LivoxPointRecord


@dataclass(frozen=True)
class AdaptiveVoxelPolicy:
    """Parameters for range-aware voxel sizing."""

    base_voxel_size_m: float = 0.5
    range_reference_m: float = 10.0
    range_exponent: float = 0.5
    min_voxel_size_m: float = 0.2
    max_voxel_size_m: float = 1.5
    min_points_per_voxel: int = 3
    range_origin_m: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class AdaptiveVoxelPlane:
    """One plane cell produced by the adaptive voxel map."""

    level: int
    key: tuple[int, int, int]
    centroid: Vector3
    normal: Vector3
    support_count: int
    voxel_size_m: float
    range_m: float


@dataclass(frozen=True)
class AdaptiveVoxelPlaneMap:
    """Immutable adaptive plane map with deterministic nearest-plane queries."""

    cells: tuple[AdaptiveVoxelPlane, ...]
    origin_m: Vector3
    _index: Mapping[int, Mapping[tuple[int, int, int], AdaptiveVoxelPlane]]

    def nearest_plane(
        self,
        point: Vector3,
        *,
        correspondence_gate_m: float,
    ) -> AdaptiveVoxelPlane | None:
        """Return the closest adaptive plane within the declared gate."""

        if correspondence_gate_m <= 0.0:
            raise ValueError("correspondence_gate_m must be positive")
        best: AdaptiveVoxelPlane | None = None
        best_distance = float("inf")
        for level, cells in self._index.items():
            voxel_size = _level_voxel_size(level, self.cells)
            key = _voxel_key(point, self.origin_m, voxel_size)
            radius = max(1, math.ceil(correspondence_gate_m / voxel_size) + 1)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        cell = cells.get((key[0] + dx, key[1] + dy, key[2] + dz))
                        if cell is None:
                            continue
                        distance = math.dist(point, cell.centroid)
                        if distance < best_distance:
                            best = cell
                            best_distance = distance
        return best if best is not None and best_distance <= correspondence_gate_m else None


@dataclass
class _Accumulator:
    points: list[Vector3]
    normals: list[Vector3]


def adaptive_voxel_size_m(range_m: float, policy: AdaptiveVoxelPolicy) -> float:
    """Return a voxel size that follows approximately constant angular support."""

    _validate_policy(policy)
    if range_m < 0.0:
        raise ValueError("range_m must be non-negative")
    scaled = float(policy.base_voxel_size_m * (
        max(range_m, 1.0e-9) / policy.range_reference_m
    ) ** policy.range_exponent)
    return float(min(policy.max_voxel_size_m, max(policy.min_voxel_size_m, scaled)))


def build_adaptive_voxel_plane_map(
    records: Sequence[LivoxPointRecord],
    policy: AdaptiveVoxelPolicy | None = None,
) -> AdaptiveVoxelPlaneMap:
    """Build a range-adaptive plane map with deterministic support filtering."""

    settings = policy or AdaptiveVoxelPolicy()
    _validate_policy(settings)
    grouped: dict[tuple[int, tuple[int, int, int]], _Accumulator] = defaultdict(
        lambda: _Accumulator(points=[], normals=[])
    )
    for record in records:
        point = record.point[:3]
        range_m = math.dist(point, settings.range_origin_m)
        voxel_size = adaptive_voxel_size_m(range_m, settings)
        level = _voxel_level(voxel_size, settings)
        actual_size = _level_size(level, settings)
        key = _voxel_key(point, settings.range_origin_m, actual_size)
        accumulator = grouped.setdefault((level, key), _Accumulator([], []))
        accumulator.points.append(point)
        if record.normal_xyz is not None:
            accumulator.normals.append(record.normal_xyz)

    cells: list[AdaptiveVoxelPlane] = []
    index: dict[int, dict[tuple[int, int, int], AdaptiveVoxelPlane]] = defaultdict(dict)
    for (level, key), accumulator in sorted(grouped.items()):
        if len(accumulator.points) < settings.min_points_per_voxel:
            continue
        normal = _plane_normal(accumulator.points, accumulator.normals)
        if normal is None:
            continue
        centroid = _centroid(accumulator.points)
        cell = AdaptiveVoxelPlane(
            level=level,
            key=key,
            centroid=centroid,
            normal=normal,
            support_count=len(accumulator.points),
            voxel_size_m=_level_size(level, settings),
            range_m=math.dist(centroid, settings.range_origin_m),
        )
        cells.append(cell)
        index[level][key] = cell
    return AdaptiveVoxelPlaneMap(
        cells=tuple(cells),
        origin_m=settings.range_origin_m,
        _index={level: dict(cells_by_key) for level, cells_by_key in index.items()},
    )


def _validate_policy(policy: AdaptiveVoxelPolicy) -> None:
    if policy.base_voxel_size_m <= 0.0:
        raise ValueError("base_voxel_size_m must be positive")
    if policy.range_reference_m <= 0.0:
        raise ValueError("range_reference_m must be positive")
    if policy.min_voxel_size_m <= 0.0 or policy.max_voxel_size_m < policy.min_voxel_size_m:
        raise ValueError("adaptive voxel size bounds are invalid")
    if policy.min_points_per_voxel < 1:
        raise ValueError("min_points_per_voxel must be positive")
    if policy.range_exponent < 0.0:
        raise ValueError("range_exponent must be non-negative")


def _voxel_level(voxel_size_m: float, policy: AdaptiveVoxelPolicy) -> int:
    return round(math.log(voxel_size_m / policy.base_voxel_size_m, 2.0))


def _level_size(level: int, policy: AdaptiveVoxelPolicy) -> float:
    return min(
        policy.max_voxel_size_m,
        max(policy.min_voxel_size_m, policy.base_voxel_size_m * (2.0**level)),
    )


def _level_voxel_size(
    level: int,
    cells: Sequence[AdaptiveVoxelPlane],
) -> float:
    for cell in cells:
        if cell.level == level:
            return cell.voxel_size_m
    raise ValueError(f"adaptive voxel level has no cells: {level}")


def _voxel_key(
    point: Vector3,
    origin: Vector3,
    voxel_size_m: float,
) -> tuple[int, int, int]:
    return tuple(
        math.floor((point[index] - origin[index]) / voxel_size_m)
        for index in range(3)
    )  # type: ignore[return-value]


def _centroid(points: Sequence[Vector3]) -> Vector3:
    count = float(len(points))
    return tuple(sum(point[index] for point in points) / count for index in range(3))  # type: ignore[return-value]


def _plane_normal(points: Sequence[Vector3], normals: Sequence[Vector3]) -> Vector3 | None:
    if normals:
        reference = normals[0]
        aligned = tuple(
            normal if _dot(normal, reference) >= 0.0 else _scale(normal, -1.0)
            for normal in normals
        )
        normal = _normalize(
            tuple(sum(value[index] for value in aligned) for index in range(3))
        )
        if normal is not None:
            return normal
    return _fallback_plane_normal(points)


def _fallback_plane_normal(points: Sequence[Vector3]) -> Vector3 | None:
    """Estimate a deterministic geometric normal from the widest local triangle.

    Solid-state public recordings often omit per-point normals.  Choosing the
    first non-collinear triple makes the result depend on packet ordering and
    is especially unstable for fine range-adaptive cells.  A bounded, widest
    triangle fallback matches the robust geometric policy used by the uniform
    voxel map while keeping map construction deterministic.
    """

    sampled = tuple(points[: min(len(points), 12)])
    if len(sampled) < 3:
        return None
    origin = sampled[0]
    first = max(sampled[1:], key=lambda point: _norm(_subtract(point, origin)))
    first_vector = _subtract(first, origin)
    second = max(
        sampled[1:],
        key=lambda point: _norm(_cross(first_vector, _subtract(point, origin))),
    )
    return _normalize(_cross(first_vector, _subtract(second, origin)))


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(values: Sequence[float]) -> Vector3 | None:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1.0e-12:
        return None
    return (values[0] / norm, values[1] / norm, values[2] / norm)
