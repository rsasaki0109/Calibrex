"""Range-adaptive voxel planes for non-repetitive solid-state LiDAR."""

from __future__ import annotations

import math
import sys
from bisect import bisect_left, bisect_right
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock

import numpy as np
from numpy.typing import NDArray

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
class _PlaneRow:
    coordinates: tuple[int, ...]
    cells: tuple[AdaptiveVoxelPlane, ...]


@dataclass(frozen=True)
class _PlaneSlice:
    coordinates: tuple[int, ...]
    rows: tuple[_PlaneRow, ...]


class _NeighborhoodCache:
    """Bound estimated tuple/key/control storage across a map's levels."""

    def __init__(self, budget: int = 2 * 1024 * 1024) -> None:
        self.budget = budget
        self.retained_bytes = 0
        self._lock = Lock()
        self._entries: OrderedDict[
            tuple[int, tuple[int, int, int], int], tuple[tuple[AdaptiveVoxelPlane, ...], int]
        ] = OrderedDict()

    def get(
        self, key: tuple[int, tuple[int, int, int], int],
    ) -> tuple[AdaptiveVoxelPlane, ...] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(
        self, key: tuple[int, tuple[int, int, int], int], cells: tuple[AdaptiveVoxelPlane, ...],
    ) -> None:
        # Conservative per-entry control/tuple/typical-int allowance, plus
        # actual tuple slots and any unusually large integer storage.
        size = 512 + len(cells) * 8 + sum(
            max(0, sys.getsizeof(value) - 28) for value in (key[0], *key[1], key[2])
        )
        if size > self.budget:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self.retained_bytes -= previous[1]
            while self.retained_bytes + size > self.budget:
                _old_key, (_old_cells, old_size) = self._entries.popitem(last=False)
                self.retained_bytes -= old_size
            self._entries[key] = (cells, size)
            self.retained_bytes += size

    def __getstate__(self) -> tuple[int]:
        return (self.budget,)

    def __setstate__(self, state: tuple[int]) -> None:
        _NeighborhoodCache.__init__(self, state[0])


class _SparsePlaneCells(Mapping[tuple[int, int, int], AdaptiveVoxelPlane]):
    """Owned cell mapping with an ordered occupied-only box index."""

    def __init__(
        self, cells: Mapping[tuple[int, int, int], AdaptiveVoxelPlane], *,
        cache: _NeighborhoodCache | None = None, level: int = 0,
    ) -> None:
        self._cells = dict(cells)
        self._cache = cache if cache is not None else _NeighborhoodCache()
        self._level = level
        grouped: dict[int, dict[int, list[tuple[int, AdaptiveVoxelPlane]]]] = {}
        for (x, y, z), cell in sorted(cells.items()):
            grouped.setdefault(x, {}).setdefault(y, []).append((z, cell))
        self._coordinates = tuple(grouped)
        self._slices = tuple(
            _PlaneSlice(
                coordinates=tuple(rows),
                rows=tuple(
                    _PlaneRow(
                        coordinates=tuple(z for z, _cell in row),
                        cells=tuple(cell for _z, cell in row),
                    )
                    for row in rows.values()
                ),
            )
            for rows in grouped.values()
        )

    def __getitem__(self, key: tuple[int, int, int]) -> AdaptiveVoxelPlane:
        return self._cells[key]

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        return iter(self._cells)

    def __len__(self) -> int:
        return len(self._cells)

    def nearby(
        self, key: tuple[int, int, int], radius: int,
    ) -> Iterator[AdaptiveVoxelPlane]:
        """Iterate exactly the old box's occupied cells in x/y/z order."""

        cache_key = (self._level, key, radius)
        cached = self._cache.get(cache_key)
        if cached is None:
            cached = tuple(self._nearby_uncached(key, radius))
            self._cache.put(cache_key, cached)
        return iter(cached)

    def _nearby_uncached(
        self, key: tuple[int, int, int], radius: int,
    ) -> Iterator[AdaptiveVoxelPlane]:
        for x in range(
            bisect_left(self._coordinates, key[0] - radius),
            bisect_right(self._coordinates, key[0] + radius),
        ):
            plane_slice = self._slices[x]
            for y in range(
                bisect_left(plane_slice.coordinates, key[1] - radius),
                bisect_right(plane_slice.coordinates, key[1] + radius),
            ):
                row = plane_slice.rows[y]
                for z in range(
                    bisect_left(row.coordinates, key[2] - radius),
                    bisect_right(row.coordinates, key[2] + radius),
                ):
                    yield row.cells[z]


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
            if isinstance(cells, _SparsePlaneCells):
                for candidate in cells.nearby(key, radius):
                    distance = math.dist(point, candidate.centroid)
                    if distance < best_distance:
                        best = candidate
                        best_distance = distance
                continue
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


class AdaptivePlaneQueryContext:
    """Capture ordered bounded-cell geometry for exact gated batch queries."""

    def __init__(self, plane_map: AdaptiveVoxelPlaneMap) -> None:
        from scipy.spatial import cKDTree

        self._map = plane_map
        ordered: list[AdaptiveVoxelPlane] = []
        supported = type(plane_map) is AdaptiveVoxelPlaneMap
        for level, cells in plane_map._index.items():
            if not isinstance(cells, _SparsePlaneCells):
                supported = False
                break
            size = _level_voxel_size(level, plane_map.cells)
            for key, cell in sorted(cells.items()):
                if (not all(math.isfinite(value) and abs(value) < 1e100 for value in cell.centroid)
                    or cell.voxel_size_m != size
                    or _voxel_key(cell.centroid, plane_map.origin_m, size) != key):
                    supported = False
                    break
                ordered.append(cell)
        self._planes = tuple(ordered)
        self._native_geometry = supported and all(
            type(vector) is tuple and len(vector) == 3
            and all(type(value) is float and math.isfinite(value) and abs(value) < 1e100
                    for value in vector)
            for plane in ordered for vector in (plane.centroid, plane.normal)
        )
        values = np.asarray([cell.centroid for cell in ordered], dtype=np.float64).reshape((-1, 3))
        values.flags.writeable = False
        self._centroids = values
        self._tree = cKDTree(values) if supported and ordered else None

    def nearest_planes(
        self, points: NDArray[np.float64], *, correspondence_gate_m: float,
    ) -> list[AdaptiveVoxelPlane | None]:
        """Preserve scalar gates, exact math.dist distances and first ordered ties."""
        gate = correspondence_gate_m
        tree = self._tree
        if (len(points) < 64 or tree is None or not math.isfinite(gate) or gate <= 0.0
            or not np.isfinite(points).all() or np.any(np.abs(points) >= 1e100)):
            return [self._map.nearest_plane(tuple(point), correspondence_gate_m=gate)
                    for point in points.tolist()]
        # Centroids lie in their declared cells. Every centroid within the gate
        # is therefore inside the scalar ceil(gate/size)+1 search box.
        distances, indexes = tree.query(points, k=2, eps=0.0, workers=1)
        result: list[AdaptiveVoxelPlane | None] = []
        for row, point_values in enumerate(points):
            point = (float(point_values[0]), float(point_values[1]), float(point_values[2]))
            distance = float(distances[row, 0])
            if distance < 1e-150 or not math.isfinite(distance):
                result.append(self._map.nearest_plane(point, correspondence_gate_m=gate))
                continue
            position = int(indexes[row, 0])
            # 3D positive sum/sqrt roundoff is bounded away from overflow and
            # underflow. Repair close contenders with the original scalar norm.
            radius = math.nextafter(distance * (1.0 + 32.0 * np.finfo(np.float64).eps), math.inf)
            if float(distances[row, 1]) <= radius:
                candidates = sorted(tree.query_ball_point(point, radius, eps=0.0, workers=1))
                position = min(candidates, key=lambda index: (
                    math.dist(point, self._planes[index].centroid), index,
                ))
            plane = self._planes[position]
            result.append(plane if math.dist(point, plane.centroid) <= gate else None)
        return result


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
    neighborhood_cache = _NeighborhoodCache()
    return AdaptiveVoxelPlaneMap(
        cells=tuple(cells),
        origin_m=settings.range_origin_m,
        _index={
            level: _SparsePlaneCells(cells_by_key, cache=neighborhood_cache, level=level)
            for level, cells_by_key in index.items()
        },
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
