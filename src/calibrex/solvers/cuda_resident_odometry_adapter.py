"""Optional resident geometry for repeated LiDAR-IMU bag odometry passes.

One session owns a CUDA stream and a bounded content cache of decoded scans.
Crop, deterministic voxel centroids, deskew, local maps and spatial queries stay
on device. Small motion solves and ambiguous geometry decisions retain the
native registration policy. Import this module only for an explicit CUDA run.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import cupy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from calibrex.solvers import cuda_scan_odometry_adapter as cuda
from calibrex.solvers import scan_to_scan_odometry as native
from calibrex.solvers._cuda_resident_kernels import (
    _FUSED_GRID_SOURCE,
    _GEOMETRY_SOURCE,
    _GRID_SOURCE,
    _GYRO_SOURCE,
    _VOXEL_SOURCE,
)
from calibrex.solvers.cuda_lidar_map_adapter import _coordinates
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries, _GyroRotationModel
from calibrex.solvers.scan_to_scan_odometry import (
    FloatArray,
    OdometryBackend,
    RotationModel,
    ScanOdometryOptions,
    ScanOdometryResult,
    ScanRegistration,
    _IncrementalOdometry,
)

_CACHE_BYTES = 512 * 1024**2
_WORKSPACE_BYTES = 1024**3
_BLOCK = (128,)


def _blocks(count: int) -> tuple[int]:
    return ((count + 127) // 128,)


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def _identity() -> str:
    fields = cuda.execution_identity().split(";")
    fields = [
        field for field in fields if not field.startswith(("query_sha256=", "system_sha256="))
    ]
    return ";".join(
        [
            *fields,
            f"query_sha256={_digest(_GRID_SOURCE)}",
            "system_sha256="
            + _digest(
                cuda._SYSTEM_SOURCE
                + _FUSED_GRID_SOURCE
                + cuda._COVARIANCE_SOURCE
                + cuda._NORMAL_SOURCE,
            ),
            "geometry_pipeline=resident-v1",
            f"geometry_sha256={_digest(_VOXEL_SOURCE + _GYRO_SOURCE + _GEOMETRY_SOURCE)}",
        ]
    )


def _validate_options(options: ScanOdometryOptions) -> None:
    cuda._validate_options(options)
    if type(options.local_map_scans) is not int or options.local_map_scans < 1:
        raise ValueError("resident local_map_scans must be a positive integer")
    if type(options.deskew_iterations) is not int or options.deskew_iterations < 0:
        raise ValueError("resident deskew_iterations must be a nonnegative integer")
    if not math.isfinite(options.voxel_size_m) or options.voxel_size_m <= 0:
        raise ValueError("resident voxel_size_m must be finite and positive")
    quanta = (
        options.geometry_quantum_m,
        options.motion_translation_quantum_m,
        options.motion_rotation_quantum_rad,
    )
    if any(not math.isfinite(value) or value <= 0 for value in quanta):
        raise ValueError("resident odometry requires three explicit positive odometry quanta")


@dataclass(frozen=True)
class _CapturedScan:
    points: cp.ndarray
    offsets: cp.ndarray | None
    finite_points: bool
    offset_bounds: tuple[float, float] | None

    def covers(self, start: float, first: float, last: float) -> bool:
        # Addition by a finite scalar is monotone in FP64. These extrema
        # describe the same queries as start + offsets on device, including
        # rounded timestamp boundaries and overflow to infinity.
        start = float(start)
        bounds = self.offset_bounds
        return (
            self.finite_points
            and bounds is not None
            and math.isfinite(start)
            and first <= start <= last
            and start + bounds[0] >= first
            and start + bounds[1] <= last
        )


class _Geometry:
    def __init__(self) -> None:
        module = cp.RawModule(code=_VOXEL_SOURCE, options=("--fmad=false",))
        self.voxel = {
            name: module.get_function(name)
            for name in ("prepare_crop", "insert_masked", "compact_cells", "centroids")
        }
        module = cp.RawModule(code=_GYRO_SOURCE + _GEOMETRY_SOURCE, options=("--fmad=false",))
        self.kernels = {
            name: module.get_function(name)
            for name in ("gyro_rotate3", "transform_points", "translate_points", "rigid_deskew")
        }
        self.scans: dict[
            tuple[tuple[int, ...], bytes, tuple[int, ...] | None],
            _CapturedScan,
        ] = {}
        self.bytes = 0
        self.preprocess_calls = 0
        self.preprocess_rows = 0
        self.voxel_fallbacks = 0
        self.transforms = 0
        self.gyro_rows = 0
        self.source_captures = 0
        self.source_hits = 0
        self.source_validation_rows = 0
        self.gyro_cached_validations = 0
        self.gyro_fallbacks = 0
        self.native_model_fallbacks = 0

    def capture(
        self,
        scan: FloatArray,
        offsets: FloatArray | None,
    ) -> _CapturedScan:
        values = np.asarray(scan, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError("resident scans require (N, 3+) coordinates")
        points = np.ascontiguousarray(values[:, :3])
        times = None if offsets is None else np.ascontiguousarray(offsets, dtype=np.float64)
        if times is not None and (times.shape != (len(points),) or not np.isfinite(times).all()):
            raise ValueError("resident point offsets require one finite value per point")
        size = points.nbytes + (0 if times is None else times.nbytes)
        if points.nbytes * 12 > _WORKSPACE_BYTES:
            raise ValueError("resident scan exceeds the 1 GiB preprocessing workspace budget")
        digest = hashlib.blake2b(digest_size=32)
        digest.update(points.tobytes())
        if times is not None:
            digest.update(times.tobytes())
        key = (points.shape, digest.digest(), None if times is None else times.shape)
        if key in self.scans:
            self.source_hits += 1
            return self.scans[key]
        captured = _CapturedScan(
            cp.asarray(points),
            None if times is None else cp.asarray(times),
            bool(np.isfinite(points).all()),
            None if times is None or not len(times) else (float(times.min()), float(times.max())),
        )
        self.source_validation_rows += len(points)
        if self.bytes + size <= _CACHE_BYTES:
            self.scans[key] = captured
            self.bytes += size
        self.source_captures += 1
        return captured

    def preprocess(self, points: cp.ndarray, options: ScanOdometryOptions) -> cp.ndarray:
        self.preprocess_calls += 1
        count = len(points)
        self.preprocess_rows += count
        if not count:
            return cp.empty((0, 3), dtype=cp.float64)
        if points.nbytes * 12 > _WORKSPACE_BYTES:
            raise ValueError("resident preprocessing exceeds the 1 GiB workspace budget")
        xyz = cp.empty_like(points)
        keys = cp.empty(count, dtype=cp.uint64)
        state = cp.zeros(2, dtype=cp.int32)
        self.voxel["prepare_crop"](
            _blocks(count),
            _BLOCK,
            (
                points,
                np.int32(count),
                np.float64(options.geometry_quantum_m),
                np.float64(options.voxel_size_m),
                np.float64(options.min_range_m),
                np.float64(options.max_range_m),
                xyz,
                keys,
                state,
            ),
        )
        size = 1 << (2 * count - 1).bit_length()
        table = cp.full(size, cp.uint64(2**64 - 1), dtype=cp.uint64)
        heads = cp.full(size, -1, dtype=cp.int32)
        next_rows = cp.empty(count, dtype=cp.int32)
        self.voxel["insert_masked"](
            _blocks(count),
            _BLOCK,
            (
                keys,
                np.int32(count),
                np.int32(size - 1),
                table,
                heads,
                next_rows,
            ),
        )
        occupied = cp.empty(size, dtype=cp.int64)
        compact_keys = cp.empty(size, dtype=cp.uint64)
        self.voxel["compact_cells"](
            _blocks(size),
            _BLOCK,
            (
                table,
                heads,
                np.int32(size),
                occupied,
                compact_keys,
                state,
            ),
        )
        invalid, groups = map(int, state.get())
        if invalid:
            return self._native_preprocess(points, options)
        if not groups:
            return cp.empty((0, 3), dtype=cp.float64)
        order = cp.argsort(compact_keys[:groups])
        result = cp.empty((groups, 3), dtype=cp.float64)
        self.voxel["centroids"](
            _blocks(groups),
            _BLOCK,
            (
                xyz,
                heads,
                next_rows,
                occupied,
                order,
                np.int32(groups),
                np.float64(options.geometry_quantum_m),
                result,
                state,
            ),
        )
        if int(state[:1].get()[0]):
            return self._native_preprocess(points, options)
        return result

    def _native_preprocess(self, points: cp.ndarray, options: ScanOdometryOptions) -> cp.ndarray:
        # Out-of-domain voxel keys and >32 points in one cell preserve the
        # native sum order through a complete, explicit fallback.
        self.voxel_fallbacks += 1
        return cp.asarray(native.preprocess_scan(cp.asnumpy(points), options))

    def transform(
        self,
        points: cp.ndarray,
        matrix: FloatArray,
        quantum: float = 0.0,
    ) -> cp.ndarray:
        result = cp.empty_like(points)
        self.transforms += 1
        if len(points):
            self.kernels["transform_points"](
                _blocks(len(points)),
                _BLOCK,
                (
                    points,
                    cp.asarray(np.ascontiguousarray(matrix)),
                    np.int32(len(points)),
                    np.float64(quantum),
                    result,
                ),
            )
        return result

    def translate(
        self,
        points: cp.ndarray,
        offsets: cp.ndarray,
        motion: FloatArray,
        gap: float,
    ) -> cp.ndarray:
        result = cp.empty_like(points)
        if len(points):
            self.kernels["translate_points"](
                _blocks(len(points)),
                _BLOCK,
                (
                    points,
                    offsets,
                    cp.asarray(np.ascontiguousarray(motion[:3, 3])),
                    np.int32(len(points)),
                    np.float64(gap),
                    result,
                ),
            )
        return result

    def rigid(
        self,
        points: cp.ndarray,
        offsets: cp.ndarray,
        motion: FloatArray,
        gap: float,
    ) -> cp.ndarray:
        result = cp.empty_like(points)
        vector = Rotation.from_matrix(motion[:3, :3]).as_rotvec()
        if len(points):
            self.kernels["rigid_deskew"](
                _blocks(len(points)),
                _BLOCK,
                (
                    points,
                    offsets,
                    cp.asarray(vector),
                    cp.asarray(np.ascontiguousarray(motion[:3, 3])),
                    np.int32(len(points)),
                    np.float64(gap),
                    result,
                ),
            )
        return result

    def statistics(self) -> dict[str, int]:
        return {
            "source_cache_bytes": self.bytes,
            "source_captures": self.source_captures,
            "source_cache_hits": self.source_hits,
            "source_validation_rows": self.source_validation_rows,
            "gyro_cached_validations": self.gyro_cached_validations,
            "preprocess_calls": self.preprocess_calls,
            "preprocess_rows": self.preprocess_rows,
            "voxel_native_fallbacks": self.voxel_fallbacks,
            "transforms": self.transforms,
            "gyro_rows": self.gyro_rows,
            "gyro_native_fallbacks": self.gyro_fallbacks,
            "native_model_fallbacks": self.native_model_fallbacks,
        }


class _GridDomainError(ValueError):
    pass


class _DeviceGrid:
    def __init__(self, points: cp.ndarray) -> None:
        self.data = points
        self._host: FloatArray | None = None
        self._cpu: cKDTree | None = None
        self.capture_method = "device_grid"
        self.query_calls = 0
        self.query_rows = 0
        self.parallel_dispatches = 0
        count = len(points)
        size = 1 << (2 * count - 1).bit_length()
        self.bytes = int(points.nbytes * 8 + size * 12)
        if self.bytes > _WORKSPACE_BYTES:
            raise ValueError("resident grid capture exceeds the 1 GiB workspace budget")
        self.scale = max(1.0, float(cp.max(cp.abs(points)).get()))
        keys = cp.full(size, -1, dtype=cp.int64)
        heads = cp.full(size, -1, dtype=cp.int32)
        next_rows = cp.empty(count, dtype=cp.int32)
        mask = cp.asarray([size - 1], dtype=cp.int32)
        brute = cp.zeros(65537, dtype=cp.int64)
        info = cp.asarray([0.16, self.scale], dtype=cp.float64)
        self.arrays = (keys, heads, next_rows, mask, brute, info)
        module = cp.RawModule(code=_GRID_SOURCE, options=("--fmad=false",))
        build = module.get_function("build_grid")
        self.kernel = module.get_function("query_grid")
        self.parallel = module.get_function("parallel_query")
        invalid = cp.zeros(1, dtype=cp.int32)
        build(
            _blocks(count),
            _BLOCK,
            (
                points,
                np.int32(count),
                np.float64(0.16),
                keys,
                heads,
                next_rows,
                np.int32(size - 1),
                invalid,
            ),
        )
        if int(invalid.get()[0]):
            raise _GridDomainError("resident grid coordinates exceed its key domain")

    @property
    def host(self) -> FloatArray:
        if self._host is None:
            self._host = _coordinates(cp.asnumpy(self.data), "odometry target")
        return self._host

    @property
    def cpu(self) -> cKDTree:
        if self._cpu is None:
            self._cpu = cKDTree(self.host, copy_data=True)
        return self._cpu

    def reserve(self, count: int) -> None:
        old = self.arrays[4]
        if count + 1 <= len(old):
            return
        if self.bytes + (count + 1) * 8 > _WORKSPACE_BYTES:
            raise ValueError("resident grid query worklist exceeds the 1 GiB workspace budget")
        brute = cp.zeros(count + 1, dtype=cp.int64)
        brute[:1] = old[:1]
        self.arrays = (*self.arrays[:4], brute, self.arrays[5])

    def query(self, points: cp.ndarray, k: int) -> tuple[cp.ndarray, cp.ndarray]:
        if not 1 <= k <= 17:
            raise ValueError("resident spatial queries support 1 through 17 neighbors")
        self.reserve(len(points))
        distances = cp.empty((len(points), k), dtype=cp.float64)
        indexes = cp.empty((len(points), k), dtype=cp.int64)
        if len(points):
            arguments = (
                points,
                self.data,
                *self.arrays,
                np.int32(len(points)),
                np.int32(len(self.data)),
                np.int32(k),
                distances,
                indexes,
            )
            self.kernel(_blocks(len(points)), _BLOCK, arguments)
            # A grid result is accepted only with a distance certificate.
            # Every remaining query scans all targets in parallel.
            self.parallel((len(points),), _BLOCK, arguments)
            self.parallel_dispatches += 1
            self.query_calls += 1
            self.query_rows += len(points)
        return distances, indexes


class _ParallelFused:
    def __init__(self, tree: _DeviceGrid) -> None:
        self.tree = tree
        self.primary = cp.RawKernel(
            _FUSED_GRID_SOURCE,
            "project_query_system3",
            options=("--fmad=false",),
        )
        self.parallel = cp.RawKernel(
            _FUSED_GRID_SOURCE,
            "parallel_fused3",
            options=("--fmad=false",),
        )

    @property
    def attributes(self) -> Any:
        return self.primary.attributes

    def __call__(
        self,
        grid: tuple[int],
        block: tuple[int],
        arguments: tuple[Any, ...],
    ) -> None:
        count = int(arguments[12])
        self.primary(grid, block, arguments)
        self.parallel((count,), _BLOCK, arguments)
        self.tree.parallel_dispatches += 1


class _ResidentTarget(cuda.CudaOdometryTarget):
    def __init__(self, points: cp.ndarray, options: ScanOdometryOptions, identity: str) -> None:
        cuda._validate_options(options)
        if len(points) < 3:
            raise ValueError("CUDA odometry targets require at least three points")
        if points.nbytes * 8 > _WORKSPACE_BYTES:
            raise ValueError("CUDA odometry target exceeds the 1 GiB payload budget")
        self.grid: _DeviceGrid | None = None
        self._normal_payload: tuple[cp.ndarray, cp.ndarray] | None = None
        self.sparse_native_normal_rows = 0
        self.spatial_native_normal_rows = 0
        try:
            tree = _DeviceGrid(points)
        except _GridDomainError:
            super().__init__(cp.asnumpy(points), options)
            self.identity = identity
            return
        self.grid = tree
        self._host = None
        self._initialize_target(tree, options, tree.scale, identity)
        self.fused_kernel = _ParallelFused(tree)

    @property
    def host(self) -> FloatArray:
        return super().host if self.grid is None else self.grid.host

    def _reserve_queries(self, count: int) -> None:
        if self.grid is not None:
            self.grid.reserve(count)

    def _launch_normals(
        self,
        rows: cp.ndarray,
        queries: cp.ndarray,
        k: int,
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        result = super()._launch_normals(rows, queries, k)
        self._normal_payload = (result[1], result[2])
        return result

    def _repair_normals(self, rows: cp.ndarray, k: int, flags: NDArray[np.uint8]) -> None:
        if self.grid is None:
            super()._repair_normals(rows, k, flags)
            return
        repair = np.flatnonzero(flags)
        if not len(repair):
            return
        assert self._normal_payload is not None
        positions = cp.asarray(repair)
        distances = cp.asnumpy(self._normal_payload[0][positions])
        indexes = cp.asnumpy(self._normal_payload[1][positions])[:, :k]
        error = 128 * np.finfo(np.float64).eps * max(1.0, self.scale)
        tied = np.any(np.diff(distances, axis=1) <= error, axis=1)
        if tied.any():
            native_rows = cp.asnumpy(rows[positions])
            _, fixed = self.grid.cpu.query(self.host[native_rows[tied]], k=k)
            indexes[tied] = fixed
            self.spatial_native_normal_rows += int(tied.sum())
        neighbours = cp.asnumpy(self.tree.data[cp.asarray(indexes)]).reshape(-1, 3)
        compact: NDArray[np.intp] = np.arange(len(indexes) * k, dtype=np.intp).reshape(
            len(indexes), k
        )
        normals, planar = native._normals_from_neighbours(neighbours, compact, k, self.options)
        selected = rows[positions]
        self._normals[selected] = cp.asarray(normals)
        self._planar[selected] = cp.asarray(planar)
        self.query_repairs += int(tied.sum())
        self.normal_repairs += len(repair)
        self.sparse_native_normal_rows += len(repair)

    def statistics(self) -> dict[str, int]:
        result = super().statistics()
        grid = self.grid
        result.update(
            {
                "gpu_grid_captures": int(grid is not None),
                "gpu_grid_native_fallbacks": int(grid is None),
                "gpu_grid_brute_rows": 0 if grid is None else int(grid.arrays[4][:1].get()[0]),
                "native_tree_captures": int(grid is None or grid._cpu is not None),
                "sparse_native_normal_rows": self.sparse_native_normal_rows,
                "spatial_native_normal_rows": self.spatial_native_normal_rows,
                "gpu_parallel_fallback_dispatches": 0 if grid is None else grid.parallel_dispatches,
            }
        )
        return result


class _DeviceGyro:
    def __init__(self, model: _GyroRotationModel, geometry: _Geometry) -> None:
        self.model = model
        self.geometry = geometry
        self.times = np.array(model.gyro.times_s, dtype=np.float64, copy=True)
        self.rotation = np.array(model.rotation, dtype=np.float64, copy=True)
        self.offset = model.time_offset_s
        self.orientations: FloatArray = model.gyro._orientations  # type: ignore[attr-defined]
        self.rates: FloatArray = model.gyro._rates  # type: ignore[attr-defined]
        sums: FloatArray = model.gyro._bias_sum  # type: ignore[attr-defined]
        self.projected = sums @ model.bias
        self.valid = bool(
            np.isfinite(self.rates).all()
            and np.isfinite(self.projected).all()
            and np.isfinite(self.rotation).all()
            and np.isfinite(model.bias).all()
            and np.max(np.abs(model.bias)) <= 1.0
            and np.allclose(self.rotation.T @ self.rotation, np.eye(3), rtol=0, atol=1e-10)
            and np.allclose(
                np.transpose(self.orientations, (0, 2, 1)) @ self.orientations,
                np.eye(3),
                rtol=0,
                atol=1e-10,
            )
            and np.all(np.linalg.det(self.orientations) > 0)
        )
        self.arrays = tuple(
            cp.asarray(np.ascontiguousarray(array, dtype=np.float64))
            for array in (self.times, self.orientations, self.rates, self.projected, self.rotation)
        )

    def rotate(
        self,
        scan_time: float,
        scan: _CapturedScan,
    ) -> cp.ndarray:
        points, offsets = scan.points, scan.offsets
        assert offsets is not None
        start = scan_time + self.offset
        self.geometry.gyro_cached_validations += 1
        if not self.valid or not scan.covers(start, float(self.times[0]), float(self.times[-1])):
            self.geometry.gyro_fallbacks += 1
            return cp.asarray(
                self.model.rotate_points(
                    scan_time,
                    cp.asnumpy(offsets),
                    cp.asnumpy(points),
                )
            )
        queries = start + offsets
        index = int(
            np.clip(
                np.searchsorted(self.times, start, side="right") - 1,
                0,
                len(self.rates) - 1,
            )
        )
        elapsed = start - self.times[index]
        fraction = elapsed / (self.times[index + 1] - self.times[index])
        start_sum = self.projected[index] + fraction * (
            self.projected[index + 1] - self.projected[index]
        )
        start_orientation = (
            self.orientations[index]
            @ Rotation.from_rotvec(
                self.rates[index] * elapsed,
            ).as_matrix()
        )
        back = self.rotation @ start_orientation.T
        result = cp.empty_like(points)
        self.geometry.kernels["gyro_rotate3"](
            _blocks(len(points)),
            _BLOCK,
            (
                points,
                queries,
                *self.arrays,
                cp.asarray(np.ascontiguousarray(back)),
                cp.asarray(start_sum),
                np.int32(len(points)),
                np.int32(len(self.times)),
                result,
            ),
        )
        self.geometry.gyro_rows += len(points)
        return result


class _ResidentIncremental:
    def __init__(
        self,
        geometry: _Geometry,
        identity: str,
        options: ScanOdometryOptions,
        rotation_model: _GyroRotationModel | None,
        require_context: Callable[[], None],
    ) -> None:
        self.options = options
        self.geometry = geometry
        self._require_context = require_context
        self.gyro = None if rotation_model is None else _DeviceGyro(rotation_model, geometry)
        self.poses: list[FloatArray] = []
        self.registrations: list[ScanRegistration] = []
        self._recent: deque[cp.ndarray] = deque(maxlen=options.local_map_scans)
        self._previous_motion = np.eye(4)
        self._previous_time: float | None = None
        self._previous_gap: float | None = None
        self.execution_backend = identity
        self.execution_statistics: dict[str, int] = {}

    def add(
        self,
        scan: FloatArray,
        time_s: float | None = None,
        point_offsets_s: FloatArray | None = None,
    ) -> ScanRegistration | None:
        self._require_context()
        captured = self.geometry.capture(scan, point_offsets_s)
        points, offsets = captured.points, captured.offsets
        if not self.poses:
            self.poses.append(np.eye(4))
            self._recent.append(self.geometry.preprocess(points, self.options))
            self._previous_time = time_s
            return None
        initial = self._previous_motion
        gap: float | None = None
        if time_s is not None and self._previous_time is not None:
            gap = time_s - self._previous_time
            if self._previous_gap is not None and self._previous_gap > 0 and gap > 0:
                initial = native.scale_motion(self._previous_motion, gap / self._previous_gap)
            self._previous_gap = gap
        self._previous_time = time_s
        if len(self._recent) == 1 and self.options.local_map_scans == 1:
            local = self._recent[0]
        else:
            local = self.geometry.preprocess(
                cp.concatenate(tuple(self._recent)),
                native._unbounded(self.options),
            )
        target_points = self.geometry.transform(
            local,
            np.linalg.inv(self.poses[-1]),
            self.options.geometry_quantum_m,
        )
        target = _ResidentTarget(target_points, self.options, self.execution_backend)
        target._prime_normals(min(len(scan), len(self._recent[-1])))
        deskew = offsets is not None and gap is not None and gap > 0
        motion = initial
        rotated: cp.ndarray | None = None
        registration: ScanRegistration | None = None
        for _ in range(1 + (self.options.deskew_iterations if deskew else 0)):
            source = points
            if deskew:
                assert offsets is not None and gap is not None
                if self.gyro is not None and time_s is not None:
                    if rotated is None:
                        rotated = self.gyro.rotate(time_s, captured)
                    source = self.geometry.translate(rotated, offsets, motion, gap)
                else:
                    source = self.geometry.rigid(points, offsets, motion, gap)
            processed = self.geometry.preprocess(source, self.options)
            registration = target._register_captured(cuda._CapturedSource(None, processed), motion)
            motion = registration.transform
        assert registration is not None
        for name, value in target.statistics().items():
            self.execution_statistics[name] = self.execution_statistics.get(name, 0) + value
        self.registrations.append(registration)
        self._previous_motion = registration.transform
        pose = self.poses[-1] @ registration.transform
        self.poses.append(pose)
        self._recent.append(self.geometry.transform(processed, pose))
        return registration

    def result(self) -> ScanOdometryResult:
        return ScanOdometryResult(
            tuple(self.poses),
            tuple(self.registrations),
            self.execution_backend,
            dict(self.execution_statistics),
        )


class ResidentGeometrySession:
    """Own geometry resources for one explicit bag calibration.

    Internal callers create odometry only while this context is active, and
    supply host scans and the bag's native gyro model. Separate sessions share
    neither a cache nor a stream. The caller's stream is restored on exit,
    including exceptional exit; no process-global backend is replaced.
    """

    def __init__(self) -> None:
        self._active = False
        self._used = False
        self._geometry: _Geometry | None = None
        self.device_id = cp.cuda.runtime.getDevice()
        self._device = cp.cuda.Device(self.device_id)
        self._stream = cp.cuda.Stream(non_blocking=True)
        self.identity = ""

    def __enter__(self) -> ResidentGeometrySession:
        if self._used:
            raise ValueError("resident geometry sessions cannot be re-entered")
        self._used = True
        self._device.__enter__()
        self._stream.__enter__()
        try:
            self._geometry = _Geometry()
            self.identity = _identity()
            self._active = True
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._stream.synchronize()
        finally:
            self._active = False
            if self._geometry is not None:
                self._geometry.scans.clear()
            self._stream.__exit__(exc_type, exc, traceback)
            self._device.__exit__(exc_type, exc, traceback)

    def create_odometry(
        self,
        options: ScanOdometryOptions | None = None,
        *,
        rotation_model: RotationModel | None = None,
        backend: OdometryBackend = "cpu",
    ) -> _IncrementalOdometry:
        """Create one pass on this session's stream, with native callback fallback."""
        self._require_context()
        if backend != "cuda":
            raise ValueError("resident odometry requires the CUDA odometry backend")
        opts = options or ScanOdometryOptions()
        _validate_options(opts)
        assert self._geometry is not None
        if rotation_model is not None and (
            type(rotation_model) is not _GyroRotationModel
            or type(rotation_model.gyro) is not GyroSeries
        ):
            self._geometry.native_model_fallbacks += 1
            return native.IncrementalScanOdometry(
                opts,
                rotation_model=rotation_model,
                backend=backend,
            )
        return _ResidentIncremental(
            self._geometry,
            self.identity,
            opts,
            rotation_model,
            self._require_context,
        )

    def _require_context(self) -> None:
        if (
            not self._active
            or cp.cuda.runtime.getDevice() != self.device_id
            or cp.cuda.get_current_stream().ptr != self._stream.ptr
        ):
            raise ValueError("resident odometry requires its active session device and stream")

    def statistics(self) -> dict[str, int]:
        """Return work and retention counters without recording hardware names."""
        return {} if self._geometry is None else self._geometry.statistics()
