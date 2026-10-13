"""Optional FP64 spatial acceleration for native robust point-to-point ICP."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import TypeAlias, cast

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.solvers._exact_neighbors import _ExactNeighbors, _tree_safe
from calibrex.solvers.cuda_lidar_map_adapter import _SOURCE, _CudaTree
from calibrex.solvers.robust_point_to_point_icp_solver import (
    FloatArray,
    IcpPoint,
    RobustPointToPointIcpOptions,
    RobustPointToPointIcpResult,
    RobustPointToPointIcpSolver,
    _rotation_matrix,
)

IntArray: TypeAlias = NDArray[np.int64]


class _CudaNeighbors(_ExactNeighbors):
    """GPU proposals, original NumPy distances and native uncertainty repairs."""

    def __init__(self, reference: FloatArray, device_id: int) -> None:
        super().__init__(reference)
        self.device_id = device_id
        self.calls = self.rows = self.repairs = 0
        self._lock = Lock()
        self.gpu: _CudaTree | None = None
        with cp.cuda.Device(device_id):
            self._stream = cp.cuda.Stream(non_blocking=True)
            if len(reference) >= 1024 and _tree_safe(reference) and np.max(
                np.abs(reference), initial=0.) < 1e50:
                with self._stream:
                    self.gpu = _CudaTree(self._reference)
                self._stream.synchronize()

    def _propose(self, points: FloatArray, width: int) -> tuple[IntArray, FloatArray] | None:
        if (self.gpu is None or width < 1 or not _tree_safe(points)
            or np.max(np.abs(points), initial=0.) >= 1e50 or width > 17):
            return None
        with self._lock, cp.cuda.Device(self.device_id), self._stream:
            distances, indices = self.gpu.query(cp.asarray(points), width)
            host_i = cast(IntArray, cp.asnumpy(indices))
            host_d = cast(FloatArray, cp.asnumpy(distances))
            self.calls += 1
            self.rows += len(points)
        return host_i, host_d

    def query(self, points: FloatArray, count: int) -> tuple[IntArray, FloatArray]:
        if count < 1 or not len(self._reference) or not len(points):
            return super().query(points, count)
        count = min(count, len(self._reference))
        width = min(count+1, len(self._reference))
        proposal = self._propose(points, width)
        if proposal is None:
            return super().query(points, count)
        candidates, _distance = proposal
        squared = np.sum((points[:, None, :]-self._reference[candidates])**2, axis=2)
        order = np.lexsort((candidates, squared), axis=1)
        candidates = np.take_along_axis(candidates, order, axis=1)
        squared = np.take_along_axis(squared, order, axis=1)
        indices, distances = candidates[:, :count].copy(), squared[:, :count].copy()
        if width > count:
            cutoff = squared[:, count-1]
            guard = np.maximum(128*np.finfo(np.float64).eps*np.maximum(1., cutoff),
                               np.finfo(np.float64).tiny)
            rows = np.flatnonzero(squared[:, count] <= cutoff+guard)
            if len(rows):
                repaired_i, repaired_d = super().query(points[rows], count)
                indices[rows], distances[rows] = repaired_i, repaired_d
                with self._lock:
                    self.repairs += len(rows)
        return indices, distances


class _CudaRigidSource:
    """Reuse raw-source topology; certify membership in the original moving frame."""

    def __init__(self, source: FloatArray, device_id: int) -> None:
        self.search = _CudaNeighbors(source, device_id)
        self.repairs = 0

    def nearest(
        self, transformed_source: FloatArray, points: FloatArray, transform: SE3,
    ) -> IntArray:
        rotation = _rotation_matrix(transform)
        inverse = (points-np.asarray(transform.translation_m))@rotation
        width = min(2, len(transformed_source))
        proposed = self.search._propose(inverse, width)
        if proposed is None or self.search._tree is None:
            return _ExactNeighbors(transformed_source).nearest(points)[0]
        candidates, distance = proposed
        squared = np.sum((points[:, None, :]-transformed_source[candidates])**2, axis=2)
        order = np.lexsort((candidates, squared), axis=1)
        result = np.take_along_axis(candidates, order, axis=1)[:, 0].copy()
        if width == 1:
            return result
        native_best = np.take_along_axis(squared, order, axis=1)[:, 0]
        scale = max(1., float(np.max(np.abs(self.search._reference), initial=0.)),
                    float(np.max(np.abs(points), initial=0.)),
                    float(np.max(np.abs(transformed_source), initial=0.)),
                    float(np.max(np.abs(transform.translation_m), initial=0.)))
        error = 512*np.finfo(np.float64).eps*scale*max(1., float(np.linalg.norm(rotation, 1)))
        minimum = float(np.linalg.svd(rotation, compute_uv=False)[-1])
        if not np.isfinite(error) or not .999999999999 <= minimum <= 1.000000000001:
            return _ExactNeighbors(transformed_source).nearest(points)[0]
        rows = np.flatnonzero(distance[:, 1]-distance[:, 0] <= 2*error)
        for row in rows:
            radius = np.nextafter(np.sqrt(native_best[row])/minimum+2*error, np.inf)
            nearby = np.asarray(self.search._tree.query_ball_point(inverse[row], radius, eps=0.),
                                dtype=np.int64)
            nearby = np.union1d(nearby, candidates[row])
            exact = np.sum((points[row]-transformed_source[nearby])**2, axis=1)
            result[row] = nearby[np.lexsort((nearby, exact))[0]]
        with self.search._lock:
            self.repairs += len(rows)
        return result


@dataclass(frozen=True)
class CudaPointToPointIcpResult:
    result: RobustPointToPointIcpResult
    provenance: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {"result": self.result.as_dict(), "provenance": self.provenance}


class CudaPointToPointIcpAdapter:
    """Accelerate mutual spatial searches while retaining the native ICP evaluator."""

    def __init__(self, device_id: int | None = None) -> None:
        self.device_id = cp.cuda.runtime.getDevice() if device_id is None else device_id

    def solve(
        self, source_points: Sequence[IcpPoint], target_points: Sequence[IcpPoint],
        initial_transform: SE3 | None = None, options: RobustPointToPointIcpOptions | None = None,
    ) -> CudaPointToPointIcpResult:
        targets: list[_CudaNeighbors] = []
        sources: list[_CudaRigidSource] = []
        lock = Lock()

        def target_factory(points: FloatArray) -> _ExactNeighbors:
            with lock:
                for value in targets:
                    if np.array_equal(value._reference, points):
                        return value
                value = _CudaNeighbors(points, self.device_id)
                targets.append(value)
            return value

        def source_factory(points: FloatArray) -> _CudaRigidSource:
            with lock:
                for value in sources:
                    if np.array_equal(value.search._reference, points):
                        return value
                value = _CudaRigidSource(points, self.device_id)
                sources.append(value)
            return value

        result = RobustPointToPointIcpSolver().solve(
            source_points, target_points, initial_transform, options,
            _target_search_factory=target_factory, _source_search_factory=source_factory,
        )
        def digest(points: Sequence[IcpPoint]) -> str:
            payload = "\n".join(f"{p.point_id}:{p.position_m!r}" for p in sorted(
                points, key=lambda p: p.point_id))
            return hashlib.sha256(payload.encode()).hexdigest()

        searches = [*targets, *(source.search for source in sources)]
        return CudaPointToPointIcpResult(result, {
            "method": "calibrex.cuda_point_to_point_icp/v0.1",
            "fp64": True, "query_fma": False, "pose_updates_cpu": True,
            "cupy": cp.__version__, "cuda": cp.cuda.runtime.runtimeGetVersion(),
            "driver": cp.cuda.runtime.driverGetVersion(),
            "query_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
            "source_sha256": digest(source_points), "target_sha256": digest(target_points),
            "initial_transform": (initial_transform or SE3.identity()).as_dict(),
            "options": (options or RobustPointToPointIcpOptions()).__dict__,
            "gpu_query_calls": sum(value.calls for value in searches),
            "gpu_query_rows": sum(value.rows for value in searches),
            "target_query_repairs": sum(value.repairs for value in targets),
            "source_query_repairs": sum(value.repairs for value in sources),
            "static_source_snapshots": len(sources), "target_snapshots": len(targets),
        })
