"""Optional FP64 spatial acceleration for native robust point-to-point ICP."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, TypeAlias, cast

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.solvers._exact_neighbors import _ExactNeighbors, _first_of_two, _tree_safe
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


_NATIVE_QUERY_SOURCE = r'''
extern "C" __global__ void native_query3(
    const double* axis_xyz, const double* native_xyz, const double* native_reference,
    const long long* order, const int* axis, const double* split, const int* child,
    const long long* span, const int n, const int m,
    const double lower_scale, const double error, const double initial_best,
    double* squared, long long* index) {
    const int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    const double ax=axis_xyz[3*row], ay=axis_xyz[3*row+1], az=axis_xyz[3*row+2];
#if RIGID_BOUNDS
    const double px=native_xyz[3*row], py=native_xyz[3*row+1], pz=native_xyz[3*row+2];
#else
    const double px=ax, py=ay, pz=az;
#endif
    double best=initial_best; long long chosen=m;
    int stack[STACK_CAPACITY]; double bound[STACK_CAPACITY]; int top=1;
    stack[0]=0; bound[0]=0;
    while(top) {
        --top; const int node=stack[top];
        if(bound[top]>best) continue;
        const int direction=axis[node];
        if(direction<0) {
            for(long long j=span[2*node]; j<span[2*node+1]; ++j) {
                const long long position=order[j];
                const double x=px-native_reference[3*position];
                const double y=py-native_reference[3*position+1];
                const double z=pz-native_reference[3*position+2];
                const double d=x*x+y*y+z*z;
                if(d<best || (d==best && position<chosen)) { best=d; chosen=position; }
            }
        } else {
            const double p=direction==0 ? ax : (direction==1 ? ay : az);
            const double delta=p-split[node]; const int side=delta<0 ? 0 : 1;
            const int far=child[2*node+1-side], near=child[2*node+side];
#if RIGID_BOUNDS
            double safe=__dadd_rd(fabs(delta),-error);
            if(safe<0) safe=0;
            safe=__dmul_rd(safe,lower_scale);
            const double lower=__dmul_rd(safe,safe);
#else
            const double lower=__dmul_rd(delta,delta);
#endif
            if(lower<=best) { stack[top]=far; bound[top]=lower; ++top; }
            stack[top]=near; bound[top]=0; ++top;
        }
    }
    squared[row]=chosen==m ? 1.0/0.0 : best; index[row]=chosen==m ? 0 : chosen;
}
'''


def _gate_squared(distance_m: float) -> float:
    """Include every squared distance whose correctly rounded sqrt passes the gate."""

    if not math.isfinite(distance_m) or distance_m >= 1e150:
        return math.inf
    enlarged = float(np.nextafter(distance_m, np.inf))
    return float(np.nextafter(enlarged*enlarged, np.inf))


class _CudaNeighbors(_ExactNeighbors):
    """Exact FP64 nearest distances; retain guarded proposals for wider queries."""

    def __init__(self, reference: FloatArray, device_id: int) -> None:
        super().__init__(reference)
        self.device_id = device_id
        self.calls = self.rows = self.repairs = 0
        self.direct_calls = self.native_upload_bytes = 0
        self.stack_capacity = 0
        self._native_kernels: dict[bool, Any] = {}
        self._lock = Lock()
        self.gpu: _CudaTree | None = None
        with cp.cuda.Device(device_id):
            self._stream = cp.cuda.Stream(non_blocking=True)
            if len(reference) >= 1024 and _tree_safe(reference) and np.max(
                np.abs(self._reference), initial=0.).item() < 1e50:
                with self._stream:
                    self.gpu = _CudaTree(self._reference)
                self._stream.synchronize()
                pending = [(self.gpu.cpu.tree, 0)]
                depth = 0
                while pending:
                    node, level = pending.pop()
                    depth = max(depth, level)
                    if node.split_dim >= 0:
                        pending.extend(((node.lesser, level+1), (node.greater, level+1)))
                self.stack_capacity = depth+1

    def _native_nearest(
        self, points: FloatArray, *, raw_points: FloatArray | None = None,
        native_reference: FloatArray | None = None, lower_scale: float = 1., error: float = 0.,
        max_squared: float = math.inf,
    ) -> tuple[IntArray, FloatArray] | None:
        axis_points = points if raw_points is None else raw_points
        if (self.gpu is None or not len(points)
            or points.dtype not in (np.float32, np.float64) or not _tree_safe(points)
            or not _tree_safe(axis_points)
            or float(np.max(np.abs(points), initial=0.)) >= 1e50
            or float(np.max(np.abs(axis_points), initial=0.)) >= 1e50):
            return None
        if native_reference is not None and (
            native_reference.shape != self._reference.shape
            or native_reference.dtype != np.float64 or not _tree_safe(native_reference)
            or np.max(np.abs(native_reference), initial=0.) >= 1e50
        ):
            return None
        with self._lock, cp.cuda.Device(self.device_id), self._stream:
            rigid = (raw_points is not None or native_reference is not None
                     or lower_scale != 1. or error != 0.)
            if rigid not in self._native_kernels:
                self._native_kernels[rigid] = cp.RawKernel(
                    _NATIVE_QUERY_SOURCE, "native_query3", options=(
                        "--fmad=false", f"-DSTACK_CAPACITY={self.stack_capacity}",
                        f"-DRIGID_BOUNDS={int(rigid)}"),
                )
            native = cp.asarray(points, dtype=cp.float64, order="C")
            axes = native if raw_points is None else cp.asarray(
                axis_points, dtype=cp.float64, order="C")
            reference = self.gpu.data if native_reference is None else cp.asarray(
                native_reference, dtype=cp.float64, order="C")
            squared = cp.empty((len(points), 1), dtype=cp.float64)
            indices = cp.empty((len(points), 1), dtype=cp.int64)
            self._native_kernels[rigid](((len(points)+127)//128,), (128,), (
                axes, native, reference, *self.gpu.arrays, np.int32(len(points)),
                np.int32(len(self._reference)), np.float64(lower_scale), np.float64(error),
                np.float64(max_squared),
                squared, indices,
            ))
            host_i = cast(IntArray, cp.asnumpy(indices))
            host_d = cast(FloatArray, cp.asnumpy(squared))
            self.calls += 1
            self.rows += len(points)
            self.direct_calls += 1
            self.native_upload_bytes += int(native.nbytes)
            if raw_points is not None:
                self.native_upload_bytes += int(axes.nbytes)
            if native_reference is not None:
                self.native_upload_bytes += int(reference.nbytes)
        return host_i, host_d

    def _nearest_within(self, points: FloatArray, distance_m: float) -> tuple[IntArray, FloatArray]:
        if ("nearest" not in self.__dict__ and type(self).nearest is _ExactNeighbors.nearest
            and "query" not in self.__dict__ and type(self).query is _CudaNeighbors.query):
            direct = self._native_nearest(points, max_squared=_gate_squared(distance_m))
            if direct is not None:
                return direct[0][:, 0], direct[1][:, 0]
        return self.nearest(points)

    def _propose(self, points: FloatArray, width: int) -> tuple[IntArray, FloatArray] | None:
        if (self.gpu is None or width < 1 or points.dtype != np.float64 or not _tree_safe(points)
            or float(np.max(np.abs(points), initial=0.)) >= 1e50 or width > 17):
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
        if count == 1:
            direct = self._native_nearest(points)
            if direct is not None:
                return direct
        width = min(count+1, len(self._reference))
        proposal = self._propose(points, width)
        if proposal is None:
            return super().query(points, count)
        candidates, _distance = proposal
        squared = np.sum((points[:, None, :]-self._reference[candidates])**2, axis=2)
        if count == 1 and width == 2:
            first = _first_of_two(candidates, squared)
            indices = np.where(first[:, None], candidates[:, :1], candidates[:, 1:2])
            distances = np.where(first[:, None], squared[:, :1], squared[:, 1:2])
            remaining = np.where(first, squared[:, 1], squared[:, 0])
        else:
            order = np.lexsort((candidates, squared), axis=1)
            candidates = np.take_along_axis(candidates, order, axis=1)
            squared = np.take_along_axis(squared, order, axis=1)
            indices, distances = candidates[:, :count].copy(), squared[:, :count].copy()
            remaining = squared[:, count] if width > count else None
        if width > count:
            cutoff = distances[:, count-1]
            guard = np.maximum(128*np.finfo(np.float64).eps*np.maximum(1., cutoff),
                               np.finfo(np.float64).tiny)
            assert remaining is not None
            rows = np.flatnonzero(remaining <= cutoff+guard)
            if len(rows):
                repaired_i, repaired_d = super().query(points[rows], count)
                indices[rows], distances[rows] = repaired_i, repaired_d
                with self._lock:
                    self.repairs += len(rows)
        return indices, distances


class _CudaRigidSource:
    """Prune raw-source topology conservatively; compare original moving-world points."""

    def __init__(self, source: FloatArray, device_id: int) -> None:
        self.search = _CudaNeighbors(source, device_id)
        self.repairs = 0
        self.world_calls = self.cpu_fallbacks = 0

    def nearest(
        self, transformed_source: FloatArray, points: FloatArray, transform: SE3,
        *, _distance_m: float | None = None,
    ) -> IntArray:
        rotation = _rotation_matrix(transform)
        inverse = (points-np.asarray(transform.translation_m))@rotation
        scale = max(1., float(np.max(np.abs(self.search._reference), initial=0.)),
                    float(np.max(np.abs(points), initial=0.)),
                    float(np.max(np.abs(transformed_source), initial=0.)),
                    float(np.max(np.abs(transform.translation_m), initial=0.)))
        error = 512*np.finfo(np.float64).eps*scale*max(1., float(np.linalg.norm(rotation, 1)))
        singular = np.linalg.svd(rotation, compute_uv=False)
        minimum, maximum = float(singular[-1]), float(singular[0])
        if (scale < 1e50 and np.isfinite(error)
            and rotation.dtype == np.float64
            and .999999999999 <= minimum <= maximum <= 1.000000000001):
            defect = float(np.sum(np.abs(rotation.T@rotation-np.eye(3))))
            error += 8*scale*defect/minimum
            # Shrink the singular-value estimate and round every bound
            # operation down. Leaves use the original moving-world points.
            lower = minimum-1024*np.finfo(np.float64).eps*max(
                1., float(np.linalg.norm(rotation, 1)))
            direct = self.search._native_nearest(
                points, raw_points=inverse, native_reference=transformed_source,
                lower_scale=lower, error=2*error,
                max_squared=math.inf if _distance_m is None else _gate_squared(_distance_m),
            )
            if direct is not None:
                with self.search._lock:
                    self.world_calls += 1
                return direct[0][:, 0].copy()
        with self.search._lock:
            self.cpu_fallbacks += 1
        return _ExactNeighbors(transformed_source).nearest(points)[0]

    def _nearest_within(
        self, transformed_source: FloatArray, points: FloatArray, transform: SE3, distance_m: float,
    ) -> IntArray:
        if "nearest" in self.__dict__ or type(self).nearest is not _CudaRigidSource.nearest:
            return self.nearest(transformed_source, points, transform)
        return self.nearest(transformed_source, points, transform, _distance_m=distance_m)


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
            "native_query_sha256": hashlib.sha256(_NATIVE_QUERY_SOURCE.encode()).hexdigest(),
            "gpu_native_query_calls": sum(value.direct_calls for value in searches),
            "gpu_world_source_query_calls": sum(value.world_calls for value in sources),
            "gpu_native_query_upload_bytes": sum(value.native_upload_bytes for value in searches),
            "native_query_stack_capacities": [value.stack_capacity for value in searches
                                              if value.gpu is not None],
            "source_cpu_fallback_calls": sum(value.cpu_fallbacks for value in sources),
            "source_sha256": digest(source_points), "target_sha256": digest(target_points),
            "initial_transform": (initial_transform or SE3.identity()).as_dict(),
            "options": asdict(options or RobustPointToPointIcpOptions()),
            "gpu_query_calls": sum(value.calls for value in searches),
            "gpu_query_rows": sum(value.rows for value in searches),
            "target_query_repairs": sum(value.repairs for value in targets),
            "source_query_repairs": sum(value.repairs for value in sources),
            "static_source_snapshots": len(sources), "target_snapshots": len(targets),
        })
