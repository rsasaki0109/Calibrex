"""Optional exact-order CUDA residuals and finite differences for LiDAR rigs."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from importlib.metadata import version
from threading import Lock
from typing import TYPE_CHECKING, TypeAlias

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3

if TYPE_CHECKING:
    from calibrex.graph.lidar_point_to_plane import _ObservationArrays

FloatArray: TypeAlias = NDArray[np.float64]
_LINEARIZATION_CACHE_BYTES = 64 * 1024**2

_SOURCE = r"""
struct Q { double x,y,z,w; };
__device__ Q normalized(Q q) {
    double n=sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
    if(n==1.0) return q;
    return {q.x/n,q.y/n,q.z/n,q.w/n};
}
__device__ Q product(Q a,Q b) {
    return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
            a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
            a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
            a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};
}
__device__ Q rotated(Q q, const double* p) {
    q=normalized(q); Q c=normalized(q);
    return product(product(q,{p[0],p[1],p[2],0.0}),{-c.x,-c.y,-c.z,c.w});
}
__device__ double residual(const double* p,const double* plane,const double* normal,
                          const double* pose,double weight,const double* transform) {
    Q q={pose[3],pose[4],pose[5],pose[6]};
    Q translated=rotated(q,transform);
    double wt[3]={pose[0]+translated.x,pose[1]+translated.y,pose[2]+translated.z};
    Q wq=normalized(normalized(product(q,{transform[3],transform[4],transform[5],transform[6]})));
    Q r=rotated(wq,p);
    double dx=r.x+wt[0]-plane[0],dy=r.y+wt[1]-plane[1],dz=r.z+wt[2]-plane[2];
    return weight*(normal[0]*dx+normal[1]*dy+normal[2]*dz);
}
extern "C" __global__ void residuals(const double* points,const double* planes,
    const double* normals,const double* poses,const double* weights,
    const double* transforms,const int n,double* result) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row<n) result[row]=residual(points+3*row,planes+3*row,normals+3*row,
                                 poses+7*row,weights[row],transforms);
}
extern "C" __global__ void linearize(const double* points,const double* planes,
    const double* normals,const double* poses,const double* weights,
    const double* transforms,const double* steps,const int n,double* result,double* jacobian) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    const double* p=points+3*row;const double* plane=planes+3*row;
    const double* normal=normals+3*row;const double* pose=poses+7*row;
    result[row]=residual(p,plane,normal,pose,weights[row],transforms);
    for(int col=0;col<6;col++) {
        double plus=residual(p,plane,normal,pose,weights[row],transforms+7*(1+2*col));
        double minus=residual(p,plane,normal,pose,weights[row],transforms+7*(2+2*col));
        jacobian[6*row+col]=(plus-minus)/(2.0*steps[col]);
    }
}
"""


def cuda_continuous_time_identity(
    *, associations_gpu: bool = False, captured_summary_gpu: bool = False,
    summary_parallel_cpu: bool = False,
    initial_projection_reused: bool = False,
) -> str:
    """Identify arithmetic and runtime without recording a device model."""
    association_identity = ""
    if associations_gpu:
        from calibrex.solvers.cuda_plane_query_adapter import cuda_plane_query_source_sha256

        association_identity = f";association_kernel_sha256={cuda_plane_query_source_sha256()}"
    return (
        "calibrex.cuda_continuous_time/v0.9;fp64=true;fma=false;"
        f"cupy={cp.__version__};cuda={cp.cuda.runtime.runtimeGetVersion()};"
        f"driver={cp.cuda.runtime.driverGetVersion()};"
        f"numba={version('numba')};trajectory_cpu=true;normal_cpu=true;"
        "standalone_residual_cpu=true;"
        "linearization_cache=last;linearization_cache_bytes=67108864;"
        f"captured_summary_gpu={str(captured_summary_gpu).lower()};summary_reduction_cpu=true;"
        f"summary_parallel_cpu={str(summary_parallel_cpu).lower()};summary_workers=3;"
        "summary_min_rows=32768;summary_workspace_bytes=16777216;"
        f"initial_projection_reused={str(initial_projection_reused).lower()};"
        "initial_projection_bytes=16777216;"
        f"associations_gpu={str(associations_gpu).lower()};"
        f"kernel_sha256={hashlib.sha256(_SOURCE.encode()).hexdigest()}"
        + association_identity
    )


class CudaRigLinearizer:
    """Retain an immutable payload and its last exact finite differences."""

    def __init__(
        self, device_id: int | None = None,
        *, _kernels: tuple[cp.RawKernel, cp.RawKernel] | None = None,
    ) -> None:
        self.device_id = cp.cuda.runtime.getDevice() if device_id is None else device_id
        self._arrays: _ObservationArrays | None = None
        self._payload: tuple[cp.ndarray, ...] = ()
        self.linearization_calls = 0
        self.linearization_cache_hits = 0
        self.linearization_cache_bytes = 0
        self._cached_key: tuple[bytes, bytes] | None = None
        self._cached_result: tuple[FloatArray, FloatArray] | None = None
        self.residual_calls = 0
        self.rows = 0
        self._lock = Lock()
        with cp.cuda.Device(self.device_id):
            self._stream = cp.cuda.Stream(non_blocking=True)
            if _kernels is None:
                self._linearize = cp.RawKernel(_SOURCE, "linearize", options=("--fmad=false",))
                self._residuals = cp.RawKernel(_SOURCE, "residuals", options=("--fmad=false",))
                self._linearize.compile()
                self._residuals.compile()
            else:
                self._linearize, self._residuals = _kernels

    def fork(self) -> CudaRigLinearizer:
        """Share compiled kernels with an independently owned stream and payload."""
        return CudaRigLinearizer(
            self.device_id, _kernels=(self._linearize, self._residuals),
        )

    def _capture(self, arrays: _ObservationArrays) -> tuple[cp.ndarray, ...]:
        if self._arrays is not arrays:
            values = (arrays.points, arrays.plane_points, arrays.normals,
                      arrays.poses, arrays.sqrt_weights)
            if sum(value.nbytes for value in values) * 2 > 256 * 1024**2:
                raise ValueError("CUDA rig observations exceed the 256 MiB payload budget")
            if any(value.flags.writeable for value in values):
                raise ValueError("CUDA rig observations require immutable arrays")
            n = len(arrays.points)
            if any(value.shape != shape for value, shape in zip(
                values, ((n, 3), (n, 3), (n, 3), (n, 7), (n,)), strict=True,
            )) or any(value.dtype != np.float64 or not np.isfinite(value).all()
                      for value in values):
                raise ValueError("CUDA rig observations require finite matching columns")
            self._payload = tuple(cp.asarray(value) for value in values)
            self._arrays = arrays
            self._clear_linearization_cache()
        return self._payload

    def _clear_linearization_cache(self) -> None:
        self._cached_key = None
        self._cached_result = None
        self.linearization_cache_bytes = 0

    def residuals(self, arrays: _ObservationArrays, transform: SE3) -> list[float]:
        """Evaluate the ordered native residual formula on the captured device."""
        n = len(arrays.points)
        with self._lock, cp.cuda.Device(self.device_id), self._stream:
            payload = self._capture(arrays)
            values = cp.asarray((*transform.translation_m, *transform.rotation_quat_xyzw))
            result = cp.empty(n, dtype=cp.float64)
            if n:
                self._residuals(((n+127)//128,), (128,), (*payload, values, np.int32(n), result))
            host: FloatArray = result.get()
            self.residual_calls += 1
            self.rows += n
        return list(host.tolist())

    def _captured_residuals(self, arrays: _ObservationArrays, transform: SE3) -> FloatArray | None:
        """Reuse a previously uploaded batch for internal error summaries only."""
        with self._lock, cp.cuda.Device(self.device_id), self._stream:
            if self._arrays is not arrays:
                return None
            host_values = np.asarray((*transform.translation_m, *transform.rotation_quat_xyzw),
                                     dtype=np.float64)
            if (self._cached_key is not None and self._cached_result is not None
                and self._cached_key[0][:host_values.nbytes] == host_values.tobytes()):
                return self._cached_result[0]
            n = len(arrays.points)
            values = cp.asarray(host_values)
            result = cp.empty(n, dtype=cp.float64)
            if n:
                self._residuals(((n+127)//128,), (128,),
                                (*self._payload, values, np.int32(n), result))
            host: FloatArray = result.get()
            host.flags.writeable = False
            self.residual_calls += 1
            self.rows += n
            return host

    def linearize(
        self, arrays: _ObservationArrays, transforms: Sequence[SE3], steps: Sequence[float],
    ) -> tuple[FloatArray, FloatArray]:
        """Evaluate all thirteen native differences while retaining their arithmetic."""
        if len(transforms) != 13 or len(steps) != 6:
            raise ValueError("CUDA rig differences require thirteen transforms and six steps")
        if any(step == 0.0 for step in steps):
            raise ZeroDivisionError("division by zero")
        n = len(arrays.points)
        with self._lock, cp.cuda.Device(self.device_id), self._stream:
            payload = self._capture(arrays)
            host_transforms = np.asarray(
                [(*t.translation_m, *t.rotation_quat_xyzw) for t in transforms], dtype=np.float64,
            )
            host_steps = np.asarray(steps, dtype=np.float64)
            key = (host_transforms.tobytes(), host_steps.tobytes())
            if self._cached_key == key and self._cached_result is not None:
                self.linearization_cache_hits += 1
                # Public callers own writable results, including on a cache hit.
                return self._cached_result[0].copy(), self._cached_result[1].copy()
            self._clear_linearization_cache()
            values = cp.asarray(host_transforms)
            device_steps = cp.asarray(host_steps)
            result, jacobian = cp.empty(n, dtype=cp.float64), cp.empty((n, 6), dtype=cp.float64)
            if n:
                self._linearize(((n+127)//128,), (128,),
                    (*payload, values, device_steps, np.int32(n), result, jacobian))
            host_r: FloatArray = result.get()
            host_j: FloatArray = jacobian.get()
            self.linearization_calls += 1
            self.rows += n
            cache_bytes = host_r.nbytes + host_j.nbytes
            if cache_bytes <= _LINEARIZATION_CACHE_BYTES:
                saved_r, saved_j = host_r.copy(), host_j.copy()
                saved_r.flags.writeable = saved_j.flags.writeable = False
                self._cached_result = saved_r, saved_j
                self._cached_key = key
                self.linearization_cache_bytes = cache_bytes
        return host_r, host_j
