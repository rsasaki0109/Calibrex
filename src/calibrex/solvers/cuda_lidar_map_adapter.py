"""Optional FP64 CUDA map registration with captured SciPy spatial topology.

Tree construction and six-dimensional updates remain native. Queries, local
normal covariance/eigensolves, point transforms and robust systems run on GPU.
CuPy is imported only when this optional adapter is requested.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import cupy as cp
import numpy as np
from scipy.spatial import cKDTree

from calibrex.solvers.lidar_lidar_map_solver import (
    _NATIVE_MAP_RESIDUALS,
    FloatArray,
    MapSample,
    MapSampleOptions,
)
from calibrex.solvers.scipy_spatial_snapshot import capture_spatial_tree

_SOURCE = r'''
extern "C" __global__ void query3(
    const double* xyz, const double* reference, const long long* order,
    const int* axis, const double* split, const int* child,
    const long long* span, const double* boxes, const int n, const int m, const int k,
    double* distance, long long* index) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n) return;
    double p[3] = {xyz[3*row], xyz[3*row+1], xyz[3*row+2]};
    double best[17]; long long chosen[17];
    for(int j=0;j<k;j++){best[j]=1.0/0.0;chosen[j]=m;}
    int stack[64]; int top=1;
    stack[0]=0;
    while(top){
        --top; int node=stack[top];
        double d[3];
        for(int a=0;a<3;a++)
            d[a]=fmax(0.0,fmax(boxes[6*node+a]-p[a],p[a]-boxes[6*node+3+a]));
        // Retain equal and roundoff-close boxes; candidate distance arithmetic is unchanged.
        double lower=(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])*(1.0-16.0*2.2204460492503131e-16);
        if(lower > best[k-1]) continue;
        if(axis[node]<0){
            for(long long j=span[2*node];j<span[2*node+1];j++){
                long long position=order[j];
                double x=p[0]-reference[3*position];
                double y=p[1]-reference[3*position+1];
                double z=p[2]-reference[3*position+2];
                double d=x*x+y*y+z*z;
                if(d<best[k-1] || (d==best[k-1] && position<chosen[k-1])){
                    int slot=k-1;
                    while(slot>0 && (d<best[slot-1] ||
                         (d==best[slot-1] && position<chosen[slot-1]))){
                        best[slot]=best[slot-1];chosen[slot]=chosen[slot-1];--slot;
                    }
                    best[slot]=d;chosen[slot]=position;
                }
            }
        }else{
            double delta=p[axis[node]]-split[node];
            int side=delta<0?0:1;
            int far=child[2*node+1-side], near=child[2*node+side];
            if(delta*delta <= best[k-1]){
                stack[top]=far;++top;
            }
            stack[top]=near;++top;
        }
    }
    for(int j=0;j<k;j++){distance[row*k+j]=sqrt(best[j]);index[row*k+j]=chosen[j];}
}
'''

_NORMAL_SOURCE = r'''

// Map PCA uses ratio*mid with no eigenvalue floor. Native repairs retain
// ambiguous neighborhoods, eigenspaces and planarity boundaries.
extern "C" __global__ void map_normals3(
    const double* points, const long long* indices, const double* distances,
    const long long* rows, const int n, const int k, const double ratio,
    double* normal, bool* planar, bool* done, unsigned char* repair) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    const double eps=2.2204460492503131e-16;
    double mean[3]={0,0,0},c[9]={0,0,0,0,0,0,0,0,0};
    for(int j=0;j<k;j++) for(int a=0;a<3;a++)
        mean[a]+=points[3*indices[row*(k+1)+j]+a];
    for(int a=0;a<3;a++) mean[a]/=k;
    for(int j=0;j<k;j++) {
        double d[3];
        for(int a=0;a<3;a++) d[a]=points[3*indices[row*(k+1)+j]+a]-mean[a];
        for(int a=0;a<3;a++) for(int b=0;b<3;b++) c[3*a+b]+=d[a]*d[b];
    }
    for(int a=0;a<9;a++) c[a]/=k;
    double scale=fmax(fmax(fabs(c[0]),fabs(c[4])),fabs(c[8]));
    bool bad=!(scale>0) || !isfinite(scale);
    double unit=fmax(scale,2.2250738585072014e-308);
    for(int a=0;a<9;a++) c[a]/=unit;
    double trace=c[0]+c[4]+c[8],q=trace/3;
    double x=c[0]-q,y=c[4]-q,z=c[8]-q;
    double p=sqrt((x*x+y*y+z*z+2*(c[1]*c[1]+c[2]*c[2]+c[5]*c[5]))/6);
    bad|=!(p>64*eps);
    double inv=1/fmax(p,2.2250738585072014e-308);
    double b0=x*inv,b1=y*inv,b2=z*inv;
    double b3=c[1]*inv,b4=c[2]*inv,b5=c[5]*inv;
    double determinant=b0*b1*b2+2*b3*b4*b5-b0*b5*b5-b1*b4*b4-b2*b3*b3;
    double phi=acos(fmax(-1.0,fmin(1.0,determinant/2)))/3;
    double low=q+2*p*cos(phi+2.0943951023931954923);
    double high=q+2*p*cos(phi),mid=trace-low-high;
    double a[9];for(int j=0;j<9;j++) a[j]=c[j];
    a[0]-=low;a[4]-=low;a[8]-=low;
    double best=-1,v[3]={0,0,0};
    for(int i=0;i<3;i++) {
        int j=(i+1)%3;
        double w[3]={a[3*i+1]*a[3*j+2]-a[3*i+2]*a[3*j+1],
                     a[3*i+2]*a[3*j]-a[3*i]*a[3*j+2],
                     a[3*i]*a[3*j+1]-a[3*i+1]*a[3*j]};
        double norm=w[0]*w[0]+w[1]*w[1]+w[2]*w[2];
        if(norm>best) {best=norm;for(int u=0;u<3;u++) v[u]=w[u];}
    }
    bad|=!(best>0) || !isfinite(best);
    double length=sqrt(fmax(best,2.2250738585072014e-308));
    for(int u=0;u<3;u++) v[u]/=length;
    double residual=0;
    for(int i=0;i<3;i++) {
        double r=c[3*i]*v[0]+c[3*i+1]*v[1]+c[3*i+2]*v[2]-low*v[i];
        residual+=r*r;
    }
    double spectral=fmax(fabs(high),2.2250738585072014e-308);
    double threshold=ratio*mid;
    // Cover closed-form root error near repeated upper eigenvalues.
    bad|=fabs(mid)<=64*eps*spectral;
    bad|=mid-low<=1e-4*spectral || fabs(low-threshold)<=2e-8*spectral;
    bad|=!isfinite(residual) || sqrt(residual)>64*eps*spectral;
    bool tied=distances[row*(k+1)+k]-distances[row*(k+1)+k-1]
              <=64*eps*fmax(1.0,distances[row*(k+1)+k]);
    long long target=rows[row];
    for(int u=0;u<3;u++) normal[3*target+u]=v[u];
    planar[target]=low<=threshold;
    done[target]=true;
    repair[row]=(bad?1:0)|(tied?2:0);
}
'''

_CHUNK = 65536
_MAX_RETAINED_BYTES = 1024 * 1024 * 1024


def _host(values: Any) -> FloatArray:
    return np.asarray(cp.asnumpy(values), dtype=np.float64)


def _coordinates(values: FloatArray, name: str) -> FloatArray:
    captured = np.array(values, dtype=np.float64, order="C", copy=True)
    if (captured.ndim != 2 or captured.shape[1] != 3 or len(captured) >= 2**30
        or not np.isfinite(captured).all() or np.any(np.abs(captured) >= 1e50)):
        raise ValueError(f"CUDA {name} requires finite bounded (N, 3) coordinates")
    captured.flags.writeable = False
    return captured


class _CudaTree:
    def __init__(self, points: FloatArray) -> None:
        self.cpu = cKDTree(points, copy_data=True)
        self.data = cp.asarray(points)
        topology, bounds, self.capture_method = capture_spatial_tree(self.cpu)
        axes, splits, children, spans = topology
        self.arrays = (
            cp.asarray(self.cpu.indices, dtype=cp.int64), cp.asarray(axes, dtype=cp.int32),
            cp.asarray(splits), cp.asarray(children, dtype=cp.int32),
            cp.asarray(spans, dtype=cp.int64), cp.asarray(bounds),
        )
        self.kernel = cp.RawKernel(_SOURCE, "query3", options=("--fmad=false",))
        self.bytes = int(self.data.nbytes + sum(array.nbytes for array in self.arrays))
        self.query_calls = 0
        self.query_rows = 0

    def query(self, points: Any, k: int) -> tuple[Any, Any]:
        if not 1 <= k <= 17:
            raise ValueError("CUDA spatial queries support 1 through 17 neighbors")
        distances = cp.empty((len(points), k), dtype=cp.float64)
        indexes = cp.empty((len(points), k), dtype=cp.int64)
        if len(points):
            self.kernel(((len(points) + 127) // 128,), (128,), (
                points, self.data, *self.arrays, np.int32(len(points)),
                np.int32(len(self.data)), np.int32(k), distances, indexes,
            ))
            self.query_calls += 1
            self.query_rows += len(points)
        return distances, indexes


class _CudaSample:
    def __init__(self, sample: MapSample, options: MapSampleOptions) -> None:
        self.source = sample
        self.pose = np.array(sample.reference_pose, dtype=np.float64, copy=True)
        if self.pose.shape != (4, 4) or not np.isfinite(self.pose).all():
            raise ValueError("CUDA reference poses require finite 4x4 matrices")
        self.pose.flags.writeable = False
        self.points_host = _coordinates(sample.target_points, "target")
        self.maps_host = _coordinates(sample.map_points, "map")
        if len(self.maps_host) < 3:
            raise ValueError("CUDA maps require at least three points")
        self.coordinate_scale = max(
            1.0, float(np.max(np.abs(self.maps_host))),
            float(np.max(np.abs(self.points_host))) if len(self.points_host) else 0.0,
        )
        self.tree = _CudaTree(self.maps_host)
        self.maps = self.tree.data
        self.points = cp.asarray(self.points_host)
        self.pose_device = cp.asarray(self.pose)
        self.normals = cp.empty_like(self.maps)
        self.planar = cp.empty(len(self.maps), dtype=cp.bool_)
        self.repaired_queries = 0
        self.normal_rows = 0
        self.normal_kernel_calls = 0
        self.normal_repairs = 0
        self.normal_neighbours = min(options.normal_neighbours, len(self.maps))
        self.max_planarity_ratio = options.max_planarity_ratio
        self.normal_kernel = cp.RawKernel(_NORMAL_SOURCE, "map_normals3", options=("--fmad=false",))
        # A pre-existing native preparation may encode its caller's captured
        # neighborhood policy; retain those normals exactly when present.
        if sample._normals is not None and sample._planar is not None:
            self.normals[:] = cp.asarray(sample._normals)
            self.planar[:] = cp.asarray(sample._planar)
            self.done = cp.ones(len(self.maps), dtype=cp.bool_)
        else:
            self.done = cp.zeros(len(self.maps), dtype=cp.bool_)
        self.bytes = int(self.tree.bytes + self.points.nbytes + self.pose_device.nbytes
                         + self.normals.nbytes + self.planar.nbytes + self.done.nbytes)

    def _ensure(self, requested: Any) -> None:
        """Compute each requested normal once within the captured map snapshot."""
        rows = cp.unique(requested[~self.done[requested]])
        k = self.normal_neighbours
        for start in range(0, len(rows), _CHUNK):
            selected = rows[start:start + _CHUNK]
            distances, indexes = self.tree.query(self.maps[selected], k + 1)
            flags = cp.empty(len(selected), dtype=cp.uint8)
            self.normal_kernel(((len(selected) + 127) // 128,), (128,), (
                self.maps, indexes, distances, selected, np.int32(len(selected)), np.int32(k),
                np.float64(self.max_planarity_ratio), self.normals, self.planar, self.done, flags,
            ))
            self.normal_kernel_calls += 1
            native_flags = cp.asnumpy(flags)
            repair = np.flatnonzero(native_flags)
            try:
                if len(repair):
                    # Re-query the original tree, retaining native membership/order
                    # at ties as well as its covariance and degenerate eigenspaces.
                    host_rows = cp.asnumpy(selected[cp.asarray(repair)])
                    _, positions = self.tree.cpu.query(self.maps_host[host_rows], k=k)
                    neighbors = self.maps_host[positions]
                    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
                    matrix = np.einsum("nki,nkj->nij", centered, centered) / k
                    values, vectors = np.linalg.eigh(matrix)
                    self.normals[cp.asarray(host_rows)] = cp.asarray(vectors[:, :, 0])
                    self.planar[cp.asarray(host_rows)] = cp.asarray(
                        values[:, 0] <= self.max_planarity_ratio * values[:, 1],
                    )
            except Exception:
                self.done[selected] = False
                raise
            self.normal_rows += len(selected)
            self.normal_repairs += len(repair)
            self.repaired_queries += int(np.count_nonzero(native_flags & 2))

    def residuals(self, extrinsic: FloatArray, gate: float) -> tuple[Any, Any, Any]:
        if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
            raise ValueError("CUDA extrinsics require finite 4x4 matrices")
        transform = self.pose @ extrinsic
        matrix = cp.asarray(transform)
        world = self.points @ matrix[:3, :3].T + matrix[:3, 3]
        distances, indexes = self.tree.query(world, 2)
        # Account conservatively for distance arithmetic and transform roundoff.
        error = (128 * np.finfo(np.float64).eps * self.coordinate_scale
                 * max(1.0, np.linalg.norm(transform, 1)))
        ambiguous = (distances[:, 1] - distances[:, 0] <= 2 * error) | (
            cp.abs(distances[:, 0] - gate) <= error
        )
        rows = cp.asnumpy(cp.flatnonzero(ambiguous))
        if len(rows):
            moving = (self.points_host[rows] @ transform[:3, :3].T + transform[:3, 3])
            native_distance, native_index = self.tree.cpu.query(moving, distance_upper_bound=gate)
            indexes[cp.asarray(rows), 0] = cp.asarray(native_index)
            distances[cp.asarray(rows), 0] = cp.asarray(native_distance)
            world[cp.asarray(rows)] = cp.asarray(moving)
            self.repaired_queries += len(rows)
        valid = (distances[:, 0] < gate) & (indexes[:, 0] < len(self.maps))
        self._ensure(indexes[valid, 0])
        valid[valid] &= self.planar[indexes[valid, 0]]
        selected = indexes[valid, 0]
        normals = self.normals[selected]
        residual = cp.sum((world[valid] - self.maps[selected]) * normals, axis=1)
        return residual, normals, self.points[valid]


class CudaMapRuntime:
    """Own one bounded GPU snapshot for a solve or complete blocked evaluation."""

    def __init__(self, samples: Sequence[MapSample], options: MapSampleOptions) -> None:
        if not 2 <= options.normal_neighbours <= 16:
            raise ValueError("CUDA normal neighborhoods require 2 through 16 neighbors")
        if not np.isfinite(options.max_correspondence_m) or options.max_correspondence_m <= 0:
            raise ValueError("CUDA correspondence gate must be finite and positive")
        if (MapSample.residuals is not _NATIVE_MAP_RESIDUALS
            or any(type(sample) is not MapSample or "residuals" in sample.__dict__
                   for sample in samples)):
            raise ValueError("CUDA requires native MapSample residual semantics")
        self.options = options
        self.device_id = cp.cuda.runtime.getDevice()
        self.samples: dict[int, _CudaSample] = {}
        self.retained_bytes = 0
        with cp.cuda.Device(self.device_id):
            for sample in samples:
                # Preflight the main payloads before constructing a new snapshot.
                estimate = sample.map_points.size * 32 + sample.target_points.size * 16
                if self.retained_bytes + estimate > _MAX_RETAINED_BYTES:
                    raise ValueError("CUDA map inputs exceed the 1 GiB retained-payload limit")
                prepared = _CudaSample(sample, options)
                if self.retained_bytes + prepared.bytes > _MAX_RETAINED_BYTES:
                    raise ValueError("CUDA map inputs exceed the 1 GiB retained-payload limit")
                self.samples[id(sample)] = prepared
                self.retained_bytes += prepared.bytes

    def identity(self) -> str:
        """Identify code/arithmetic/runtime without publishing a device model name."""
        properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        capability = f"{properties['major']}.{properties['minor']}"
        return (
            "calibrex.cuda_lidar_map/v0.1;fp64=true;query_fma=false;"
            f"cupy={cp.__version__};cuda={cp.cuda.runtime.runtimeGetVersion()};"
            f"driver={cp.cuda.runtime.driverGetVersion()};cc={capability};"
            f"query_sha256={hashlib.sha256(_SOURCE.encode()).hexdigest()};"
            f"normal_sha256={hashlib.sha256(_NORMAL_SOURCE.encode()).hexdigest()};normal_cache=lazy"
        )

    def statistics(self) -> dict[str, int]:
        """Report actual device queries and scalar uncertainty repairs."""
        return {
            "retained_input_bytes": self.retained_bytes,
            "prepared_samples": len(self.samples),
            "gpu_normal_rows": sum(sample.normal_rows for sample in self.samples.values()),
            "gpu_normal_kernel_calls": sum(
                sample.normal_kernel_calls for sample in self.samples.values()
            ),
            "scalar_normal_repairs": sum(sample.normal_repairs for sample in self.samples.values()),
            "gpu_query_calls": sum(sample.tree.query_calls for sample in self.samples.values()),
            "gpu_query_rows": sum(sample.tree.query_rows for sample in self.samples.values()),
            "scalar_query_repairs": sum(
                sample.repaired_queries for sample in self.samples.values()
            ),
        }

    def _sample(self, sample: MapSample) -> _CudaSample:
        captured = self.samples.get(id(sample))
        if captured is None or captured.source is not sample:
            raise ValueError("sample does not belong to this CUDA snapshot")
        return captured

    def residuals(self, sample: MapSample, extrinsic: FloatArray) -> FloatArray:
        """Return source-ordered residuals for native holdout and control scoring."""
        with cp.cuda.Device(self.device_id):
            values, _, _ = self._sample(sample).residuals(
                extrinsic, self.options.max_correspondence_m,
            )
            return _host(values)

    def normal_equations(
        self, samples: Sequence[MapSample], extrinsic: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Keep correspondence, analytic Jacobian and Huber products on device."""
        with cp.cuda.Device(self.device_id):
            hessian = cp.zeros((6, 6), dtype=cp.float64)
            gradient = cp.zeros(6, dtype=cp.float64)
            parts = []
            transform = cp.asarray(extrinsic)
            for sample in samples:
                prepared = self._sample(sample)
                residual, normals, points = prepared.residuals(
                    extrinsic, self.options.max_correspondence_m,
                )
                if not len(residual):
                    continue
                reference = points @ transform[:3, :3].T + transform[:3, 3]
                normal_reference = normals @ prepared.pose_device[:3, :3]
                jacobian = cp.column_stack([
                    cp.cross(reference, normal_reference), normal_reference,
                ])
                weight = cp.minimum(
                    1.0, self.options.huber_m / cp.maximum(cp.abs(residual), 1e-12),
                )
                hessian += jacobian.T @ (jacobian * weight[:, None])
                gradient += jacobian.T @ (weight * residual)
                parts.append(residual)
            residuals = cp.concatenate(parts) if parts else cp.empty(0, dtype=cp.float64)
            return _host(hessian), _host(gradient), _host(residuals)
