"""Optional FP64 CUDA map registration with device-built spatial topology.

Spatial codes, topology and bounds run on device; native trees serve repairs.
Six-dimensional updates remain native. Queries, local
normal covariance/eigensolves, point transforms and robust systems run on GPU.
CuPy is imported only when this optional adapter is requested.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from threading import Lock
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

_SOURCE = r'''
extern "C" __global__ void query3(
    const double* xyz, const double* reference, const double* packed_reference,
    const long long* order,
    const int* axis, const double* split, const int* child,
    const long long* span, const double* boxes, const int n, const int m, const int k,
    double* distance, long long* index) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n) return;
    double p[3] = {xyz[3*row], xyz[3*row+1], xyz[3*row+2]};
    double best[17]; long long chosen[17];
    for(int j=0;j<k;j++){best[j]=1.0/0.0;chosen[j]=m;}
    // 63 Morton bits plus fewer than 30 point-index bits bound radix depth below 96.
    int stack[96]; int top=1;
    stack[0]=0;
    while(top){
        --top; int node=stack[top];
        double d[3];
        for(int a=0;a<3;a++)
            d[a]=fmax(0.0,fmax(boxes[6*node+a]-p[a],p[a]-boxes[6*node+3+a]));
        // Retain equal and roundoff-close boxes; candidate distance arithmetic is unchanged.
        double lower=(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])*(1.0-16.0*2.2204460492503131e-16);
        if(lower > best[0]) continue;
        if(axis[node]<0){
            for(long long j=span[2*node];j<span[2*node+1];j++){
                long long position=order[j];
                double x=p[0]-packed_reference[3*j];
                double y=p[1]-packed_reference[3*j+1];
                double z=p[2]-packed_reference[3*j+2];
                double d=x*x+y*y+z*z;
                if(d<best[0] || (d==best[0] && position<chosen[0])){
                    int slot=0;
                    while(2*slot+1<k){
                        int child=2*slot+1;
                        if(child+1<k && (best[child+1]>best[child] ||
                           (best[child+1]==best[child] && chosen[child+1]>chosen[child]))) ++child;
                        if(best[child]<d || (best[child]==d && chosen[child]<=position)) break;
                        best[slot]=best[child];chosen[slot]=chosen[child];slot=child;
                    }
                    best[slot]=d;chosen[slot]=position;
                }
            }
        }else if(axis[node]==3){
            int left=child[2*node],right=child[2*node+1];
            double lower_child[2];
            for(int side=0;side<2;side++){
                int next=side==0?left:right;double v[3];
                for(int a=0;a<3;a++)
                    v[a]=fmax(0.0,fmax(boxes[6*next+a]-p[a],p[a]-boxes[6*next+3+a]));
                lower_child[side]=(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
                                 *(1.0-16.0*2.2204460492503131e-16);
            }
            int near=lower_child[0]<=lower_child[1]?0:1,far=1-near;
            if(lower_child[far]<=best[0]) stack[top++]=far==0?left:right;
            if(lower_child[near]<=best[0]) stack[top++]=near==0?left:right;
        }else{
            double delta=p[axis[node]]-split[node];
            int side=delta<0?0:1;
            int far=child[2*node+1-side], near=child[2*node+side];
            if(delta*delta <= best[0]){
                stack[top]=far;++top;
            }
            stack[top]=near;++top;
        }
    }
    for(int end=k-1;end>0;--end){
        double d=best[end];long long position=chosen[end];
        best[end]=best[0];chosen[end]=chosen[0];
        int slot=0;
        while(2*slot+1<end){
            int child=2*slot+1;
            if(child+1<end && (best[child+1]>best[child] ||
               (best[child+1]==best[child] && chosen[child+1]>chosen[child]))) ++child;
            if(best[child]<d || (best[child]==d && chosen[child]<=position)) break;
            best[slot]=best[child];chosen[slot]=chosen[child];slot=child;
        }
        best[slot]=d;chosen[slot]=position;
    }
    for(int j=0;j<k;j++){distance[row*k+j]=sqrt(best[j]);index[row*k+j]=chosen[j];}
}

extern "C" __global__ void morton3(
    const double* points, const int n, const double lx, const double ly, const double lz,
    const double hx, const double hy, const double hz, unsigned long long* codes) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    double low[3]={lx,ly,lz},high[3]={hx,hy,hz};
    double width=fmax(high[0]-low[0],fmax(high[1]-low[1],high[2]-low[2]));
    unsigned long long code=0;
    for(int a=0;a<3;a++){
        double fraction=width>0?(points[3*row+a]-low[a])/width:0;
        unsigned int key=(unsigned int)(fmax(0.0,fmin(1.0,fraction))*2097151.0);
        for(int bit=0;bit<21;bit++)
            code|=((unsigned long long)((key>>bit)&1))<<(3*bit+2-a);
    }
    codes[row]=code;
}

// Independent binary-prefix construction, following Karras (HPG 2012):
// https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees
__device__ int common3(const unsigned long long* codes, const long long* order,
                      const int leaves, const int a, const int b) {
    if(b<0 || b>=leaves) return -1;
    unsigned long long bits=codes[order[a]]^codes[order[b]];
    return bits?__clzll(bits):64+__clz((unsigned int)(a^b));
}

extern "C" __global__ void radix_nodes3(
    const unsigned long long* codes, const long long* order, const int leaves,
    int* axis, int* children, int* parents) {
    int node=blockDim.x*blockIdx.x+threadIdx.x;
    if(node>=leaves-1) return;
    int direction=common3(codes,order,leaves,node,node+1)
                  >common3(codes,order,leaves,node,node-1)?1:-1;
    int threshold=common3(codes,order,leaves,node,node-direction);
    int upper=2;
    while(common3(codes,order,leaves,node,node+upper*direction)>threshold) upper*=2;
    int distance=0;
    for(int step=upper/2;step;step/=2)
        if(common3(codes,order,leaves,node,node+(distance+step)*direction)>threshold)
            distance+=step;
    int end=node+distance*direction,first=min(node,end),last=max(node,end);
    int prefix=common3(codes,order,leaves,first,last),middle=first,step=last-first;
    do {
        step=(step+1)/2;
        int candidate=middle+step;
        if(candidate<last && common3(codes,order,leaves,first,candidate)>prefix) middle=candidate;
    } while(step>1);
    int left=middle==first?leaves-1+middle:middle;
    int right=middle+1==last?leaves+middle:middle+1;
    axis[node]=3;children[2*node]=left;children[2*node+1]=right;
    parents[left]=node;parents[right]=node;
}

extern "C" __global__ void leaf_refit3(
    const double* points, const long long* order, const int n, const int leaves,
    int* axis, int* children, const int* parents, int* arrivals,
    long long* spans, double* boxes, double* splits) {
    int leaf=blockDim.x*blockIdx.x+threadIdx.x;
    if(leaf>=leaves) return;
    int node=leaves-1+leaf;
    long long start=leaf,end=start+1;
    if(start>n) start=n;if(end>n) end=n;
    axis[node]=-1;children[2*node]=children[2*node+1]=0;
    spans[2*node]=start;spans[2*node+1]=end;
    for(int a=0;a<3;a++){
        double low=1.0/0.0,high=-1.0/0.0;
        for(long long j=start;j<end;j++){
            double value=points[3*order[j]+a];low=fmin(low,value);high=fmax(high,value);
        }
        boxes[6*node+a]=low;boxes[6*node+3+a]=high;
    }
    volatile double* visible_boxes=boxes;
    volatile long long* visible_spans=spans;
    while(parents[node]>=0){
        __threadfence();
        node=parents[node];
        if(atomicAdd(arrivals+node,1)==0) return;
        int left=children[2*node],right=children[2*node+1];
        visible_spans[2*node]=min(visible_spans[2*left],visible_spans[2*right]);
        visible_spans[2*node+1]=max(visible_spans[2*left+1],visible_spans[2*right+1]);
        for(int a=0;a<3;a++){
            visible_boxes[6*node+a]=fmin(visible_boxes[6*left+a],visible_boxes[6*right+a]);
            visible_boxes[6*node+3+a]=fmax(visible_boxes[6*left+3+a],visible_boxes[6*right+3+a]);
        }
        // Disjoint child boxes admit the cheaper axis-plane traversal. Leaf
        // blocks straddling a Morton prefix retain full box traversal instead.
        double gap=-1;int dimension=-1;bool reverse=false;
        for(int a=0;a<3;a++){
            double forward=visible_boxes[6*right+a]-visible_boxes[6*left+3+a];
            double backward=visible_boxes[6*left+a]-visible_boxes[6*right+3+a];
            if(forward>=0 && forward>gap){gap=forward;dimension=a;reverse=false;}
            if(backward>=0 && backward>gap){gap=backward;dimension=a;reverse=true;}
        }
        if(dimension>=0){
            if(reverse){int swap=left;left=right;right=swap;}
            double low=visible_boxes[6*left+3+dimension],high=visible_boxes[6*right+dimension];
            splits[node]=fmax(low,fmin(high,low+(high-low)*0.5));
            children[2*node]=left;children[2*node+1]=right;axis[node]=dimension;
        }
        if(visible_spans[2*node+1]-visible_spans[2*node]<=8) axis[node]=-1;
    }
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
        self._host = _coordinates(points, "spatial tree")
        if self._host.nbytes * 17 > _MAX_RETAINED_BYTES:
            raise ValueError("CUDA spatial construction exceeds the 1 GiB payload budget")
        self._cpu: cKDTree | None = None
        self._cpu_lock = Lock()
        self.native_tree_builds = 0
        self.data = cp.asarray(self._host)
        self.capture_method = "device_morton_bvh"
        count = len(self._host)
        leaves = max(1, count)
        nodes = 2 * leaves - 1
        axes = cp.empty(nodes, dtype=cp.int32)
        splits = cp.zeros(nodes, dtype=cp.float64)
        children = cp.empty((nodes, 2), dtype=cp.int32)
        spans = cp.empty((nodes, 2), dtype=cp.int64)
        boxes = cp.empty((nodes, 2, 3), dtype=cp.float64)
        codes = cp.empty(count, dtype=cp.uint64)
        parents = cp.full(nodes, -1, dtype=cp.int32)
        arrivals = cp.zeros(leaves - 1, dtype=cp.int32)
        morton = cp.RawKernel(_SOURCE, "morton3", options=("--fmad=false",))
        leaf_kernel = cp.RawKernel(_SOURCE, "leaf_refit3", options=("--fmad=false",))
        topology_kernel = cp.RawKernel(_SOURCE, "radix_nodes3", options=("--fmad=false",))
        self.build_kernel_calls = 0
        if count:
            lows = [np.float64(self._host[:, a].min()) for a in range(3)]
            highs = [np.float64(self._host[:, a].max()) for a in range(3)]
            morton(((count + 127) // 128,), (128,),
                   (self.data, np.int32(count), *lows, *highs, codes))
            self.build_kernel_calls += 1
        order = cp.lexsort(cp.stack((cp.arange(count, dtype=cp.uint64), codes)))
        if leaves > 1:
            topology_kernel(((leaves - 1 + 127) // 128,), (128,),
                            (codes, order, np.int32(leaves), axes, children, parents))
            self.build_kernel_calls += 1
        leaf_kernel(((leaves + 127) // 128,), (128,),
                    (self.data, order, np.int32(count), np.int32(leaves),
                     axes, children, parents, arrivals, spans, boxes, splits))
        self.build_kernel_calls += 1
        active = (parents < 0) | (axes[cp.maximum(parents, 0)] >= 0)
        mapping = cp.cumsum(active, dtype=cp.int32) - 1
        selected = cp.flatnonzero(active)
        # Keep the original data for normal/system indexing and pack the query
        # coordinates in leaf order once. Candidate IDs retain original indices.
        self.packed = cp.ascontiguousarray(self.data[order])
        self.arrays = (order, axes[selected], splits[selected], mapping[children[selected]],
                       spans[selected], boxes[selected])
        self.kernel = cp.RawKernel(_SOURCE, "query3", options=("--fmad=false",))
        self.bytes = int(self.data.nbytes + self.packed.nbytes
                         + sum(array.nbytes for array in self.arrays))
        self.query_calls = 0
        self.query_rows = 0

    @property
    def cpu(self) -> cKDTree:
        """Build the original native tree only for a requested uncertainty repair."""
        with self._cpu_lock:
            if self._cpu is None:
                self._cpu = cKDTree(self._host, copy_data=True)
                self.native_tree_builds += 1
            return self._cpu

    def query(self, points: Any, k: int) -> tuple[Any, Any]:
        if not 1 <= k <= 17:
            raise ValueError("CUDA spatial queries support 1 through 17 neighbors")
        distances = cp.empty((len(points), k), dtype=cp.float64)
        indexes = cp.empty((len(points), k), dtype=cp.int64)
        if len(points):
            self.kernel(((len(points) + 127) // 128,), (128,), (
                points, self.data, self.packed, *self.arrays, np.int32(len(points)),
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
        # A pre-existing native preparation may encode its caller's captured
        # neighborhood policy; retain those normals exactly when present.
        if sample._normals is not None and sample._planar is not None:
            self.normals[:] = cp.asarray(sample._normals)
            self.planar[:] = cp.asarray(sample._planar)
        else:
            k = min(options.normal_neighbours, len(self.maps))
            for start in range(0, len(self.maps), _CHUNK):
                stop = min(start + _CHUNK, len(self.maps))
                distances, indexes = self.tree.query(self.maps[start:stop], k + 1)
                # Equal/roundoff-close boundary neighbors retain SciPy's exact
                # membership/order. Only the uncertain rows cross back to CPU.
                margin = 64 * np.finfo(np.float64).eps
                ambiguous = distances[:, k] - distances[:, k - 1] <= (
                    margin * cp.maximum(1.0, distances[:, k])
                )
                rows = cp.asnumpy(cp.flatnonzero(ambiguous))
                if len(rows):
                    _, native = self.tree.cpu.query(self.maps_host[start + rows], k=k)
                    indexes[cp.asarray(rows), :k] = cp.asarray(native)
                    self.repaired_queries += len(rows)
                neighbors = self.maps[indexes[:, :k]]
                centered = neighbors - neighbors.mean(axis=1, keepdims=True)
                covariance = cp.einsum("nki,nkj->nij", centered, centered) / max(k, 1)
                values, vectors = cp.linalg.eigh(covariance)
                self.normals[start:stop] = vectors[:, :, 0]
                self.planar[start:stop] = values[:, 0] <= options.max_planarity_ratio * values[:, 1]
                # Repair ill-conditioned neighborhoods and planarity boundaries
                # with the original native covariance/eigensolve.
                scale = cp.maximum(cp.abs(values[:, 2]), np.finfo(np.float64).tiny)
                uncertain = (cp.abs(values[:, 1]) <= margin * scale) | (
                    cp.abs(values[:, 0] - options.max_planarity_ratio * values[:, 1])
                    <= margin * scale
                )
                rows = cp.asnumpy(cp.flatnonzero(uncertain))
                if len(rows):
                    native_indices = cp.asnumpy(indexes[cp.asarray(rows), :k])
                    native = self.maps_host[native_indices]
                    centered_host = native - native.mean(axis=1, keepdims=True)
                    matrix = np.einsum("nki,nkj->nij", centered_host, centered_host) / max(k, 1)
                    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
                    self.normals[start + cp.asarray(rows)] = cp.asarray(eigenvectors[:, :, 0])
                    self.planar[start + cp.asarray(rows)] = cp.asarray(
                        eigenvalues[:, 0] <= options.max_planarity_ratio * eigenvalues[:, 1]
                    )
        self.bytes = int(self.tree.bytes + self.points.nbytes + self.pose_device.nbytes
                         + self.normals.nbytes + self.planar.nbytes)

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
            f"query_sha256={hashlib.sha256(_SOURCE.encode()).hexdigest()}"
        )

    def statistics(self) -> dict[str, int]:
        """Report actual device queries and scalar uncertainty repairs."""
        return {
            "retained_input_bytes": self.retained_bytes,
            "prepared_samples": len(self.samples),
            "gpu_query_calls": sum(sample.tree.query_calls for sample in self.samples.values()),
            "gpu_query_rows": sum(sample.tree.query_rows for sample in self.samples.values()),
            "gpu_packed_reference_rows": sum(
                len(sample.tree.packed) for sample in self.samples.values()
            ),
            "gpu_packed_reference_bytes": sum(
                sample.tree.packed.nbytes for sample in self.samples.values()
            ),
            "gpu_tree_builds": len(self.samples),
            "gpu_tree_build_kernel_calls": sum(
                sample.tree.build_kernel_calls for sample in self.samples.values()
            ),
            "native_tree_builds": sum(
                sample.tree.native_tree_builds for sample in self.samples.values()
            ),
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
