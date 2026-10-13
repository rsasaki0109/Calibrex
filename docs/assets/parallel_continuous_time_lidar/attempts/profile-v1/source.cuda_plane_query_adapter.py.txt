"""Optional exact CUDA associations for captured continuous-time plane maps."""

from __future__ import annotations

import hashlib
import math
from threading import Lock
from typing import Any

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from calibrex.data.adaptive_voxel import AdaptivePlaneQueryContext
from calibrex.data.livox import VoxelPlaneQueryContext

_SOURCE = r'''
extern "C" __global__ void uniform_positions(
    const double* xyz, const long long* cells, const long long* lower,
    const long long* upper, const long long wy, const long long wz,
    const long long* offsets, const long long* positions, const double* centroids,
    const bool* valid, const double gate, const int n, long long* result) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row >= n) return;
    result[row] = -1;
    for(int a=0;a<3;a++)
        if(cells[3*row+a]<lower[a] || cells[3*row+a]>upper[a]) return;
    long long cell = ((cells[3*row]-lower[0])*wy + cells[3*row+1]-lower[1])*wz
                    + cells[3*row+2]-lower[2];
    double best = 1.0/0.0; long long chosen = -1;
    for(long long i=offsets[cell];i<offsets[cell+1];i++) {
        long long p=positions[i];
        double x=xyz[3*row]-centroids[3*p];
        double y=xyz[3*row+1]-centroids[3*p+1];
        double z=xyz[3*row+2]-centroids[3*p+2];
        double d=sqrt(x*x+y*y+z*z);
        if(d<best) {best=d;chosen=p;}
    }
    if(chosen>=0 && !(best>gate) && valid[chosen]) result[row]=chosen;
}
extern "C" __global__ void adaptive_neighbors(
    const double* xyz, const double* reference, const long long* order,
    const int* axis, const double* split, const int* child, const long long* span,
    const int n, const int m, double* distance, long long* index) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    double p[3]={xyz[3*row],xyz[3*row+1],xyz[3*row+2]};
    double best[2]={1.0/0.0,1.0/0.0}; long long chosen[2]={m,m};
    int stack[64]; double bound[64]; int top=1;
    stack[0]=0;bound[0]=0;
    while(top) {
        --top;int node=stack[top];
        if(bound[top]>best[1]) continue;
        if(axis[node]<0) {
            for(long long j=span[2*node];j<span[2*node+1];j++) {
                long long position=order[j];
                double x=p[0]-reference[3*position];
                double y=p[1]-reference[3*position+1];
                double z=p[2]-reference[3*position+2];
                double d=x*x+y*y+z*z;
                if(d<best[1] || (d==best[1] && position<chosen[1])) {
                    int slot=1;
                    if(d<best[0] || (d==best[0] && position<chosen[0])) {
                        best[1]=best[0];chosen[1]=chosen[0];slot=0;
                    }
                    best[slot]=d;chosen[slot]=position;
                }
            }
        } else {
            double delta=p[axis[node]]-split[node];int side=delta<0?0:1;
            if(delta*delta<=best[1]) {
                stack[top]=child[2*node+1-side];bound[top]=delta*delta;++top;
            }
            stack[top]=child[2*node+side];bound[top]=0;++top;
        }
    }
    for(int j=0;j<2;j++) {distance[2*row+j]=sqrt(best[j]);index[2*row+j]=chosen[j];}
}
'''

_MAX_MAP_BYTES = 64 * 1024**2
_MAX_QUERY_BYTES = 128 * 1024**2


def cuda_plane_query_source_sha256() -> str:
    """Identify the exact association kernels without device model information."""
    return hashlib.sha256(_SOURCE.encode()).hexdigest()


def _tree_columns(tree: Any) -> tuple[NDArray[Any], ...] | None:
    axes: list[int] = []
    splits: list[float] = []
    children: list[list[int]] = []
    spans: list[list[int]] = []

    def append(node: Any, depth: int = 0) -> int:
        if depth >= 63:
            raise ValueError("CUDA plane tree exceeds its bounded traversal stack")
        index = len(axes)
        axes.append(node.split_dim)
        splits.append(node.split)
        children.append([0, 0])
        spans.append([node.start_idx, node.end_idx])
        if node.split_dim >= 0:
            children[index] = [append(node.lesser, depth+1), append(node.greater, depth+1)]
        return index

    try:
        append(tree.tree)
    except ValueError:
        return None
    return (
        np.array(tree.data, dtype=np.float64, order="C", copy=True),
        np.asarray(tree.indices, dtype=np.int64), np.asarray(axes, dtype=np.int32),
        np.asarray(splits, dtype=np.float64), np.asarray(children, dtype=np.int32),
        np.asarray(spans, dtype=np.int64),
    )


class CudaPlaneQueries:
    """Own bounded immutable map payloads and serialize one private CUDA stream."""

    @classmethod
    def capture(
        cls, uniform: VoxelPlaneQueryContext | None, adaptive: AdaptivePlaneQueryContext | None,
    ) -> CudaPlaneQueries | None:
        """Capture supported numeric maps; unsupported maps retain CPU queries."""
        columns: tuple[NDArray[Any], ...] = ()
        widths = (0, 0, 0)
        if uniform is not None:
            if type(uniform) is not VoxelPlaneQueryContext:
                return None
            index = uniform._index
            if index is None or index.neighborhoods is None or not uniform._native_geometry:
                return None
            neighbors = index.neighborhoods
            widths = neighbors.widths
            columns = (neighbors.lower, neighbors.upper, neighbors.offsets, neighbors.positions,
                       index.centroids, index.valid_normal)
        tree_columns: tuple[NDArray[Any], ...] = ()
        if adaptive is not None:
            if (type(adaptive) is not AdaptivePlaneQueryContext or adaptive._tree is None
                or not adaptive._native_geometry or adaptive._tree.data.nbytes > _MAX_MAP_BYTES):
                return None
            captured = _tree_columns(adaptive._tree)
            if captured is None:
                return None
            tree_columns = captured
        if not columns and not tree_columns:
            return None
        if sum(a.nbytes for a in (*columns, *tree_columns)) > _MAX_MAP_BYTES:
            return None
        return cls(uniform, adaptive, columns, widths, tree_columns)

    def __init__(
        self, uniform: VoxelPlaneQueryContext | None, adaptive: AdaptivePlaneQueryContext | None,
        columns: tuple[NDArray[Any], ...], widths: tuple[int, int, int],
        tree_columns: tuple[NDArray[Any], ...],
    ) -> None:
        self.uniform, self.adaptive = uniform, adaptive
        self.device_id = cp.cuda.runtime.getDevice()
        self._lock = Lock()
        self._widths = widths
        self.query_calls = self.uniform_rows = self.adaptive_rows = 0
        self.query_upload_bytes = 0
        self.map_upload_bytes = sum(a.nbytes for a in (*columns, *tree_columns))
        with cp.cuda.Device(self.device_id):
            self._stream = cp.cuda.Stream(non_blocking=True)
            with self._stream:
                self._uniform = tuple(cp.asarray(a) for a in columns)
                self._adaptive = tuple(cp.asarray(a) for a in tree_columns)
                self._uniform_kernel = cp.RawKernel(
                    _SOURCE, "uniform_positions", options=("--fmad=false",),
                )
                self._adaptive_kernel = cp.RawKernel(
                    _SOURCE, "adaptive_neighbors", options=("--fmad=false",),
                )
                self._uniform_kernel.compile()
                self._adaptive_kernel.compile()
            self._stream.synchronize()

    def query(
        self, points: NDArray[np.float64], *, voxel_size_m: float, correspondence_gate_m: float,
    ) -> tuple[NDArray[np.int64] | None, NDArray[np.int64] | None] | None:
        """Upload coordinates once, then repair uncertain adaptive neighbors on CPU."""
        n, gate = len(points), correspondence_gate_m
        if (n < 64 or n*96 > _MAX_QUERY_BYTES or points.shape != (n, 3)
            or points.dtype != np.float64 or not np.isfinite(points).all()
            or np.any(np.abs(points) >= 1e100)
            or (self.adaptive is not None and (not math.isfinite(gate) or gate <= 0.0))):
            return None
        cells = None
        if self.uniform is not None:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                floating = np.floor(points/voxel_size_m)
            if (not np.isfinite(floating).all() or np.any(floating < -(2.0**63)+2048)
                or np.any(floating >= 2.0**63-2048)):
                return None
            cells = floating.astype(np.int64)
        positions = distances = indexes = None
        with self._lock, cp.cuda.Device(self.device_id), self._stream:
            xyz = cp.asarray(points, order="C")
            if cells is not None:
                device_cells = cp.asarray(cells)
                result = cp.empty(n, dtype=cp.int64)
                lower, upper, offsets, lookup, centroids, valid = self._uniform
                self._uniform_kernel(((n+127)//128,), (128,), (
                    xyz, device_cells, lower, upper, np.int64(self._widths[1]),
                    np.int64(self._widths[2]), offsets, lookup, centroids, valid,
                    np.float64(gate), np.int32(n), result,
                ))
                positions = result.get()
                self.uniform_rows += n
            if self.adaptive is not None:
                device_distances = cp.empty((n, 2), dtype=cp.float64)
                device_indexes = cp.empty((n, 2), dtype=cp.int64)
                reference, *topology = self._adaptive
                self._adaptive_kernel(((n+127)//128,), (128,), (
                    xyz, reference, *topology, np.int32(n), np.int32(len(reference)),
                    device_distances, device_indexes,
                ))
                distances, indexes = device_distances.get(), device_indexes.get()
                self.adaptive_rows += n
            self.query_calls += 1
            self.query_upload_bytes += points.nbytes + (cells.nbytes if cells is not None else 0)
        adaptive_positions = None
        if self.adaptive is not None:
            assert distances is not None and indexes is not None
            adaptive_positions = self.adaptive._positions_from_neighbors(
                points, distances, indexes, gate,
            )
        return positions, adaptive_positions
