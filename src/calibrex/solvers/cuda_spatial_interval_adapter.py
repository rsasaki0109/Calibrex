"""Optional conservative FP32 pruning around unchanged FP64 spatial candidates."""

from __future__ import annotations

import cupy as cp
import numpy as np

from calibrex.solvers.cuda_lidar_map_adapter import _CudaTree
from calibrex.solvers.scan_to_scan_odometry import FloatArray

_SOURCE = r"""
extern "C" __global__ void query3(
    const double* xyz, const double* reference, const long long* order,
    const int* axis, const float* split, const int* child,
    const long long* span, const float* boxes, const float* intervals,
    const int n, const int m, const int k,
    double* distance, long long* index) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n) return;
    double p[3] = {xyz[3*row], xyz[3*row+1], xyz[3*row+2]};
    double best[17]; long long chosen[17];
    for(int j=0;j<k;j++){best[j]=1.0/0.0;chosen[j]=m;}
    float pl[3],ph[3];
    for(int a=0;a<3;a++){pl[a]=__double2float_rd(p[a]);ph[a]=__double2float_ru(p[a]);}
    int stack[64]; int top=1;
    stack[0]=0;
    while(top){
        --top; int node=stack[top];
        float d[3];
        for(int a=0;a<3;a++)
            d[a]=fmaxf(0.0f,fmaxf(__fsub_rd(boxes[6*node+a],ph[a]),__fsub_rd(pl[a],boxes[6*node+3+a])));
        float lower=__fmul_rd(__fadd_rd(__fadd_rd(__fmul_rd(d[0],d[0]),
                          __fmul_rd(d[1],d[1])),__fmul_rd(d[2],d[2])),0.9999998807907104f);
        if(!isfinite(lower))lower=0.0f;
        if((double)lower>best[k-1]) continue;
        if(axis[node]<0){
            for(long long j=span[2*node];j<span[2*node+1];j++){
                long long position=order[j];
                float lower_point[3];
                for(int a=0;a<3;a++) lower_point[a]=fmaxf(0.0f,fmaxf(
                    __fsub_rd(intervals[6*position+a],ph[a]),
                    __fsub_rd(pl[a],intervals[6*position+3+a])));
                float candidate=__fmul_rd(__fadd_rd(__fadd_rd(
                    __fmul_rd(lower_point[0],lower_point[0]),
                    __fmul_rd(lower_point[1],lower_point[1])),
                    __fmul_rd(lower_point[2],lower_point[2])),0.9999998807907104f);
                if(isfinite(candidate) && (double)candidate>best[k-1])continue;
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
            int coordinate=axis[node];
            float delta=fmaxf(0.0f,fmaxf(__fsub_rd(split[2*node],ph[coordinate]),
                                       __fsub_rd(pl[coordinate],split[2*node+1])));
            float separation=__fmul_rd(__fmul_rd(delta,delta),0.9999998807907104f);
            if(!isfinite(separation))separation=0.0f;
            
            int side=pl[coordinate]<split[2*node]?0:1;
            int far=child[2*node+1-side], near=child[2*node+side];
            if((double)separation <= best[k-1]){
                stack[top]=far;++top;
            }
            stack[top]=near;++top;
        }
    }
    for(int j=0;j<k;j++){distance[row*k+j]=sqrt(best[j]);index[row*k+j]=chosen[j];}
}

// Directed conversions enclose the original FP64 coordinates and split planes.
extern "C" __global__ void prepare_intervals3(
    const double* points, const double* splits, const double* boxes,
    const int n, const int nodes, float* point_intervals,
    float* split_intervals, float* box_intervals) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row<n) for(int a=0;a<3;a++) {
        point_intervals[6*row+a]=__double2float_rd(points[3*row+a]);
        point_intervals[6*row+3+a]=__double2float_ru(points[3*row+a]);
    }
    if(row<nodes) {
        split_intervals[2*row]=__double2float_rd(splits[row]);
        split_intervals[2*row+1]=__double2float_ru(splits[row]);
        for(int a=0;a<3;a++) {
            box_intervals[6*row+a]=__double2float_rd(boxes[6*row+a]);
            box_intervals[6*row+3+a]=__double2float_ru(boxes[6*row+3+a]);
        }
    }
}
"""


class _CudaIntervalTree(_CudaTree):
    """Retain a native tree and optionally prepare bounded outward intervals.

    Intervals only reject candidates using lower bounds. Point distances, tie
    ordering and returned values use the original FP64 arithmetic. Keeping the
    native arrays also permits wide inputs to use the original kernels.
    """

    def __init__(self, points: FloatArray) -> None:
        self._ready: cp.cuda.Event | None = None
        self.interval_arrays: tuple[cp.ndarray, ...] | None = None
        self.use_intervals = False
        self.interval_query_rows = 0
        super().__init__(points)
        # Limit temporary ownership and avoid intervals that erase a tiny cloud
        # translated far from the origin. Unsupported targets keep the native path.
        scale = max(1.0, float(np.max(np.abs(points))))
        extent = float(np.max(np.ptp(points, axis=0)))
        if (not 32 <= len(points) <= 65536 or scale > 1e6
                or extent <= 256 * np.finfo(np.float32).eps * scale):
            return
        self._device_id = int(cp.cuda.runtime.getDevice())
        self._build_stream = cp.cuda.get_current_stream()
        order, axes, splits, children, spans, boxes = self.arrays
        split_intervals = cp.empty((len(splits), 2), dtype=cp.float32)
        box_intervals = cp.empty(boxes.shape, dtype=cp.float32)
        point_intervals = cp.empty((len(points), 6), dtype=cp.float32)
        # Own all outputs before submitting writes, including the failure path.
        self.interval_arrays = (
            order, axes, split_intervals, children, spans, box_intervals, point_intervals,
        )
        self.interval_kernel = cp.RawKernel(_SOURCE, "query3", options=("--fmad=false",))
        prepare = cp.RawKernel(_SOURCE, "prepare_intervals3", options=("--fmad=false",))
        try:
            count = max(len(points), len(splits))
            prepare(((count + 127) // 128,), (128,), (
                self.data, splits, boxes, np.int32(len(points)), np.int32(len(splits)),
                point_intervals, split_intervals, box_intervals,
            ))
            ready = cp.cuda.Event()
            ready.record()
            self._ready = ready
        except Exception:
            self._build_stream.synchronize()
            raise
        self.bytes += split_intervals.nbytes + box_intervals.nbytes + point_intervals.nbytes
        self.use_intervals = True

    def __del__(self) -> None:
        ready = getattr(self, "_ready", None)
        if ready is not None:
            try:
                with cp.cuda.Device(self._device_id):
                    ready.synchronize()
            except Exception:
                pass  # The CUDA context may already be closed during shutdown.

    def select_intervals(self, enabled: bool) -> bool:
        """Select eligible prepared arrays and order cross-stream readers."""
        self.use_intervals = enabled and self.interval_arrays is not None
        if self.use_intervals:
            stream = cp.cuda.get_current_stream()
            if stream.ptr != self._build_stream.ptr:
                stream.wait_event(self._ready)
        return self.use_intervals

    def query(self, points: cp.ndarray, k: int) -> tuple[cp.ndarray, cp.ndarray]:
        """Keep candidate distances and sorted neighbor membership unchanged."""
        if not self.use_intervals:
            return super().query(points, k)
        if not 1 <= k <= 17:
            raise ValueError("CUDA spatial queries support 1 through 17 neighbors")
        self.select_intervals(True)
        distances = cp.empty((len(points), k), dtype=cp.float64)
        indexes = cp.empty((len(points), k), dtype=cp.int64)
        if len(points):
            assert self.interval_arrays is not None
            self.interval_kernel(((len(points) + 127) // 128,), (128,), (
                points, self.data, *self.interval_arrays, np.int32(len(points)),
                np.int32(len(self.data)), np.int32(k), distances, indexes,
            ))
            self.query_calls += 1
            self.query_rows += len(points)
            self.interval_query_rows += len(points)
        return distances, indexes
