"""Optional device-resident point-to-plane registration for LiDAR odometry."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import cupy as cp
import numpy as np

from calibrex.solvers.cuda_lidar_map_adapter import (
    _SOURCE,
    _coordinates,
    _CudaTree,
    _host,
)
from calibrex.solvers.scan_to_scan_odometry import (
    FloatArray,
    ScanOdometryOptions,
    ScanRegistration,
    _exp_se3,
    _norm3,
    _normals_from_neighbours,
    _quantize_geometry,
    register_point_to_plane,
)

_SYSTEM_SOURCE = r"""
extern "C" __global__ void system3(
    const double* moved, const double* target, const double* normal,
    const bool* planar, const double* distances, const long long* indices,
    const int n, const int m, const double gate, const double huber,
    double* terms) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    long long hit=indices[2*row];
    bool valid=distances[2*row]<gate && hit<m;
    if(valid) valid=planar[hit];
    double j[6]={0,0,0,0,0,0}, r=0, w=1;
    if(valid){
        const double* p=moved+3*row; const double* q=target+3*hit;
        const double* a=normal+3*hit;
        r=(p[0]-q[0])*a[0]+(p[1]-q[1])*a[1]+(p[2]-q[2])*a[2];
        j[0]=p[1]*a[2]-p[2]*a[1];j[1]=p[2]*a[0]-p[0]*a[2];
        j[2]=p[0]*a[1]-p[1]*a[0];j[3]=a[0];j[4]=a[1];j[5]=a[2];
        w=fabs(r)<=huber?1.0:huber/fmax(fabs(r),1e-12);
    }
    int column=0;
    for(int a=0;a<6;a++) for(int b=a;b<6;b++)
        terms[(column++)*n+row]=(j[a]*w)*j[b];
    for(int a=0;a<6;a++) terms[(column++)*n+row]=(j[a]*w)*r;
    terms[27*n+row]=r*r; terms[28*n+row]=valid?1.0:0.0;
}
"""


_COVARIANCE_SOURCE = r"""
extern "C" __global__ void covariance3(
    const double* points, const long long* indices, const int n, const int k,
    double* covariance) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    double mean[3]={0,0,0};
    for(int j=0;j<k;j++) for(int a=0;a<3;a++) mean[a]+=points[3*indices[row*(k+1)+j]+a];
    for(int a=0;a<3;a++) mean[a]/=k;
    double c[9]={0,0,0,0,0,0,0,0,0};
    for(int j=0;j<k;j++){
        double d[3];for(int a=0;a<3;a++) d[a]=points[3*indices[row*(k+1)+j]+a]-mean[a];
        for(int a=0;a<3;a++) for(int b=0;b<3;b++) c[3*a+b]+=d[a]*d[b];
    }
    for(int j=0;j<9;j++) covariance[9*row+j]=c[j]/k;
}
"""

_FUSED_SOURCE = r"""
extern "C" __global__ void project_query_system3(
    const double* xyz, const double* matrix, const double* reference,
    const long long* order, const int* axis, const double* split,
    const int* child, const long long* span, const double* normal,
    const bool* planar, const int n, const int m, const double gate,
    const double huber, const double error, double* uncertain, double* terms) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    const double* s=xyz+3*row;
    double p[3];
    for(int a=0;a<3;a++)
        p[a]=matrix[4*a]*s[0]+matrix[4*a+1]*s[1]+matrix[4*a+2]*s[2]+matrix[4*a+3];
    double best[2]={1.0/0.0,1.0/0.0};long long chosen[2]={m,m};
    int stack[64];double bound[64];int top=1;stack[0]=0;bound[0]=0;
    while(top){
        --top;int node=stack[top];
        if(bound[top]>best[1]) continue;
        if(axis[node]<0){
            for(long long v=span[2*node];v<span[2*node+1];v++){
                long long position=order[v];
                double x=p[0]-reference[3*position];
                double y=p[1]-reference[3*position+1];
                double z=p[2]-reference[3*position+2];
                double d=x*x+y*y+z*z;
                if(d<best[1] || (d==best[1] && position<chosen[1])){
                    if(d<best[0] || (d==best[0] && position<chosen[0])){
                        best[1]=best[0];chosen[1]=chosen[0];best[0]=d;chosen[0]=position;
                    }else{best[1]=d;chosen[1]=position;}
                }
            }
        }else{
            double delta=p[axis[node]]-split[node];int side=delta<0?0:1;
            if(delta*delta<=best[1]){
                stack[top]=child[2*node+1-side];bound[top]=delta*delta;++top;
            }
            stack[top]=child[2*node+side];bound[top]=0;++top;
        }
    }
    double distance=sqrt(best[0]);long long hit=chosen[0];
    bool repair=sqrt(best[1])-distance<=error || fabs(distance-gate)<=error;
    uncertain[row]=repair?1.0:0.0;
    bool valid=distance<gate && hit<m;
    if(valid) valid=planar[hit];
    double j[6]={0,0,0,0,0,0},r=0,w=1;
    if(valid){
        const double* q=reference+3*hit;const double* a=normal+3*hit;
        r=(p[0]-q[0])*a[0]+(p[1]-q[1])*a[1]+(p[2]-q[2])*a[2];
        j[0]=p[1]*a[2]-p[2]*a[1];j[1]=p[2]*a[0]-p[0]*a[2];
        j[2]=p[0]*a[1]-p[1]*a[0];j[3]=a[0];j[4]=a[1];j[5]=a[2];
        w=fabs(r)<=huber?1.0:huber/fmax(fabs(r),1e-12);
    }
    int column=0;
    for(int a=0;a<6;a++) for(int b=a;b<6;b++)
        terms[(column++)*n+row]=(j[a]*w)*j[b];
    for(int a=0;a<6;a++) terms[(column++)*n+row]=(j[a]*w)*r;
    terms[27*n+row]=r*r;terms[28*n+row]=valid?1.0:0.0;
    terms[29*n+row]=repair?1.0:0.0;
}
"""


def execution_identity() -> str:
    """Identify arithmetic, query code and runtime without the device model."""
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    system_digest = hashlib.sha256(
        (_SYSTEM_SOURCE + _FUSED_SOURCE + _COVARIANCE_SOURCE).encode(),
    ).hexdigest()
    return (
        "calibrex.cuda_scan_odometry/v0.1;fp64=true;query_fma=false;"
        f"cupy={cp.__version__};cuda={cp.cuda.runtime.runtimeGetVersion()};"
        f"driver={cp.cuda.runtime.driverGetVersion()};"
        f"cc={properties['major']}.{properties['minor']};"
        f"query_sha256={hashlib.sha256(_SOURCE.encode()).hexdigest()};"
        f"system_sha256={system_digest}"
    )


class CudaOdometryTarget:
    """Capture one target and reuse its tree/normals across deskew rounds.

    Correspondences, normal PCA, Jacobians and robust systems stay on
    device. Only the 6x6 system, count and squared-error sum cross back per
    iteration. Native small solves, stopping policy and pose updates remain
    on CPU; roundoff-sensitive neighbour/planarity decisions use native repairs.
    """

    def __init__(self, target: FloatArray, options: ScanOdometryOptions) -> None:
        if not 2 <= options.normal_neighbours <= 16:
            raise ValueError("CUDA normal neighborhoods require 2 through 16 neighbors")
        if not np.isfinite(options.max_correspondence_m) or options.max_correspondence_m <= 0:
            raise ValueError("CUDA correspondence gate must be finite and positive")
        self.options = options
        self.device_id = cp.cuda.runtime.getDevice()
        self.host = _coordinates(
            _quantize_geometry(target, options.geometry_quantum_m), "odometry target",
        )
        if len(self.host) < 3:
            raise ValueError("CUDA odometry targets require at least three points")
        if self.host.nbytes * 8 > 1024**3:
            raise ValueError("CUDA odometry target exceeds the 1 GiB payload budget")
        with cp.cuda.Device(self.device_id):
            self.tree = _CudaTree(self.host)
            self.normals = cp.zeros_like(self.tree.data)
            self.planar = cp.zeros(len(self.host), dtype=cp.bool_)
            self.done = cp.zeros(len(self.host), dtype=cp.bool_)
        self.scale = max(1.0, float(np.max(np.abs(self.host))))
        self.query_repairs = 0
        self.normal_repairs = 0
        self.fused_calls = 0
        self.registration_repairs = 0
        self.identity = execution_identity()
        with cp.cuda.Device(self.device_id):
            self.system_kernel = cp.RawKernel(_SYSTEM_SOURCE, "system3", options=("--fmad=false",))
            self.fused_kernel = cp.RawKernel(
                _FUSED_SOURCE,
                "project_query_system3",
                options=("--fmad=false",),
            )
            self.covariance_kernel = cp.RawKernel(
                _COVARIANCE_SOURCE,
                "covariance3",
                options=("--fmad=false",),
            )
            self._ensure(cp.arange(len(self.host)))

    def _ensure(self, hit: cp.ndarray) -> None:
        missing = cp.unique(hit[~self.done[hit]])
        k = min(self.options.normal_neighbours, len(self.host))
        margin = 64 * np.finfo(np.float64).eps
        for start in range(0, len(missing), 65536):
            rows = missing[start : start + 65536]
            distances, indexes = self.tree.query(self.tree.data[rows], k + 1)
            ambiguous = distances[:, k] - distances[:, k - 1] <= (
                margin * cp.maximum(1.0, distances[:, k])
            )
            repair = (
                cp.asnumpy(cp.flatnonzero(ambiguous))
                if int(cp.count_nonzero(ambiguous))
                else np.empty(0, dtype=np.intp)
            )
            if len(repair):
                native_rows = cp.asnumpy(rows[cp.asarray(repair)])
                _, native = self.tree.cpu.query(self.host[native_rows], k=k)
                indexes[cp.asarray(repair), :k] = cp.asarray(native)
                self.query_repairs += len(repair)
            covariance = cp.empty((len(rows), 3, 3), dtype=cp.float64)
            self.covariance_kernel(
                ((len(rows) + 127) // 128,),
                (128,),
                (
                    self.tree.data,
                    indexes,
                    np.int32(len(rows)),
                    np.int32(k),
                    covariance,
                ),
            )
            values, vectors = cp.linalg.eigh(covariance)
            ratio = self.options.max_planarity_ratio
            self.normals[rows] = vectors[:, :, 0]
            self.planar[rows] = values[:, 0] <= ratio * cp.maximum(values[:, 1], 1e-12)
            scale = cp.maximum(cp.abs(values[:, 2]), np.finfo(np.float64).tiny)
            # Thin scan-line neighbourhoods can amplify eigensolver roundoff
            # in the normal direction even away from the planarity threshold.
            uncertain = (values[:, 1] - values[:, 0] <= 1e-4 * scale) | (
                cp.abs(values[:, 0] - ratio * cp.maximum(values[:, 1], 1e-12)) <= margin * scale
            )
            repair = (
                cp.asnumpy(cp.flatnonzero(uncertain))
                if int(cp.count_nonzero(uncertain))
                else np.empty(0, dtype=np.intp)
            )
            if len(repair):
                native_rows = cp.asnumpy(rows[cp.asarray(repair)])
                _, native_indices = self.tree.cpu.query(self.host[native_rows], k=k)
                self.query_repairs += len(repair)
                normals, planar = _normals_from_neighbours(
                    self.host,
                    native_indices,
                    k,
                    self.options,
                )
                selected = rows[cp.asarray(repair)]
                self.normals[selected] = cp.asarray(normals)
                self.planar[selected] = cp.asarray(planar)
                self.normal_repairs += len(repair)
            self.done[rows] = True

    def register(self, source: FloatArray, initial: FloatArray) -> ScanRegistration:
        """Run native odometry iteration/stopping semantics on this captured target."""
        host_source = _coordinates(
            _quantize_geometry(source, self.options.geometry_quantum_m), "odometry source",
        )
        if self.host.nbytes * 8 + host_source.nbytes * 12 > 1024**3:
            raise ValueError("CUDA odometry inputs exceed the 1 GiB payload budget")
        transform = np.array(initial, dtype=np.float64, copy=True)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("CUDA initial odometry poses require finite 4x4 matrices")
        opts = self.options
        count, iteration, last_count = 0, 0, 0
        converged, rmse = False, math.inf
        last_hessian: FloatArray | None = None
        uncertain_stopping = False
        scale = max(self.scale, float(np.max(np.abs(host_source))) if len(host_source) else 1.0)
        with cp.cuda.Device(self.device_id):
            points = cp.asarray(host_source)
            terms = cp.empty((30, len(points)), dtype=cp.float64)
            uncertain = cp.empty(len(points), dtype=cp.float64)
            for iteration in range(1, opts.max_iterations + 1):  # noqa: B007
                matrix = cp.asarray(transform)
                error = (
                    128
                    * np.finfo(np.float64).eps
                    * scale
                    * max(
                        1.0,
                        float(np.linalg.norm(transform, ord=1)),
                    )
                )
                if len(points):
                    self.fused_kernel(
                        ((len(points) + 127) // 128,),
                        (128,),
                        (
                            points,
                            matrix,
                            self.tree.data,
                            *self.tree.arrays,
                            self.normals,
                            self.planar,
                            np.int32(len(points)),
                            np.int32(len(self.host)),
                            np.float64(opts.max_correspondence_m),
                            np.float64(opts.huber_delta_m),
                            np.float64(error),
                            uncertain,
                            terms,
                        ),
                    )
                    self.tree.query_calls += 1
                    self.tree.query_rows += len(points)
                    self.fused_calls += 1
                totals = _host(cp.sum(terms, axis=1))
                if totals[29]:
                    rows = cp.asnumpy(cp.flatnonzero(uncertain))
                    # Recompute after rare native gate/tie repairs. Ordinary
                    # iterations synchronize only for the reduced system.
                    moved = points @ matrix[:3, :3].T + matrix[:3, 3]
                    distances, indexes = self.tree.query(moved, 2)
                    native_moved = host_source[rows] @ transform[:3, :3].T + transform[:3, 3]
                    distance, index = self.tree.cpu.query(
                        native_moved,
                        k=1,
                        distance_upper_bound=opts.max_correspondence_m,
                    )
                    positions = cp.asarray(rows)
                    distances[positions, 0] = cp.asarray(distance)
                    indexes[positions, 0] = cp.asarray(index)
                    moved[positions] = cp.asarray(native_moved)
                    self.query_repairs += len(rows)
                    self.system_kernel(
                        ((len(points) + 127) // 128,),
                        (128,),
                        (
                            moved,
                            self.tree.data,
                            self.normals,
                            self.planar,
                            distances,
                            indexes,
                            np.int32(len(points)),
                            np.int32(len(self.host)),
                            np.float64(opts.max_correspondence_m),
                            np.float64(opts.huber_delta_m),
                            terms,
                        ),
                    )
                    totals = _host(cp.sum(terms[:29], axis=1))
                count = int(totals[28])
                if count < opts.min_correspondences:
                    break
                upper = np.triu_indices(6)
                hessian = np.zeros((6, 6))
                hessian[upper] = totals[:21]
                hessian[(upper[1], upper[0])] = totals[:21]
                gradient = totals[21:27]
                rmse = math.sqrt(float(totals[27]) / count)
                try:
                    delta = -np.linalg.solve(hessian, gradient)
                except np.linalg.LinAlgError:
                    break
                last_hessian, last_count = hessian, count
                for norm, tolerance in (
                    (_norm3(delta[3:]), opts.convergence_translation_m),
                    (_norm3(delta[:3]), opts.convergence_rotation_rad),
                ):
                    uncertain_stopping |= abs(norm - tolerance) <= 1e-3 * tolerance
                transform = _exp_se3(delta) @ transform
                if (
                    _norm3(delta[3:]) < opts.convergence_translation_m
                    and _norm3(delta[:3]) < opts.convergence_rotation_rad
                ):
                    converged = True
                    break
        if uncertain_stopping:
            native = register_point_to_plane(host_source, self.host, initial, opts)
            self.registration_repairs += 1
            return replace(native, execution_backend=self.identity)
        eigenvalue = (
            0.0 if last_hessian is None else float(np.linalg.eigvalsh(last_hessian)[0]) / last_count
        )
        return ScanRegistration(
            transform,
            converged and count >= opts.min_correspondences,
            iteration,
            count,
            rmse,
            eigenvalue,
            execution_backend=self.identity,
        )

    def statistics(self) -> dict[str, int]:
        """Report actual work, including the two kinds of native repairs."""
        return {
            "gpu_query_calls": self.tree.query_calls,
            "gpu_query_rows": self.tree.query_rows,
            "scalar_query_repairs": self.query_repairs,
            "scalar_normal_repairs": self.normal_repairs,
            "gpu_fused_system_calls": self.fused_calls,
            "scalar_registration_repairs": self.registration_repairs,
        }
