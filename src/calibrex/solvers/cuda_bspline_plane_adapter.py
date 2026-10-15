"""Optional FP64 fused spline plane residuals, derivatives and banded products.

The core still owns LM decisions, IMU residuals, the solve and covariance.
One thread owns each band entry, avoiding nondeterministic atomic reductions.
"""

from __future__ import annotations

import hashlib

import cupy as cp
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from calibrex.core.continuous_time_bspline import (
    CubicBSplineTrajectory,
    FloatArray,
    cumulative_basis,
    so3_log_batch,
)
from calibrex.core.continuous_time_bspline_derivatives import _right_jacobian
from calibrex.core.continuous_time_bspline_fit import BSplinePlaneReturns

_PAYLOAD_BUDGET = 1024**3

_SOURCE = r"""
__device__ void multiply(const double* a, const double* b, double* out) {
    for (int i=0;i<3;++i) for (int j=0;j<3;++j)
        out[3*i+j]=(a[3*i]*b[j]+a[3*i+1]*b[3+j])+a[3*i+2]*b[6+j];
}
__device__ void transposed(const double* a, double* out) {
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) out[3*i+j]=a[3*j+i];
}
__device__ void exponential(const double* v, double* step, double* jr) {
    double squared=(v[0]*v[0]+v[1]*v[1])+v[2]*v[2], angle=sqrt(squared);
    double a,b,ja,jb;
    if (angle<1e-8) {a=1.-angle*angle/6.;b=.5-angle*angle/24.;}
    else {a=sin(angle)/angle;b=(1.-cos(angle))/(angle*angle);}
    if (squared<1e-8) {
        ja=.5-squared/24.+squared*squared/720.;
        jb=1./6.-squared/120.+squared*squared/5040.;
    }
    else {ja=(1.-cos(angle))/(angle*angle);jb=(angle-sin(angle))/(angle*angle*angle);}
    double k[9]={0.,-v[2],v[1],v[2],0.,-v[0],-v[1],v[0],0.},kk[9];
    multiply(k,k,kk);
    for(int i=0;i<9;++i) {
        double identity=(i%4==0)?1.:0.;
        step[i]=(identity+a*k[i])+b*kk[i];
        jr[i]=(identity-ja*k[i])+jb*kk[i];
    }
}
extern "C" __global__ void planes(
    const double* rotations,const double* positions,const long long* segment,
    const double* basis,const double* relative,const double* delta,const double* logj,
    const double* points,const double* anchors,const double* normals,
    double sigma,int count,int derivatives,double* residual,double* local) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=count)return;
    int s=segment[row];
    double rotation[9],position[3],steps[27],jrs[27],v[3],scratch[9];
    for(int i=0;i<9;++i)rotation[i]=rotations[9*s+i];
    for(int a=0;a<3;++a)position[a]=positions[3*s+a];
    for(int slot=1;slot<4;++slot) {
        for(int a=0;a<3;++a)v[a]=basis[4*row+slot]*delta[3*(s+slot-1)+a];
        exponential(v,steps+9*(slot-1),jrs+9*(slot-1));
        multiply(rotation,steps+9*(slot-1),scratch);
        for(int i=0;i<9;++i)rotation[i]=scratch[i];
        for(int a=0;a<3;++a)
            position[a]+=basis[4*row+slot]*(positions[3*(s+slot)+a]-positions[3*(s+slot-1)+a]);
    }
    double r=0.;
    for(int a=0;a<3;++a) {
        double world=(rotation[3*a]*points[3*row]+rotation[3*a+1]*points[3*row+1])
                     +rotation[3*a+2]*points[3*row+2];
        r+=normals[3*row+a]*((world+position[a])-anchors[3*row+a]);
    }
    residual[row]=r/sigma;
    if(!derivatives)return;
    double tangent[36]={0.},suffix[9]={1.,0.,0.,0.,1.,0.,0.,0.,1.};
    double scratch2[9],block[9],previous[9],transpose[9];
    for(int slot=3;slot>0;--slot) {
        int edge=s+slot-1;
        multiply(suffix,jrs+9*(slot-1),scratch);
        multiply(scratch,logj+9*edge,scratch2);
        for(int i=0;i<9;++i)block[i]=scratch2[i]*basis[4*row+slot];
        transposed(relative+9*edge,transpose);
        multiply(block,transpose,previous);
        for(int a=0;a<3;++a)for(int b=0;b<3;++b) {
            tangent[12*a+3*slot+b]+=block[3*a+b];
            tangent[12*a+3*(slot-1)+b]-=previous[3*a+b];
        }
        transposed(steps+9*(slot-1),transpose);
        multiply(suffix,transpose,scratch);
        for(int i=0;i<9;++i)suffix[i]=scratch[i];
    }
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)tangent[12*a+b]+=suffix[3*a+b];
    double x=points[3*row],y=points[3*row+1],z=points[3*row+2];
    double k[9]={0.,-z,y,z,0.,-x,-y,x,0.},world_derivative[9];
    multiply(rotation,k,world_derivative);
    for(int i=0;i<9;++i)world_derivative[i]=-world_derivative[i];
    for(int slot=0;slot<4;++slot) {
        double weight=basis[4*row+slot]-(slot<3?basis[4*row+slot+1]:0.);
        for(int axis=0;axis<3;++axis) {
            local[24*row+6*slot+axis]=(normals[3*row+axis]/sigma)*weight;
            double d=0.;
            for(int a=0;a<3;++a) {
                double value=(world_derivative[3*a]*tangent[3*slot+axis]
                             +world_derivative[3*a+1]*tangent[12+3*slot+axis])
                             +world_derivative[3*a+2]*tangent[24+3*slot+axis];
                d+=normals[3*row+a]*value;
            }
            local[24*row+6*slot+3+axis]=d/sigma;
        }
    }
}

extern "C" __global__ void normal_system(const long long* row_ids,const long long* bounds,
    const double* local,const double* residual,int controls,double* band,double* gradient) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=controls*7*36)return;
    int axis_b=i%6,axis_a=(i/6)%6,offset=(i/36)%7-3,a=i/(36*7),b=a+offset;
    double value=0.,g=0.;
    if(b>=0 && b<controls) {
        int lo=max(0,max(a,b)-3), hi=min(controls-4,min(a,b));
        for(int s=lo;s<=hi;++s) for(long long p=bounds[s];p<bounds[s+1];++p) {
            long long row=row_ids[p];
            value+=local[24*row+6*(a-s)+axis_a]*local[24*row+6*(b-s)+axis_b];
        }
    }
    band[i]=value;
    if(offset==0 && axis_b==0) {
        int lo=max(0,a-3),hi=min(controls-4,a);
        for(int s=lo;s<=hi;++s) for(long long p=bounds[s];p<bounds[s+1];++p) {
            long long row=row_ids[p];
            g+=local[24*row+6*(a-s)+axis_a]*residual[row];
        }
        gradient[6*a+axis_a]=g;
    }
}
"""


class _CudaPlaneEvaluator:
    """Capture immutable measurements and one fit's bounded device workspace."""

    def __init__(
        self, trajectory: CubicBSplineTrajectory, planes: BSplinePlaneReturns,
    ) -> None:
        self.segment, fraction = trajectory.segments(planes.times_sec)
        controls = trajectory.control_point_count
        # Include retained host/device outputs and conservative transfer scratch.
        if len(self.segment) * 640 + controls * 7 * 36 * 48 > _PAYLOAD_BUDGET:
            raise ValueError("CUDA spline planes exceed the 1 GiB payload budget")
        self._domain = (trajectory.start_time_sec, trajectory.knot_spacing_sec, controls)
        self._edges = np.unique(self.segment[:, None] + np.arange(3)[None, :])
        self._device = int(cp.cuda.runtime.getDevice())
        self._sigma = planes.sigma_m
        self._trajectory: CubicBSplineTrajectory | None = None
        self._has_derivatives = False
        self._host_residual = np.empty(0)
        self._host_local = np.empty((0, 24))
        self._statistics = {
            "gpu_plane_calls": 0, "gpu_plane_rows": 0,
            "gpu_plane_derivative_calls": 0, "gpu_plane_derivative_rows": 0,
            "gpu_plane_normal_calls": 0,
        }
        with cp.cuda.Device(self._device):
            self._segment, self._basis, self._points, self._anchors, self._normals = (
                cp.asarray(np.ascontiguousarray(value, dtype=dtype))
                for value, dtype in (
                    (self.segment, np.int64),
                    (cumulative_basis(fraction, trajectory.knot_spacing_sec)[0], np.float64),
                    (planes.points_body_m, np.float64),
                    (planes.plane_points_world_m, np.float64),
                    (planes.plane_normals_world, np.float64),
                )
            )
            self._row_ids = cp.asarray(
                np.argsort(self.segment, kind="stable").astype(np.int64)
            )
            self._bounds = cp.asarray(np.concatenate((
                [0], np.cumsum(np.bincount(self.segment, minlength=controls - 3)),
            )).astype(np.int64))
            module = cp.RawModule(code=_SOURCE, options=("--fmad=false",))
            self._kernel = module.get_function("planes")
            self._normal_kernel = module.get_function("normal_system")
            self._residual = cp.empty(len(self.segment), dtype=cp.float64)
            self._local = cp.empty((len(self.segment), 24), dtype=cp.float64)
            self._band = cp.empty((controls, 7, 6, 6), dtype=cp.float64)
            self._gradient = cp.empty(6 * controls, dtype=cp.float64)
        shape = (controls, 7, 6, 6)
        index = np.arange(controls)[:, None] + np.arange(-3, 4)[None, :]
        self._valid = np.broadcast_to(
            ((index >= 0) & (index < controls))[:, :, None, None], shape,
        )
        self._rows = np.broadcast_to(
            6 * np.arange(controls)[:, None, None, None] + np.arange(6)[None, None, :, None],
            shape,
        )[self._valid]
        self._columns = np.broadcast_to(
            6 * index[:, :, None, None] + np.arange(6)[None, None, None, :], shape,
        )[self._valid]
        self.identity = (
            "calibrex.cuda_bspline_planes/v0.1;fp64=true;fma=false;"
            "normal=deterministic_banded;imu=numpy;covariance=numpy;"
            f"sha256={hashlib.sha256(_SOURCE.encode()).hexdigest()}"
        )

    def evaluate(
        self, trajectory: CubicBSplineTrajectory, *, derivatives: bool,
    ) -> tuple[FloatArray, FloatArray]:
        """Evaluate referenced edges on CPU, and all per-return work on device."""
        domain = (trajectory.start_time_sec, trajectory.knot_spacing_sec,
                  trajectory.control_point_count)
        if domain != self._domain:
            raise ValueError("CUDA plane schedule requires the initial spline time domain")
        if self._trajectory is trajectory and (not derivatives or self._has_derivatives):
            return self._host_residual, self._host_local
        # Only referenced edges enter Log: an unused near-pi edge stays unused.
        relative = np.zeros((trajectory.control_point_count - 1, 3, 3))
        relative[self._edges] = np.einsum(
            "tji,tjk->tik", trajectory.rotations[self._edges],
            trajectory.rotations[self._edges + 1],
        )
        delta = np.zeros((len(relative), 3))
        delta[self._edges] = so3_log_batch(relative[self._edges])
        log_jacobian = np.zeros_like(relative)
        if derivatives:
            log_jacobian[self._edges] = _right_jacobian(delta[self._edges], inverse=True)
        self._trajectory = None
        with cp.cuda.Device(self._device):
            inputs = tuple(cp.asarray(np.ascontiguousarray(value, dtype=np.float64))
                           for value in (trajectory.rotations, trajectory.positions,
                                         relative, delta, log_jacobian))
            rotations, positions, relative_device, delta_device, log_device = inputs
            count = len(self.segment)
            self._kernel(((count + 127) // 128,), (128,), (
                rotations, positions, self._segment, self._basis,
                relative_device, delta_device, log_device,
                self._points, self._anchors, self._normals, np.float64(self._sigma),
                np.int32(count), np.int32(derivatives), self._residual, self._local,
            ))
            residual = cp.asnumpy(self._residual)
            local = cp.asnumpy(self._local) if derivatives else np.empty((0, 24))
        # Publish the cache only after the synchronized downloads succeed.
        self._trajectory, self._has_derivatives = trajectory, derivatives
        self._host_residual, self._host_local = residual, local
        self._statistics["gpu_plane_calls"] += 1
        self._statistics["gpu_plane_rows"] += count
        self._statistics["gpu_plane_derivative_calls"] += int(derivatives)
        self._statistics["gpu_plane_derivative_rows"] += count * int(derivatives)
        return residual, local

    def normal_system(
        self, trajectory: CubicBSplineTrajectory, imu_normal: csc_matrix,
        imu_gradient: FloatArray,
    ) -> tuple[csc_matrix, FloatArray]:
        """Reduce the 7-block control band directly, without a large sparse J."""
        self.evaluate(trajectory, derivatives=True)
        controls = trajectory.control_point_count
        with cp.cuda.Device(self._device):
            self._normal_kernel(((self._band.size + 127) // 128,), (128,), (
                self._row_ids, self._bounds, self._local, self._residual,
                np.int32(controls), self._band, self._gradient,
            ))
            band, plane_gradient = cp.asnumpy(self._band), cp.asnumpy(self._gradient)
        normal = coo_matrix(
            (band[self._valid], (self._rows, self._columns)), shape=imu_normal.shape,
        ).tocsc()
        gradient = imu_gradient.copy()
        gradient[:6 * controls] += plane_gradient
        self._statistics["gpu_plane_normal_calls"] += 1
        return (normal + imu_normal).tocsc(), gradient

    def statistics(self) -> dict[str, int]:
        """Return actual kernel work for the fit's execution provenance."""
        return self._statistics.copy()
