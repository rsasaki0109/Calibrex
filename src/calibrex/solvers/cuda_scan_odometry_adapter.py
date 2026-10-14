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
from calibrex.solvers.imu_lidar_rotation_solver import _GyroPointInputs, _GyroRotationModel
from calibrex.solvers.scan_to_scan_odometry import (
    FloatArray,
    ScanOdometryOptions,
    ScanRegistration,
    _canonicalize_motion,
    _exp_se3,
    _norm3,
    _normals_from_neighbours,
    _quantize_geometry,
    preprocess_scan,
    register_point_to_plane,
)

_PREPROCESS_SOURCE = r"""
extern "C" __global__ void crop_keys3(
    const double* input, const int n, const double quantum, const double voxel,
    const double minimum, const double maximum, double* points,
    long long* keys, unsigned char* valid) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    double p[3];
    for(int a=0;a<3;a++){
        p[a]=input[3*row+a];
        if(quantum>0) p[a]=nearbyint(p[a]/quantum)*quantum;
        points[3*row+a]=p[a];
        keys[3*row+a]=(long long)floor(p[a]/voxel);
    }
    double range=sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
    valid[row]=(range>=minimum && range<=maximum)?1:0;
}

extern "C" __global__ void voxel_centers3(
    const double* points, const long long* order, const long long* starts,
    const int n, const int groups, const double quantum, double* output) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=groups) return;
    long long begin=starts[row],end=row+1<groups?starts[row+1]:n;
    double sum[3]={0,0,0};
    // Each voxel retains original input order, matching native bincount/add.at.
    for(long long j=begin;j<end;j++)
        for(int a=0;a<3;a++) sum[a]+=points[3*order[j]+a];
    for(int a=0;a<3;a++){
        double value=sum[a]/(double)(end-begin);
        if(quantum>0) value=nearbyint(value/quantum)*quantum;
        output[3*row+a]=value;
    }
}

__device__ void gyro_matrix_vector3(const double* a,const double* x,double* y,bool transpose=false){
    for(int i=0;i<3;i++) y[i]=(transpose?a[i]:a[3*i])*x[0]
        +(transpose?a[3+i]:a[3*i+1])*x[1]+(transpose?a[6+i]:a[3*i+2])*x[2];
}
__device__ void gyro_rotvec_apply3(const double* v,const double* p,double* out){
    double angle=sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    double square=angle*angle;
    double scale=angle<=1e-3?0.5-square/48.0+(square*square)/3840.0:sin(0.5*angle)/angle;
    double x=v[0]*scale,y=v[1]*scale,z=v[2]*scale,w=cos(0.5*angle);
    double matrix[9]={x*x-y*y-z*z+w*w,2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),-x*x+y*y-z*z+w*w,2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),-x*x-y*y+z*z+w*w};
    gyro_matrix_vector3(matrix,p,out);
}
extern "C" __global__ void gyro_rotate3(
    const double* points,const double* query,const double* times,
    const double* orientations,const double* rates,const double* projected,
    const double* mounting,const double* back,const double* start_sum,
    const int n,const int m,double* output){
    int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=n)return;
    int low=0,high=m;
    while(low<high){
        int middle=low+(high-low)/2;
        if(times[middle]<=query[row])low=middle+1;else high=middle;
    }
    int index=max(0,min(m-2,low-1));
    double elapsed=query[row]-times[index],fraction=elapsed/(times[index+1]-times[index]);
    double p[3],partial[3],world[3],corrected[3],v[3];
    for(int a=0;a<3;a++)p[a]=points[3*row+a];
    for(int a=0;a<3;a++)v[a]=rates[3*index+a]*elapsed;
    gyro_rotvec_apply3(v,p,partial);gyro_matrix_vector3(orientations+9*index,partial,world);
    for(int a=0;a<3;a++)v[a]=-(projected[3*index+a]
        +fraction*(projected[3*(index+1)+a]-projected[3*index+a])-start_sum[a]);
    gyro_rotvec_apply3(v,world,corrected);gyro_matrix_vector3(back,corrected,output+3*row);
}

extern "C" __global__ void gyro_translate3(
    const double* rotated,const double* fraction,const double* translation,
    const int n,const double quantum,const double scale,double* source,unsigned char* flags){
    int row=blockDim.x*blockIdx.x+threadIdx.x;if(row>=n)return;
    bool sensitive=false;
    for(int a=0;a<3;a++){
        double value=rotated[3*row+a]+fraction[row]*translation[a];
        source[3*row+a]=value;
        double bound=2048.0*2.2204460492503131e-16
            *fmax(scale,fmax(fabs(rotated[3*row+a]),fabs(fraction[row]*translation[a])));
        double units=value/quantum;
        sensitive|=!isfinite(value) || fabs(units-nearbyint(units))>=0.5-bound/quantum;
    }
    flags[row]=sensitive?1:0;
}
"""

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

_NORMAL_SOURCE = r"""
// FP64 symmetric 3x3 PCA; uncertain memberships/eigenspaces retain native repairs.
extern "C" __global__ void normals3(
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
    double floor=1e-12/unit,spectral=fmax(fabs(high),2.2250738585072014e-308);
    double threshold=ratio*fmax(mid,floor);
    // Cover closed-form root error near repeated upper eigenvalues.
    bad|=mid-low<=1e-4*spectral || fabs(low-threshold)<=2e-8*spectral;
    bad|=!isfinite(residual) || sqrt(residual)>64*eps*spectral;
    bool tied=distances[row*(k+1)+k]-distances[row*(k+1)+k-1]
              <=64*eps*fmax(1.0,distances[row*(k+1)+k]);
    long long target=rows[row];
    for(int u=0;u<3;u++) normal[3*target+u]=v[u];
    planar[target]=low<=threshold;
    done[target]=true;
    repair[row]=(bad||tied)?1:0;
}
"""


_FUSED_SOURCE = r"""
extern "C" __global__ void project_query_system3(
    const double* xyz, const double* matrix, const double* reference,
    const long long* order, const int* axis, const double* split,
    const int* child, const long long* span, const double* boxes, const double* normal,
    const bool* planar, const bool* done, const int n, const int m, const double gate,
    const double huber, const double error, double* uncertain, double* terms,
    double* moved_out, double* distances_out, long long* indices_out,
    long long* pending_out) {
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if(row>=n) return;
    const double* s=xyz+3*row;
    double p[3];
    for(int a=0;a<3;a++)
        p[a]=matrix[4*a]*s[0]+matrix[4*a+1]*s[1]+matrix[4*a+2]*s[2]+matrix[4*a+3];
    double best[2]={1.0/0.0,1.0/0.0};long long chosen[2]={m,m};
    int stack[96];int top=1;stack[0]=0;
    while(top){
        --top;int node=stack[top];
        double d[3];
        for(int a=0;a<3;a++)
            d[a]=fmax(0.0,fmax(boxes[6*node+a]-p[a],p[a]-boxes[6*node+3+a]));
        double lower=(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])*(1.0-16.0*2.2204460492503131e-16);
        if(lower>best[1]) continue;
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
            if(lower_child[far]<=best[1]) stack[top++]=far==0?left:right;
            if(lower_child[near]<=best[1]) stack[top++]=near==0?left:right;
        }else{
            double delta=p[axis[node]]-split[node];int side=delta<0?0:1;
            if(delta*delta<=best[1]){
                stack[top]=child[2*node+1-side];++top;
            }
            stack[top]=child[2*node+side];++top;
        }
    }
    double distance=sqrt(best[0]);long long hit=chosen[0];
    for(int a=0;a<3;a++) moved_out[3*row+a]=p[a];
    distances_out[2*row]=distance;distances_out[2*row+1]=sqrt(best[1]);
    indices_out[2*row]=hit;indices_out[2*row+1]=chosen[1];
    bool repair=sqrt(best[1])-distance<=error || fabs(distance-gate)<=error;
    uncertain[row]=repair?1.0:0.0;
    bool valid=distance<gate && hit<m;
    bool pending=valid && !done[hit];
    pending_out[row]=pending?hit:-1;
    if(valid) valid=done[hit] && planar[hit];
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
    terms[30*n+row]=pending?1.0:0.0;
}
"""


def execution_identity() -> str:
    """Identify arithmetic, query code and runtime without the device model."""
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    system_digest = hashlib.sha256(
        (_SYSTEM_SOURCE + _FUSED_SOURCE + _COVARIANCE_SOURCE + _NORMAL_SOURCE
         + _PREPROCESS_SOURCE).encode(),
    ).hexdigest()
    return (
        "calibrex.cuda_scan_odometry/v0.1;fp64=true;query_fma=false;"
        f"cupy={cp.__version__};cuda={cp.cuda.runtime.runtimeGetVersion()};"
        f"driver={cp.cuda.runtime.driverGetVersion()};"
        f"cc={properties['major']}.{properties['minor']};"
        f"query_sha256={hashlib.sha256(_SOURCE.encode()).hexdigest()};"
        f"system_sha256={system_digest}"
    )


class CudaScanPreprocessor:
    """Exact source-ordered FP64 voxel means with native unsupported-input fallback."""

    def __init__(self) -> None:
        self.device_id = cp.cuda.runtime.getDevice()
        self.calls = self.input_rows = self.output_rows = self.kernel_calls = self.native_calls = 0
        self.crop = cp.RawKernel(_PREPROCESS_SOURCE, "crop_keys3", options=("--fmad=false",))
        self.centers = cp.RawKernel(_PREPROCESS_SOURCE, "voxel_centers3", options=("--fmad=false",))
        self.gyro_kernel = cp.RawKernel(
            _PREPROCESS_SOURCE, "gyro_rotate3", options=("--fmad=false",),
        )
        self.translate_kernel = cp.RawKernel(
            _PREPROCESS_SOURCE, "gyro_translate3", options=("--fmad=false",),
        )
        self.gyro_calls = self.gyro_rows = self.translation_calls = self.repaired_points = 0
        self._prefix_key: tuple[int, ...] | None = None
        self._prefix_arrays: tuple[cp.ndarray, ...] | None = None
        self._prefix_host: tuple[FloatArray, ...] | None = None
        self.begin_scan()

    def begin_scan(self) -> None:
        """Reuse point rotations within this scan only; keep captured gyro prefixes."""
        self._gyro_inputs: _GyroPointInputs | None = None
        self._rotated: cp.ndarray | None = None
        self._fractions: cp.ndarray | None = None
        self._host_fractions: FloatArray | None = None

    def gyro_deskew(
        self, model: object, scan_time_s: float, offsets_s: FloatArray, points: FloatArray,
        motion: FloatArray, gap_s: float, options: ScanOdometryOptions,
    ) -> FloatArray | None:
        """Prepare quantized native gyro sweeps; unsupported models use the caller's path."""
        quantum = options.geometry_quantum_m
        if (type(model) is not _GyroRotationModel or not np.isfinite(quantum)
                or quantum <= 0 or quantum > 1e-3 or not np.isfinite(gap_s) or gap_s <= 0
                or not np.isfinite(options.voxel_size_m) or options.voxel_size_m <= 0
                or motion.shape != (4, 4) or not np.isfinite(motion).all()
                or np.max(np.abs(motion[:3, 3]), initial=0.) >= 1e6):
            return None
        fractions = np.asarray(offsets_s, dtype=np.float64) / gap_s
        if (not np.isfinite(fractions).all()
                or np.max(np.abs(fractions), initial=0.)
                * np.max(np.abs(motion[:3, 3]), initial=0.) >= 1e6):
            return None
        if self._gyro_inputs is None:
            assert isinstance(model, _GyroRotationModel)
            data = model._point_inputs(scan_time_s, offsets_s, points)
            if data is None:
                return None
            magnitude = float(np.max(np.abs(data.mounted), initial=0.))
            prefixes = (data.times, data.orientations, data.rates, data.projected)
            if (len(points) < 3 or magnitude >= 1e6
                    or np.max(np.abs(offsets_s), initial=0.) > .25
                    or np.max(np.abs(data.rates), initial=0.)
                    * np.max(np.diff(data.times), initial=0.) * math.sqrt(3) > .5
                    or magnitude / options.voxel_size_m >= 2**61
                    or points.nbytes * 18 + sum(p.nbytes for p in prefixes) > 1024**3):
                return None
            with cp.cuda.Device(self.device_id):
                key = tuple(id(p) for p in prefixes)
                if key != self._prefix_key:
                    self._prefix_host = prefixes  # prevent identity reuse after release
                    self._prefix_arrays = tuple(cp.asarray(p) for p in prefixes)
                    self._prefix_key = key
                assert self._prefix_arrays is not None
                self._rotated = cp.empty(data.mounted.shape, dtype=cp.float64)
                self.gyro_kernel(((len(points) + 127) // 128,), (128,), (
                    cp.asarray(data.mounted), cp.asarray(data.queries), *self._prefix_arrays,
                    cp.asarray(data.mounting), cp.asarray(data.back), cp.asarray(data.start_sum),
                    np.int32(len(points)), np.int32(len(data.times)), self._rotated,
                ))
                self.gyro_calls += 1
                self.gyro_rows += len(points)
                self._host_fractions = fractions
                self._fractions = cp.asarray(self._host_fractions)
                self._gyro_inputs = data
        assert self._rotated is not None and self._fractions is not None
        assert self._host_fractions is not None and self._gyro_inputs is not None
        with cp.cuda.Device(self.device_id):
            source = cp.empty_like(self._rotated)
            flags = cp.empty(len(source), dtype=cp.uint8)
            scale = max(1., float(np.max(np.abs(self._gyro_inputs.mounted), initial=0.)))
            self.translate_kernel(((len(source) + 127) // 128,), (128,), (
                self._rotated, self._fractions, cp.asarray(motion[:3, 3]), np.int32(len(source)),
                np.float64(quantum), np.float64(scale), source, flags,
            ))
            self.translation_calls += 1
            repair = np.flatnonzero(cp.asnumpy(flags)).astype(np.int64, copy=False)
            if len(repair):
                native = self._gyro_inputs.native_points(repair) + np.outer(
                    self._host_fractions[repair], motion[:3, 3],
                )
                source[cp.asarray(repair)] = cp.asarray(native)
                self.repaired_points += len(repair)
            return self._device(source, options)

    def __call__(self, points: FloatArray, options: ScanOdometryOptions) -> FloatArray:
        xyz = np.asarray(points, dtype=np.float64)[:, :3]
        magnitude = float(np.max(np.abs(xyz), initial=0.))
        voxel, quantum = options.voxel_size_m, options.geometry_quantum_m
        if (xyz.shape[1] != 3 or not len(xyz) or xyz.nbytes * 12 > 1024**3
                or not np.isfinite(xyz).all()
                or not np.isfinite(voxel) or voxel <= 0 or magnitude >= 1e50
                or magnitude / voxel >= 2**61
                or (quantum < 1 and quantum > 0
                    and magnitude > np.finfo(np.float64).max * quantum)):
            self.native_calls += 1
            return preprocess_scan(points, options)
        with cp.cuda.Device(self.device_id):
            source = cp.asarray(np.ascontiguousarray(xyz))
            return self._device(source, options)

    def _device(self, source: cp.ndarray, options: ScanOdometryOptions) -> FloatArray:
        voxel, quantum = options.voxel_size_m, options.geometry_quantum_m
        with cp.cuda.Device(self.device_id):
            data = cp.empty_like(source)
            keys = cp.empty(source.shape, dtype=cp.int64)
            valid = cp.empty(len(source), dtype=cp.uint8)
            self.crop(((len(source) + 127) // 128,), (128,),
                      (source, np.int32(len(source)), np.float64(quantum), np.float64(voxel),
                       np.float64(options.min_range_m), np.float64(options.max_range_m),
                       data, keys, valid))
            self.calls += 1
            self.input_rows += len(source)
            self.kernel_calls += 1
            rows = cp.flatnonzero(valid)
            if not len(rows):
                return np.empty((0, 3), dtype=np.float64)
            selected = keys[rows]
            sort_keys = cp.stack((rows, selected[:, 2], selected[:, 1], selected[:, 0]))
            order = rows[cp.lexsort(sort_keys)]
            sorted_keys = keys[order]
            changes = cp.flatnonzero(cp.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)) + 1
            starts = cp.concatenate((cp.zeros(1, dtype=cp.int64), changes))
            output = cp.empty((len(starts), 3), dtype=cp.float64)
            self.centers(((len(starts) + 127) // 128,), (128,),
                         (data, order, starts, np.int32(len(order)), np.int32(len(starts)),
                          np.float64(quantum), output))
            self.kernel_calls += 1
            self.output_rows += len(output)
            return _host(output)

    def statistics(self) -> dict[str, int]:
        """Report actual crop/mean work and native fallback calls."""
        return {"gpu_preprocess_calls": self.calls, "gpu_preprocess_input_rows": self.input_rows,
                "gpu_preprocess_output_rows": self.output_rows,
                "gpu_preprocess_kernel_calls": self.kernel_calls,
                "native_preprocess_calls": self.native_calls,
                "gpu_gyro_rotation_calls": self.gyro_calls,
                "gpu_gyro_rotation_rows": self.gyro_rows,
                "gpu_gyro_translation_calls": self.translation_calls,
                "native_gyro_point_repairs": self.repaired_points}


class CudaOdometryTarget:
    """Capture one target and reuse its tree/normals across deskew rounds.

    Correspondences, normal PCA, Jacobians and robust systems stay on
    device. Only the 6x6 system, count and squared-error sum cross back per
    iteration. Native small solves, stopping policy and pose updates remain
    on CPU; roundoff-sensitive neighbour/planarity decisions use native repairs.
    """

    preprocessor: type[CudaScanPreprocessor] = CudaScanPreprocessor

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
            self._normals = cp.zeros_like(self.tree.data)
            self._planar = cp.zeros(len(self.host), dtype=cp.bool_)
            self.done = cp.zeros(len(self.host), dtype=cp.bool_)
        self.scale = max(1.0, float(np.max(np.abs(self.host))))
        self.query_repairs = 0
        self.normal_repairs = 0
        self.fused_calls = 0
        self.system_refresh_calls = 0
        self.normal_rows = 0
        self.registration_repairs = 0
        self.motion_repairs = 0
        self.normal_kernel_calls = 0
        self._all_done = False
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
            self.normal_kernel = cp.RawKernel(
                _NORMAL_SOURCE, "normals3", options=("--fmad=false",),
            )

    @property
    def normals(self) -> cp.ndarray:
        """Return the complete normal array, materializing it for explicit readers."""
        with cp.cuda.Device(self.device_id):
            self._ensure_all()
        return self._normals

    @property
    def planar(self) -> cp.ndarray:
        """Return all planarity decisions; registration requests only its hits."""
        with cp.cuda.Device(self.device_id):
            self._ensure_all()
        return self._planar

    def _ensure(self, hit: cp.ndarray) -> None:
        if self._all_done:
            return
        missing = cp.unique(hit[~self.done[hit]])
        self._compute_normals(missing)

    def _ensure_all(self) -> None:
        if self._all_done:
            return
        if self.normal_rows:
            self._ensure(cp.asarray(np.arange(len(self.host), dtype=np.int64)))
        else:
            self._compute_normals(
                self.tree.arrays[0], complete=True,
            )

    def _compute_normals(self, missing: cp.ndarray, *, complete: bool = False) -> None:
        k = min(self.options.normal_neighbours, len(self.host))
        for start in range(0, len(missing), 65536):
            rows = missing[start : start + 65536]
            self.normal_rows += len(rows)
            queries = self.tree.data[rows]
            distances, indexes = self.tree.query(queries, k + 1)
            flags = cp.empty(len(rows), dtype=cp.uint8)
            self.normal_kernel(
                ((len(rows) + 127) // 128,),
                (128,),
                (
                    self.tree.data, indexes, distances, rows, np.int32(len(rows)), np.int32(k),
                    np.float64(self.options.max_planarity_ratio), self._normals, self._planar,
                    self.done, flags,
                ),
            )
            self.normal_kernel_calls += 1
            repair = np.flatnonzero(cp.asnumpy(flags))
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
                self._normals[selected] = cp.asarray(normals)
                self._planar[selected] = cp.asarray(planar)
                self.normal_repairs += len(repair)
        self._all_done = self.normal_rows == len(self.host)

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
            # Dense overlap amortizes one batched normal query/eigensolve.
            # Sparse sources request only their hits to avoid preparing a large,
            # mostly unused target. This changes work scheduling, not geometry.
            if len(points) >= len(self.host) / 10:
                self._ensure_all()
            terms = cp.empty((31, len(points)), dtype=cp.float64)
            uncertain = cp.empty(len(points), dtype=cp.float64)
            moved = cp.empty_like(points)
            distances = cp.empty((len(points), 2), dtype=cp.float64)
            indexes = cp.empty((len(points), 2), dtype=cp.int64)
            pending = cp.empty(len(points), dtype=cp.int64)
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
                            self._normals,
                            self._planar,
                            self.done,
                            np.int32(len(points)),
                            np.int32(len(self.host)),
                            np.float64(opts.max_correspondence_m),
                            np.float64(opts.huber_delta_m),
                            np.float64(error),
                            uncertain,
                            terms,
                            moved,
                            distances,
                            indexes,
                            pending,
                        ),
                    )
                    self.tree.query_calls += 1
                    self.tree.query_rows += len(points)
                    self.fused_calls += 1
                totals = _host(cp.sum(terms, axis=1))
                refresh = False
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
                    valid_hits = indexes[:, 0][
                        (distances[:, 0] < opts.max_correspondence_m)
                        & (indexes[:, 0] < len(self.host))
                    ]
                    self._ensure(valid_hits)
                    refresh = True
                elif totals[30]:
                    # The fused query has already captured the exact moved
                    # coordinates and matches. Compute only newly hit normals,
                    # then refresh the system without repeating spatial queries.
                    self._ensure(pending[pending >= 0])
                    refresh = True
                if refresh:
                    self.system_kernel(
                        ((len(points) + 127) // 128,),
                        (128,),
                        (
                            moved,
                            self.tree.data,
                            self._normals,
                            self._planar,
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
                    self.system_refresh_calls += 1
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
        near_motion_boundary = self._near_motion_boundary(transform)
        if uncertain_stopping or near_motion_boundary:
            native = register_point_to_plane(
                host_source, self.host, initial, opts, target_tree=self.tree.cpu,
            )
            self.registration_repairs += 1
            self.motion_repairs += int(near_motion_boundary)
            return replace(native, execution_backend=self.identity)
        eigenvalue = (
            0.0 if last_hessian is None else float(np.linalg.eigvalsh(last_hessian)[0]) / last_count
        )
        return ScanRegistration(
            _canonicalize_motion(transform, opts),
            converged and count >= opts.min_correspondences,
            iteration,
            count,
            rmse,
            eigenvalue,
            execution_backend=self.identity,
        )

    def _near_motion_boundary(self, transform: FloatArray) -> bool:
        """Use the native registration when the requested motion grid is ambiguous."""
        if (self.options.motion_translation_quantum_m == 0
                and self.options.motion_rotation_quantum_rad == 0):
            return False
        from scipy.spatial.transform import Rotation

        for values, quantum in (
            (transform[:3, 3], self.options.motion_translation_quantum_m),
            (Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
             self.options.motion_rotation_quantum_rad),
        ):
            if quantum and np.any(np.abs(np.remainder(values / quantum, 1) - 0.5) <= 1e-3):
                return True
        return False

    def statistics(self) -> dict[str, int]:
        """Report actual work, including the two kinds of native repairs."""
        return {
            "serialized_tree_captures": int(self.tree.capture_method == "validated_serialized"),
            "public_node_tree_captures": int(self.tree.capture_method == "public_nodes"),
            "gpu_tree_builds": int(self.tree.capture_method == "device_morton_bvh"),
            "gpu_tree_build_kernel_calls": self.tree.build_kernel_calls,
            "native_tree_builds": self.tree.native_tree_builds,
            "gpu_query_calls": self.tree.query_calls,
            "gpu_query_rows": self.tree.query_rows,
            "scalar_query_repairs": self.query_repairs,
            "scalar_normal_repairs": self.normal_repairs,
            "gpu_fused_system_calls": self.fused_calls,
            "gpu_system_refresh_calls": self.system_refresh_calls,
            "gpu_normal_rows": self.normal_rows,
            "gpu_normal_kernel_calls": self.normal_kernel_calls,
            "scalar_registration_repairs": self.registration_repairs,
            "scalar_motion_repairs": self.motion_repairs,
        }
