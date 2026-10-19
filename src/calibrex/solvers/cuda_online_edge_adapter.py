"""Optional bounded CUDA grid scores with native ambiguity and finalist checks.

Only raw noncenter grid scores use this adapter. Centers, final evaluations,
controls, curvature, and every ambiguous comparison retain native arithmetic.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.solvers.levinson_thrun_online_solver import (
    LevinsonObjectiveEvaluation,
    LevinsonOnlineFrame,
    OnlineEdgeExecution,
    _project_response,
    _rotation_matrix,
    evaluate_levinson_objective,
)

_SNAPSHOT_LIMIT: Final = 64 * 1024 * 1024
_OWNED_DEVICE_LIMIT: Final = 128 * 1024 * 1024
_LIBRARY_RESERVE: Final = 4 * 1024 * 1024
_MARGIN: Final = 2.0**-40
_POLICY: Final = "fp64_raw_score_enclosure/v0.1;native_ambiguous_and_finalists"

_KERNEL = r"""
struct ScoreBound { double score, error; unsigned int uncertain; };
__device__ __forceinline__ ScoreBound point_score(
    const double* rotations, const double* source, const double* translations,
    const double* image, const double* weights, int i, int n, int width, int height,
    double fx, double fy, double cx, double cy, double image_max, int forward_axes) {
  ScoreBound result={0.0, 0.0, 0};
  if (image_max == 0.0 || weights[i%n] == 0.0) return result;
  int point=i%n, pose=i/n;
  const double* r=rotations+9*pose;
  const double* t=translations+3*pose;
  const double* q=source+3*point;
  double x=(q[0]*r[0]+q[1]*r[1])+q[2]*r[2]+t[0];
  double y=(q[0]*r[3]+q[1]*r[4])+q[2]*r[5]+t[1];
  double z=(q[0]*r[6]+q[1]*r[7])+q[2]*r[8]+t[2];
  if (forward_axes) { double xx=-y, yy=-z; z=x; x=xx; y=yy; }
  double scale=1.0+4.0*(fabs(source[3*point])+fabs(source[3*point+1])+fabs(source[3*point+2]))
      +fabs(translations[3*pose])+fabs(translations[3*pose+1])+fabs(translations[3*pose+2]);
  double delta=9.094947017729282379150390625e-13*scale;
  if (!isfinite(x) || !isfinite(y) || !isfinite(z) || !isfinite(delta)) {
    result.uncertain=1; return result;
  }
  if (z+delta <= 1e-6) return result;
  if (z-delta <= 1e-6 || z<=2.0*delta) { result.uncertain=1; return result; }
  double reciprocal=__drcp_rn(z);
  double xx=x*reciprocal, yy=y*reciprocal;
  double u=fx*xx+cx, v=fy*yy+cy;
  // z>2*delta gives 1/(z-delta)<2/z; 4*rounded(1/z) is an upper bound.
  double inverse=4.0*reciprocal;
  double du_bound=fabs(fx)*delta*(1.0+fabs(xx))*inverse
      +9.094947017729282379150390625e-13*(1.0+fabs(fx*xx)+fabs(cx)+fabs(u));
  double dv_bound=fabs(fy)*delta*(1.0+fabs(yy))*inverse
      +9.094947017729282379150390625e-13*(1.0+fabs(fy*yy)+fabs(cy)+fabs(v));
  if (!isfinite(u) || !isfinite(v) || !isfinite(du_bound) || !isfinite(dv_bound)) {
    result.uncertain=1; return result;
  }
  if (u+du_bound < 0.0 || v+dv_bound < 0.0 || u-du_bound >= width-1 || v-dv_bound >= height-1)
    return result;
  if (u-du_bound < 0.0 || v-dv_bound < 0.0 || u+du_bound >= width-1 || v+dv_bound >= height-1) {
    result.uncertain=1; return result;
  }
  int left=(int)floor(u), top=(int)floor(v);
  double du=u-left, dv=v-top;
  double response=(1.0-du)*(1.0-dv)*image[top*width+left]
      +du*(1.0-dv)*image[top*width+left+1]
      +(1.0-du)*dv*image[(top+1)*width+left]
      +du*dv*image[(top+1)*width+left+1];
  double weighted=response*weights[point];
  double error=2.0*(image_max*(du_bound+dv_bound+9.094947017729282379150390625e-13)*weights[point]
      +fabs(weighted)*9.094947017729282379150390625e-13
      +2.2250738585072014e-308*(1.0+image_max+fabs(weights[point])));
  if (!isfinite(weighted) || !isfinite(error)) {
    result.uncertain=1; return result;
  }
  result.score=weighted; result.error=error;
  return result;
}

extern "C" __global__ void guarded(
    const double* rotations, const double* source, const double* translations,
    const double* image, const double* weights, double* output,
    int n, int count, int width, int height,
    double fx, double fy, double cx, double cy, double image_max, int forward_axes) {
  const int pose=blockIdx.x, thread=threadIdx.x;
  if (pose*n >= count) return;
  double raw=0.0, error=0.0;
  unsigned int uncertain=0;
  for (int point=thread; point<n; point+=256) {
    ScoreBound value=point_score(rotations, source, translations, image, weights,
        pose*n+point, n, width, height, fx, fy, cx, cy, image_max, forward_axes);
    raw+=value.score; error+=value.error; uncertain|=value.uncertain;
  }
  __shared__ double raw_parts[256], error_parts[256];
  __shared__ unsigned int flags[256];
  raw_parts[thread]=raw; error_parts[thread]=error; flags[thread]=uncertain;
  __syncthreads();
  for (int stride=128; stride>0; stride/=2) {
    if (thread<stride) {
      raw_parts[thread]+=raw_parts[thread+stride];
      error_parts[thread]+=error_parts[thread+stride];
      flags[thread]|=flags[thread+stride];
    }
    __syncthreads();
  }
  if (thread == 0) {
    output[3*pose]=raw_parts[0];
    output[3*pose+1]=2.0*error_parts[0]
        +fabs(raw_parts[0])*((n+128)*9.094947017729282379150390625e-13);
    output[3*pose+2]=(double)flags[0];
  }
}
"""
_KERNEL_SHA256: Final = hashlib.sha256(_KERNEL.encode()).hexdigest()

_IMAGE_KERNEL = r"""
extern "C" __global__ void gradient(const double* image, double* edge, int h, int w) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=h*w) return;
  int row=i/w, column=i%w;
  double value=0.0;
  for (int y=-1; y<=1; ++y) for (int x=-1; x<=1; ++x) {
    int yy=max(0,min(h-1,row+y)), xx=max(0,min(w-1,column+x));
    value=fmax(value,fabs(image[i]-image[yy*w+xx]));
  }
  edge[i]=value;
}
extern "C" __global__ void spread_step(
    const double* previous, double* next, double* spread, int h, int w, double gamma) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=h*w) return;
  int row=i/w, column=i%w;
  double value=0.0;
  for (int y=-1; y<=1; ++y) for (int x=-1; x<=1; ++x) {
    int yy=max(0,min(h-1,row+y)), xx=max(0,min(w-1,column+x));
    value=fmax(value,previous[yy*w+xx]);
  }
  value=gamma*value;
  next[i]=value;
  spread[i]=fmax(spread[i],value);
}
extern "C" __global__ void combine(
    const double* edge, const double* spread, double* output, int n, double alpha) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) output[i]=alpha*edge[i]+(1.0-alpha)*spread[i];
}
"""
IMAGE_KERNEL_SHA256: Final = hashlib.sha256(_IMAGE_KERNEL.encode()).hexdigest()


class CudaEdgeImageRuntime:
    """Initialize image code once, with all CUDA work on the caller's thread."""

    def __init__(self) -> None:
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise ValueError("CUDA online edge evaluation requires an available CUDA device")
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise ValueError("CUDA online edge evaluation could not initialize a device") from exc
        self._module = cp.RawModule(code=_IMAGE_KERNEL, options=("--std=c++11", "--fmad=false"))
        self._functions: tuple[Any, Any, Any] | None = None

    def evaluate(
        self, image: NDArray[np.float64], *, alpha: float, gamma: float, max_radius: int
    ) -> NDArray[np.float64]:
        """Run bounded spilloff without a host round trip at every radius."""
        from calibrex.solvers.levinson_thrun_online_solver import (
            _edge_response_input,
            levinson_edge_response,
        )

        image = _edge_response_input(image, alpha, gamma, max_radius)
        height, width = image.shape
        if height < 1 or width < 1 or image.size >= 2**31 or 6 * image.nbytes > _OWNED_DEVICE_LIMIT:
            return levinson_edge_response(image, alpha=alpha, gamma=gamma, max_radius=max_radius)
        if self._functions is None:
            self._functions = (
                self._module.get_function("gradient"),
                self._module.get_function("spread_step"),
                self._module.get_function("combine"),
            )
        gradient, spread_step, combine = self._functions
        source = cp.asarray(np.ascontiguousarray(image))
        edge = cp.empty(image.shape, dtype=cp.float64)
        frontier = cp.empty_like(edge)
        next_frontier = cp.empty_like(edge)
        output = cp.empty_like(edge)
        launch = ((int(image.size) + 255) // 256,), (256,)
        dimensions = np.int32(height), np.int32(width)
        gradient(*launch, (source, edge, *dimensions))
        spread = edge.copy()
        previous = edge
        for _ in range(max_radius):
            spread_step(*launch, (previous, frontier, spread, *dimensions, np.float64(gamma)))
            previous, frontier, next_frontier = frontier, next_frontier, frontier
        combine(*launch, (edge, spread, output, np.int32(image.size), np.float64(alpha)))
        return np.asarray(output.get(), dtype=np.float64)


def cuda_edge_response(
    image: NDArray[np.float64], *, alpha: float, gamma: float, max_radius: int
) -> NDArray[np.float64]:
    """Evaluate one image; batch preparation reuses CudaEdgeImageRuntime."""
    return CudaEdgeImageRuntime().evaluate(image, alpha=alpha, gamma=gamma, max_radius=max_radius)


@dataclass
class _Counters:
    grid_calls: int = 0
    grid_pose_evaluations: int = 0
    cuda_frame_pose_evaluations: int = 0
    native_frame_pose_evaluations: int = 0
    projection_kernel_launches: int = 0
    score_readback_bytes: int = 0
    image_upload_bytes: int = 0
    geometry_upload_bytes: int = 0
    pose_upload_bytes: int = 0
    native_resolved_poses: int = 0
    native_boundary_poses: int = 0
    native_comparison_poses: int = 0
    native_winner_poses: int = 0
    max_device_snapshot_bytes: int = 0
    max_scratch_bytes: int = 0


@dataclass(frozen=True)
class _DeviceFrame:
    owner: LevinsonOnlineFrame
    points: Any
    image: Any
    weights: Any
    image_max: float
    size: int


class CudaOnlineEdgeRuntime:
    """Own immutable run inputs and bounded device data, without global patches."""

    def __init__(self, frames: Sequence[LevinsonOnlineFrame]) -> None:
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise ValueError("CUDA online edge evaluation requires an available CUDA device")
            version = cp.cuda.runtime.runtimeGetVersion()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise ValueError("CUDA online edge evaluation could not initialize a device") from exc
        self.identity = (
            f"calibrex.cuda_online_edge/v0.1;cupy={cp.__version__};"
            f"cuda_runtime={version};numpy={np.__version__}"
        )
        self.frames = tuple(self._own(frame) for frame in frames)
        self._owners = {id(frame): frame for frame in self.frames}
        self._counts = _Counters()
        self._cache: OrderedDict[int, _DeviceFrame] = OrderedDict()
        self._cache_bytes = 0
        self._kernel = cp.RawKernel(_KERNEL, "guarded", options=("--std=c++11", "--fmad=false"))
        digest = hashlib.sha256()
        for frame in self.frames:
            digest.update(repr((frame.frame_id, frame.camera)).encode("utf-8"))
            for array in (frame.lidar_points, frame.discontinuity_weights, frame.edge_response):
                digest.update(repr((array.shape, array.dtype.str)).encode())
                digest.update(array.tobytes(order="C"))
        self._capture_sha256 = digest.hexdigest()

    @staticmethod
    def _own(frame: LevinsonOnlineFrame) -> LevinsonOnlineFrame:
        arrays = [
            np.array(array, dtype=np.float64, copy=True, order="K")
            for array in (frame.edge_response, frame.lidar_points, frame.discontinuity_weights)
        ]
        for array in arrays:
            array.setflags(write=False)
        return LevinsonOnlineFrame(frame.frame_id, arrays[0], arrays[1], arrays[2], frame.camera)

    @staticmethod
    def _eligible(frame: LevinsonOnlineFrame) -> bool:
        camera = frame.camera
        values = (camera.fx, camera.fy, camera.cx, camera.cy)
        return bool(
            camera.projection == "pinhole"
            and camera.axes in {"optical_z", "forward_x_left_y_up_z"}
            and len(frame.lidar_points) >= 32
            and max(frame.edge_response.shape) < 2**31
            and np.all(np.isfinite(values))
            and np.max(np.abs(values)) < 1e12
            and np.max(np.abs(frame.lidar_points), initial=0.0) < 1e12
            and np.max(frame.discontinuity_weights, initial=0.0) < 1e12
            and np.max(frame.edge_response, initial=0.0) < 1e12
        )

    def _device_frame(self, frame: LevinsonOnlineFrame) -> _DeviceFrame | None:
        key = id(frame)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        if not self._eligible(frame):
            return None
        size = sum(
            int(array.nbytes)
            for array in (frame.edge_response, frame.lidar_points, frame.discontinuity_weights)
        )
        if size > _SNAPSHOT_LIMIT:
            return None
        while self._cache and self._cache_bytes + size > _SNAPSHOT_LIMIT:
            _, removed = self._cache.popitem(last=False)
            self._cache_bytes -= removed.size
            del removed
        value = _DeviceFrame(
            frame,
            cp.asarray(np.ascontiguousarray(frame.lidar_points)),
            cp.asarray(np.ascontiguousarray(frame.edge_response)),
            cp.asarray(np.ascontiguousarray(frame.discontinuity_weights)),
            float(np.max(frame.edge_response)),
            size,
        )
        self._cache[key] = value
        self._cache_bytes += size
        self._counts.image_upload_bytes += int(frame.edge_response.nbytes)
        self._counts.geometry_upload_bytes += int(
            frame.lidar_points.nbytes + frame.discontinuity_weights.nbytes
        )
        self._counts.max_device_snapshot_bytes = max(
            self._counts.max_device_snapshot_bytes,
            self._cache_bytes,
        )
        return value

    def _native_score(self, frames: Sequence[LevinsonOnlineFrame], pose: SE3) -> float:
        self._counts.native_frame_pose_evaluations += len(frames)
        return evaluate_levinson_objective(frames, pose).raw_objective

    def _score_chunk(
        self,
        data: _DeviceFrame,
        rotations: Any,
        translations: Any,
    ) -> Any:
        frame, camera = data.owner, data.owner.camera
        n, p = len(frame.lidar_points), int(translations.shape[0])
        rotations = cp.ascontiguousarray(rotations)
        output = cp.empty((p, 3), dtype=cp.float64)
        count = n * p
        self._kernel(
            (p,),
            (256,),
            (
                rotations,
                data.points,
                translations,
                data.image,
                data.weights,
                output,
                np.int32(n),
                np.int32(count),
                np.int32(camera.width),
                np.int32(camera.height),
                np.float64(camera.fx),
                np.float64(camera.fy),
                np.float64(camera.cx),
                np.float64(camera.cy),
                np.float64(data.image_max),
                np.int32(camera.axes == "forward_x_left_y_up_z"),
            ),
        )
        stats = output.get()
        self._counts.cuda_frame_pose_evaluations += p
        self._counts.projection_kernel_launches += 1
        self._counts.score_readback_bytes += int(stats.nbytes)
        scratch = p * 128
        self._counts.max_scratch_bytes = max(self._counts.max_scratch_bytes, scratch)
        return stats

    def raw_scores(
        self,
        frames: Sequence[LevinsonOnlineFrame],
        poses: Sequence[SE3],
        center: LevinsonObjectiveEvaluation,
    ) -> list[float]:
        """Return decision-compatible scores, resolving ambiguous signs and finalists.

        Unresolved scores stay inside conservative native-score enclosures; their
        signs and inability to win are proved before returning. Returned winner
        scores and every potentially ambiguous comparison use native arithmetic.
        """
        for frame in frames:
            if self._owners.get(id(frame)) is not frame:
                raise ValueError("online CUDA grid requires its owned input snapshot")
        self._counts.grid_calls += 1
        self._counts.grid_pose_evaluations += len(poses)
        if len(poses) > 728:
            raise ValueError("online CUDA grid accepts at most 728 noncenter poses")
        if not poses:
            return []
        rotations = np.asarray([_rotation_matrix(pose) for pose in poses], dtype=float)
        translations = np.asarray([pose.translation_m for pose in poses], dtype=float)
        if (
            not math.isfinite(center.raw_objective)
            or not np.all(np.isfinite(rotations))
            or not np.all(np.isfinite(translations))
            or np.max(np.abs(rotations), initial=0.0) > 2.0
            or np.max(np.abs(translations), initial=0.0) >= 1e12
        ):
            return [self._native_score(frames, pose) for pose in poses]
        dr, dt = cp.asarray(rotations.reshape(-1, 3)), cp.asarray(translations)
        pose_bytes = int(rotations.nbytes + translations.nbytes)
        self._counts.pose_upload_bytes += pose_bytes
        raw, errors, boundary = [0.0] * len(poses), [0.0] * len(poses), [False] * len(poses)
        for frame in frames:
            data = self._device_frame(frame)
            room = _OWNED_DEVICE_LIMIT - self._cache_bytes - _LIBRARY_RESERVE - pose_bytes
            chunk = min(len(poses), room // 128)
            if data is None or chunk < 1:
                for index, pose in enumerate(poses):
                    response, weights = _project_response(frame, pose)
                    value = float(response @ weights)
                    if errors[index] > 0.0:
                        errors[index] += _MARGIN * (abs(raw[index]) + abs(value))
                    raw[index] += value
                    self._counts.native_frame_pose_evaluations += 1
                del data
                continue
            for start in range(0, len(poses), chunk):
                stop = min(start + chunk, len(poses))
                stats = self._score_chunk(data, dr[3 * start : 3 * stop], dt[start:stop])
                for local, index in enumerate(range(start, stop)):
                    score, bound, flag = stats[local]
                    errors[index] += float(bound) + _MARGIN * (abs(raw[index]) + abs(float(score)))
                    raw[index] += float(score)
                    boundary[index] |= bool(flag)
            del data
        exact = [error == 0.0 for error in errors]
        for index, pose in enumerate(poses):
            close = errors[index] > 0.0 and abs(raw[index] - center.raw_objective) <= errors[index]
            if boundary[index] or close or not math.isfinite(raw[index] + errors[index]):
                raw[index] = self._native_score(frames, pose)
                errors[index] = 0.0
                exact[index] = True
                self._counts.native_resolved_poses += 1
                self._counts.native_boundary_poses += int(boundary[index])
                self._counts.native_comparison_poses += int(close)
        if not all(math.isfinite(value) for value in raw):
            return [self._native_score(frames, pose) for pose in poses]
        lower = max(
            [
                center.raw_objective,
                *(value - error for value, error in zip(raw, errors, strict=True)),
            ]
        )
        for index, pose in enumerate(poses):
            if not exact[index] and raw[index] + errors[index] >= lower:
                raw[index] = self._native_score(frames, pose)
                self._counts.native_resolved_poses += 1
                self._counts.native_winner_poses += 1
        return raw

    def execution(self) -> OnlineEdgeExecution:
        """Return typed software provenance without hardware model identifiers."""
        return OnlineEdgeExecution(
            backend="cuda",
            runtime=self.identity,
            kernel_sha256=_KERNEL_SHA256,
            capture_sha256=self._capture_sha256,
            policy=_POLICY,
            device_snapshot_limit_bytes=_SNAPSHOT_LIMIT,
            owned_device_limit_bytes=_OWNED_DEVICE_LIMIT,
            **asdict(self._counts),
        )
