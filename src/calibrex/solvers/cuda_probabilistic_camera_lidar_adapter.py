"""Optional CUDA point evaluation with unchanged native NumPy reductions.

Each solver call owns its snapshots, device buffers and counters. The core
optimizer never imports CuPy on the default NumPy path.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np

from calibrex.core.geometry import SE3, normalize_quaternion_xyzw, quaternion_conjugate_xyzw
from calibrex.core.probabilistic_correspondence import ProbabilisticRefinementExecution
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    _POSE_CAPTURE_BUDGET_BYTES,
    FloatArray,
    ProbabilisticCameraLidarRefinementOptions,
    ProbabilisticPoseEvaluation,
    _PoseFrame,
    _PreparedPoseEvaluation,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from calibrex.core.probabilistic_correspondence import ProbabilisticCorrespondenceFrame

_KERNEL = r"""
extern "C" __global__ void evaluate(
    const double* data, const int* frame, const double* cameras, int n, int covariance,
    const double* poses, double* out
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const double* pose = poses + 11*blockIdx.y;
    double lx=pose[0], ly=pose[1], lz=pose[2], lw=pose[3];
    double rx=pose[4], ry=pose[5], rz=pose[6], rw=pose[7];
    double tx=pose[8], ty=pose[9], tz=pose[10];
    out += 3*n*blockIdx.y;
    out[2*n+i] = 0.0;
    const double* p = data + 10*i;
    const double* c = cameras + 14*frame[i];
    double x=p[0], y=p[1], z=p[2];
    double ax=lw*x+lx*0.0+ly*z-lz*y;
    double ay=lw*y-lx*z+ly*0.0+lz*x;
    double az=lw*z+lx*y-ly*x+lz*0.0;
    double aw=lw*0.0-lx*x-ly*y-lz*z;
    double xx=aw*rx+ax*rw+ay*rz-az*ry+tx;
    double yy=aw*ry-ax*rz+ay*rw+az*rx+ty;
    double zz=aw*rz+ax*ry-ay*rx+az*rw+tz;
    if (!(zz > 1e-9)) return;
    double denominator=zz;
    if (c[0] != 0.0) {
        double norm=sqrt(xx*xx+yy*yy+zz*zz);
        if (c[0] == 1.0) denominator=zz+c[7]*norm;
        else {
            double zeta=c[7]*norm+zz;
            double second=sqrt(xx*xx+yy*yy+zeta*zeta);
            denominator=c[8]*second+(1.0-c[8])*zeta;
        }
        if (!(denominator > 1e-12)) return;
    }
    xx=xx/denominator; yy=yy/denominator;
    if (c[13] != 0.0) {
        double radius=xx*xx+yy*yy;
        if (isfinite(radius) && fabs(radius)>1e100) { out[2*n+i]=2.0; return; }
        double squared=radius*radius;
        double cubed=squared*radius;
        double radial=1.0+c[9]*radius+c[10]*squared+0.0*cubed;
        double dx=xx*radial+2.0*c[11]*xx*yy+c[12]*(radius+2.0*xx*xx);
        double dy=yy*radial+c[11]*(radius+2.0*yy*yy)+2.0*c[12]*xx*yy;
        xx=dx; yy=dy;
    }
    double u=c[1]*xx+c[3], v=c[2]*yy+c[4];
    if (!(u>=0.0 && u<c[5] && v>=0.0 && v<c[6])) return;
    double du=u-p[3], dv=v-p[4];
    double pixel=du*du+dv*dv;
    double squared_error=pixel;
    if (covariance) {
        double a=p[5], b=p[6], cc=p[7], d=p[8];
        squared_error=(d*du*du-(b+cc)*du*dv+a*dv*dv)/(a*d-b*cc);
    }
    out[i]=pixel;
    out[n+i]=sqrt(squared_error>0.0 ? squared_error : 0.0);
    out[2*n+i]=1.0;
}
"""

_KERNEL_SHA256 = hashlib.sha256(_KERNEL.encode()).hexdigest()


@dataclass(frozen=True)
class _FrameSnapshot:
    owner: ProbabilisticCorrespondenceFrame
    frame: _PoseFrame
    reliability: FloatArray
    outlier_probability: FloatArray


class CudaProbabilisticPoseRuntime:
    """Own one input snapshot, bounded caches and execution lineage.

    A runtime is sequential and scoped to one immutable input set. Construct a
    new runtime to observe changed frame objects. Prepared evaluators own their
    confidence weights while geometry and device inputs may be shared.
    """

    def __init__(self) -> None:
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise ValueError("CUDA pose evaluation requires an available CUDA device")
            runtime = cp.cuda.runtime.runtimeGetVersion()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise ValueError("CUDA pose evaluation could not initialize a CUDA device") from exc
        self.identity = (
            f"calibrex.cuda_probabilistic_pose/v0.1;cupy={cp.__version__};"
            f"cuda_runtime={runtime};numpy={np.__version__}"
        )
        self.kernel = cp.RawKernel(_KERNEL, "evaluate", options=("--std=c++11", "--fmad=false"))
        self.calls = 0
        self.cuda_calls = 0
        self.kernel_launches = 0
        self.pose_upload_bytes = 0
        self.numpy_fallback_calls = 0
        self.upload_bytes = 0
        self.download_bytes = 0
        self._snapshots: dict[str, _FrameSnapshot] = {}
        self._snapshot_bytes = 0
        self._device_cache: dict[str, tuple[Any, Any, Any, Any]] = {}
        self._device_cache_bytes = 0

    def prepare(
        self,
        frames: Sequence[ProbabilisticCorrespondenceFrame],
        settings: ProbabilisticCameraLidarRefinementOptions,
    ) -> _PreparedPoseEvaluation:
        """Capture a private evaluator; unsupported inputs keep native arithmetic."""
        return _CudaPreparedPoseEvaluation(frames, settings, self)

    def _capture_frame(self, frame: ProbabilisticCorrespondenceFrame) -> _FrameSnapshot | None:
        previous = self._snapshots.get(frame.frame_id)
        if previous is not None:
            if previous.owner is not frame:
                raise ValueError(
                    "a CUDA pose snapshot cannot bind different objects with one frame ID"
                )
            return previous
        estimate = len(frame.correspondences) * 128
        if self._snapshot_bytes + estimate > _POSE_CAPTURE_BUDGET_BYTES:
            return None
        captured = _PoseFrame(
            frame.intrinsics.model_copy(deep=True),
            np.asarray([c.point_lidar_m for c in frame.correspondences], dtype=float),
            np.asarray([c.image_mean_px for c in frame.correspondences], dtype=float),
            np.asarray([c.image_covariance_px2 for c in frame.correspondences], dtype=float),
            np.empty(0, dtype=float),
        )
        reliability = np.asarray([c.reliability for c in frame.correspondences], dtype=float)
        outlier = np.asarray([c.outlier_probability for c in frame.correspondences], dtype=float)
        for array in (captured.points, captured.means, captured.covariance, reliability, outlier):
            array.setflags(write=False)
        snapshot = _FrameSnapshot(frame, captured, reliability, outlier)
        self._snapshots[frame.frame_id] = snapshot
        self._snapshot_bytes += estimate
        return snapshot

    def execution(
        self,
        since: ProbabilisticRefinementExecution | None = None,
    ) -> ProbabilisticRefinementExecution:
        """Return counters without device names or other host identifiers."""
        fields = (
            "calls",
            "cuda_calls",
            "numpy_fallback_calls",
            "upload_bytes",
            "download_bytes",
            "kernel_launches",
            "pose_upload_bytes",
        )
        counters = {
            name: int(getattr(self, name)) - (int(getattr(since, name)) if since is not None else 0)
            for name in fields
        }
        if since is not None and (
            since.runtime != self.identity or since.kernel_sha256 != _KERNEL_SHA256
        ):
            raise ValueError("execution counters belong to a different pose runtime")
        return ProbabilisticRefinementExecution(
            backend="cuda",
            runtime=self.identity,
            kernel_sha256=_KERNEL_SHA256,
            calls=counters["calls"],
            cuda_calls=counters["cuda_calls"],
            kernel_launches=counters["kernel_launches"],
            pose_upload_bytes=counters["pose_upload_bytes"],
            numpy_fallback_calls=counters["numpy_fallback_calls"],
            upload_bytes=counters["upload_bytes"],
            download_bytes=counters["download_bytes"],
        )


class _CudaPreparedPoseEvaluation(_PreparedPoseEvaluation):
    def __init__(
        self,
        frames: Sequence[ProbabilisticCorrespondenceFrame],
        settings: ProbabilisticCameraLidarRefinementOptions,
        runtime: CudaProbabilisticPoseRuntime,
    ) -> None:
        self.runtime = runtime
        self.device: tuple[Any, Any, Any, Any] | None = None
        self._batch_out: Any = None
        self.settings = settings
        self.total = sum(len(frame.correspondences) for frame in frames)
        tiny = max((len(f.correspondences) for f in frames), default=0) < 64
        unsupported = any(
            (f.intrinsics.projection != "pinhole" and f.intrinsics.xi is None)
            or (f.intrinsics.projection == "double_sphere" and f.intrinsics.alpha is None)
            or (
                f.intrinsics.distortion_model == "radial-tangential"
                and (
                    len(f.intrinsics.distortion) not in {4, 5}
                    or (len(f.intrinsics.distortion) == 5 and f.intrinsics.distortion[4] != 0.0)
                )
            )
            for f in frames
        )
        if tiny or unsupported or self.total * 384 > _POSE_CAPTURE_BUDGET_BYTES:
            super().__init__(frames, settings)
            return
        snapshots = [runtime._capture_frame(frame) for frame in frames]
        if any(snapshot is None for snapshot in snapshots):
            super().__init__(frames, settings)
            return
        self.fallback = ()
        prepared = []
        for snapshot in snapshots:
            assert snapshot is not None
            frame = snapshot.frame
            weights = (
                snapshot.reliability
                if settings.use_reliability
                else np.ones(len(frame.points), dtype=float)
            ) * (
                1.0 - snapshot.outlier_probability
                if settings.use_outlier_probability
                else np.ones(len(frame.points), dtype=float)
            )
            selected = weights >= settings.minimum_confidence
            if not np.any(selected):
                continue
            if np.all(selected):
                points, means, covariance = frame.points, frame.means, frame.covariance
            else:
                points, means, covariance = (
                    frame.points[selected],
                    frame.means[selected],
                    frame.covariance[selected],
                )
            prepared.append(
                _PoseFrame(frame.intrinsics, points, means, covariance, weights[selected])
            )
        self.frames = tuple(prepared)
        if not self.frames:
            return
        rows, frame_ids, cameras = [], [], []
        for index, frame in enumerate(self.frames):
            intr = frame.intrinsics
            rows.append(
                np.column_stack(
                    (
                        frame.points,
                        frame.means,
                        frame.covariance,
                        frame.weights,
                    )
                )
            )
            frame_ids.append(np.full(len(frame.points), index, dtype=np.int32))
            distortion = list(intr.distortion[:4]) + [0.0] * 4
            cameras.append(
                [
                    {"pinhole": 0.0, "mei": 1.0, "double_sphere": 2.0}[intr.projection],
                    intr.fx,
                    intr.fy,
                    intr.cx,
                    intr.cy,
                    intr.width,
                    intr.height,
                    intr.xi or 0.0,
                    intr.alpha or 0.0,
                    *distortion[:4],
                    float(intr.distortion_model == "radial-tangential"),
                ]
            )
        packed = np.concatenate(rows)
        ids = np.concatenate(frame_ids)
        camera_array = np.asarray(cameras, dtype=np.float64)
        self.host_weights = packed[:, 9].copy()
        self.n = len(packed)
        digest = hashlib.sha256(packed[:, :9].tobytes())
        digest.update(ids.tobytes())
        digest.update(camera_array.tobytes())
        key = digest.hexdigest()
        cached = runtime._device_cache.get(key)
        if cached is not None:
            self.device = cached
            return
        try:
            self.device = (
                cp.asarray(packed),
                cp.asarray(ids),
                cp.asarray(camera_array),
                cp.empty((3, self.n), dtype=cp.float64),
            )
        except cp.cuda.memory.OutOfMemoryError as exc:
            raise ValueError(
                "CUDA pose evaluation could not allocate its bounded snapshot"
            ) from exc
        runtime.upload_bytes += packed.nbytes + ids.nbytes + camera_array.nbytes
        size = sum(int(array.nbytes) for array in self.device)
        if runtime._device_cache_bytes + size <= _POSE_CAPTURE_BUDGET_BYTES:
            runtime._device_cache[key] = self.device
            runtime._device_cache_bytes += size

    def evaluate(self, transform: SE3) -> ProbabilisticPoseEvaluation:
        return self.evaluate_many((transform,))[0]

    def evaluate_many(self, transforms: Sequence[SE3]) -> tuple[ProbabilisticPoseEvaluation, ...]:
        if not transforms:
            return ()
        self.runtime.calls += len(transforms)
        if self.device is None:
            self.runtime.numpy_fallback_calls += len(transforms)
            return tuple(super(_CudaPreparedPoseEvaluation, self).evaluate(t) for t in transforms)
        # Account for the original snapshot and both host/device batch outputs.
        capacity = max(
            1,
            min(
                32,
                1 + (_POSE_CAPTURE_BUDGET_BYTES - 384 * self.total) // (48 * self.n),
            ),
        )
        data, ids, cameras, single_out = self.device
        results = []
        for start in range(0, len(transforms), capacity):
            poses = transforms[start : start + capacity]
            host_poses = []
            for transform in poses:
                left = normalize_quaternion_xyzw(transform.rotation_quat_xyzw)
                right = quaternion_conjugate_xyzw(left)
                host_poses.append((*left, *right, *transform.translation_m))
            pose_array = np.asarray(host_poses, dtype=np.float64)
            device_poses = cp.asarray(pose_array)
            self.runtime.pose_upload_bytes += pose_array.nbytes
            if len(poses) == 1:
                out = single_out.reshape(1, 3, self.n)
            else:
                if self._batch_out is None or len(self._batch_out) < len(poses):
                    self._batch_out = cp.empty((len(poses), 3, self.n), dtype=cp.float64)
                out = self._batch_out[: len(poses)]
            self.runtime.kernel(
                ((self.n + 255) // 256, len(poses)),
                (256,),
                (
                    data,
                    ids,
                    cameras,
                    np.int32(self.n),
                    np.int32(self.settings.use_covariance),
                    device_poses,
                    out,
                ),
            )
            values = out.get()
            self.runtime.kernel_launches += 1
            self.runtime.cuda_calls += len(poses)
            self.runtime.download_bytes += values.nbytes
            for transform, value in zip(poses, values, strict=True):
                if np.any(value[2] == 2.0):
                    self.runtime.numpy_fallback_calls += 1
                    results.append(super().evaluate(transform))
                    continue
                valid = value[2] == 1.0
                count = int(np.count_nonzero(valid))
                if not count:
                    results.append(ProbabilisticPoseEvaluation(math.inf, None, None, 0, self.total))
                    continue
                pixels, errors = value[0, valid], value[1, valid]
                weights = self.host_weights[valid]
                robust = np.log1p((errors / self.settings.cauchy_scale) ** 2)
                results.append(
                    ProbabilisticPoseEvaluation(
                        float(np.sum(weights * robust) / np.sum(weights)),
                        math.sqrt(float(np.sum(weights * pixels)) / float(np.sum(weights))),
                        float(np.sum(weights * errors) / np.sum(weights)),
                        count,
                        self.total - count,
                    )
                )
        return tuple(results)
