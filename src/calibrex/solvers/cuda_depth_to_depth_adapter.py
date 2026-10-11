"""Optional CUDA adapter for repeated D2D projection, isolated behind CuPy.

Projection, stable visibility and histogram counts run in FP64 on the GPU.
Entropy/MI arithmetic stays on the native NumPy path. Prepared observations are
snapshots: do not modify their arrays while using this projector.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import cupy as cp
import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthPairHistogram,
    DepthPairProjection,
    DepthToDepthObservation,
    _empty_projection,
    _rotation_matrix,
)
from calibrex.solvers.depth_to_depth_runtime import DEFAULT_CUDA_CACHE_BYTES

if TYPE_CHECKING:
    from calibrex.solvers.cuda_depth_to_depth_batch import CudaPoseBatch

CUDA_DEPTH_PAIR_PROJECTOR_VERSION: Final = "calibrex.cuda_depth_pair_projector/v0.4"
_THREADS: Final = 256
_HISTOGRAM_BATCH_FRAMES: Final = 32


def _validate_histogram_configuration(
    bins: int, camera_range: tuple[float, float], lidar_range: tuple[float, float],
) -> None:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("histogram bins must be a positive integer")
    if not all(
        len(limits) == 2
        and all(math.isfinite(value) for value in limits)
        and limits[0] < limits[1]
        for limits in (camera_range, lidar_range)
    ):
        raise ValueError("histogram ranges must be finite and increasing")

# Positive double bit patterns are ordered like unsigned integers. Two separate
# atomic minima retain the nearest range, then the first input index on ties.
# FMA is disabled to preserve the CPU expression order at pixel boundaries.
_SOURCE: Final = r"""
__device__ void initialize_pixel(
    const long long pixel, const long long pixel_count, const long long n,
    unsigned long long* best_range, unsigned long long* best_index)
{
    if (pixel < pixel_count) {
        best_range[pixel] = 0x7ff0000000000000ULL; // positive infinity
        best_index[pixel] = n;
    } else if (pixel == pixel_count) {
        best_index[pixel] = 0; // projected count, before visibility/depth gates
    }
}

__device__ void project_point(
    const long long i, const double* points, const long long n, const double* rt,
    const int width, const int height, const double fx, const double fy,
    const double cx, const double cy, const int kind, const double xi,
    const double alpha, const double k1, const double k2,
    const double p1, const double p2, const int z_buffer,
    long long* pixels, double* ranges, unsigned long long* best_range,
    unsigned long long* best_index)
{
    if (i >= n) return;
    pixels[i] = -1;
    const double px = points[3*i], py = points[3*i+1], pz = points[3*i+2];
    const double x = ((px*rt[0] + py*rt[1]) + pz*rt[2]) + rt[9];
    const double y = ((px*rt[3] + py*rt[4]) + pz*rt[5]) + rt[10];
    const double z = ((px*rt[6] + py*rt[7]) + pz*rt[8]) + rt[11];
    double distance = 0., u = 0., v = 0.;
    if (kind == 0) {
        if (!(z > 1.e-9)) return;
        u = fx*x/z + cx;
        v = fy*y/z + cy;
    } else if (kind == 1) {
        distance = sqrt((x*x + y*y) + z*z);
        const double z1 = xi*distance + z;
        const double d2 = sqrt((x*x + y*y) + z1*z1);
        const double denominator = alpha*d2 + (1.-alpha)*z1;
        const double w1 = alpha <= .5 ? alpha/(1.-alpha) : (1.-alpha)/alpha;
        const double w2 = (w1+xi)/sqrt((2.*w1*xi + xi*xi) + 1.);
        if (!(distance > 1.e-9 && denominator > 1.e-9 && z > -w2*distance)) return;
        u = fx*x/denominator + cx;
        v = fy*y/denominator + cy;
    } else {
        distance = sqrt((x*x + y*y) + z*z);
        if (!(distance > 1.e-9 && z > 1.e-9)) return;
        const double denominator = z/distance + xi;
        if (!(denominator > 1.e-9)) return;
        const double nx = (x/distance)/denominator;
        const double ny = (y/distance)/denominator;
        const double r2 = nx*nx + ny*ny;
        const double radial = (1. + k1*r2) + k2*(r2*r2);
        const double dx = (nx*radial + 2.*p1*nx*ny) + p2*(r2 + 2.*nx*nx);
        const double dy = (ny*radial + p1*(r2 + 2.*ny*ny)) + 2.*p2*nx*ny;
        u = fx*dx + cx;
        v = fy*dy + cy;
    }
    if (!(isfinite(u) && isfinite(v))) return;
    // rint is round-to-nearest-even, matching np.rint. Check bounds before
    // converting to integer so huge finite projections cannot overflow.
    u = rint(u);
    v = rint(v);
    if (!(u >= 0. && u < width && v >= 0. && v < height)) return;
    const long long pixel = (long long)v * width + (long long)u;
    pixels[i] = pixel;
    if (kind == 0) distance = sqrt((x*x + y*y) + z*z);
    ranges[i] = distance;
    if (z_buffer) {
        atomicMin(best_range + pixel, (unsigned long long)__double_as_longlong(distance));
    }
    const unsigned int active = __activemask();
    if ((threadIdx.x & 31) == __ffs(active)-1)
        atomicAdd(best_index + (long long)width*height, (unsigned long long)__popc(active));
}

__device__ void select_visible_point(
    const long long i, const long long* pixels, const double* ranges, const long long n,
    const unsigned long long* best_range, unsigned long long* best_index)
{
    if (i >= n || pixels[i] < 0) return;
    const long long pixel = pixels[i];
    if ((unsigned long long)__double_as_longlong(ranges[i]) == best_range[pixel])
        atomicMin(best_index + pixel, (unsigned long long)i);
}

__device__ int histogram_bin(const double value, const double* edges, const int bins)
{
    if (value < edges[0] || value > edges[bins]) return -1;
    if (value == edges[bins]) return bins-1; // NumPy includes the rightmost edge
    int lower = 0, upper = bins+1;
    while (lower < upper) {
        const int middle = (lower+upper)/2;
        if (value >= edges[middle]) lower = middle+1;
        else upper = middle;
    }
    return lower-1;
}

__device__ void histogram_pixel(
    const long long i, const long long n, const long long pixel_count, const int z_buffer,
    const long long* pixels, const unsigned long long* best_index,
    const double* depth, const double* lidar_range, const int bins,
    const double* edges, unsigned long long* histogram)
{
    const long long bins_squared = (long long)bins*bins;
    if (i == 0) histogram[bins_squared+1] = best_index[pixel_count];
    if (i >= (z_buffer ? pixel_count : n)) return;
    const long long index = z_buffer ? (long long)best_index[i] : i;
    if (index >= n) return;
    const long long pixel = z_buffer ? i : pixels[i];
    if (pixel < 0) return;
    const double camera_value = depth[pixel], lidar_value = lidar_range[index];
    if (!(isfinite(camera_value) && camera_value > 0. &&
          isfinite(lidar_value) && lidar_value > 0.)) return;
    // Count valid visible features even if the histogram range excludes them.
    const unsigned int active = __activemask();
    if ((threadIdx.x & 31) == __ffs(active)-1)
        atomicAdd(histogram+bins_squared, (unsigned long long)__popc(active));
    const int camera_bin = histogram_bin(camera_value, edges, bins);
    const int lidar_bin = histogram_bin(lidar_value, edges+bins+1, bins);
    if (camera_bin >= 0 && lidar_bin >= 0)
        atomicAdd(histogram + (long long)camera_bin*bins + lidar_bin, 1ULL);
}

extern "C" __global__ void initialize(
    const long long pixel_count, const long long n,
    unsigned long long* best_range, unsigned long long* best_index)
{
    initialize_pixel((long long)blockIdx.x * blockDim.x + threadIdx.x,
                     pixel_count, n, best_range, best_index);
}

extern "C" __global__ void project(
    const double* points, const long long n, const double* rt,
    const int width, const int height, const double fx, const double fy,
    const double cx, const double cy, const int kind, const double xi,
    const double alpha, const double k1, const double k2,
    const double p1, const double p2, const int z_buffer,
    long long* pixels, double* ranges, unsigned long long* best_range,
    unsigned long long* best_index)
{
    project_point((long long)blockIdx.x * blockDim.x + threadIdx.x,
                  points, n, rt, width, height, fx, fy, cx, cy, kind, xi,
                  alpha, k1, k2, p1, p2, z_buffer, pixels, ranges, best_range, best_index);
}

extern "C" __global__ void select_visible(
    const long long* pixels, const double* ranges, const long long n,
    const unsigned long long* best_range, unsigned long long* best_index)
{
    select_visible_point((long long)blockIdx.x * blockDim.x + threadIdx.x,
                         pixels, ranges, n, best_range, best_index);
}

extern "C" __global__ void histogram_pairs(
    const long long n, const long long pixel_count, const int z_buffer,
    const long long* pixels, const unsigned long long* best_index,
    const double* depth, const double* lidar_range, const int bins,
    const double* edges, unsigned long long* histogram)
{
    histogram_pixel((long long)blockIdx.x * blockDim.x + threadIdx.x,
                    n, pixel_count, z_buffer, pixels, best_index, depth,
                    lidar_range, bins, edges, histogram);
}
"""


@dataclass(frozen=True)
class _DeviceObservation:
    points: Any
    depth: Any
    lidar_range: Any


@dataclass(frozen=True)
class _DeviceProjectionWorkspace:
    pixels: Any
    ranges: Any
    best_range: Any
    best_index: Any


class CudaDepthPairProjector:
    """Callable D2D projector with a bounded, explicitly prepared GPU snapshot.

    Point coordinates, depth and original LiDAR ranges are prepared on the GPU.
    Unprepared or over-budget observations are transferred per call, so memory
    never grows with trials. Observations must remain unchanged during a run.
    The default 256 MiB budget covers cached inputs, excluding working buffers
    and CuPy's allocator pool. Use one benchmark worker per GPU.
    """

    def __init__(
        self,
        observations: Sequence[DepthToDepthObservation] = (),
        *,
        cache_bytes: int = DEFAULT_CUDA_CACHE_BYTES,
        candidate_workspace_bytes: int = 1024**3,
    ) -> None:
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be non-negative")
        if candidate_workspace_bytes < 0:
            raise ValueError("candidate_workspace_bytes must be non-negative")
        try:
            self._device = cp.cuda.Device()
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise ValueError("cuda projection requires an NVIDIA CUDA device")
            self._project = cp.RawKernel(_SOURCE, "project", options=("--fmad=false",))
            self._select = cp.RawKernel(_SOURCE, "select_visible", options=("--fmad=false",))
            self._initialize = cp.RawKernel(_SOURCE, "initialize")
            self._histogram = cp.RawKernel(_SOURCE, "histogram_pairs", options=("--fmad=false",))
            # Validate NVRTC/toolkit availability before a benchmark writes traces.
            self._project.compile()
            self._select.compile()
            self._initialize.compile()
            self._histogram.compile()
        except (RuntimeError, OSError, cp.cuda.compiler.CompileException) as exc:
            raise ValueError(
                f"cuda projection requires a working CuPy/CUDA runtime and CUDA toolkit: {exc}"
            ) from exc
        self._prepared: dict[int, tuple[DepthToDepthObservation, _DeviceObservation]] = {}
        self.cache_budget_bytes = cache_bytes
        self.cached_input_bytes: int = 0
        self._input_upload_calls = 0
        self._input_upload_bytes = 0
        self._histogram_download_calls = 0
        self._histogram_download_bytes = 0
        self._pose_batch: CudaPoseBatch | None = None
        self._candidate_workspace_bytes = candidate_workspace_bytes
        self._candidate_workspace_peak_bytes = 0
        try:
            for observation in observations:
                if id(observation) in self._prepared:
                    continue
                size = (
                    observation.lidar_points.nbytes
                    + observation.depth_map.nbytes
                    + observation.lidar_range_m.nbytes
                )
                if self.cached_input_bytes + size <= cache_bytes:
                    self._prepared[id(observation)] = (observation, self._upload(observation))
                    self.cached_input_bytes += size
        except cp.cuda.memory.OutOfMemoryError as exc:
            raise ValueError(
                "CUDA input cache allocation failed; lower --cuda-cache-mib "
                "(0 disables caching) and leave room for projection scratch buffers"
            ) from exc
        self._preparation_upload_calls = self._input_upload_calls
        self._preparation_upload_bytes = self._input_upload_bytes

    def _upload(self, observation: DepthToDepthObservation) -> _DeviceObservation:
        inputs = _DeviceObservation(
            cp.asarray(np.ascontiguousarray(observation.lidar_points, dtype=np.float64)),
            cp.asarray(np.ascontiguousarray(observation.depth_map, dtype=np.float64)),
            cp.asarray(np.ascontiguousarray(observation.lidar_range_m, dtype=np.float64)),
        )
        self._input_upload_calls += 1
        self._input_upload_bytes += int(
            inputs.points.nbytes + inputs.depth.nbytes + inputs.lidar_range.nbytes
        )
        return inputs

    def transfer_statistics(self) -> dict[str, int]:
        """Snapshot successful input uploads and histogram downloads.

        Input counters exclude the small pose and histogram edges. Histogram
        counters cover count downloads; pair-array downloads are excluded.
        They add no device synchronizations to the normal evaluation path.
        """

        return {
            "cache_budget_bytes": self.cache_budget_bytes,
            "cached_input_bytes": self.cached_input_bytes,
            "prepared_observation_count": len(self._prepared),
            "preparation_input_upload_calls": self._preparation_upload_calls,
            "preparation_input_upload_bytes": self._preparation_upload_bytes,
            "evaluation_input_upload_calls": self._input_upload_calls
            - self._preparation_upload_calls,
            "evaluation_input_upload_bytes": self._input_upload_bytes
            - self._preparation_upload_bytes,
            "histogram_download_calls": self._histogram_download_calls,
            "histogram_download_bytes": self._histogram_download_bytes,
            "candidate_cached_input_bytes": self._pose_batch.input_bytes if self._pose_batch else 0,
            "candidate_original_point_count": (
                self._pose_batch.original_points if self._pose_batch else 0
            ),
            "candidate_retained_point_count": (
                self._pose_batch.retained_points if self._pose_batch else 0
            ),
            "candidate_workspace_peak_bytes": self._candidate_workspace_peak_bytes,
        }

    def identity(self) -> str:
        """Return dependency, device and numerical-policy provenance."""

        with self._device:
            from calibrex.solvers.cuda_depth_to_depth_batch import _BATCH_SOURCE

            properties = cp.cuda.runtime.getDeviceProperties(self._device.id)
            name = properties["name"].decode("utf-8")
            nvrtc_major, nvrtc_minor = cp.cuda.nvrtc.getVersion()
            identity = (
                f"{CUDA_DEPTH_PAIR_PROJECTOR_VERSION};cupy={cp.__version__}"
                f";numpy={np.__version__};cuda_runtime={cp.cuda.runtime.runtimeGetVersion()}"
                f";cuda_driver={cp.cuda.runtime.driverGetVersion()};device={name}"
                f";compute={properties['major']}.{properties['minor']}"
                f";nvrtc={nvrtc_major}.{nvrtc_minor}"
                f";kernel_sha256={hashlib.sha256(_SOURCE.encode()).hexdigest()};fp64;fmad=false"
                f";histogram_batch_frames={_HISTOGRAM_BATCH_FRAMES}"
                f";candidate_batch_size=4;candidate_workspace_bytes={self._candidate_workspace_bytes}"
                f";batch_kernel_sha256={hashlib.sha256((_SOURCE+_BATCH_SOURCE).encode()).hexdigest()}"
            )
            # Default-budget aliases share an identity; explicit changes in
            # performance configuration must have a distinct runtime identity.
            if self.cache_budget_bytes != DEFAULT_CUDA_CACHE_BYTES:
                identity += f";cache_bytes={self.cache_budget_bytes}"
            return identity

    def prepare_pose_search(
        self, observations: Sequence[DepthToDepthObservation], initial: SE3, *,
        rotation_bound_deg: float, translation_bound_m: float,
    ) -> None:
        """Prepare one search; input copies share the configured cache byte budget.

        Scratch is separately bounded to 1 GiB by default. Uncached inputs,
        more than 32 frames, and allocation failures use the ordinary path.
        """

        from calibrex.solvers.cuda_depth_to_depth_batch import MAX_BATCH_FRAMES, CudaPoseBatch

        if not all(math.isfinite(value) and value > 0.0
                   for value in (rotation_bound_deg, translation_bound_m)):
            raise ValueError("pose search bounds must be finite and positive")
        self._pose_batch = None
        if (not observations or len(observations) > MAX_BATCH_FRAMES
                or not self._candidate_workspace_bytes
                or any(id(item) not in self._prepared for item in observations)):
            return
        # Refuse oversized scratch before any compact copies or allocation.
        minimum = sum((item.camera.width*item.camera.height+1)*16 for item in observations)
        if minimum+16 > self._candidate_workspace_bytes:
            return
        with self._device:
            try:
                batch = CudaPoseBatch(
                    observations, self._prepared, initial, rotation_bound_deg, translation_bound_m,
                    input_budget_bytes=self.cache_budget_bytes-self.cached_input_bytes,
                    workspace_budget_bytes=self._candidate_workspace_bytes,
                )
            except (cp.cuda.memory.OutOfMemoryError, ValueError):
                return
        self._pose_batch = batch
        self._candidate_workspace_peak_bytes = max(
            self._candidate_workspace_peak_bytes, batch.workspace_bytes,
        )

    def histogram_depth_pairs_poses(
        self, observations: Sequence[DepthToDepthObservation], transforms: Sequence[SE3], *,
        use_z_buffer: bool, bins: int, camera_range: tuple[float, float],
        lidar_range: tuple[float, float],
    ) -> Iterator[tuple[DepthPairHistogram, ...]]:
        """Yield one ordered frame tuple per pose, using bounded prepared scratch."""

        _validate_histogram_configuration(bins, camera_range, lidar_range)
        batch = self._pose_batch
        if batch is None or not batch.supports(observations, transforms):
            for transform in transforms:
                yield tuple(self.histogram_depth_pairs_batch(
                    tuple(observations), transform, use_z_buffer=use_z_buffer, bins=bins,
                    camera_range=camera_range, lidar_range=lidar_range,
                ))
            return
        try:
            with self._device:
                for host in batch.histogram_chunks(
                    transforms, use_z_buffer=use_z_buffer, bins=bins,
                    camera_range=camera_range, lidar_range=lidar_range,
                    download=self._download_histograms,
                ):
                    for pose_rows in host:
                        yield tuple(self._host_histogram(row, bins) for row in pose_rows)
        except Exception:
            # A failed launch/download cannot leave reusable, partially updated scratch.
            self._pose_batch = None
            raise

    def __call__(
        self,
        observation: DepthToDepthObservation,
        transform_camera_lidar: SE3,
        *,
        use_z_buffer: bool = True,
    ) -> DepthPairProjection:
        """Project one observation, retaining CPU ordering and tie semantics."""

        if not observation.lidar_points.size:
            return _empty_projection()
        with self._device:
            return self._project_pairs(observation, transform_camera_lidar, use_z_buffer)

    def _project_pairs(
        self,
        observation: DepthToDepthObservation,
        transform: SE3,
        use_z_buffer: bool,
    ) -> DepthPairProjection:
        pixels, best_index, _inputs = self._device_projection(observation, transform, use_z_buffer)
        count = observation.lidar_points.shape[0]
        camera = observation.camera
        if use_z_buffer:
            selected = cp.asnumpy(best_index)
            projected_count = int(selected[-1])
            occupied = np.flatnonzero(selected[:-1] < count)
            indices = selected[occupied].astype(np.int64)
        else:
            projected_pixels = cp.asnumpy(pixels)
            indices = np.flatnonzero(projected_pixels >= 0)
            occupied = projected_pixels[indices]
            projected_count = int(indices.size)
        pixel_u = occupied % camera.width
        pixel_v = occupied // camera.width
        camera_depth = observation.depth_map[pixel_v, pixel_u]
        lidar_range = observation.lidar_range_m[indices]
        finite = (
            np.isfinite(camera_depth)
            & (camera_depth > 0.0)
            & np.isfinite(lidar_range)
            & (lidar_range > 0.0)
        )
        return DepthPairProjection(
            camera_depth=np.asarray(camera_depth[finite], dtype=np.float64),
            lidar_range_m=np.asarray(lidar_range[finite], dtype=np.float64),
            pixel_u=pixel_u[finite],
            pixel_v=pixel_v[finite],
            projected_count_before_visibility=projected_count,
        )

    def _device_projection(
        self,
        observation: DepthToDepthObservation,
        transform: SE3,
        use_z_buffer: bool,
        *,
        rt: Any | None = None,
        workspace: _DeviceProjectionWorkspace | None = None,
    ) -> tuple[Any, Any, _DeviceObservation]:
        prepared = self._prepared.get(id(observation))
        inputs = prepared[1] if prepared is not None else self._upload(observation)
        count = observation.lidar_points.shape[0]
        camera = observation.camera
        pixel_count = camera.width * camera.height
        if rt is None:
            rotation = _rotation_matrix(transform.rotation_quat_xyzw)
            rt = cp.asarray(np.concatenate((rotation.ravel(), transform.translation_m)))
        if workspace is None:
            workspace = self._projection_workspace(count, pixel_count)
        pixels, ranges = workspace.pixels, workspace.ranges
        best_range, best_index = workspace.best_range, workspace.best_index
        self._initialize(
            ((pixel_count + 1 + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (np.int64(pixel_count), np.int64(count), best_range, best_index),
        )
        distortion = (*camera.distortion, 0.0, 0.0, 0.0, 0.0)
        grid = ((count + _THREADS - 1) // _THREADS,)
        self._project(
            grid,
            (_THREADS,),
            (
                inputs.points,
                np.int64(count),
                rt,
                np.int32(camera.width),
                np.int32(camera.height),
                np.float64(camera.fx),
                np.float64(camera.fy),
                np.float64(camera.cx),
                np.float64(camera.cy),
                np.int32({"pinhole": 0, "double_sphere": 1, "mei": 2}[camera.projection]),
                np.float64(camera.xi),
                np.float64(camera.alpha),
                *(np.float64(value) for value in distortion[:4]),
                np.int32(use_z_buffer),
                pixels,
                ranges,
                best_range,
                best_index,
            ),
        )
        if use_z_buffer:
            self._select(
                grid, (_THREADS,), (pixels, ranges, np.int64(count), best_range, best_index)
            )
        return pixels, best_index, inputs

    @staticmethod
    def _projection_workspace(count: int, pixel_count: int) -> _DeviceProjectionWorkspace:
        return _DeviceProjectionWorkspace(
            cp.empty(count, dtype=cp.int64),
            cp.empty(count, dtype=cp.float64),
            cp.empty(pixel_count, dtype=cp.uint64),
            cp.empty(pixel_count + 1, dtype=cp.uint64),
        )

    def histogram_depth_pairs(
        self,
        observation: DepthToDepthObservation,
        transform_camera_lidar: SE3,
        *,
        use_z_buffer: bool,
        bins: int,
        camera_range: tuple[float, float],
        lidar_range: tuple[float, float],
    ) -> DepthPairHistogram:
        """Return exact NumPy-compatible histogram counts with one small transfer."""

        if not observation.lidar_points.size:
            return DepthPairHistogram(np.zeros((bins, bins)), 0, 0)
        with self._device:
            pixels, best_index, inputs = self._device_projection(
                observation, transform_camera_lidar, use_z_buffer
            )
            edges = cp.asarray(
                np.concatenate(
                    (
                        np.linspace(*camera_range, bins + 1),
                        np.linspace(*lidar_range, bins + 1),
                    )
                )
            )
            result = cp.zeros(bins * bins + 2, dtype=cp.uint64)
            self._enqueue_histogram(
                observation, pixels, best_index, inputs, use_z_buffer, bins, edges, result
            )
            host = self._download_histograms(result)
        return self._host_histogram(host, bins)

    def _enqueue_histogram(
        self,
        observation: DepthToDepthObservation,
        pixels: Any,
        best_index: Any,
        inputs: _DeviceObservation,
        use_z_buffer: bool,
        bins: int,
        edges: Any,
        result: Any,
    ) -> None:
        count = observation.lidar_points.shape[0]
        pixel_count = observation.camera.width * observation.camera.height
        work = pixel_count if use_z_buffer else count
        self._histogram(
            ((work + _THREADS - 1) // _THREADS,),
            (_THREADS,),
            (
                np.int64(count),
                np.int64(pixel_count),
                np.int32(use_z_buffer),
                pixels,
                best_index,
                inputs.depth,
                inputs.lidar_range,
                np.int32(bins),
                edges,
                result,
            ),
        )

    def _download_histograms(self, result: Any) -> Any:
        host = cp.asnumpy(result)
        self._histogram_download_calls += 1
        self._histogram_download_bytes += int(host.nbytes)
        return host

    @staticmethod
    def _host_histogram(host: Any, bins: int) -> DepthPairHistogram:
        return DepthPairHistogram(
            histogram=host[: bins * bins].astype(np.float64).reshape(bins, bins),
            visible_point_count=int(host[-2]),
            projected_count_before_visibility=int(host[-1]),
        )

    def histogram_depth_pairs_batch(
        self,
        observations: tuple[DepthToDepthObservation, ...] | list[DepthToDepthObservation],
        transform_camera_lidar: SE3,
        *,
        use_z_buffer: bool,
        bins: int,
        camera_range: tuple[float, float],
        lidar_range: tuple[float, float],
    ) -> Iterator[DepthPairHistogram]:
        """Stream ordered histograms with one download per at most 32 frames.

        One pose/edge upload and one scratch workspace are shared within this
        invocation. Kernels for successive frames reuse scratch in current-stream
        order. Device results are bounded to 32 rows; no trial cache is retained.
        """

        _validate_histogram_configuration(bins, camera_range, lidar_range)
        if not observations:
            return
        with self._device:
            rotation = _rotation_matrix(transform_camera_lidar.rotation_quat_xyzw)
            rt = cp.asarray(
                np.concatenate(
                    (
                        rotation.ravel(),
                        transform_camera_lidar.translation_m,
                    )
                )
            )
            edges = cp.asarray(
                np.concatenate(
                    (
                        np.linspace(*camera_range, bins + 1),
                        np.linspace(*lidar_range, bins + 1),
                    )
                )
            )
            workspace = self._projection_workspace(
                max(item.lidar_points.shape[0] for item in observations),
                max(item.camera.width * item.camera.height for item in observations),
            )
        for start in range(0, len(observations), _HISTOGRAM_BATCH_FRAMES):
            chunk = observations[start : start + _HISTOGRAM_BATCH_FRAMES]
            with self._device:
                result = cp.zeros((len(chunk), bins * bins + 2), dtype=cp.uint64)
                for index, observation in enumerate(chunk):
                    if not observation.lidar_points.size:
                        continue
                    pixels, best_index, inputs = self._device_projection(
                        observation,
                        transform_camera_lidar,
                        use_z_buffer,
                        rt=rt,
                        workspace=workspace,
                    )
                    self._enqueue_histogram(
                        observation,
                        pixels,
                        best_index,
                        inputs,
                        use_z_buffer,
                        bins,
                        edges,
                        result[index],
                    )
                host = self._download_histograms(result)
                del result
            for row in host:
                yield self._host_histogram(row, bins)
