"""Bounded CUDA candidate batches with conservative pinhole frustum pruning."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Final

import cupy as cp
import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthObservation, _rotation_matrix
from calibrex.solvers.cuda_depth_to_depth_adapter import (
    _SOURCE,
    _THREADS,
    _DeviceObservation,
    _DeviceProjectionWorkspace,
)

CANDIDATE_BATCH_SIZE: Final = 4
CANDIDATE_WORKSPACE_BYTES: Final = 1024**3
MAX_BATCH_FRAMES: Final = 32

# A frame/pose pair owns a disjoint slice. Every phase is a separate launch:
# reset finishes before projection, and selection finishes before histogramming.
# Reset only pixels touched in that slot's previous pose. Atomic exchanges avoid
# concurrent writes even when several points formerly occupied the same pixel.
_BATCH_SOURCE: Final = r"""
extern "C" __global__ void conservative_mask(
    const double* points, const long long n, const double* rt,
    const int width, const int height, const double fx, const double fy,
    const double cx, const double cy, const double rotation_factor,
    const double translation_bound, const double initial_translation_norm, bool* keep)
{
    const long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    const double px=points[3*i], py=points[3*i+1], pz=points[3*i+2];
    const double norm=sqrt((px*px+py*py)+pz*pz);
    const double x=((px*rt[0]+py*rt[1])+pz*rt[2])+rt[9];
    const double y=((px*rt[3]+py*rt[4])+pz*rt[5])+rt[10];
    const double z=((px*rt[6]+py*rt[7])+pz*rt[8])+rt[11];
    const double radius=rotation_factor*norm+1.e-9*(norm+initial_translation_norm+1.);
    const double a[5]={0.,fx,-fx,fy,-fy};
    const double b[5]={1.,cx+.5,width-.5-cx,cy+.5,height-.5-cy};
    bool retained=true;
    for (int plane=0;plane<5;++plane) {
        const double coordinate=plane<3 ? x : y;
        const double score=coordinate*a[plane]+z*b[plane];
        const double margin=sqrt(a[plane]*a[plane]+b[plane]*b[plane])*radius
                            +(fabs(a[plane])+fabs(b[plane]))*translation_bound;
        if (isfinite(score) && isfinite(margin) && score < -margin) retained=false;
    }
    keep[i]=retained;
}

extern "C" __global__ void initialize_batch(
    const long long* metadata, const long long point_stride, const long long pixel_stride,
    unsigned long long* best_range, unsigned long long* best_index)
{
    const long long* m = metadata + blockIdx.y*7;
    const long long offset = m[6] + blockIdx.z*pixel_stride;
    initialize_pixel((long long)blockIdx.x*blockDim.x+threadIdx.x,
                     m[1], m[0], best_range+offset, best_index+offset);
}

extern "C" __global__ void reset_batch(
    const long long* metadata, const long long point_stride, const long long pixel_stride,
    const long long* pixels, unsigned long long* best_range, unsigned long long* best_index)
{
    const long long* m = metadata + blockIdx.y*7;
    const long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long po = m[5]+blockIdx.z*point_stride;
    const long long bo = m[6]+blockIdx.z*pixel_stride;
    if (i == 0) best_index[bo+m[1]] = 0;
    if (i >= m[0]) return;
    const long long pixel = pixels[po+i];
    if (pixel >= 0) {
        atomicExch(best_range+bo+pixel, 0x7ff0000000000000ULL);
        atomicExch(best_index+bo+pixel, (unsigned long long)m[0]);
    }
}

extern "C" __global__ void project_batch(
    const long long* metadata, const unsigned long long* pointers, const double* camera,
    const double* rt, const long long point_stride, const long long pixel_stride,
    const int z_buffer, long long* pixels, double* ranges,
    unsigned long long* best_range, unsigned long long* best_index)
{
    const long long* m = metadata + blockIdx.y*7;
    const long long po = m[5]+blockIdx.z*point_stride;
    const long long bo = m[6]+blockIdx.z*pixel_stride;
    const double* c = camera+blockIdx.y*10;
    project_point((long long)blockIdx.x*blockDim.x+threadIdx.x,
        (const double*)pointers[blockIdx.y*3], m[0], rt+blockIdx.z*12,
        (int)m[2], (int)m[3], c[0], c[1], c[2], c[3], (int)m[4],
        c[4], c[5], c[6], c[7], c[8], c[9], z_buffer,
        pixels+po, ranges+po, best_range+bo, best_index+bo);
}

extern "C" __global__ void select_batch(
    const long long* metadata, const long long point_stride, const long long pixel_stride,
    const long long* pixels, const double* ranges,
    const unsigned long long* best_range, unsigned long long* best_index)
{
    const long long* m = metadata + blockIdx.y*7;
    const long long po = m[5]+blockIdx.z*point_stride;
    const long long bo = m[6]+blockIdx.z*pixel_stride;
    select_visible_point((long long)blockIdx.x*blockDim.x+threadIdx.x,
                         pixels+po, ranges+po, m[0], best_range+bo, best_index+bo);
}

extern "C" __global__ void histogram_batch(
    const long long* metadata, const unsigned long long* pointers,
    const long long point_stride, const long long pixel_stride, const int frames,
    const int z_buffer, const long long* pixels, const unsigned long long* best_index,
    const int bins, const double* edges, unsigned long long* histogram)
{
    const long long* m = metadata + blockIdx.y*7;
    const long long i = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long po = m[5]+blockIdx.z*point_stride;
    const long long bo = m[6]+blockIdx.z*pixel_stride;
    const long long bins_squared = (long long)bins*bins;
    histogram += (blockIdx.z*frames+blockIdx.y)*(bins_squared+2);
    if (i == 0) histogram[bins_squared+1] = best_index[bo+m[1]];
    if (i >= m[0]) return;
    const long long pixel = pixels[po+i];
    if (pixel < 0 || (z_buffer && best_index[bo+pixel] != (unsigned long long)i)) return;
    const double camera_value = ((const double*)pointers[blockIdx.y*3+1])[pixel];
    const double lidar_value = ((const double*)pointers[blockIdx.y*3+2])[i];
    if (!(isfinite(camera_value) && camera_value > 0. &&
          isfinite(lidar_value) && lidar_value > 0.)) return;
    const unsigned int active = __activemask();
    if ((threadIdx.x & 31) == __ffs(active)-1)
        atomicAdd(histogram+bins_squared, (unsigned long long)__popc(active));
    const int camera_bin = histogram_bin(camera_value, edges, bins);
    const int lidar_bin = histogram_bin(lidar_value, edges+bins+1, bins);
    if (camera_bin < 0 || lidar_bin < 0) return;
    const long long bin = (long long)camera_bin*bins+lidar_bin;
    #if __CUDA_ARCH__ >= 700
    const unsigned int matches = __match_any_sync(__activemask(), bin);
    if ((threadIdx.x & 31) == __ffs(matches)-1)
        atomicAdd(histogram+bin, (unsigned long long)__popc(matches));
    #else
    atomicAdd(histogram+bin, 1ULL);
    #endif
}
"""


def conservative_pinhole_mask(
    observation: DepthToDepthObservation, initial: SE3,
    rotation_bound_deg: float, translation_bound_m: float,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Retain every point that could project anywhere in the declared search.

    Three Euler bounds imply a rotation angle at most their sum. Rotation moves
    a point by at most 2*sin(angle/2)*||point||. Each translation component can
    move by its bound. A frustum plane therefore excludes a point only beyond
    ||normal||*rotation_displacement + ||normal||_1*translation_bound.
    Pixel limits include both half-pixel boundaries; a generous floating-point
    guard keeps ambiguous points. Other camera models retain all their points.
    """

    points, camera = observation.lidar_points, observation.camera
    keep = np.ones(len(points), dtype=np.bool_)
    if camera.projection != "pinhole" or not len(points):
        return keep
    rotation = _rotation_matrix(initial.rotation_quat_xyzw)
    angle = min(math.pi, math.radians(3.0*rotation_bound_deg))
    planes = (
        (0., 0., 1.),
        (camera.fx, 0., camera.cx+0.5),
        (-camera.fx, 0., camera.width-0.5-camera.cx),
        (0., camera.fy, camera.cy+0.5),
        (0., -camera.fy, camera.height-0.5-camera.cy),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        norm = np.linalg.norm(points, axis=1)
        xyz = ((points[:, 0, None]*rotation[:, 0] + points[:, 1, None]*rotation[:, 1])
               + points[:, 2, None]*rotation[:, 2]) + initial.translation_m
        radius = 2.0*math.sin(angle/2.0)*norm
        guard = 1.e-9*(norm+np.linalg.norm(initial.translation_m)+1.0)
        for coefficients in planes:
            plane = np.asarray(coefficients)
            score = (xyz[:, 0]*plane[0]+xyz[:, 1]*plane[1])+xyz[:, 2]*plane[2]
            margin = np.linalg.norm(plane)*(radius+guard)
            margin += np.abs(plane).sum()*translation_bound_m
            # Non-finite interval arithmetic is ambiguous and retains the point.
            keep &= ~(np.isfinite(score) & np.isfinite(margin) & (score < -margin))
    return keep


class CudaPoseBatch:
    """One search's bounded scratch and optional compact, ordered input copies."""

    def __init__(
        self, observations: Sequence[DepthToDepthObservation],
        prepared: dict[int, tuple[DepthToDepthObservation, _DeviceObservation]],
        initial: SE3, rotation_bound_deg: float, translation_bound_m: float,
        *, input_budget_bytes: int, workspace_budget_bytes: int,
    ) -> None:
        self.observations = tuple(observations)
        self.initial = initial
        self.angle = min(math.pi, math.radians(3.0*rotation_bound_deg))
        self.translation_bound = translation_bound_m
        self.input_bytes = 0
        self.original_points = sum(len(item.lidar_points) for item in observations)
        self.inputs: list[_DeviceObservation] = []
        names = (
            "conservative_mask", "initialize_batch", "reset_batch", "project_batch",
            "select_batch", "histogram_batch",
        )
        kernels = [
            cp.RawKernel(_SOURCE+_BATCH_SOURCE, name, options=("--fmad=false",)) for name in names
        ]
        for kernel in kernels:
            kernel.compile()
        cull, initialize, self.reset, self.project, self.select, self.histogram = kernels
        initial_rt = cp.asarray(np.concatenate((
            _rotation_matrix(initial.rotation_quat_xyzw).ravel(), initial.translation_m,
        )))
        metadata, pointers, cameras = [], [], []
        point_offset = pixel_offset = 0
        for observation in observations:
            inputs = prepared[id(observation)][1]
            camera = observation.camera
            original_count = len(observation.lidar_points)
            if camera.projection == "pinhole" and original_count:
                mask = cp.empty(original_count, dtype=cp.bool_)
                cull(((original_count+_THREADS-1)//_THREADS,), (_THREADS,), (
                    inputs.points, np.int64(original_count), initial_rt,
                    np.int32(camera.width), np.int32(camera.height),
                    np.float64(camera.fx), np.float64(camera.fy),
                    np.float64(camera.cx), np.float64(camera.cy),
                    np.float64(2.0*math.sin(self.angle/2.0)), np.float64(translation_bound_m),
                    np.float64(np.linalg.norm(initial.translation_m)), mask,
                ))
                indices = cp.flatnonzero(mask)  # Stable input order, including visibility ties.
                size = int(indices.size)*32
                if indices.size < original_count and self.input_bytes+size <= input_budget_bytes:
                    inputs = _DeviceObservation(
                        inputs.points[indices], inputs.depth, inputs.lidar_range[indices],
                    )
                    self.input_bytes += size
            self.inputs.append(inputs)  # Keeps all pointer-table targets alive.
            count = int(inputs.points.shape[0])
            metadata.append((count, camera.width*camera.height, camera.width, camera.height,
                             {"pinhole": 0, "double_sphere": 1, "mei": 2}[camera.projection],
                             point_offset, pixel_offset))
            pointers.append((
                inputs.points.data.ptr, inputs.depth.data.ptr, inputs.lidar_range.data.ptr,
            ))
            cameras.append((camera.fx, camera.fy, camera.cx, camera.cy, camera.xi, camera.alpha,
                            *(*camera.distortion, 0., 0., 0., 0.)[:4]))
            point_offset += count
            pixel_offset += camera.width*camera.height+1
        self.retained_points = point_offset
        self.point_stride = np.int64(max(1, point_offset))
        self.pixel_stride = np.int64(pixel_offset)
        per_pose = int(self.point_stride*16+self.pixel_stride*16)
        self.capacity = min(CANDIDATE_BATCH_SIZE, workspace_budget_bytes//per_pose)
        if self.capacity < 1:
            raise ValueError("candidate workspace exceeds its byte budget")
        self.metadata = cp.asarray(metadata, dtype=cp.int64)
        self.pointers = cp.asarray(pointers, dtype=cp.uint64)
        self.cameras = cp.asarray(cameras, dtype=cp.float64)
        self.workspace = _DeviceProjectionWorkspace(
            cp.full(int(self.point_stride)*self.capacity, -1, dtype=cp.int64),
            cp.empty(int(self.point_stride)*self.capacity, dtype=cp.float64),
            cp.empty(pixel_offset*self.capacity, dtype=cp.uint64),
            cp.empty(pixel_offset*self.capacity, dtype=cp.uint64),
        )
        self.workspace_bytes = per_pose*self.capacity
        self.grid = (max(1, (max(row[0] for row in metadata)+_THREADS-1)//_THREADS),
                     len(observations))
        initialize(
            ((max(row[1] for row in metadata)+_THREADS)//_THREADS,
             len(observations), self.capacity),
            (_THREADS,),
            (self.metadata, self.point_stride, self.pixel_stride,
             self.workspace.best_range, self.workspace.best_index),
        )
        self.ready = cp.cuda.Event()
        self.ready.record()

    def supports(
        self, observations: Sequence[DepthToDepthObservation], poses: Sequence[SE3],
    ) -> bool:
        """Fall back to full inputs for different frames or poses outside the bound."""

        if tuple(map(id, observations)) != tuple(map(id, self.observations)):
            return False
        for pose in poses:
            dot = abs(sum(a*b for a, b in zip(
                pose.rotation_quat_xyzw, self.initial.rotation_quat_xyzw, strict=True,
            )))
            if dot < math.cos(self.angle/2.0)-128.0*np.finfo(float).eps:
                return False
            if any(abs(a-b) > self.translation_bound+1.e-10*(abs(b)+1.0)
                   for a, b in zip(pose.translation_m, self.initial.translation_m, strict=True)):
                return False
        return True

    def histogram_chunks(
        self, poses: Sequence[SE3], *, use_z_buffer: bool, bins: int,
        camera_range: tuple[float, float], lidar_range: tuple[float, float],
        download: Callable[[Any], Any],
    ) -> Iterator[Any]:
        """Download ordered pose/frame counts, retaining only one device chunk."""

        cp.cuda.get_current_stream().wait_event(self.ready)
        edges = cp.asarray(np.concatenate((np.linspace(*camera_range, bins+1),
                                           np.linspace(*lidar_range, bins+1))))
        w = self.workspace
        z_buffer = np.int32(use_z_buffer)
        for start in range(0, len(poses), self.capacity):
            chunk = poses[start:start+self.capacity]
            rt = cp.asarray(np.asarray([
                np.concatenate((
                    _rotation_matrix(pose.rotation_quat_xyzw).ravel(), pose.translation_m,
                ))
                for pose in chunk
            ]))
            result = cp.zeros((len(chunk), len(self.observations), bins*bins+2), dtype=cp.uint64)
            grid = (*self.grid, len(chunk))
            self.reset(grid, (_THREADS,), (self.metadata, self.point_stride, self.pixel_stride,
                                         w.pixels, w.best_range, w.best_index))
            self.project(grid, (_THREADS,), (
                self.metadata, self.pointers, self.cameras, rt,
                self.point_stride, self.pixel_stride,
                z_buffer, w.pixels, w.ranges, w.best_range, w.best_index,
            ))
            if use_z_buffer:
                self.select(grid, (_THREADS,), (self.metadata, self.point_stride, self.pixel_stride,
                                              w.pixels, w.ranges, w.best_range, w.best_index))
            self.histogram(grid, (_THREADS,), (
                self.metadata, self.pointers, self.point_stride, self.pixel_stride,
                np.int32(len(self.observations)), z_buffer, w.pixels, w.best_index,
                np.int32(bins), edges, result,
            ))
            self.ready.record()
            host = download(result)
            del result
            yield host
