"""Optional ordered CPU projection and separable bounded image spilloff."""

from __future__ import annotations

import hashlib
import inspect
import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
from numba import njit
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.solvers.levinson_thrun_online_solver import (
    FloatArray,
    LevinsonObjectiveEvaluation,
    LevinsonOnlineFrame,
    OnlineCpuExecution,
    _edge_response_input,
    _project_response,
    _rotation_matrix,
)


@njit(cache=False, nogil=True, fastmath=False)  # type: ignore[untyped-decorator]
def _interpolate(
    rotated: FloatArray,
    rotation_indices: NDArray[np.int64],
    translations: FloatArray,
    source_weights: FloatArray,
    image: FloatArray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    forward_axes: bool,
    apply_translation: bool,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    responses = np.empty((len(rotation_indices), rotated.shape[1]), dtype=np.float64)
    weights = np.empty_like(responses)
    counts = np.empty(len(rotation_indices), dtype=np.int64)
    height, width = image.shape
    for pose in range(len(rotation_indices)):
        points = rotated[rotation_indices[pose]]
        tx, ty, tz = translations[pose]
        count = 0
        for index in range(len(points)):
            if apply_translation:
                x0, y0, z0 = points[index, 0] + tx, points[index, 1] + ty, points[index, 2] + tz
            else:
                x0, y0, z0 = points[index, 0], points[index, 1], points[index, 2]
            if forward_axes:
                px, py, pz = -y0, -z0, x0
            else:
                px, py, pz = x0, y0, z0
            if not pz > 1.0e-6:
                continue
            x, y = px / pz, py / pz
            u, v = fx * x + cx, fy * y + cy
            if not (u >= 0.0 and v >= 0.0 and u < width - 1 and v < height - 1):
                continue
            left, top = math.floor(u), math.floor(v)
            du, dv = u - left, v - top
            responses[pose, count] = (
                (1.0 - du) * (1.0 - dv) * image[top, left]
                + du * (1.0 - dv) * image[top, left + 1]
                + (1.0 - du) * dv * image[top + 1, left]
                + du * dv * image[top + 1, left + 1]
            )
            weights[pose, count] = source_weights[index]
            count += 1
        counts[pose] = count
    return responses, weights, counts


def _image_response(image: FloatArray, alpha: float, gamma: float, radius: int) -> FloatArray:
    from calibrex.solvers.levinson_thrun_online_solver import _maximum_neighbor_difference

    edge = _maximum_neighbor_difference(image)
    spread = edge.copy()
    previous = edge.copy()
    horizontal = np.empty_like(image)
    next_frontier = np.empty_like(image)
    for _ in range(radius):
        np.copyto(horizontal, previous)
        np.maximum(horizontal[:, 1:], previous[:, :-1], out=horizontal[:, 1:])
        np.maximum(horizontal[:, :-1], previous[:, 1:], out=horizontal[:, :-1])
        np.copyto(next_frontier, horizontal)
        np.maximum(next_frontier[1:], horizontal[:-1], out=next_frontier[1:])
        np.maximum(next_frontier[:-1], horizontal[1:], out=next_frontier[:-1])
        np.multiply(gamma, next_frontier, out=next_frontier)
        np.maximum(spread, next_frontier, out=spread)
        previous, next_frontier = next_frontier, previous
    return alpha * edge + (1.0 - alpha) * spread


KERNEL_SHA256 = hashlib.sha256(inspect.getsource(_interpolate.py_func).encode()).hexdigest()
IMAGE_KERNEL_SHA256 = hashlib.sha256(inspect.getsource(_image_response).encode()).hexdigest()
_GRID_CHUNK_SIZE = 64
_GRID_MAX_POSES = 729
_GRID_BUFFER_BYTES = 64 * 1024**2


def _raw_chunk(
    frame: LevinsonOnlineFrame,
    rotated: FloatArray,
    indices: NDArray[np.int64],
    translations: FloatArray,
) -> list[float]:
    camera = frame.camera
    responses, weights, counts = _interpolate(
        rotated,
        indices,
        translations,
        frame.discontinuity_weights,
        frame.edge_response,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.axes == "forward_x_left_y_up_z",
        True,
    )
    return [
        float(responses[index, :count] @ weights[index, :count])
        for index, count in enumerate(counts)
    ]


def numba_edge_response(
    image: FloatArray, *, alpha: float, gamma: float, max_radius: int
) -> FloatArray:
    """Keep native FP64 operations while avoiding stacked neighborhood arrays."""
    image = _edge_response_input(image, alpha, gamma, max_radius)
    if not image.size:
        from calibrex.solvers.levinson_thrun_online_solver import levinson_edge_response

        return levinson_edge_response(image, alpha=alpha, gamma=gamma, max_radius=max_radius)
    result: FloatArray = _image_response(np.ascontiguousarray(image), alpha, gamma, max_radius)
    return result


class NumbaOnlineEdgeRuntime:
    """Per-solve counters with native transforms, dot products and sum order.

    Like NumPy evaluation, this synchronous path requires callers to leave input
    arrays unchanged during a solve. It retains no input or result array cache.
    """

    def __init__(self) -> None:
        self._counts = dict.fromkeys(
            (
                "objective_evaluations",
                "point_rows",
                "grid_calls",
                "grid_pose_evaluations",
                "numba_frame_pose_evaluations",
                "native_frame_pose_evaluations",
            ),
            0,
        )

    def project(self, frame: LevinsonOnlineFrame, pose: SE3) -> tuple[FloatArray, FloatArray]:
        """Fuse pinhole interpolation; retain native unsupported projections."""
        if frame.camera.projection != "pinhole":
            self._counts["native_frame_pose_evaluations"] += 1
            return _project_response(frame, pose)
        return self._project_prepared(
            frame, _rotation_matrix(pose), np.asarray(pose.translation_m, dtype=float)
        )

    def _project_prepared(
        self, frame: LevinsonOnlineFrame, rotation: FloatArray, translation: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        translated = frame.lidar_points @ rotation.T + translation
        camera = frame.camera
        responses, weights, counts = _interpolate(
            translated.reshape(1, len(translated), 3),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 3), dtype=float),
            frame.discontinuity_weights,
            frame.edge_response,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera.axes == "forward_x_left_y_up_z",
            False,
        )
        self._counts["numba_frame_pose_evaluations"] += 1
        return responses[0, : counts[0]], weights[0, : counts[0]]

    def _evaluate(
        self, frames: Sequence[LevinsonOnlineFrame], pose: SE3, *, statistics: bool
    ) -> LevinsonObjectiveEvaluation:
        self._counts["objective_evaluations"] += 1
        raw, total_weight, projected_count = 0.0, 0.0, 0
        rotation = _rotation_matrix(pose)
        translation = np.asarray(pose.translation_m, dtype=float)
        for frame in frames:
            if frame.camera.projection == "pinhole":
                response, weights = self._project_prepared(frame, rotation, translation)
            else:
                response, weights = self.project(frame, pose)
            self._counts["point_rows"] += len(frame.lidar_points)
            raw += float(response @ weights)
            if statistics:
                total_weight += float(np.sum(weights))
                projected_count += int(weights.size)
        return LevinsonObjectiveEvaluation(
            raw,
            raw / total_weight if total_weight > 0.0 else 0.0,
            projected_count,
            total_weight,
            len(frames),
        )

    def evaluate(
        self, frames: Sequence[LevinsonOnlineFrame], pose: SE3
    ) -> LevinsonObjectiveEvaluation:
        """Evaluate complete support and normalized statistics in native order."""
        return self._evaluate(frames, pose, statistics=True)

    def raw_scores(
        self,
        frames: Sequence[LevinsonOnlineFrame],
        poses: Sequence[SE3],
        center: LevinsonObjectiveEvaluation,
    ) -> list[float]:
        """Share identical rotations within a bounded, ordered grid evaluation."""
        del center
        self._counts["grid_calls"] += 1
        self._counts["grid_pose_evaluations"] += len(poses)
        if len(poses) < _GRID_CHUNK_SIZE or len(poses) > _GRID_MAX_POSES:
            return [self._evaluate(frames, pose, statistics=False).raw_objective for pose in poses]
        rotations: list[FloatArray] = []
        lookup: dict[bytes, int] = {}
        ids: list[int] = []
        for pose in poses:
            matrix = _rotation_matrix(pose)
            key = matrix.tobytes()
            if key not in lookup:
                lookup[key] = len(rotations)
                rotations.append(matrix)
            ids.append(lookup[key])
        if len(rotations) == len(poses):
            return [self._evaluate(frames, pose, statistics=False).raw_objective for pose in poses]
        indices = np.asarray(ids, dtype=np.int64)
        translations = np.asarray([pose.translation_m for pose in poses], dtype=float)
        starts = tuple(range(0, len(poses), _GRID_CHUNK_SIZE))
        workers = min(4, os.cpu_count() or 1, len(starts))
        scores = [0.0] * len(poses)
        self._counts["objective_evaluations"] += len(poses)
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="calibrex-online-grid")
        try:
            for frame in frames:
                rows = len(frame.lidar_points)
                self._counts["point_rows"] += rows * len(poses)
                # Include rotated rows, one native matmul temporary and every
                # active chunk's response/weight/count buffers. Larger frames
                # retain the single-pose path, as do unsupported projections.
                buffer_bytes = (
                    indices.nbytes
                    + translations.nbytes
                    + sum(matrix.nbytes for matrix in rotations)
                    + rows * 24 * (len(rotations) + 1)
                    + workers * (rows * 16 * _GRID_CHUNK_SIZE + _GRID_CHUNK_SIZE * 8)
                )
                if frame.camera.projection != "pinhole" or buffer_bytes > _GRID_BUFFER_BYTES:
                    for index, pose in enumerate(poses):
                        response, weights = self.project(frame, pose)
                        scores[index] += float(response @ weights)
                    continue
                rotated = np.empty((len(rotations), rows, 3), dtype=float)
                for index, matrix in enumerate(rotations):
                    rotated[index] = frame.lidar_points @ matrix.T

                def chunk(
                    start: int,
                    frame: LevinsonOnlineFrame = frame,
                    rotated: FloatArray = rotated,
                ) -> list[float]:
                    stop = start + _GRID_CHUNK_SIZE
                    return _raw_chunk(frame, rotated, indices[start:stop], translations[start:stop])

                for start, group in zip(starts, pool.map(chunk, starts), strict=True):
                    for offset, value in enumerate(group):
                        scores[start + offset] += value
                self._counts["numba_frame_pose_evaluations"] += len(poses)
                del chunk, rotated
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return scores

    def execution(self) -> OnlineCpuExecution:
        """Record software/kernel lineage and all evaluated work."""
        return OnlineCpuExecution(
            backend="numba",
            runtime=f"calibrex.numba_online_edge/v0.2;numba={numba.__version__};numpy={np.__version__}",
            kernel_sha256=KERNEL_SHA256,
            policy="ordered_fp64_projection/native_transform_dot_sum;fastmath=false;cache=false",
            **self._counts,
        )
