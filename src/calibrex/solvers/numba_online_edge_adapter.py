"""Optional ordered CPU projection and separable bounded image spilloff."""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Sequence

import numba
import numpy as np
from numba import njit

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
    points: FloatArray,
    source_weights: FloatArray,
    image: FloatArray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    forward_axes: bool,
) -> tuple[FloatArray, FloatArray]:
    response = np.empty(len(points), dtype=np.float64)
    weights = np.empty(len(points), dtype=np.float64)
    count = 0
    height, width = image.shape
    for index in range(len(points)):
        if forward_axes:
            px, py, pz = -points[index, 1], -points[index, 2], points[index, 0]
        else:
            px, py, pz = points[index, 0], points[index, 1], points[index, 2]
        if not pz > 1.0e-6:
            continue
        x, y = px / pz, py / pz
        u, v = fx * x + cx, fy * y + cy
        if not (u >= 0.0 and v >= 0.0 and u < width - 1 and v < height - 1):
            continue
        left, top = math.floor(u), math.floor(v)
        du, dv = u - left, v - top
        response[count] = (
            (1.0 - du) * (1.0 - dv) * image[top, left]
            + du * (1.0 - dv) * image[top, left + 1]
            + (1.0 - du) * dv * image[top + 1, left]
            + du * dv * image[top + 1, left + 1]
        )
        weights[count] = source_weights[index]
        count += 1
    return response[:count], weights[:count]


@njit(cache=False, nogil=True, fastmath=False)  # type: ignore[untyped-decorator]
def _image_response(image: FloatArray, alpha: float, gamma: float, radius: int) -> FloatArray:
    height, width = image.shape
    edge = np.empty_like(image)
    for row in range(height):
        for column in range(width):
            value = 0.0
            for dy in range(-1, 2):
                y = min(max(row + dy, 0), height - 1)
                for dx in range(-1, 2):
                    x = min(max(column + dx, 0), width - 1)
                    value = max(value, abs(image[row, column] - image[y, x]))
            edge[row, column] = value
    spread = edge.copy()
    previous = edge.copy()
    horizontal = np.empty_like(image)
    next_frontier = np.empty_like(image)
    for _ in range(radius):
        for row in range(height):
            for column in range(width):
                horizontal[row, column] = max(
                    previous[row, max(column - 1, 0)],
                    previous[row, column],
                    previous[row, min(column + 1, width - 1)],
                )
        for row in range(height):
            for column in range(width):
                value = gamma * max(
                    horizontal[max(row - 1, 0), column],
                    horizontal[row, column],
                    horizontal[min(row + 1, height - 1), column],
                )
                next_frontier[row, column] = value
                spread[row, column] = max(spread[row, column], value)
        previous, next_frontier = next_frontier, previous
    return alpha * edge + (1.0 - alpha) * spread


KERNEL_SHA256 = hashlib.sha256(inspect.getsource(_interpolate.py_func).encode()).hexdigest()
IMAGE_KERNEL_SHA256 = hashlib.sha256(
    inspect.getsource(_image_response.py_func).encode()
).hexdigest()


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
        translated = frame.lidar_points @ _rotation_matrix(pose).T + np.asarray(
            pose.translation_m, dtype=float
        )
        camera = frame.camera
        result: tuple[FloatArray, FloatArray] = _interpolate(
            translated,
            frame.discontinuity_weights,
            frame.edge_response,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera.axes == "forward_x_left_y_up_z",
        )
        self._counts["numba_frame_pose_evaluations"] += 1
        return result

    def _evaluate(
        self, frames: Sequence[LevinsonOnlineFrame], pose: SE3, *, statistics: bool
    ) -> LevinsonObjectiveEvaluation:
        self._counts["objective_evaluations"] += 1
        raw, total_weight, projected_count = 0.0, 0.0, 0
        for frame in frames:
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
        """Omit only the unused normalization/support of noncenter candidates."""
        del center
        self._counts["grid_calls"] += 1
        self._counts["grid_pose_evaluations"] += len(poses)
        return [self._evaluate(frames, pose, statistics=False).raw_objective for pose in poses]

    def execution(self) -> OnlineCpuExecution:
        """Record software/kernel lineage and all evaluated work."""
        return OnlineCpuExecution(
            backend="numba",
            runtime=f"calibrex.numba_online_edge/v0.1;numba={numba.__version__};numpy={np.__version__}",
            kernel_sha256=KERNEL_SHA256,
            policy="ordered_fp64_projection/native_transform_dot_sum;fastmath=false;cache=false",
            **self._counts,
        )
