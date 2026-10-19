"""Compiled CPU benchmark control; native BLAS and reductions retained."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit


class NativeRawGridControl:
    """Omit unused candidate normalization while preserving native dot order."""

    def __init__(self, online: Any) -> None:
        self.online = online
        self.pose_evaluations = 0
        self.point_rows = 0

    def raw_scores(self, frames: Any, poses: Any, center: Any) -> list[float]:
        del center
        values = []
        for pose in poses:
            raw = 0.0
            for frame in frames:
                response, weights = self.online._project_response(frame, pose)
                raw += float(response @ weights)
            values.append(raw)
        self.pose_evaluations += len(poses)
        self.point_rows += len(poses) * sum(len(frame.lidar_points) for frame in frames)
        return values


@njit(cache=False, fastmath=False)
def interpolate(points, source_weights, image, fx, fy, cx, cy, forward_axes):
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


def install(online):
    native = online._project_response

    def project(frame, transform):
        if frame.camera.projection != "pinhole":
            return native(frame, transform)
        translated = frame.lidar_points @ online._rotation_matrix(transform).T + np.asarray(
            transform.translation_m, dtype=float
        )
        camera = frame.camera
        return interpolate(
            translated,
            frame.discontinuity_weights,
            frame.edge_response,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera.axes == "forward_x_left_y_up_z",
        )

    online._project_response = project
    return native, project
