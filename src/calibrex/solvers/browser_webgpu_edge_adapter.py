"""Experimental asynchronous WebGPU batches for the offline edge objective.

The JavaScript runtime is injected; importing this module needs neither a browser
nor a GPU package. This adapter does not replace the synchronous check solver.
It evaluates every supplied pose, in order, with resident images and points.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from calibrex.evaluation.camera_lidar_edge import (
    EdgeCamera,
    EdgeObjective,
    FloatArray,
    PreparedFrame,
    _rotation_from_vector,
)


def pack_edge_objective(objective: EdgeObjective) -> dict[str, Any]:
    """Pack the existing objective, including its fixed denominator, without subsampling.

    Samples use twelve 32-bit words: xyz/weight, four unsigned image metadata
    words (offset, width, height, reserved), then shift/inverse scale/reserved.
    Integer offsets must never be rounded through float32.
    """

    count = len(objective._points)
    samples = np.zeros((count, 12), dtype=np.float32)
    samples[:, :3] = objective._points
    samples[:, 3] = objective._weights
    integers = samples.view(np.uint32)
    if objective._flat.size >= 2**32:
        raise ValueError("edge images exceed the WebGPU uint32 address space")
    integers[:, 4] = objective._offset
    integers[:, 5] = objective._stride
    integers[:, 6] = objective._v_limit.astype(np.uint32) + 1
    samples[:, 8] = objective._shift
    samples[:, 9] = objective._inverse_scale
    camera = objective.camera
    coefficients = camera.distortion
    kind = 0
    if camera.distortion_model == "equidistant" and len(coefficients) >= 4:
        kind = 2
    elif camera.distortion_model == "radtan" and coefficients:
        if len(coefficients) < 2:
            raise ValueError("radtan needs at least two distortion coefficients")
        kind = 1
    parameters = np.zeros(16, dtype=np.float32)
    parameters[:4] = camera.fx, camera.fy, camera.cx, camera.cy
    parameters[4 : 4 + min(8, len(coefficients))] = coefficients[:8]
    parameters[12:14] = kind, int(kind == 1 and len(coefficients) >= 8)
    return {
        "samples": samples.ravel(),
        "images": np.ascontiguousarray(objective._flat),
        "camera": parameters,
        "totalWeight": objective._total_weight,
    }


class BrowserWebGpuEdgeObjective:
    """Explicit, asynchronous candidate scoring; errors propagate without CPU fallback.

    ``runtime`` is ``CalibrexEdgeGpu`` from ``docs/app/edge-gpu.js`` in a Worker.
    JavaScript owns all GPU buffers and copies borrowed Pyodide buffers before
    its first await. Call :meth:`close` even after a failed measurement.
    """

    def __init__(self, objective: EdgeObjective, session: Any) -> None:
        self.objective = objective
        self._session = session
        self._closed = False

    @classmethod
    async def create(
        cls,
        frames: Sequence[PreparedFrame],
        camera: EdgeCamera,
        start: FloatArray,
        *,
        runtime: Any,
    ) -> BrowserWebGpuEdgeObjective:
        """Upload one complete prepared objective to an injected browser runtime."""

        objective = EdgeObjective(frames, camera, start)
        session = await runtime.createFromPython(pack_edge_objective(objective))
        return cls(objective, session)

    async def score_many(self, theta: FloatArray) -> FloatArray:
        """Score an N by 6 array (rotation vectors in degrees, translation in metres)."""

        if self._closed:
            raise RuntimeError("WebGPU edge objective is closed")
        values = np.asarray(theta, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6 or not np.all(np.isfinite(values)):
            raise ValueError("theta must be a finite N by 6 array")
        if not len(values):
            return np.empty(0, dtype=np.float64)
        rows = np.empty((len(values), 3, 4), dtype=np.float32)
        for index, value in enumerate(values):
            rows[index, :, :3] = _rotation_from_vector(np.radians(value[:3]))
            rows[index, :, 3] = self.objective._start_translation + value[3:].astype(np.float32)
        result = await self._session.scoreFromPython(rows.ravel())
        scores: NDArray[np.float64] = np.array(
            result.to_py() if hasattr(result, "to_py") else result, dtype=np.float64, copy=True
        )
        if scores.shape != (len(values),) or not np.all(np.isfinite(scores)):
            raise RuntimeError("WebGPU returned invalid edge scores")
        return scores

    def close(self) -> None:
        """Destroy every buffer and the owned device; repeated calls are harmless."""

        if not self._closed:
            self._closed = True
            self._session.destroy()
