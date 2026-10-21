"""Experimental asynchronous WebGPU batches for the offline edge objective.

The JavaScript runtime is injected; importing this module needs neither a browser
nor a GPU package. This adapter does not replace the synchronous check solver.
It evaluates every supplied pose, in order, with resident images and points.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from calibrex import __version__
from calibrex.core.camera_lidar_edge import (
    CameraLidarEdgeArtifact,
    CameraLidarEdgeMotion,
    CameraLidarEdgeProvenance,
)
from calibrex.core.result import StrictModel
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.evaluation.camera_lidar_edge import (
    EdgeAlignmentOptions,
    EdgeCamera,
    EdgeObjective,
    FloatArray,
    PreparedFrame,
    _EdgeReply,
    _EdgeScoreRequest,
    _EdgeWorkflow,
    _estimate_workflow,
    _rotation_from_vector,
)
from calibrex.evaluation.edge_preparation_cache import EdgePreparationCache


def pack_edge_objective(objective: EdgeObjective) -> dict[str, Any]:
    """Pack the existing objective, including its fixed denominator, without subsampling.

    Samples use twelve 32-bit words: xyz/weight, four unsigned image metadata
    words (offset, width, height, frame index), then shift/inverse scale/reserved.
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
    integers[:, 7] = np.repeat(
        np.arange(len(objective._source_frames), dtype=np.uint32),
        [len(frame.points) for frame in objective._source_frames],
    )
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

        return await self._score_rows(theta, None)

    async def _score_subset(self, theta: FloatArray, subset: EdgeObjective) -> FloatArray:
        """Use the resident frame family, retaining the subset's CPU denominator."""

        original = self.objective
        if len(original._source_frames) > 64:
            raise ValueError("WebGPU calibration supports at most 64 prepared frames")
        if subset.camera != original.camera or not np.array_equal(
            subset._source_start, original._source_start
        ):
            raise ValueError("WebGPU subset has a different camera or candidate transform")
        indices = {id(frame): index for index, frame in enumerate(original._source_frames)}
        if len(indices) != len(original._source_frames):
            raise ValueError("WebGPU subset must contain distinct resident frames")
        selected = [indices.get(id(frame)) for frame in subset._source_frames]
        if any(index is None for index in selected) or len(set(selected)) != len(selected):
            raise ValueError("WebGPU subset must contain distinct resident frames")
        mask = sum(1 << index for index in selected if index is not None)
        return await self._score_rows(theta, (mask & 0xFFFFFFFF, mask >> 32, subset._total_weight))

    async def _score_rows(
        self, theta: FloatArray, selection: tuple[int, int, float] | None
    ) -> FloatArray:
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
        result = await (
            self._session.scoreFromPython(rows.ravel())
            if selection is None
            else self._session.scoreSubsetFromPython(rows.ravel(), *selection)
        )
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


class BrowserWebGpuEdgeDiagnostics(StrictModel):
    """Schema-valid lineage and work counts for experimental GPU screening."""

    schema_version: Literal["calibrex.browser_webgpu_edge_diagnostics/v0.1"] = (
        "calibrex.browser_webgpu_edge_diagnostics/v0.1"
    )
    backend: Literal["webgpu_screening_cpu_confirmation"] = "webgpu_screening_cpu_confirmation"
    caller_generator: str
    caller_generator_version: str
    absolute_tolerance: float = Field(default=0.000002, ge=0.000002, le=0.000002)
    relative_tolerance: float = Field(default=0.00002, ge=0.00002, le=0.00002)
    gpu_candidate_scores: int = Field(default=0, ge=0)
    gpu_batches: int = Field(default=0, ge=0)
    cpu_scalar_scores: int = Field(default=0, ge=0)
    cpu_confirmed_candidates: int = Field(default=0, ge=0)
    objectives_uploaded: int = Field(default=0, ge=0)
    peak_owned_gpu_buffer_bytes: int = Field(default=0, ge=0)
    fallback_adapter: bool | None = None
    setup_and_upload_seconds: float = Field(default=0.0, ge=0.0)
    gpu_scoring_seconds: float = Field(default=0.0, ge=0.0)
    cpu_scoring_seconds: float = Field(default=0.0, ge=0.0)


class _BrowserEdgeDriver:
    """Batch independent proposals; confirm every possible winner on the CPU.

    The interval is an empirical precision screen, not a bound proved for every
    WebGPU implementation. A checked candidate outside it aborts the run. Scalar
    line-search steps and all held-out control scores use the existing CPU math.
    """

    def __init__(
        self, runtime: Any, caller_generator: str, caller_version: str = __version__
    ) -> None:
        self.runtime = runtime
        self.diagnostics = BrowserWebGpuEdgeDiagnostics(
            caller_generator=caller_generator, caller_generator_version=caller_version
        )
        self._context: Any = None
        self._adapter: BrowserWebGpuEdgeObjective | None = None

    async def evaluate(self, request: _EdgeScoreRequest) -> _EdgeReply:
        diagnostic = self.diagnostics
        if not request.select_maximum:
            began = time.perf_counter()
            value = request.evaluate()
            diagnostic.cpu_scoring_seconds += time.perf_counter() - began
            diagnostic.cpu_scalar_scores += 1
            return value
        began = time.perf_counter()
        if self._adapter is None and len(request.objective._source_frames) > 64:
            raise ValueError("WebGPU calibration supports at most 64 prepared frames")
        if self._context is None:
            pending = asyncio.ensure_future(self.runtime.createContext())
            try:
                self._context = await asyncio.shield(pending)
            except BaseException:
                # Cancelling a Python await does not cancel a JavaScript promise.
                # A context acquired afterwards still belongs to this attempt.
                def dispose_late(done: asyncio.Future[Any]) -> None:
                    with suppress(BaseException):
                        done.result().destroy()

                pending.add_done_callback(dispose_late)
                raise
            diagnostic.fallback_adapter = self._context.fallbackAdapter
        if self._adapter is None:
            session = await self._context.createFromPython(pack_edge_objective(request.objective))
            self._adapter = BrowserWebGpuEdgeObjective(request.objective, session)
            diagnostic.objectives_uploaded += 1
            diagnostic.peak_owned_gpu_buffer_bytes = int(self._context.peakAllocatedBytes)
        diagnostic.setup_and_upload_seconds += time.perf_counter() - began
        began = time.perf_counter()
        values = await (
            self._adapter.score_many(request.theta)
            if request.objective is self._adapter.objective
            else self._adapter._score_subset(request.theta, request.objective)
        )
        diagnostic.gpu_scoring_seconds += time.perf_counter() - began
        diagnostic.gpu_candidate_scores += len(values)
        diagnostic.gpu_batches += 1
        # If every checked score is within its interval, no excluded candidate
        # can beat the maximum lower endpoint. Include equality to keep first-
        # occurrence tie breaking and account for both candidates' errors.
        # Convert the CPU-relative acceptance rule into conservative intervals
        # around GPU values, before any CPU score is known.
        error = (diagnostic.absolute_tolerance + diagnostic.relative_tolerance * np.abs(values)) / (
            1.0 - diagnostic.relative_tolerance
        )
        possible = np.flatnonzero(values + error >= np.max(values - error))
        began = time.perf_counter()
        confirmed = np.asarray([request.objective.score(request.theta[i]) for i in possible])
        diagnostic.cpu_scoring_seconds += time.perf_counter() - began
        diagnostic.cpu_confirmed_candidates += len(possible)
        if not np.all(np.isfinite(confirmed)) or np.any(
            np.abs(confirmed - values[possible])
            > (diagnostic.absolute_tolerance + diagnostic.relative_tolerance * np.abs(confirmed))
        ):
            raise RuntimeError("WebGPU candidate precision screen failed CPU confirmation")
        best = int(np.argmax(confirmed))
        return int(possible[best]), float(confirmed[best])

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
        if self._context is not None:
            self._context.destroy()
            self._context = None


_Result = TypeVar("_Result")


async def _drive_browser_edge_workflow(
    workflow: _EdgeWorkflow[_Result],
    driver: _BrowserEdgeDriver,
    *,
    observer: Callable[[_EdgeScoreRequest, _EdgeReply], None] | None = None,
) -> _Result:
    try:
        request = next(workflow)
        while True:
            reply = await driver.evaluate(request)
            if observer is not None:
                observer(request, reply)
            request = workflow.send(reply)
    except StopIteration as done:
        return done.value  # type: ignore[no-any-return]
    finally:
        workflow.close()
        driver.close()


async def estimate_from_frames_webgpu(
    frames: Sequence[CameraLidarFrame],
    camera: EdgeCamera,
    candidate: FloatArray,
    *,
    runtime: Any,
    options: EdgeAlignmentOptions | None = None,
    provenance: CameraLidarEdgeProvenance,
    parent_frame: str = "camera",
    child_frame: str = "lidar",
    reference_label: str | None = None,
    start_note: str | None = None,
    motion_compensation: CameraLidarEdgeMotion | None = None,
    preparation_cache: EdgePreparationCache | None = None,
) -> CameraLidarEdgeArtifact:
    """Run the complete offline estimator with explicit experimental WebGPU screening.

    Frame preparation, candidate budgets, sequential moves, jackknife and known-
    bad controls share the CPU workflow. Winning scores are CPU-confirmed;
    ambiguous candidates are checked in original order. No backend error falls
    back to a successful CPU run. Insufficient data returns before acquiring a
    device. The returned artifact retains the existing schema and records the
    backend, work counts and precision assumptions in its options and lineage.
    """

    candidate = np.array(candidate, dtype=np.float64, copy=True)
    provenance = provenance.model_copy(deep=True)
    driver = _BrowserEdgeDriver(runtime, provenance.generator, provenance.generator_version)
    artifact = await _drive_browser_edge_workflow(
        _estimate_workflow(
            frames,
            camera,
            candidate,
            options=options,
            provenance=provenance,
            parent_frame=parent_frame,
            child_frame=child_frame,
            reference_label=reference_label,
            start_note=start_note,
            motion_compensation=motion_compensation,
            preparation_cache=preparation_cache,
        ),
        driver,
    )
    return _record_webgpu_backend(artifact, driver.diagnostics)


def _record_webgpu_backend(
    artifact: CameraLidarEdgeArtifact, diagnostics: BrowserWebGpuEdgeDiagnostics
) -> CameraLidarEdgeArtifact:
    payload = artifact.model_dump(mode="python")
    payload["options"]["experimental_scoring"] = diagnostics.model_dump(mode="json")
    payload["provenance"]["generator"] = "calibrex.solvers.browser_webgpu_edge_adapter"
    payload["provenance"]["generator_version"] = __version__
    payload["limitations"].append(
        "Experimental WebGPU candidate screening uses empirical float32 precision intervals, "
        "not a universal error bound. Possible winners are CPU-confirmed; scalar search "
        "steps and held-out controls use CPU scores. Full calibration decision parity must "
        "be checked on the intended inputs before relying on this backend."
    )
    return CameraLidarEdgeArtifact.model_validate(payload)
