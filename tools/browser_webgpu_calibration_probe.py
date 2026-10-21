"""Complete, instrumented in-memory estimator timing in the pinned Pyodide Worker.

Fixtures contain raw gray images and point clouds. Preparation, all search starts,
jackknife, held-out controls and artifact validation are timed on each run. Runtime
startup, NPZ loading and bag decoding are reported separately or excluded explicitly.
"""

from __future__ import annotations

import hashlib
import json
import time
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from pydantic import Field

from calibrex.core.camera_lidar_edge import CameraLidarEdgeArtifact, CameraLidarEdgeProvenance
from calibrex.core.result import StrictModel
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.evaluation import camera_lidar_edge as edge
from calibrex.solvers.browser_webgpu_edge_adapter import (
    _BrowserEdgeDriver,
    _drive_browser_edge_workflow,
    _record_webgpu_backend,
    estimate_from_frames_webgpu,
)

_Digest: TypeAlias = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
_Pose: TypeAlias = Annotated[list[float], Field(min_length=6, max_length=6)]
_Decision: TypeAlias = tuple[_Digest, bool, list[_Pose], float | tuple[int, float]]


class CalibrationFixtureProvenance(StrictModel):
    schema_version: Literal["calibrex.browser_webgpu_calibration_fixture/v0.1"]
    created_at: str
    source_head: str = Field(pattern=r"^[0-9a-f]{40}$")
    generator: str
    sources: dict[str, _Digest]
    accuracy_claim: Literal[False]
    origin: dict[str, Any]
    frames: int = Field(ge=1, le=48)
    estimate_translation: bool


class _TraceSummary(StrictModel):
    sha256: _Digest
    requests: int = Field(ge=0)
    poses: int = Field(ge=0)


class _BackendRun(StrictModel):
    seconds: float = Field(gt=0)
    artifact: CameraLidarEdgeArtifact
    trace: _TraceSummary
    untraced_seconds: float = Field(gt=0)
    untraced_artifact: CameraLidarEdgeArtifact


class _Comparison(StrictModel):
    artifact_equal: bool
    decision_trace_equal: bool
    pose_budget_equal: bool


class _Attempt(StrictModel):
    repetition: int = Field(ge=0)
    order: list[Literal["cpu", "gpu"]] = Field(min_length=2, max_length=2)
    cpu: _BackendRun
    gpu: _BackendRun
    comparison: _Comparison
    ratio: float = Field(gt=0)
    untraced_ratio: float = Field(gt=0)


class _FirstTraces(StrictModel):
    cpu: list[_Decision]
    gpu: list[_Decision]


class _CalibrationRun(StrictModel):
    status: Literal["PASS", "FAIL"]
    fixture_provenance: CalibrationFixtureProvenance
    frames: int = Field(ge=1, le=48)
    raw_points: int = Field(ge=1)
    estimate_translation: bool
    input_load_seconds: float = Field(ge=0)
    cpu_first: _BackendRun
    gpu_first: _BackendRun
    first_comparison: _Comparison
    attempts: list[_Attempt] = Field(min_length=3)
    first_estimator_ratio: float = Field(gt=0)
    paired_median_ratio: float = Field(gt=0)
    first_untraced_estimator_ratio: float = Field(gt=0)
    paired_median_untraced_ratio: float = Field(gt=0)
    first_traces: _FirstTraces
    scope: str
    fixture_index: int = Field(ge=0)
    fixture: str


class _BenchmarkProvenance(StrictModel):
    created_at: str
    command: list[str] = Field(min_length=1)
    source_head: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_worktree_dirty: bool
    sources: dict[str, _Digest]
    pyodide_version: Literal["0.27.2"]
    whole_calibration_claim: Literal[False]
    browser_version: str | None = None


class BrowserWebGpuCalibrationReport(StrictModel):
    """Validation model for the opt-in browser calibration benchmark report."""

    schema_version: Literal["calibrex.browser_webgpu_calibration_probe/v0.2"]
    status: Literal["RUNNING", "PASS", "FAIL", "TIMEOUT"]
    provenance: _BenchmarkProvenance
    runs: list[_CalibrationRun]
    logs: list[str]
    errors: list[str]
    checks: dict[str, Any]


class _Trace:
    def __init__(self) -> None:
        self.keys: weakref.WeakKeyDictionary[edge.EdgeObjective, str] = weakref.WeakKeyDictionary()
        self.records: list[Any] = []
        self.poses = 0

    def __call__(self, request: edge._EdgeScoreRequest, reply: edge._EdgeReply) -> None:
        if request.objective not in self.keys:
            objective = request.objective
            digest = hashlib.sha256(json.dumps(objective.camera.as_dict(), sort_keys=True).encode())
            for name in (
                "_points",
                "_weights",
                "_flat",
                "_offset",
                "_stride",
                "_shift",
                "_inverse_scale",
                "_u_limit",
                "_v_limit",
                "_start_translation",
            ):
                array = getattr(objective, name)
                digest.update(str(array.dtype).encode())
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
            self.keys[objective] = digest.hexdigest()
        self.records.append(
            [
                self.keys[request.objective],
                request.select_maximum,
                request.theta.tolist(),
                list(reply) if isinstance(reply, tuple) else reply,
            ]
        )
        self.poses += len(request.theta)

    def summary(self) -> dict[str, Any]:
        return {
            "sha256": hashlib.sha256(
                json.dumps(self.records, separators=(",", ":")).encode()
            ).hexdigest(),
            "requests": len(self.records),
            "poses": self.poses,
        }


def _calibration_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    # Only the declared backend lineage, diagnostic counters/times and its
    # additional precision limitation differ; every calibration field is compared.
    normalized = json.loads(json.dumps(artifact))
    diagnostic = normalized["options"].pop("experimental_scoring", None)
    if diagnostic is not None:
        normalized["provenance"]["generator"] = diagnostic["caller_generator"]
        normalized["provenance"]["generator_version"] = diagnostic["caller_generator_version"]
        assert "Experimental WebGPU" in normalized["limitations"].pop()
    return normalized


def _compare(cpu: dict[str, Any], gpu: dict[str, Any]) -> dict[str, bool]:
    diagnostics = [
        gpu[key]["options"]["experimental_scoring"] for key in ("artifact", "untraced_artifact")
    ]
    return {
        "artifact_equal": all(
            _calibration_payload(run[key]) == cpu["artifact"]
            for run in (cpu, gpu)
            for key in ("artifact", "untraced_artifact")
        ),
        "decision_trace_equal": gpu["trace"] == cpu["trace"],
        "pose_budget_equal": all(
            d["gpu_candidate_scores"] + d["cpu_scalar_scores"] == cpu["trace"]["poses"]
            for d in diagnostics
        ),
    }


async def measure(
    path: str,
    runtime: Any,
    repetitions: int = 3,
    *,
    source_head: str,
    profile_cpu: bool = False,
) -> dict[str, Any]:
    """Compare complete estimators and exact decision traces, retaining all attempts."""

    if len(source_head) != 40 or any(c not in "0123456789abcdef" for c in source_head):
        raise ValueError("source_head must identify the measured implementation")
    began = time.perf_counter()
    raw_bytes = Path(path).read_bytes()
    with np.load(path, allow_pickle=False) as fixture:
        lineage = json.loads(str(fixture["provenance_json"]))
        camera = edge.EdgeCamera(*fixture["intrinsics"].tolist())
        candidate = fixture["candidate"].copy()
        translation = bool(fixture["estimate_translation"])
        frames = [
            CameraLidarFrame(
                time_s=float(i),
                image_time_s=float(i),
                image=fixture[f"image_{i}"].copy(),
                points=fixture[f"points_{i}"].copy(),
                cloud_frame_id="lidar",
                image_frame_id="camera",
            )
            for i in range(int(fixture["frames"]))
        ]
    load_seconds = time.perf_counter() - began
    options = edge.EdgeAlignmentOptions(estimate_translation=translation)
    provenance = CameraLidarEdgeProvenance(
        generator="tools.browser_webgpu_calibration_probe",
        generator_version="0.2",
        git_commit=source_head,
        command=["measure", path, "--source-head", source_head],
        dataset_family="synthetic",
        sequence_ids=["distinct-ray-cast-scenes"],
        image_topic="/image",
        lidar_topic="/points",
        input_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        input_digest_scope="complete NPZ bytes",
        dataset_license="Apache-2.0 synthetic fixture generator",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    del raw_bytes
    if profile_cpu:
        import cProfile
        import io
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        edge.estimate_from_frames(frames, camera, candidate, options=options, provenance=provenance)
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(22)
        print(
            "Auxiliary CPU profile, excluded from measured runs:\n" + stream.getvalue(), flush=True
        )
    records: dict[str, list[Any]] = {}

    async def run(backend: str, retain_trace: bool) -> dict[str, Any]:
        # The ordinary public API is timed independently. Full decision hashing
        # copies image arrays and is deliberately confined to a separate replay.
        began = time.perf_counter()
        untraced = (
            edge.estimate_from_frames(
                frames, camera, candidate, options=options, provenance=provenance
            )
            if backend == "cpu"
            else await estimate_from_frames_webgpu(
                frames, camera, candidate, options=options, provenance=provenance, runtime=runtime
            )
        )
        untraced_seconds = time.perf_counter() - began
        trace = _Trace()
        began = time.perf_counter()
        workflow = edge._estimate_workflow(
            frames, camera, candidate, options=options, provenance=provenance
        )
        if backend == "cpu":
            artifact = edge._drive_edge_workflow(workflow, observer=trace)
        else:
            driver = _BrowserEdgeDriver(runtime, provenance.generator, provenance.generator_version)
            artifact = await _drive_browser_edge_workflow(workflow, driver, observer=trace)
            artifact = _record_webgpu_backend(artifact, driver.diagnostics)
        seconds = time.perf_counter() - began
        result = {
            "seconds": seconds,
            "artifact": artifact.model_dump(mode="json"),
            "trace": trace.summary(),
            "untraced_seconds": untraced_seconds,
            "untraced_artifact": untraced.model_dump(mode="json"),
        }
        if retain_trace:
            records[backend] = trace.records
        print(
            json.dumps({"phase": "completed_estimator", "backend": backend, **result}), flush=True
        )
        return result

    first_cpu = await run("cpu", True)
    first_gpu = await run("gpu", True)
    comparison = _compare(first_cpu, first_gpu)
    attempts = []
    for repetition in range(repetitions):
        print(f"complete estimator repetition {repetition + 1}/{repetitions}", flush=True)
        order = ["cpu", "gpu"] if repetition % 2 == 0 else ["gpu", "cpu"]
        row: dict[str, Any] = {"repetition": repetition, "order": order}
        for backend in order:
            row[backend] = await run(backend, False)
        row["comparison"] = _compare(row["cpu"], row["gpu"])
        row["ratio"] = row["cpu"]["seconds"] / row["gpu"]["seconds"]
        row["untraced_ratio"] = row["cpu"]["untraced_seconds"] / row["gpu"]["untraced_seconds"]
        attempts.append(row)
    return {
        "status": "PASS"
        if all(comparison.values()) and all(all(row["comparison"].values()) for row in attempts)
        else "FAIL",
        "fixture_provenance": lineage,
        "frames": len(frames),
        "raw_points": sum(len(frame.points) for frame in frames),
        "estimate_translation": translation,
        "input_load_seconds": load_seconds,
        "cpu_first": first_cpu,
        "gpu_first": first_gpu,
        "first_comparison": comparison,
        "attempts": attempts,
        "first_estimator_ratio": first_cpu["seconds"] / first_gpu["seconds"],
        "paired_median_ratio": float(np.median([row["ratio"] for row in attempts])),
        "first_untraced_estimator_ratio": first_cpu["untraced_seconds"]
        / first_gpu["untraced_seconds"],
        "paired_median_untraced_ratio": float(
            np.median([row["untraced_ratio"] for row in attempts])
        ),
        "first_traces": records,
        "scope": (
            "ordinary API timings and separate instrumented decision replays, raw synchronized "
            "frames to validated artifact, including preparation "
            "and new GPU context per run; excludes runtime/package startup, NPZ input loading "
            "and bag decoding; no real-data accuracy claim"
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export the full-estimator probe JSON schema")
    parser.add_argument("--schema", type=Path, required=True)
    target = parser.parse_args().schema
    target.write_text(
        json.dumps(BrowserWebGpuCalibrationReport.model_json_schema(), indent=2) + "\n",
        encoding="utf-8",
    )
