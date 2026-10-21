"""Calibration decisions, conservative candidate screening and owned resources."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import jsonschema
import numpy as np
import pytest
from tests.unit.camera_lidar_scene import perturbed, true_transform
from tests.unit.test_camera_lidar_edge import CAMERA, FAST, _frames, _provenance
from tools.browser_webgpu_calibration_probe import _Trace

from calibrex.core.camera_lidar_edge import camera_lidar_edge_json_schema
from calibrex.evaluation import camera_lidar_edge as edge
from calibrex.solvers import browser_webgpu_edge_adapter as gpu


class Context:
    fallbackAdapter = False
    peakAllocatedBytes = 4096

    def __init__(self):
        self.closed = 0
        self.uploads = 0
        self.sessions = []

    async def createFromPython(self, packed):
        assert packed["samples"].dtype == np.float32
        self.uploads += 1
        session = Session()
        self.sessions.append(session)
        return session

    def destroy(self):
        self.closed += 1


class Session:
    def __init__(self):
        self.closed = 0

    def destroy(self):
        self.closed += 1


class Runtime:
    def __init__(self):
        self.context = Context()
        self.requests = 0

    async def createContext(self):
        self.requests += 1
        return self.context


@pytest.mark.parametrize("translation", [False, True])
def test_complete_async_workflow_retains_cpu_artifact_and_every_pose(monkeypatch, translation):
    truth = true_transform()
    frames = _frames(truth)
    candidate = perturbed(truth, (0.8, -1.2, 0.0))
    options = replace(FAST, estimate_translation=translation)
    provenance = _provenance()
    original_score = edge.EdgeObjective.score
    cpu_calls = []
    cpu_trace, gpu_trace = _Trace(), _Trace()
    cpu_drive = edge._drive_edge_workflow
    gpu_drive = gpu._drive_browser_edge_workflow

    def observe_cpu(workflow, **kwargs):
        return cpu_drive(workflow, observer=cpu_trace, **kwargs)

    async def observe_gpu(workflow, driver, **kwargs):
        return await gpu_drive(workflow, driver, observer=gpu_trace, **kwargs)

    monkeypatch.setattr(edge, "_drive_edge_workflow", observe_cpu)
    monkeypatch.setattr(gpu, "_drive_browser_edge_workflow", observe_gpu)

    def count(self, theta):
        cpu_calls.append(theta.copy())
        return original_score(self, theta)

    monkeypatch.setattr(edge.EdgeObjective, "score", count)
    cpu = edge.estimate_from_frames(
        frames, CAMERA, candidate, options=options, provenance=provenance
    )
    cpu_evaluations = len(cpu_calls)
    # Reused scores still represent the same ordered logical pose requests.
    assert 0 < cpu_evaluations < cpu_trace.poses

    async def score_many(self, theta):
        # Mock transport only; actual WGSL arithmetic has a separate browser probe.
        return np.asarray([original_score(self.objective, pose) for pose in theta])

    monkeypatch.setattr(gpu.BrowserWebGpuEdgeObjective, "score_many", score_many)

    async def score_subset(self, theta, subset):
        return np.asarray([original_score(subset, pose) for pose in theta])

    monkeypatch.setattr(gpu.BrowserWebGpuEdgeObjective, "_score_subset", score_subset)
    runtime = Runtime()
    actual = asyncio.run(
        gpu.estimate_from_frames_webgpu(
            frames, CAMERA, candidate, options=options, provenance=provenance, runtime=runtime
        )
    ).model_dump(mode="json")
    jsonschema.validate(actual, camera_lidar_edge_json_schema())
    diagnostic = actual["options"].pop("experimental_scoring")
    jsonschema.validate(diagnostic, gpu.BrowserWebGpuEdgeDiagnostics.model_json_schema())
    assert gpu_trace.records == cpu_trace.records
    assert diagnostic["gpu_candidate_scores"] + diagnostic["cpu_scalar_scores"] == cpu_trace.poses
    assert (
        diagnostic["cpu_confirmed_candidates"] + diagnostic["cpu_scalar_scores"]
        == len(cpu_calls) - cpu_evaluations
    )
    assert diagnostic["objectives_uploaded"] == runtime.context.uploads
    assert diagnostic["gpu_batches"] > 20
    assert diagnostic["peak_owned_gpu_buffer_bytes"] == 4096
    assert actual["provenance"]["generator"] == "calibrex.solvers.browser_webgpu_edge_adapter"
    actual["provenance"]["generator"] = provenance.generator
    actual["provenance"]["generator_version"] = provenance.generator_version
    assert "empirical" in actual["limitations"].pop()
    assert actual == cpu.model_dump(mode="json")
    assert runtime.requests == runtime.context.closed == 1
    assert all(session.closed == 1 for session in runtime.context.sessions)


@pytest.mark.parametrize(
    "real, approximate, expected",
    [
        ([0.5, 0.500001, 0.1], [0.500001, 0.5, 0.1], 1),
        ([0.5, 0.5, 0.1], [0.5, 0.500001, 0.1], 0),
        ([-0.5, -0.499999, -1.0], [-0.499999, -0.5, -1.0], 1),
    ],
)
def test_cpu_confirmation_corrects_close_ranks_and_preserves_first_tie(
    monkeypatch, real, approximate, expected
):
    objective = edge.EdgeObjective(
        [
            edge.PreparedFrame(
                0,
                0,
                np.ones((3, 3), np.float32),
                1,
                np.array([[0, 0, 1]], np.float32),
                np.ones(1, np.float32),
                0,
            )
        ],
        edge.EdgeCamera(1, 1, 1, 1),
        np.eye(4),
    )
    checked = []

    def score(theta):
        index = int(theta[0])
        checked.append(index)
        return real[index]

    monkeypatch.setattr(objective, "score", score)

    async def score_many(self, theta):
        return np.asarray(approximate)

    monkeypatch.setattr(gpu.BrowserWebGpuEdgeObjective, "score_many", score_many)
    runtime = Runtime()
    driver = gpu._BrowserEdgeDriver(runtime, "test")
    poses = np.zeros((3, 6))
    poses[:, 0] = np.arange(3)
    result = asyncio.run(driver.evaluate(edge._EdgeScoreRequest(objective, poses, True)))
    driver.close()
    assert result == (expected, real[expected])
    assert checked == [0, 1]
    assert runtime.context.closed == 1


@pytest.mark.parametrize("failure", ["precision", "readback"])
def test_gpu_failures_abort_calibration_and_close_resources(monkeypatch, failure):
    async def broken(self, theta):
        if failure == "readback":
            raise RuntimeError("readback failure")
        return np.full(len(theta), 10.0)

    monkeypatch.setattr(gpu.BrowserWebGpuEdgeObjective, "score_many", broken)
    truth = true_transform()
    runtime = Runtime()
    with pytest.raises(RuntimeError, match=r"precision screen|readback failure"):
        asyncio.run(
            gpu.estimate_from_frames_webgpu(
                _frames(truth),
                CAMERA,
                truth,
                options=FAST,
                provenance=_provenance(),
                runtime=runtime,
            )
        )
    assert runtime.context.closed == 1
    assert all(session.closed == 1 for session in runtime.context.sessions)


def test_insufficient_frames_return_before_requesting_a_device():
    runtime = Runtime()
    artifact = asyncio.run(
        gpu.estimate_from_frames_webgpu(
            [], CAMERA, np.eye(4), provenance=_provenance(), runtime=runtime
        )
    )
    assert artifact.solver_status == "insufficient_samples"
    assert artifact.policy_status == "inconclusive"
    assert artifact.jackknife_fits == 0
    assert runtime.requests == 0
    assert artifact.options["experimental_scoring"]["gpu_candidate_scores"] == 0


def test_cancellation_closes_a_context_returned_after_the_python_await(monkeypatch):
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        context = Context()

        class LateRuntime:
            async def createContext(self):
                started.set()
                await release.wait()
                return context

        truth = true_transform()
        task = asyncio.create_task(
            gpu.estimate_from_frames_webgpu(
                _frames(truth),
                CAMERA,
                truth,
                options=FAST,
                provenance=_provenance(),
                runtime=LateRuntime(),
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        for _ in range(4):
            await asyncio.sleep(0)
        assert context.closed == 1

    asyncio.run(run())
