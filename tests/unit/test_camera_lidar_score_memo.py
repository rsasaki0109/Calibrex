"""Exact score reuse must preserve calibration, ownership and custom objectives."""

from __future__ import annotations

import gc
import json
import weakref

import numpy as np
import pytest
from tests.unit.camera_lidar_scene import (
    CX,
    CY,
    FOCAL,
    cast_points,
    make_scene,
    perturbed,
    render_image,
    true_transform,
)
from tools.browser_edge_preparation_probe import _Trace

from calibrex.core.camera_lidar_edge import CameraLidarEdgeProvenance
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.evaluation import camera_lidar_edge as edge


def prepared() -> edge.PreparedFrame:
    return edge.PreparedFrame(
        time_s=0.0,
        gap_s=0.0,
        edge=np.arange(16, dtype=np.float32).reshape(4, 4) / 16,
        scale=1,
        points=np.array([[0, 0, 1], [0.2, 0.3, 1]], dtype=np.float32),
        weights=np.ones(2, dtype=np.float32),
        rings=1,
    )


def fixed() -> edge.EdgeObjective:
    return edge._fixed_objective([prepared()], edge.EdgeCamera(1, 1, 1, 1), np.eye(4))


def test_snapshots_own_numeric_inputs_and_next_objective_observes_changes() -> None:
    frame = prepared()
    start = np.eye(4)
    camera = edge.EdgeCamera(1, 1, 1, 1)
    objective = edge._fixed_objective([frame], camera, start)
    memo = edge._EdgeScoreMemo()
    theta = np.zeros(6)
    before = memo.score(objective, theta)
    assert before > 0
    frame.edge.fill(0)
    frame.points.fill(99)
    frame.weights.fill(10)
    start[:3, 3] = 5
    assert memo.score(objective, theta) == before
    assert edge._fixed_objective([frame], camera, start).score(theta) == 0
    for value in vars(objective).values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.flat[0] = 0
    with pytest.raises(AttributeError, match="read-only"):
        objective.camera = camera
    with pytest.raises(AttributeError, match="read-only"):
        del objective._points


def test_pose_keys_preserve_signed_zero_and_every_double_bit() -> None:
    objective = fixed()
    memo = edge._EdgeScoreMemo()
    zero = np.zeros(6)
    negative_zero = zero.copy()
    negative_zero[0] = -0.0
    next_double = zero.copy()
    next_double[0] = np.nextafter(0.0, 1.0)
    for theta in [zero, negative_zero, next_double]:
        assert memo.score(objective, theta) == objective.score(theta)
    assert memo.misses == 3 and memo.hits == 0
    assert memo.score(objective, zero.copy()) == objective.score(zero)
    assert memo.hits == 1
    assert memo.score(objective, zero.astype(np.float32)) == objective.score(
        zero.astype(np.float32)
    )
    assert memo.misses == 4 and memo.entries == 3


def test_capacity_clear_and_weak_objective_ownership() -> None:
    objective = fixed()
    memo = edge._EdgeScoreMemo(max_entries=2)
    for value in [0.0, 0.1, 0.2]:
        theta = np.array([value, 0, 0, 0, 0, 0])
        assert memo.score(objective, theta) == objective.score(theta)
        assert 0 < memo.entries <= 2
    assert memo.entries == 1 and memo.misses == 3
    reference = weakref.ref(objective)
    del objective
    gc.collect()
    assert reference() is None and len(memo._values) == 0
    memo.clear()
    assert memo.entries == 0


def test_mutable_and_custom_objectives_keep_their_original_calls() -> None:
    class Stateful(edge.EdgeObjective):
        def __init__(self) -> None:
            self.calls = 0

        def score(self, theta: edge.FloatArray) -> float:
            self.calls += 1
            return float(self.calls)

    custom = Stateful()
    memo = edge._EdgeScoreMemo()
    assert edge._EdgeScoreRequest(custom, np.zeros((3, 6)), True).evaluate(memo) == (2, 3.0)
    objective = edge.EdgeObjective([prepared()], edge.EdgeCamera(1, 1, 1, 1), np.eye(4))
    first = memo.score(objective, np.zeros(6))
    objective._flat += 1
    assert memo.score(objective, np.zeros(6)) == first + 1
    assert memo.entries == memo.hits == 0


@pytest.mark.parametrize(
    "camera",
    [
        edge.EdgeCamera(1, 1, 1, 1, "radtan", [0.1, 0.0]),
        type("CustomCamera", (edge.EdgeCamera,), {})(1, 1, 1, 1),
    ],
)
def test_noncanonical_camera_bypasses_reuse(camera: edge.EdgeCamera) -> None:
    objective = edge._fixed_objective([prepared()], camera, np.eye(4))
    assert type(objective) is edge.EdgeObjective
    memo = edge._EdgeScoreMemo()
    memo.score(objective, np.zeros(6))
    memo.score(objective, np.zeros(6))
    assert memo.hits == memo.entries == 0 and memo.misses == 2


@pytest.mark.parametrize("operation", ["get", "set"])
def test_optional_cache_allocation_failure_preserves_score(operation: str) -> None:
    class FailingMap(weakref.WeakKeyDictionary):
        def get(self, *args):
            if operation == "get":
                raise MemoryError("cache lookup")
            return super().get(*args)

        def __setitem__(self, *args):
            if operation == "set":
                raise MemoryError("cache storage")
            return super().__setitem__(*args)

    objective = fixed()
    memo = edge._EdgeScoreMemo()
    memo._values = FailingMap()
    assert memo.score(objective, np.zeros(6)) == objective.score(np.zeros(6))
    assert memo.misses == 1 and memo.entries == 0


@pytest.mark.parametrize("failure", ["none", "workflow", "observer", "close"])
def test_workflow_releases_owned_entries_on_every_exit(monkeypatch, failure: str) -> None:
    owned = []
    original = edge._EdgeScoreMemo

    class ObservedMemo(original):
        def __init__(self) -> None:
            super().__init__()
            owned.append(self)

    monkeypatch.setattr(edge, "_EdgeScoreMemo", ObservedMemo)
    objective = fixed()
    closed = []

    def workflow():
        try:
            yield edge._EdgeScoreRequest(objective, np.zeros((1, 6)))
            if failure == "workflow":
                raise ValueError("workflow")
            yield edge._EdgeScoreRequest(objective, np.zeros((1, 6)))
            return 7
        finally:
            closed.append(True)
            if failure == "close":
                raise ValueError("close")

    def observer(request, reply):
        if failure in ("observer", "close"):
            raise ValueError("observer")

    if failure == "none":
        assert edge._drive_edge_workflow(workflow(), observer=observer) == 7
        assert owned[0].hits == owned[0].misses == 1
    else:
        with pytest.raises(ValueError, match="close" if failure == "close" else failure):
            edge._drive_edge_workflow(workflow(), observer=observer)
    assert closed == [True] and len(owned) == 1
    assert owned[0].entries == 0 and len(owned[0]._values) == 0


@pytest.mark.parametrize("translation", [False, True])
def test_complete_calibration_and_logical_trace_match_uncached_math(monkeypatch, translation):
    truth = true_transform()
    frames = []
    for index in range(8):
        scene = make_scene(index)
        frames.append(
            CameraLidarFrame(
                time_s=float(index),
                image_time_s=float(index),
                image=render_image(scene),
                points=cast_points(scene, truth),
                cloud_frame_id="lidar",
                image_frame_id="camera",
            )
        )
    camera = edge.EdgeCamera(FOCAL, FOCAL, CX, CY)
    candidate = perturbed(truth, (0.8, -1.2, 0.0))
    provenance = CameraLidarEdgeProvenance(
        generator="test",
        generator_version="0",
        dataset_family="synthetic",
        sequence_ids=["scene"],
        image_topic="/image",
        lidar_topic="/points",
        input_sha256="0" * 64,
        input_digest_scope="test",
        dataset_license="none",
    )
    options = edge.EdgeAlignmentOptions(estimate_translation=translation)
    drive, memo_type = edge._drive_edge_workflow, edge._EdgeScoreMemo
    traces, memos, artifacts = [], [], []
    for enabled in [True, False]:
        trace = _Trace()
        memo = memo_type(max_entries=1024 if enabled else 0)
        monkeypatch.setattr(edge, "_EdgeScoreMemo", lambda memo=memo: memo)

        def observe(workflow, trace=trace, **kwargs):
            return drive(workflow, observer=trace, **kwargs)

        monkeypatch.setattr(edge, "_drive_edge_workflow", observe)
        artifact = edge.estimate_from_frames(
            frames,
            camera,
            candidate,
            options=options,
            provenance=provenance,
        )
        assert artifact.samples.frames == 8 and artifact.jackknife_fits == 6
        assert memo.entries == 0 and not memo._values
        assert memo.hits + memo.misses == trace.poses
        traces.append(json.dumps(trace.records, separators=(",", ":")))
        memos.append(memo)
        artifacts.append(artifact.model_dump(mode="json"))
    assert artifacts[0] == artifacts[1] and traces[0] == traces[1]
    assert memos[0].hits > 0 and memos[0].misses < memos[1].misses
