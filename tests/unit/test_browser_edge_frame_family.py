"""Frame-family transport must retain membership, starts and exact denominators."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.test_browser_webgpu_edge_adapter import frames

from calibrex.evaluation.camera_lidar_edge import EdgeCamera, EdgeObjective
from calibrex.solvers.browser_webgpu_edge_adapter import (
    BrowserWebGpuEdgeObjective,
    pack_edge_objective,
)


def test_subset_transport_keeps_both_mask_words_and_cpu_denominator():
    source = [replace(frames()[0], time_s=float(i)) for i in range(40)]
    camera = EdgeCamera(100, 110, 3, 2)
    full = EdgeObjective(source, camera, np.eye(4))
    subset = EdgeObjective([source[i] for i in (0, 32, 39)], camera, np.eye(4))
    calls = []

    class Session:
        async def scoreSubsetFromPython(self, rows, low, high, total):
            calls.append((rows.copy(), low, high, total))
            return np.array([0.125])

    adapter = BrowserWebGpuEdgeObjective(full, Session())
    actual = asyncio.run(adapter._score_subset(np.zeros((1, 6)), subset))
    np.testing.assert_array_equal(actual, [0.125])
    assert calls[0][1:] == (1, 129, subset._total_weight)
    packed = pack_edge_objective(full)["samples"].reshape(-1, 12).view(np.uint32)
    np.testing.assert_array_equal(packed[:, 7], np.repeat(np.arange(40), len(source[0].points)))


@pytest.mark.parametrize("invalid", ["foreign", "duplicate", "rotation", "camera", "too_many"])
def test_subset_rejects_changed_or_unresident_inputs_before_submission(invalid):
    source = frames()
    camera = EdgeCamera(100, 110, 3, 2)
    start = np.eye(4)
    if invalid == "too_many":
        source = [replace(source[0], time_s=float(i)) for i in range(65)]
    full = EdgeObjective(source, camera, start)
    kept = [source[0]]
    if invalid == "foreign":
        kept = [replace(source[0])]
    elif invalid == "duplicate":
        kept = [source[0], source[0]]
    elif invalid == "rotation":
        start = start.copy()
        start[:2, :2] = [[0, -1], [1, 0]]
    elif invalid == "camera":
        camera = replace(camera, fx=90)
    subset = EdgeObjective(kept, camera, start)
    adapter = BrowserWebGpuEdgeObjective(full, object())
    with pytest.raises(ValueError, match=r"at most|distinct|different"):
        asyncio.run(adapter._score_subset(np.zeros((1, 6)), subset))
