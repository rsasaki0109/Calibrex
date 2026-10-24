"""Live residual diagnostics are truthful and cannot change the calibration result."""
from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.check.browser import EventProgress
from calibrex.core.progress import ProgressEvent, emit_metric, progress_active, progress_sink
from calibrex.solvers.imu_lidar_rotation_solver import (
    GyroSeries,
    ImuLidarRotationSolver,
    RateIntervals,
    RotationOptions,
    model_residuals,
)


@pytest.mark.parametrize("model", ["mean_rate", "preintegrated"])
def test_telemetry_matches_actual_residual_and_preserves_solution(
    model: Literal["mean_rate", "preintegrated"],
) -> None:
    times = np.linspace(0, 8, 801)
    rates = np.column_stack([
        .3 * np.sin(times), .2 * np.cos(1.7 * times), .4 * np.sin(.6 * times),
    ])
    gyro = GyroSeries(times, rates)
    start = np.linspace(.2, 7.5, 60)
    end = start + .1
    true_rotation = Rotation.from_euler("xyz", [2, -3, 4], degrees=True).as_matrix()
    delta, _ = gyro.increments(start, end)
    delta = true_rotation @ delta @ true_rotation.T
    lidar_rate = Rotation.from_matrix(delta).as_rotvec() / (end - start)[:, None]
    intervals = RateIntervals(np.zeros(len(start), dtype=np.int64), start, end, lidar_rate, delta)
    opts = RotationOptions(model=model, irls_iterations=2)
    solver = ImuLidarRotationSolver()
    reference = solver.solve(gyro, intervals, opts)
    events: list[ProgressEvent] = []
    with progress_sink(events.append):
        measured = solver.solve(gyro, intervals, opts)
    np.testing.assert_array_equal(measured.rotation, reference.rotation)
    np.testing.assert_array_equal(measured.gyro_bias_rps, reference.gyro_bias_rps)
    assert measured.time_offset_s == reference.time_offset_s
    assert measured.dofs == reference.dofs
    assert measured.variance_factor == reference.variance_factor
    assert [e.step for e in events] == [0, 1, 2]
    assert all(e.kind == "metric" and e.unit == "noise-normalized RMS" for e in events)
    assert measured.rotation is not None
    final = model_residuals(
        gyro, intervals, measured.rotation, measured.gyro_bias_rps,
        measured.time_offset_s, opts,
    )
    assert events[-1].value == pytest.approx(float(np.sqrt(np.mean(final**2))))
    # Same solver invoked for a refit starts a new series, not a continuation.
    with progress_sink(events.append):
        solver.solve(gyro, intervals, opts)
    assert [e.step for e in events] == [0, 1, 2, 0, 1, 2]


def test_browser_forwards_finite_metrics_and_restores_sink() -> None:
    events: list[dict[str, object]] = []
    progress = EventProgress(lambda item: events.append(json.loads(item)))
    assert not progress_active()
    with progress_sink(progress.on_event):
        assert progress_active()
        emit_metric("Rotation training residual", 2.5, "noise-normalized RMS", 0)
        emit_metric("missing", float("nan"), "", 1)
        emit_metric("infinite", float("inf"), "", 1)
        with progress_sink(None):
            assert not progress_active()
            emit_metric("disabled", 0, "", 0)
    assert not progress_active()
    assert events == [{
        "event": "metric", "name": "Rotation training residual", "value": 2.5,
        "unit": "noise-normalized RMS", "step": 0,
    }]
