from __future__ import annotations

import struct
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.test_continuous_time_camera_lidar_solver import _captures

from calibrex.core.capture_time import LidarCaptureTimePolicy
from calibrex.core.continuous_trajectory import ContinuousTrajectoryPose, PiecewiseSE3Trajectory
from calibrex.core.geometry import SE3
from calibrex.solvers import continuous_time_camera_lidar_solver as solver


def _bytes(point: solver.DeskewedLidarPoint) -> bytes:
    return struct.pack(
        "5d",
        *point.position_lidar_at_reference_m,
        point.capture_time_reference_sec,
        point.deskew_delta_sec,
    )


def test_prepared_projection_preserves_every_coordinate_bit() -> None:
    rng = np.random.default_rng(714)
    points = [(zero, zero, zero) for zero in (0.0, -0.0)]
    points += [tuple(float(v) for v in rng.normal(size=3)) for _ in range(100)]
    for _ in range(12):
        pose = SE3(tuple(rng.normal(size=3)), tuple(rng.normal(size=4)))
        prepared = solver._prepare_point_transform(pose)
        for point in points:
            assert struct.pack("3d", *prepared(point)) == struct.pack(
                "3d", *pose.transform_point(point)
            )


@pytest.mark.parametrize("piecewise", [False, True])
@pytest.mark.parametrize("direction", ["top_to_bottom", "bottom_to_top"])
@pytest.mark.parametrize("point_times", [False, True])
@pytest.mark.parametrize("rolling", [False, True])
@pytest.mark.parametrize("nanoseconds", [False, True])
def test_prepared_deskew_preserves_bits_across_clock_states(
    piecewise: bool,
    direction: str,
    point_times: bool,
    rolling: bool,
    nanoseconds: bool,
) -> None:
    captures = _captures(SE3.identity(), true_offset_sec=0.02)
    trajectory = PiecewiseSE3Trajectory(
        (
            ContinuousTrajectoryPose(99.0, SE3.identity()),
            ContinuousTrajectoryPose(100.5, SE3((0.7, -0.1, 0.3), (0.1, -0.2, 0.05, 0.98))),
            ContinuousTrajectoryPose(102.0, SE3((1.8, -0.2, 0.6), (-0.1, 0.3, -0.15, 0.9))),
        )
    )
    body_lidar = SE3((0.42, -0.18, 0.27), (0.01, -0.02, 0.03, 0.999))
    capture = replace(
        captures[4],
        transform_body_lidar=body_lidar,
        rolling_shutter_readout_sec=0.02,
        rolling_shutter_direction=direction,
        body_trajectory=trajectory if piecewise else None,
        correspondences=tuple(
            replace(
                item, capture_offset_sec=item.capture_offset_sec * (1.0e9 if nanoseconds else 1)
            )
            for item in captures[4].correspondences
        ),
    )
    options = solver.ContinuousTimeCameraLidarOptions(
        use_per_point_time=point_times,
        use_rolling_shutter=rolling,
    )
    prepared = solver._make_deskew_evaluator([capture], options)
    # Repeated clock states, cache eviction, and both signed zeros.
    offsets = [0.0, -0.0, 0.02, -0.01, 0.03, 0.03, -0.01, 0.03, 0.0]
    for offset in offsets:
        policy = LidarCaptureTimePolicy(
            "nanoseconds" if nanoseconds else "seconds",
            "scan_end",
            offset,
        )
        for item in capture.correspondences:
            assert _bytes(prepared(capture, item, policy, options)) == _bytes(
                solver._deskew_correspondence(capture, item, policy, options)
            )


@pytest.mark.parametrize("ablated", [False, True])
def test_complete_solve_trace_and_diagnostics_match_scalar(
    monkeypatch: pytest.MonkeyPatch,
    ablated: bool,
) -> None:
    truth = SE3((0.10, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    captures = _captures(truth, true_offset_sec=0.02)
    captures[0] = replace(
        captures[0],
        correspondences=(
            replace(captures[0].correspondences[0], reliability=0.0),
            replace(captures[0].correspondences[1], point_lidar_m=(0.0, 0.0, -1.0)),
            replace(captures[0].correspondences[2], point_lidar_m=(100.0, 0.0, 1.0)),
            *captures[0].correspondences[3:],
        ),
    )
    options = solver.ContinuousTimeCameraLidarOptions(
        estimate_time_offset=not ablated,
        use_per_point_time=not ablated,
        use_covariance=not ablated,
        use_rolling_shutter=not ablated,
    )
    initial = SE3((0.05, -0.05, 0.02), (0.0, 0.0, 0.0, 1.0))
    policy = LidarCaptureTimePolicy("seconds", "scan_end")
    prepared = solver.ContinuousTimeCameraLidarSolver().solve(captures, initial, policy, options)
    monkeypatch.setattr(solver, "_make_capture_batches", lambda *_args: None)
    monkeypatch.setattr(
        solver, "_make_deskew_evaluator", lambda *_args: solver._deskew_correspondence
    )
    monkeypatch.setattr(
        solver, "_prepare_point_transform", lambda transform: transform.transform_point
    )
    scalar = solver.ContinuousTimeCameraLidarSolver().solve(captures, initial, policy, options)
    assert prepared.as_dict() == scalar.as_dict()
    assert prepared.trace and prepared.status == "converged"
    assert set(prepared.train_capture_ids).isdisjoint(prepared.holdout_capture_ids)


def test_two_clock_cache_is_bounded_and_signed_zeros_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _captures(SE3.identity(), true_offset_sec=0.02)[0]
    item = capture.correspondences[0]
    options = solver.ContinuousTimeCameraLidarOptions()
    evaluator = solver._make_deskew_evaluator([capture], options)
    original = solver._static_point_at_later_body_frame
    queries: list[float] = []

    def counted(*args: object) -> object:
        queries.append(float(args[2]))
        return original(*args)

    monkeypatch.setattr(solver, "_static_point_at_later_body_frame", counted)
    for offset in [0.0, -0.0] * 10:
        evaluator(capture, item, LidarCaptureTimePolicy("seconds", "scan_end", offset), options)
    assert len(queries) == 2
    for offset in [0.02, 0.0, -0.0]:
        evaluator(capture, item, LidarCaptureTimePolicy("seconds", "scan_end", offset), options)
    assert len(queries) == 5


def test_budget_fallback_and_foreign_queries_retain_scalar_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _captures(SE3.identity(), true_offset_sec=0.02)[0]
    options = solver.ContinuousTimeCameraLidarOptions()
    monkeypatch.setattr(solver, "_DESKEW_CACHE_MAX_BYTES", 0)
    assert solver._make_deskew_evaluator([capture], options) is solver._deskew_correspondence
    monkeypatch.setattr(solver, "_DESKEW_CACHE_MAX_BYTES", 128 * 1024 * 1024)
    evaluator = solver._make_deskew_evaluator([capture], options)
    policy = LidarCaptureTimePolicy("seconds", "scan_end")
    for foreign in [replace(capture), replace(capture, transform_body_lidar=SE3.identity())]:
        item = foreign.correspondences[0]
        assert _bytes(evaluator(foreign, item, policy, options)) == _bytes(
            solver._deskew_correspondence(foreign, item, policy, options)
        )
    other = replace(options, use_per_point_time=False)
    item = capture.correspondences[0]
    assert _bytes(evaluator(capture, item, policy, other)) == _bytes(
        solver._deskew_correspondence(capture, item, policy, other)
    )


def test_separate_evaluators_keep_changed_captures_and_settings_fresh() -> None:
    capture = _captures(SE3.identity(), true_offset_sec=0.02)[0]
    changed = replace(capture, transform_body_lidar=SE3((1.0, 2.0, 3.0), (0.2, 0.3, 0.1, 0.9)))
    options = solver.ContinuousTimeCameraLidarOptions()
    first = solver._make_deskew_evaluator([capture], options)
    second = solver._make_deskew_evaluator([changed], options)
    policy = LidarCaptureTimePolicy("seconds", "scan_end", 0.02)
    item = capture.correspondences[0]
    assert _bytes(first(capture, item, policy, options)) != _bytes(
        second(changed, item, policy, options)
    )
    for state, evaluator in [(capture, first), (changed, second)]:
        assert _bytes(evaluator(state, item, policy, options)) == _bytes(
            solver._deskew_correspondence(state, item, policy, options)
        )


def test_prepared_piecewise_still_rejects_missing_temporal_support() -> None:
    capture = _captures(SE3.identity(), true_offset_sec=0.02)[0]
    capture = replace(
        capture,
        body_trajectory=PiecewiseSE3Trajectory(
            (
                ContinuousTrajectoryPose(100.0, SE3.identity()),
                ContinuousTrajectoryPose(100.2, SE3.identity()),
            )
        ),
    )
    options = solver.ContinuousTimeCameraLidarOptions()
    evaluator = solver._make_deskew_evaluator([capture], options)
    for _ in range(2):
        with pytest.raises(ValueError, match="outside"):
            evaluator(
                capture,
                capture.correspondences[0],
                LidarCaptureTimePolicy("seconds", "scan_end"),
                options,
            )
