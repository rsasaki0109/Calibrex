from __future__ import annotations

import math
import struct
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.test_continuous_time_camera_lidar_solver import _captures

from calibrex.core.capture_time import (
    ConstantBodyTwist,
    LidarCaptureTimePolicy,
    _static_point_at_later_body_frame,
)
from calibrex.core.continuous_trajectory import ContinuousTrajectoryPose, PiecewiseSE3Trajectory
from calibrex.core.geometry import SE3
from calibrex.solvers import continuous_time_camera_lidar_solver as solver


def _trajectory() -> PiecewiseSE3Trajectory:
    return PiecewiseSE3Trajectory(
        (
            ContinuousTrajectoryPose(99.0, SE3.identity()),
            ContinuousTrajectoryPose(100.5, SE3((0.7, -0.1, 0.3), (0.1, -0.2, 0.05, 0.98))),
            ContinuousTrajectoryPose(102.0, SE3((1.8, -0.2, 0.6), (-0.1, 0.3, -0.15, 0.9))),
        )
    )


def test_batch_transform_preserves_coordinate_bits_and_signed_zeros() -> None:
    rng = np.random.default_rng(187)
    points = np.concatenate((np.zeros((1, 3)), -np.zeros((1, 3)), rng.normal(size=(97, 3))))
    poses = [SE3.identity(), SE3((0.0, -0.0, 0.0), (0.0, -0.0, 0.0, -1.0))]
    poses += [SE3(tuple(rng.normal(size=3)), tuple(rng.normal(size=4))) for _ in range(20)]
    for pose in poses:
        actual = solver._transform_points(points, pose)
        expected = np.asarray([pose.transform_point(p) for p in points])
        assert actual.tobytes() == expected.tobytes()
    assert solver._transform_points(np.empty((0, 3)), poses[0]).shape == (0, 3)


@pytest.mark.parametrize(
    "axis",
    [
        (0.0, 0.0, 0.0),
        (1.0e-16, 0.0, 0.0),
        (1.0e-15, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.7, -0.4, 1.2),
        (-0.3, 1.9, -0.7),
    ],
)
def test_constant_twist_batch_matches_scalar_for_every_rotation_branch(
    axis: tuple[float, float, float],
) -> None:
    rng = np.random.default_rng(325)
    delta = np.asarray([0.0, -0.0, 1.0e-12, -1.0e-12, 0.2, -0.2, math.pi, -math.pi, 7.0])
    points = rng.normal(size=(delta.size, 3))
    twist = ConstantBodyTwist((0.8, -0.3, 0.2), axis)
    actual = solver._constant_twist_points(points, twist, delta)
    expected = np.asarray(
        [
            _static_point_at_later_body_frame(tuple(p), twist, float(d))
            for p, d in zip(points, delta, strict=True)
        ]
    )
    assert actual.tobytes() == expected.tobytes()
    assert solver._constant_twist_points(np.empty((0, 3)), twist, np.empty(0)).shape == (0, 3)


@pytest.mark.parametrize("piecewise", [False, True])
@pytest.mark.parametrize("direction", ["top_to_bottom", "bottom_to_top"])
@pytest.mark.parametrize("point_times", [False, True])
@pytest.mark.parametrize("rolling", [False, True])
@pytest.mark.parametrize("nanoseconds", [False, True])
def test_capture_batch_deskew_preserves_all_bits_and_clock_semantics(
    piecewise: bool,
    direction: str,
    point_times: bool,
    rolling: bool,
    nanoseconds: bool,
) -> None:
    source = _captures(SE3.identity(), true_offset_sec=0.02)[4]
    capture = replace(
        source,
        transform_body_lidar=SE3((0.42, -0.18, 0.27), (0.01, -0.02, 0.03, 0.999)),
        rolling_shutter_readout_sec=0.02,
        rolling_shutter_direction=direction,
        body_trajectory=_trajectory() if piecewise else None,
        correspondences=tuple(
            replace(
                item, capture_offset_sec=item.capture_offset_sec * (1.0e9 if nanoseconds else 1)
            )
            for item in source.correspondences
        ),
    )
    options = solver.ContinuousTimeCameraLidarOptions(
        use_per_point_time=point_times, use_rolling_shutter=rolling
    )
    deskew = solver._make_deskew_evaluator([capture], options)
    batch = solver._prepare_capture_batch(capture, options, deskew)
    for offset in [0.0, -0.0, 0.02, -0.01, 0.03, 0.03, -0.01, 0.03, 0.0]:
        policy = LidarCaptureTimePolicy(
            "nanoseconds" if nanoseconds else "seconds", "scan_end", offset
        )
        actual = batch.points(struct.pack("d", offset), policy)
        expected = np.asarray(
            [
                solver._deskew_correspondence(
                    capture, p, policy, options
                ).position_lidar_at_reference_m
                for p in batch.selected
            ]
        )
        assert actual.tobytes() == expected.tobytes()
        assert not actual.flags.writeable


@pytest.mark.parametrize("covariance", [False, True])
@pytest.mark.parametrize("piecewise", [False, True])
def test_batch_evaluation_preserves_filtering_order_statistics_and_sensitivity(
    covariance: bool,
    piecewise: bool,
) -> None:
    captures = _captures(SE3.identity(), true_offset_sec=0.02)
    captures[0] = replace(
        captures[0],
        correspondences=(
            replace(captures[0].correspondences[0], reliability=0.0),
            replace(captures[0].correspondences[1], point_lidar_m=(0.0, 0.0, -1.0)),
            replace(captures[0].correspondences[2], point_lidar_m=(100.0, 0.0, 1.0)),
            *captures[0].correspondences[3:],
        ),
    )
    if piecewise:
        captures = [replace(c, body_trajectory=_trajectory()) for c in captures]
    captures += [replace(captures[0], capture_id="empty", correspondences=())]
    options = solver.ContinuousTimeCameraLidarOptions(use_covariance=covariance)
    policy = LidarCaptureTimePolicy("seconds", "scan_start", 0.02)
    pose = SE3((0.1, -0.03, 0.2), (0.01, -0.02, 0.03, 0.999))
    deskew = solver._make_deskew_evaluator(captures, options)
    batches = solver._make_capture_batches(captures, options, deskew)
    for offset in (0.02, -0.03, 0.0, 0.02):
        assert solver._evaluate(captures, pose, policy, offset, options, deskew, batches) == (
            solver.evaluate_continuous_time_camera_lidar(captures, pose, policy, offset, options)
        )
        assert solver._projected_pixels(
            captures, pose, policy, offset, options, deskew, batches
        ) == (solver._projected_pixels(captures, pose, policy, offset, options))
    assert solver._time_sensitivity(captures, pose, policy, 0.0, options, deskew, batches) == (
        solver._time_sensitivity(captures, pose, policy, 0.0, options)
    )
    empty = [replace(captures[0], correspondences=())]
    batch = solver._make_capture_batches(empty, options, deskew)
    assert solver._evaluate(empty, pose, policy, 0.0, options, deskew, batch) == (
        solver.evaluate_continuous_time_camera_lidar(empty, pose, policy, 0.0, options)
    )


def test_batch_clock_cache_is_bounded_and_keeps_policy_and_signed_zero_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = _captures(SE3.identity(), true_offset_sec=0.02)[0]
    options = solver.ContinuousTimeCameraLidarOptions()
    batch = solver._prepare_capture_batch(capture, options, solver._deskew_correspondence)
    original = solver._constant_twist_points
    calls = []

    def counted(*args: object) -> object:
        calls.append(len(args[0]))
        return original(*args)

    monkeypatch.setattr(solver, "_constant_twist_points", counted)
    for offset in [0.0, -0.0] * 10:
        policy = LidarCaptureTimePolicy("seconds", "scan_end", offset)
        batch.points(struct.pack("d", offset), policy)
    assert len(calls) == 2
    for offset in [0.02, 0.0, -0.0]:
        policy = LidarCaptureTimePolicy("seconds", "scan_end", offset)
        batch.points(struct.pack("d", offset), policy)
    assert len(calls) == 5
    for unit, stamp in [("nanoseconds", "scan_end"), ("nanoseconds", "scan_start")]:
        batch.points(struct.pack("d", 0.0), LidarCaptureTimePolicy(unit, stamp))
    assert len(calls) == 7
    assert set(calls) == {len(capture.correspondences)}


def test_batch_budget_fallback_and_new_solves_preserve_complete_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = _captures(SE3.identity(), true_offset_sec=0.02)
    options = solver.ContinuousTimeCameraLidarOptions(max_evaluations=80)
    policy = LidarCaptureTimePolicy("seconds", "scan_end")
    batch = solver.ContinuousTimeCameraLidarSolver().solve(
        captures, SE3.identity(), policy, options
    )
    changed = [
        replace(c, transform_body_lidar=SE3((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
        for c in captures
    ]
    fresh = solver.ContinuousTimeCameraLidarSolver().solve(changed, SE3.identity(), policy, options)
    monkeypatch.setattr(solver, "_DESKEW_CACHE_MAX_BYTES", 0)
    assert solver._make_capture_batches(captures, options, solver._deskew_correspondence) is None
    scalar = solver.ContinuousTimeCameraLidarSolver().solve(
        captures, SE3.identity(), policy, options
    )
    scalar_fresh = solver.ContinuousTimeCameraLidarSolver().solve(
        changed, SE3.identity(), policy, options
    )
    assert batch.as_dict() == scalar.as_dict()
    assert fresh.as_dict() == scalar_fresh.as_dict()
    assert batch.as_dict() != fresh.as_dict()


def test_piecewise_complete_trace_preserved_and_support_checked_before_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = [
        replace(c, body_trajectory=_trajectory())
        for c in _captures(SE3.identity(), true_offset_sec=0.02)
    ]
    options = solver.ContinuousTimeCameraLidarOptions(max_evaluations=60)
    policy = LidarCaptureTimePolicy("seconds", "scan_end")
    batch = solver.ContinuousTimeCameraLidarSolver().solve(
        captures, SE3.identity(), policy, options
    )
    monkeypatch.setattr(solver, "_make_capture_batches", lambda *_args: None)
    scalar = solver.ContinuousTimeCameraLidarSolver().solve(
        captures, SE3.identity(), policy, options
    )
    assert batch.as_dict() == scalar.as_dict()
    missing = replace(captures[0], lidar_message_stamp_sec=98.0)
    with pytest.raises(ValueError, match="trajectory support"):
        solver.ContinuousTimeCameraLidarSolver().solve([missing], SE3.identity(), policy, options)
