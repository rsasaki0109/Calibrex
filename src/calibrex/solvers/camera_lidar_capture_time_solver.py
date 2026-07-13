"""Per-point Camera-LiDAR capture-time offset and deskew evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from calibrex.core.capture_time import (
    ConstantBodyTwist,
    LidarCaptureTimePolicy,
    TimedLidarPoint,
    deskew_lidar_points_to_reference,
)
from calibrex.core.geometry import SE3, Vector3
from calibrex.evaluation.holdout import split_indices

CaptureTimeStatus = Literal[
    "converged",
    "evidence_only",
    "insufficient_captures",
    "missing_holdout",
    "degenerate_motion",
    "time_offset_at_bound",
]


@dataclass(frozen=True)
class CameraLidarTimedCapture:
    """One LiDAR scan evaluated at a paired camera exposure instant."""

    capture_id: str
    lidar_message_stamp_sec: float
    camera_exposure_time_sec: float
    points: tuple[TimedLidarPoint, ...]
    twist: ConstantBodyTwist
    transform_body_lidar: SE3
    reference_positions_lidar_at_camera_m: tuple[Vector3, ...] | None = None

    def __post_init__(self) -> None:
        if not self.capture_id:
            raise ValueError("capture_id must not be empty")
        if not math.isfinite(self.lidar_message_stamp_sec) or not math.isfinite(
            self.camera_exposure_time_sec
        ):
            raise ValueError("capture timestamps must be finite")
        point_ids = [point.point_id for point in self.points]
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("timed point IDs must be unique within each capture")
        if self.reference_positions_lidar_at_camera_m is not None and len(
            self.reference_positions_lidar_at_camera_m
        ) != len(self.points):
            raise ValueError("reference positions must match timed point count")
        for point in self.points:
            if not all(math.isfinite(value) for value in point.position_lidar_m):
                raise ValueError("timed LiDAR point coordinates must be finite")
            if not math.isfinite(float(point.capture_offset)):
                raise ValueError("point capture offsets must be finite")
        if self.reference_positions_lidar_at_camera_m is not None:
            for reference_position in self.reference_positions_lidar_at_camera_m:
                if not all(math.isfinite(value) for value in reference_position):
                    raise ValueError("reference point coordinates must be finite")
        twist_values = (
            *self.twist.linear_velocity_body_mps,
            *self.twist.angular_velocity_body_radps,
        )
        if not all(math.isfinite(value) for value in twist_values):
            raise ValueError("body twist must be finite")


@dataclass(frozen=True)
class CameraLidarCaptureTimeOptions:
    min_train_captures: int = 4
    min_holdout_captures: int = 2
    holdout_ratio: float = 0.2
    split_seed: int = 0
    max_abs_time_offset_sec: float = 0.1
    coarse_step_sec: float = 0.002
    refinement_tolerance_sec: float = 1.0e-8
    max_refinement_iterations: int = 80
    sensitivity_step_sec: float = 0.001
    min_time_sensitivity_mps: float = 0.05
    known_bad_offsets_sec: tuple[float, ...] = (-0.02, -0.01, -0.005, 0.005, 0.01, 0.02)
    known_bad_margin_m: float = 0.001


@dataclass(frozen=True)
class CameraLidarCaptureTimeEvaluation:
    capture_count: int
    point_count: int
    mean_point_time_span_sec: float | None
    mean_abs_camera_to_lidar_stamp_sec: float | None
    mean_abs_point_to_camera_time_sec: float | None
    deskew_displacement_rmse_m: float | None
    deskew_displacement_max_m: float | None
    reference_rmse_m: float | None


@dataclass(frozen=True)
class CameraLidarTimeProbe:
    injected_offset_sec: float
    candidate_offset_sec: float
    holdout_reference_rmse_m: float | None
    difference_from_nominal_rmse_m: float | None
    residual_increase_m: float | None
    detectable: bool | None


@dataclass(frozen=True)
class CameraLidarCaptureTimeResult:
    status: CaptureTimeStatus
    reason: str
    estimated_sensor_time_offset_sec: float | None
    train_capture_ids: tuple[str, ...]
    holdout_capture_ids: tuple[str, ...]
    train_evaluation: CameraLidarCaptureTimeEvaluation
    holdout_evaluation: CameraLidarCaptureTimeEvaluation
    time_jacobian_norm_mps: float | None
    time_sensitivity_rms_mps: float | None
    time_observability_rank: int
    time_condition_number: float | None
    probes: tuple[CameraLidarTimeProbe, ...]
    iterations: int
    policy: LidarCaptureTimePolicy
    solver_options: CameraLidarCaptureTimeOptions

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "estimated_sensor_time_offset_sec": self.estimated_sensor_time_offset_sec,
            "train_capture_ids": list(self.train_capture_ids),
            "holdout_capture_ids": list(self.holdout_capture_ids),
            "train_evaluation": asdict(self.train_evaluation),
            "holdout_evaluation": asdict(self.holdout_evaluation),
            "time_jacobian_norm_mps": self.time_jacobian_norm_mps,
            "time_sensitivity_rms_mps": self.time_sensitivity_rms_mps,
            "time_observability_rank": self.time_observability_rank,
            "time_condition_number": self.time_condition_number,
            "weak_directions": [] if self.time_observability_rank == 1 else ["time_offset"],
            "known_bad_probes": [asdict(probe) for probe in self.probes],
            "iterations": self.iterations,
            "policy": self.policy.as_dict(),
            "solver_options": asdict(self.solver_options),
            "method": "continuous_twist_per_point_time_offset/v0.1",
            "specialization": (
                "Taylor-Nieto/Park motion-based timing contract with fixed body twist; "
                "reference-point mode estimates one clock offset, while evidence-only "
                "mode applies measured firing times without claiming an estimate"
            ),
            "papers": [
                {
                    "title": (
                        "Motion-Based Calibration of Multimodal Sensor Extrinsics and "
                        "Timing Offset Estimation"
                    ),
                    "authors": "Zachary Taylor and Juan Nieto",
                    "venue": "IEEE Transactions on Robotics 32(5), 2016",
                    "doi": "10.1109/TRO.2016.2596771",
                },
                {
                    "title": (
                        "Spatiotemporal Camera-LiDAR Calibration: A Targetless and "
                        "Structureless Approach"
                    ),
                    "authors": (
                        "Chanoh Park, Peyman Moghadam, Soohwan Kim, "
                        "Sridha Sridharan, and Clinton Fookes"
                    ),
                    "venue": "IEEE Robotics and Automation Letters 5(2), 2020",
                    "doi": "10.1109/LRA.2020.2969164",
                },
            ],
            "clock_convention": "lidar_sensor_time + dt_lidar = camera_reference_time",
            "implementation": "independent ROS-free Python implementation; no paper code copied",
        }


class CameraLidarCaptureTimeSolver:
    """Estimate or falsify one LiDAR clock offset using per-point firing times."""

    def solve(
        self,
        captures: Sequence[CameraLidarTimedCapture],
        policy: LidarCaptureTimePolicy,
        options: CameraLidarCaptureTimeOptions | None = None,
    ) -> CameraLidarCaptureTimeResult:
        opts = options or CameraLidarCaptureTimeOptions()
        _validate_options(opts)
        identifiers = [capture.capture_id for capture in captures]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("capture IDs must be unique")
        ordered = sorted(captures, key=lambda capture: capture.capture_id)
        reference_flags = [
            capture.reference_positions_lidar_at_camera_m is not None for capture in ordered
        ]
        if any(reference_flags) and not all(reference_flags):
            raise ValueError("reference positions must be supplied for every capture or none")
        train_indices, holdout_indices = split_indices(
            len(ordered), opts.holdout_ratio, opts.split_seed
        )
        train = [ordered[index] for index in train_indices]
        holdout = [ordered[index] for index in holdout_indices]
        train_ids = tuple(capture.capture_id for capture in train)
        holdout_ids = tuple(capture.capture_id for capture in holdout)
        if len(train) < opts.min_train_captures:
            return _empty_result("insufficient_captures", train_ids, holdout_ids, policy, opts)
        if len(holdout) < opts.min_holdout_captures:
            return _empty_result("missing_holdout", train_ids, holdout_ids, policy, opts)
        if any(not capture.points for capture in ordered):
            return _empty_result("insufficient_captures", train_ids, holdout_ids, policy, opts)

        nominal_offset = policy.sensor_time_offset_sec
        jacobian_norm, sensitivity = _time_observability(
            train, policy, nominal_offset, opts.sensitivity_step_sec
        )
        rank = int(sensitivity is not None and sensitivity >= opts.min_time_sensitivity_mps)
        if rank == 0:
            train_evaluation = _evaluate(train, policy, nominal_offset)
            holdout_evaluation = _evaluate(holdout, policy, nominal_offset)
            return CameraLidarCaptureTimeResult(
                "degenerate_motion",
                "body motion does not make the LiDAR clock offset observable",
                None,
                train_ids,
                holdout_ids,
                train_evaluation,
                holdout_evaluation,
                jacobian_norm,
                sensitivity,
                0,
                None,
                _known_bad_probes(
                    holdout,
                    policy,
                    nominal_offset,
                    holdout_evaluation.reference_rmse_m,
                    opts,
                ),
                0,
                policy,
                opts,
            )

        has_references = all(reference_flags)
        iterations = 0
        if has_references:
            estimated, iterations = _estimate_offset(train, policy, opts)
            at_bound = abs(estimated) >= (
                opts.max_abs_time_offset_sec - opts.refinement_tolerance_sec
            )
            status: CaptureTimeStatus = "time_offset_at_bound" if at_bound else "converged"
            reason = (
                "time-offset optimum reached the configured search bound"
                if at_bound
                else "bounded one-dimensional time-offset refinement converged"
            )
        else:
            estimated = nominal_offset
            status = "evidence_only"
            reason = (
                "measured firing times were applied, but no cross-modal geometric "
                "references were supplied; clock offset was not estimated"
            )
        train_evaluation = _evaluate(train, policy, estimated)
        holdout_evaluation = _evaluate(holdout, policy, estimated)
        probes = _known_bad_probes(
            holdout,
            policy,
            estimated,
            holdout_evaluation.reference_rmse_m,
            opts,
        )
        return CameraLidarCaptureTimeResult(
            status,
            reason,
            estimated if has_references else None,
            train_ids,
            holdout_ids,
            train_evaluation,
            holdout_evaluation,
            jacobian_norm,
            sensitivity,
            1,
            1.0,
            probes,
            iterations,
            policy,
            opts,
        )


def evaluate_camera_lidar_capture_times(
    captures: Sequence[CameraLidarTimedCapture],
    policy: LidarCaptureTimePolicy,
    sensor_time_offset_sec: float,
) -> CameraLidarCaptureTimeEvaluation:
    """Evaluate an unchanged set of timed captures at one clock offset."""

    return _evaluate(captures, policy, sensor_time_offset_sec)


def _evaluate(
    captures: Sequence[CameraLidarTimedCapture],
    policy: LidarCaptureTimePolicy,
    offset_sec: float,
) -> CameraLidarCaptureTimeEvaluation:
    spans: list[float] = []
    stamp_deltas: list[float] = []
    point_time_deltas: list[float] = []
    deskew_squared: list[float] = []
    reference_squared: list[float] = []
    for capture in captures:
        positions, capture_times = _deskew_positions(capture, policy, offset_sec)
        if capture_times:
            spans.append(max(capture_times) - min(capture_times))
        stamp_deltas.append(abs(capture.camera_exposure_time_sec - capture.lidar_message_stamp_sec))
        point_time_deltas.extend(
            abs(capture.camera_exposure_time_sec - value) for value in capture_times
        )
        for point, position in zip(capture.points, positions, strict=True):
            deskew_squared.append(_squared_distance(point.position_lidar_m, position))
        references = capture.reference_positions_lidar_at_camera_m
        if references is not None:
            reference_squared.extend(
                _squared_distance(reference, position)
                for reference, position in zip(references, positions, strict=True)
            )
    displacement_max = math.sqrt(max(deskew_squared)) if deskew_squared else None
    return CameraLidarCaptureTimeEvaluation(
        capture_count=len(captures),
        point_count=sum(len(capture.points) for capture in captures),
        mean_point_time_span_sec=_mean(spans),
        mean_abs_camera_to_lidar_stamp_sec=_mean(stamp_deltas),
        mean_abs_point_to_camera_time_sec=_mean(point_time_deltas),
        deskew_displacement_rmse_m=_root_mean_square(deskew_squared),
        deskew_displacement_max_m=displacement_max,
        reference_rmse_m=_root_mean_square(reference_squared),
    )


def _deskew_positions(
    capture: CameraLidarTimedCapture,
    policy: LidarCaptureTimePolicy,
    offset_sec: float,
) -> tuple[tuple[Vector3, ...], tuple[float, ...]]:
    active_policy = LidarCaptureTimePolicy(
        point_offset_unit=policy.point_offset_unit,
        stamp_reference=policy.stamp_reference,
        sensor_time_offset_sec=offset_sec,
    )
    result = deskew_lidar_points_to_reference(
        capture.points,
        message_stamp_sec=capture.lidar_message_stamp_sec,
        reference_time_sec=capture.camera_exposure_time_sec,
        policy=active_policy,
        twist=capture.twist,
        transform_body_lidar=capture.transform_body_lidar,
    )
    return (
        tuple(point.position_lidar_at_reference_m for point in result.points),
        tuple(point.capture_time_reference_sec for point in result.points),
    )


def _time_observability(
    captures: Sequence[CameraLidarTimedCapture],
    policy: LidarCaptureTimePolicy,
    offset_sec: float,
    step_sec: float,
) -> tuple[float | None, float | None]:
    squared_derivatives: list[float] = []
    for capture in captures:
        minus, _ = _deskew_positions(capture, policy, offset_sec - step_sec)
        plus, _ = _deskew_positions(capture, policy, offset_sec + step_sec)
        for left, right in zip(minus, plus, strict=True):
            derivative_squared = _squared_distance(left, right) / (4.0 * step_sec * step_sec)
            squared_derivatives.append(derivative_squared)
    if not squared_derivatives:
        return None, None
    return math.sqrt(sum(squared_derivatives)), _root_mean_square(squared_derivatives)


def _estimate_offset(
    captures: Sequence[CameraLidarTimedCapture],
    policy: LidarCaptureTimePolicy,
    options: CameraLidarCaptureTimeOptions,
) -> tuple[float, int]:
    bound = options.max_abs_time_offset_sec
    step_count = max(2, math.ceil((2.0 * bound) / options.coarse_step_sec))
    grid = [-bound + 2.0 * bound * index / step_count for index in range(step_count + 1)]
    values = [_reference_objective(captures, policy, value) for value in grid]
    best_index = min(range(len(grid)), key=lambda index: (values[index], index))
    if best_index == 0 or best_index == len(grid) - 1:
        return grid[best_index], len(grid)
    left = grid[best_index - 1]
    right = grid[best_index + 1]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    x_left = right - golden * (right - left)
    x_right = left + golden * (right - left)
    f_left = _reference_objective(captures, policy, x_left)
    f_right = _reference_objective(captures, policy, x_right)
    iterations = len(grid)
    for _ in range(options.max_refinement_iterations):
        iterations += 1
        if right - left <= options.refinement_tolerance_sec:
            break
        if f_left <= f_right:
            right = x_right
            x_right = x_left
            f_right = f_left
            x_left = right - golden * (right - left)
            f_left = _reference_objective(captures, policy, x_left)
        else:
            left = x_left
            x_left = x_right
            f_left = f_right
            x_right = left + golden * (right - left)
            f_right = _reference_objective(captures, policy, x_right)
    return (left + right) / 2.0, iterations


def _reference_objective(
    captures: Sequence[CameraLidarTimedCapture],
    policy: LidarCaptureTimePolicy,
    offset_sec: float,
) -> float:
    squared: list[float] = []
    for capture in captures:
        references = capture.reference_positions_lidar_at_camera_m
        if references is None:
            raise ValueError("time-offset estimation requires reference positions")
        positions, _ = _deskew_positions(capture, policy, offset_sec)
        squared.extend(
            _squared_distance(reference, position)
            for reference, position in zip(references, positions, strict=True)
        )
    return sum(squared) / len(squared) if squared else math.inf


def _known_bad_probes(
    holdout: Sequence[CameraLidarTimedCapture],
    policy: LidarCaptureTimePolicy,
    baseline_offset_sec: float,
    baseline_reference_rmse_m: float | None,
    options: CameraLidarCaptureTimeOptions,
) -> tuple[CameraLidarTimeProbe, ...]:
    nominal = [_deskew_positions(capture, policy, baseline_offset_sec)[0] for capture in holdout]
    probes: list[CameraLidarTimeProbe] = []
    for injected in options.known_bad_offsets_sec:
        candidate_offset = baseline_offset_sec + injected
        candidate_squared: list[float] = []
        for capture, nominal_positions in zip(holdout, nominal, strict=True):
            candidate_positions, _ = _deskew_positions(capture, policy, candidate_offset)
            candidate_squared.extend(
                _squared_distance(left, right)
                for left, right in zip(nominal_positions, candidate_positions, strict=True)
            )
        difference = _root_mean_square(candidate_squared)
        candidate_evaluation = _evaluate(holdout, policy, candidate_offset)
        candidate_reference = candidate_evaluation.reference_rmse_m
        residual_increase = (
            candidate_reference - baseline_reference_rmse_m
            if candidate_reference is not None and baseline_reference_rmse_m is not None
            else difference
        )
        probes.append(
            CameraLidarTimeProbe(
                injected,
                candidate_offset,
                candidate_reference,
                difference,
                residual_increase,
                residual_increase > options.known_bad_margin_m
                if residual_increase is not None
                else None,
            )
        )
    return tuple(probes)


def _validate_options(options: CameraLidarCaptureTimeOptions) -> None:
    positive = (
        options.min_train_captures,
        options.min_holdout_captures,
        options.max_abs_time_offset_sec,
        options.coarse_step_sec,
        options.refinement_tolerance_sec,
        options.max_refinement_iterations,
        options.sensitivity_step_sec,
        options.min_time_sensitivity_mps,
        options.known_bad_margin_m,
    )
    if any(not math.isfinite(float(value)) or value <= 0 for value in positive):
        raise ValueError("capture-time positive options must be finite and positive")
    if not 0.0 < options.holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in (0, 1)")
    if not options.known_bad_offsets_sec or any(
        not math.isfinite(value) or value == 0.0 for value in options.known_bad_offsets_sec
    ):
        raise ValueError("known_bad_offsets_sec must contain finite non-zero values")


def _empty_result(
    status: Literal["insufficient_captures", "missing_holdout"],
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    policy: LidarCaptureTimePolicy,
    options: CameraLidarCaptureTimeOptions,
) -> CameraLidarCaptureTimeResult:
    empty = CameraLidarCaptureTimeEvaluation(0, 0, None, None, None, None, None, None)
    return CameraLidarCaptureTimeResult(
        status,
        "not enough complete timed train captures"
        if status == "insufficient_captures"
        else "the deterministic capture split did not produce a holdout set",
        None,
        train_ids,
        holdout_ids,
        empty,
        empty,
        None,
        None,
        0,
        None,
        (),
        0,
        policy,
        options,
    )


def _squared_distance(left: Vector3, right: Vector3) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _root_mean_square(squared_values: Sequence[float]) -> float | None:
    return math.sqrt(sum(squared_values) / len(squared_values)) if squared_values else None
