"""Joint camera--LiDAR extrinsic and clock-offset refinement.

The solver consumes explicit probabilistic 2D--3D correspondences, per-point
LiDAR firing times, and body twists.  It is ROS-independent and deliberately
keeps trajectory estimation outside this boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.capture_time import (
    ConstantBodyTwist,
    DeskewedLidarPoint,
    LidarCaptureTimePolicy,
    TimedLidarPoint,
    deskew_lidar_points_to_reference,
)
from calibrex.core.continuous_trajectory import PiecewiseSE3Trajectory
from calibrex.core.geometry import SE3
from calibrex.data.depth import DepthCameraIntrinsics
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta

FloatArray: TypeAlias = NDArray[np.float64]
ContinuousTimeStatus = Literal[
    "converged",
    "max_evaluations",
    "insufficient_correspondences",
    "missing_holdout",
    "degenerate_motion",
    "at_bound",
]


@dataclass(frozen=True)
class TimedProbabilisticImageCorrespondence:
    """One LiDAR firing and its image-plane probability distribution."""

    correspondence_id: str
    point_lidar_m: tuple[float, float, float]
    capture_offset_sec: float
    image_mean_px: tuple[float, float]
    image_covariance_px2: tuple[float, float, float, float]
    outlier_probability: float = 0.0
    reliability: float = 1.0

    def __post_init__(self) -> None:
        values = (
            *self.point_lidar_m,
            self.capture_offset_sec,
            *self.image_mean_px,
            *self.image_covariance_px2,
            self.outlier_probability,
            self.reliability,
        )
        if not self.correspondence_id or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("timed correspondence values must be finite")
        a, b, c, d = self.image_covariance_px2
        if not math.isclose(b, c, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError("image covariance must be symmetric")
        if a <= 0.0 or d <= 0.0 or a * d - b * c <= 0.0:
            raise ValueError("image covariance must be positive definite")
        if not 0.0 <= self.outlier_probability <= 1.0:
            raise ValueError("outlier_probability must be in [0, 1]")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be in [0, 1]")


@dataclass(frozen=True)
class ContinuousTimeCameraLidarCapture:
    """One camera exposure paired with a sequential LiDAR scan."""

    capture_id: str
    lidar_message_stamp_sec: float
    camera_exposure_time_sec: float
    correspondences: tuple[TimedProbabilisticImageCorrespondence, ...]
    twist: ConstantBodyTwist
    transform_body_lidar: SE3
    camera: DepthCameraIntrinsics
    rolling_shutter_readout_sec: float = 0.0
    rolling_shutter_direction: Literal[
        "top_to_bottom", "bottom_to_top"
    ] = "top_to_bottom"
    body_trajectory: PiecewiseSE3Trajectory | None = None

    def __post_init__(self) -> None:
        if not self.capture_id:
            raise ValueError("capture_id must not be empty")
        if not math.isfinite(self.lidar_message_stamp_sec) or not math.isfinite(
            self.camera_exposure_time_sec
        ):
            raise ValueError("capture timestamps must be finite")
        identifiers = [item.correspondence_id for item in self.correspondences]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("correspondence IDs must be unique within a capture")
        if self.camera.projection != "pinhole":
            raise ValueError("continuous-time solver v0.1 supports pinhole cameras")
        if (
            not math.isfinite(self.rolling_shutter_readout_sec)
            or self.rolling_shutter_readout_sec < 0.0
        ):
            raise ValueError("rolling_shutter_readout_sec must be non-negative")


@dataclass(frozen=True)
class ContinuousTimeCameraLidarOptions:
    """Bounds, deterministic pattern steps, and evidence gates."""

    holdout_ratio: float = 0.25
    split_seed: int = 0
    minimum_train_correspondences: int = 24
    minimum_holdout_correspondences: int = 8
    minimum_confidence: float = 0.25
    rotation_bound_deg: float = 2.0
    translation_bound_m: float = 0.25
    time_offset_bound_sec: float = 0.05
    initial_rotation_step_deg: float = 0.5
    initial_translation_step_m: float = 0.05
    initial_time_step_sec: float = 0.01
    minimum_rotation_step_deg: float = 0.02
    minimum_translation_step_m: float = 0.002
    minimum_time_step_sec: float = 0.0002
    max_evaluations: int = 500
    cauchy_scale: float = 3.0
    time_sensitivity_probe_sec: float = 0.005
    minimum_time_sensitivity_px_per_sec: float = 0.5
    estimate_time_offset: bool = True
    use_per_point_time: bool = True
    use_covariance: bool = True
    use_rolling_shutter: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.holdout_ratio < 1.0:
            raise ValueError("holdout_ratio must be in (0, 1)")
        if self.minimum_train_correspondences < 4:
            raise ValueError("minimum_train_correspondences must be at least four")
        if self.minimum_holdout_correspondences < 4:
            raise ValueError("minimum_holdout_correspondences must be at least four")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        bounds = (
            self.rotation_bound_deg,
            self.translation_bound_m,
            self.time_offset_bound_sec,
        )
        if any(value <= 0.0 for value in bounds):
            raise ValueError("continuous-time parameter bounds must be positive")
        if not (
            0.0
            < self.minimum_rotation_step_deg
            <= self.initial_rotation_step_deg
            <= self.rotation_bound_deg
        ):
            raise ValueError("rotation steps and bound are inconsistent")
        if not (
            0.0
            < self.minimum_translation_step_m
            <= self.initial_translation_step_m
            <= self.translation_bound_m
        ):
            raise ValueError("translation steps and bound are inconsistent")
        if not (
            0.0
            < self.minimum_time_step_sec
            <= self.initial_time_step_sec
            <= self.time_offset_bound_sec
        ):
            raise ValueError("time steps and bound are inconsistent")
        if self.max_evaluations < 15:
            raise ValueError("max_evaluations must be at least 15")
        if self.cauchy_scale <= 0.0:
            raise ValueError("cauchy_scale must be positive")
        if self.time_sensitivity_probe_sec <= 0.0:
            raise ValueError("time_sensitivity_probe_sec must be positive")
        if self.minimum_time_sensitivity_px_per_sec <= 0.0:
            raise ValueError(
                "minimum_time_sensitivity_px_per_sec must be positive"
            )


@dataclass(frozen=True)
class ContinuousTimeEvaluation:
    """Uncertainty-normalized reprojection evidence."""

    objective: float
    weighted_reprojection_rmse_px: float | None
    mean_mahalanobis_error: float | None
    valid_correspondence_count: int
    skipped_correspondence_count: int


@dataclass(frozen=True)
class ContinuousTimeIteration:
    """One evaluated bounded correction."""

    evaluation: int
    delta_rotation_deg_xyz: tuple[float, float, float]
    delta_translation_m_xyz: tuple[float, float, float]
    time_offset_sec: float
    objective: float
    accepted: bool


@dataclass(frozen=True)
class ContinuousTimeCameraLidarResult:
    """Joint estimate with train/holdout evidence and observability."""

    status: ContinuousTimeStatus
    reason: str
    initial_transform_camera_lidar: SE3
    transform_camera_lidar: SE3
    initial_time_offset_sec: float
    estimated_time_offset_sec: float
    train_capture_ids: tuple[str, ...]
    holdout_capture_ids: tuple[str, ...]
    initial_train_evaluation: ContinuousTimeEvaluation
    final_train_evaluation: ContinuousTimeEvaluation
    initial_holdout_evaluation: ContinuousTimeEvaluation
    final_holdout_evaluation: ContinuousTimeEvaluation
    time_sensitivity_px_per_sec: float
    time_observability_rank: int
    trajectory_model: Literal["constant_body_twist", "piecewise_se3"]
    trace: tuple[ContinuousTimeIteration, ...]
    options: ContinuousTimeCameraLidarOptions

    def as_dict(self) -> dict[str, object]:
        """Return a provenance-rich solver record."""

        return {
            "status": self.status,
            "reason": self.reason,
            "initial_transform_camera_lidar": (
                self.initial_transform_camera_lidar.as_dict()
            ),
            "transform_camera_lidar": self.transform_camera_lidar.as_dict(),
            "initial_time_offset_sec": self.initial_time_offset_sec,
            "estimated_time_offset_sec": self.estimated_time_offset_sec,
            "train_capture_ids": list(self.train_capture_ids),
            "holdout_capture_ids": list(self.holdout_capture_ids),
            "initial_train_evaluation": asdict(self.initial_train_evaluation),
            "final_train_evaluation": asdict(self.final_train_evaluation),
            "initial_holdout_evaluation": asdict(
                self.initial_holdout_evaluation
            ),
            "final_holdout_evaluation": asdict(self.final_holdout_evaluation),
            "time_sensitivity_px_per_sec": self.time_sensitivity_px_per_sec,
            "time_observability_rank": self.time_observability_rank,
            "trajectory_model": self.trajectory_model,
            "trace": [asdict(item) for item in self.trace],
            "options": asdict(self.options),
            "method": "continuous_time_probabilistic_camera_lidar/v0.1",
            "clock_convention": (
                "lidar_sensor_time + dt_lidar = camera_reference_time"
            ),
            "trajectory_boundary": (
                "body motion is supplied by a bounded piecewise SE(3) "
                "trajectory when present, otherwise by a per-capture "
                "constant body twist; trajectory estimation remains external"
            ),
            "papers": [
                {
                    "title": (
                        "Motion-Based Calibration of Multimodal Sensor "
                        "Extrinsics and Timing Offset Estimation"
                    ),
                    "doi": "10.1109/TRO.2016.2596771",
                },
                {
                    "title": (
                        "Spatiotemporal Camera-LiDAR Calibration: A "
                        "Targetless and Structureless Approach"
                    ),
                    "doi": "10.1109/LRA.2020.2969164",
                },
            ],
            "implementation": (
                "independent ROS-free bounded optimizer; no paper code copied"
            ),
        }


class ContinuousTimeCameraLidarSolver:
    """Jointly refine one extrinsic and one LiDAR-to-camera clock offset."""

    def solve(
        self,
        captures: Sequence[ContinuousTimeCameraLidarCapture],
        initial_transform_camera_lidar: SE3,
        capture_time_policy: LidarCaptureTimePolicy,
        options: ContinuousTimeCameraLidarOptions | None = None,
    ) -> ContinuousTimeCameraLidarResult:
        """Optimize training captures and report disjoint holdout evidence."""

        settings = options or ContinuousTimeCameraLidarOptions()
        identifiers = [item.capture_id for item in captures]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("capture IDs must be unique")
        trajectory_presence = {
            item.body_trajectory is not None for item in captures
        }
        if len(trajectory_presence) > 1:
            raise ValueError(
                "captures must use one consistent trajectory model"
            )
        _validate_trajectory_temporal_support(
            captures,
            capture_time_policy,
            settings,
        )
        ordered = sorted(captures, key=lambda item: item.capture_id)
        train_indices, holdout_indices = split_indices(
            len(ordered), settings.holdout_ratio, settings.split_seed
        )
        train = tuple(ordered[index] for index in train_indices)
        holdout = tuple(ordered[index] for index in holdout_indices)
        initial_time = capture_time_policy.sensor_time_offset_sec
        initial_train = evaluate_continuous_time_camera_lidar(
            train,
            initial_transform_camera_lidar,
            capture_time_policy,
            initial_time,
            settings,
        )
        initial_holdout = evaluate_continuous_time_camera_lidar(
            holdout,
            initial_transform_camera_lidar,
            capture_time_policy,
            initial_time,
            settings,
        )
        if (
            initial_train.valid_correspondence_count
            < settings.minimum_train_correspondences
        ):
            return _terminal_result(
                "insufficient_correspondences",
                "too few training correspondences passed confidence/projection gates",
                initial_transform_camera_lidar,
                initial_time,
                train,
                holdout,
                initial_train,
                initial_holdout,
                0.0,
                settings,
            )
        if (
            initial_holdout.valid_correspondence_count
            < settings.minimum_holdout_correspondences
        ):
            return _terminal_result(
                "missing_holdout",
                "too few disjoint holdout correspondences passed the gates",
                initial_transform_camera_lidar,
                initial_time,
                train,
                holdout,
                initial_train,
                initial_holdout,
                0.0,
                settings,
            )
        sensitivity = _time_sensitivity(
            train,
            initial_transform_camera_lidar,
            capture_time_policy,
            initial_time,
            settings,
        )
        if (
            settings.estimate_time_offset
            and sensitivity < settings.minimum_time_sensitivity_px_per_sec
        ):
            return _terminal_result(
                "degenerate_motion",
                "weak motion makes the clock offset unobservable",
                initial_transform_camera_lidar,
                initial_time,
                train,
                holdout,
                initial_train,
                initial_holdout,
                sensitivity,
                settings,
            )

        delta: FloatArray = np.zeros(7, dtype=float)
        best = initial_train
        steps: FloatArray = np.asarray(
            [
                settings.initial_rotation_step_deg,
                settings.initial_rotation_step_deg,
                settings.initial_rotation_step_deg,
                settings.initial_translation_step_m,
                settings.initial_translation_step_m,
                settings.initial_translation_step_m,
                settings.initial_time_step_sec,
            ],
            dtype=float,
        )
        minimum_steps: FloatArray = np.asarray(
            [
                settings.minimum_rotation_step_deg,
                settings.minimum_rotation_step_deg,
                settings.minimum_rotation_step_deg,
                settings.minimum_translation_step_m,
                settings.minimum_translation_step_m,
                settings.minimum_translation_step_m,
                settings.minimum_time_step_sec,
            ],
            dtype=float,
        )
        bounds: FloatArray = np.asarray(
            [
                settings.rotation_bound_deg,
                settings.rotation_bound_deg,
                settings.rotation_bound_deg,
                settings.translation_bound_m,
                settings.translation_bound_m,
                settings.translation_bound_m,
                settings.time_offset_bound_sec,
            ],
            dtype=float,
        )
        active: NDArray[np.bool_] = np.ones(7, dtype=bool)
        active[6] = settings.estimate_time_offset
        trace = [
            _iteration(1, delta, initial_time, best.objective, accepted=True)
        ]
        evaluations = 1
        while (
            np.any((steps >= minimum_steps) & active)
            and evaluations < settings.max_evaluations
        ):
            candidates: list[
                tuple[float, tuple[float, ...], FloatArray, ContinuousTimeEvaluation]
            ] = []
            for axis in range(7):
                if not active[axis] or steps[axis] < minimum_steps[axis]:
                    continue
                for direction in (-1.0, 1.0):
                    candidate = delta.copy()
                    candidate[axis] = float(
                        np.clip(
                            candidate[axis] + direction * steps[axis],
                            -bounds[axis],
                            bounds[axis],
                        )
                    )
                    if np.array_equal(candidate, delta):
                        continue
                    transform = apply_local_se3_delta(
                        initial_transform_camera_lidar,
                        candidate[:3],
                        candidate[3:6],
                    )
                    result = evaluate_continuous_time_camera_lidar(
                        train,
                        transform,
                        capture_time_policy,
                        initial_time + float(candidate[6]),
                        settings,
                    )
                    evaluations += 1
                    candidates.append(
                        (
                            result.objective,
                            tuple(candidate.tolist()),
                            candidate,
                            result,
                        )
                    )
                    if evaluations >= settings.max_evaluations:
                        break
                if evaluations >= settings.max_evaluations:
                    break
            if not candidates:
                break
            candidate_objective, _tie, candidate_delta, candidate_result = min(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            accepted = candidate_objective < best.objective - 1.0e-12
            for objective, _tie, candidate, _result in candidates:
                trace.append(
                    _iteration(
                        len(trace) + 1,
                        candidate,
                        initial_time,
                        objective,
                        accepted=(
                            accepted
                            and np.array_equal(candidate, candidate_delta)
                        ),
                    )
                )
            if accepted:
                delta = candidate_delta
                best = candidate_result
            else:
                steps *= 0.5

        transform = apply_local_se3_delta(
            initial_transform_camera_lidar, delta[:3], delta[3:6]
        )
        estimated_time = initial_time + float(delta[6])
        final_holdout = evaluate_continuous_time_camera_lidar(
            holdout,
            transform,
            capture_time_policy,
            estimated_time,
            settings,
        )
        at_bound = bool(
            np.any(
                np.isclose(np.abs(delta), bounds, atol=1.0e-12) & active
            )
        )
        converged = bool(np.all((steps < minimum_steps) | ~active))
        status: ContinuousTimeStatus = (
            "at_bound"
            if at_bound
            else "converged"
            if converged
            else "max_evaluations"
        )
        reason = {
            "at_bound": "joint optimum reached a configured correction bound",
            "converged": "all joint parameter steps reached their minima",
            "max_evaluations": "maximum joint objective evaluations reached",
        }[status]
        return ContinuousTimeCameraLidarResult(
            status=status,
            reason=reason,
            initial_transform_camera_lidar=initial_transform_camera_lidar,
            transform_camera_lidar=transform,
            initial_time_offset_sec=initial_time,
            estimated_time_offset_sec=estimated_time,
            train_capture_ids=tuple(item.capture_id for item in train),
            holdout_capture_ids=tuple(item.capture_id for item in holdout),
            initial_train_evaluation=initial_train,
            final_train_evaluation=best,
            initial_holdout_evaluation=initial_holdout,
            final_holdout_evaluation=final_holdout,
            time_sensitivity_px_per_sec=sensitivity,
            time_observability_rank=int(
                sensitivity >= settings.minimum_time_sensitivity_px_per_sec
            ),
            trajectory_model=(
                "piecewise_se3"
                if any(item.body_trajectory is not None for item in captures)
                else "constant_body_twist"
            ),
            trace=tuple(trace),
            options=settings,
        )


def evaluate_continuous_time_camera_lidar(
    captures: Sequence[ContinuousTimeCameraLidarCapture],
    transform_camera_lidar: SE3,
    capture_time_policy: LidarCaptureTimePolicy,
    time_offset_sec: float,
    options: ContinuousTimeCameraLidarOptions | None = None,
) -> ContinuousTimeEvaluation:
    """Evaluate robust covariance-normalized reprojection loss."""

    settings = options or ContinuousTimeCameraLidarOptions()
    policy = LidarCaptureTimePolicy(
        point_offset_unit=capture_time_policy.point_offset_unit,
        stamp_reference=capture_time_policy.stamp_reference,
        sensor_time_offset_sec=time_offset_sec,
    )
    squared_pixel_errors: list[float] = []
    mahalanobis_errors: list[float] = []
    weights: list[float] = []
    skipped = 0
    for capture in captures:
        selected = [
            item
            for item in capture.correspondences
            if item.reliability * (1.0 - item.outlier_probability)
            >= settings.minimum_confidence
        ]
        for correspondence in selected:
            point = _deskew_correspondence(
                capture, correspondence, policy, settings
            )
            point_camera = transform_camera_lidar.transform_point(
                point.position_lidar_at_reference_m
            )
            if point_camera[2] <= 1.0e-9:
                skipped += 1
                continue
            u = capture.camera.fx * point_camera[0] / point_camera[2] + capture.camera.cx
            v = capture.camera.fy * point_camera[1] / point_camera[2] + capture.camera.cy
            if not (
                0.0 <= u < capture.camera.width
                and 0.0 <= v < capture.camera.height
            ):
                skipped += 1
                continue
            residual_u = u - correspondence.image_mean_px[0]
            residual_v = v - correspondence.image_mean_px[1]
            if settings.use_covariance:
                a, b, c, d = correspondence.image_covariance_px2
                determinant = a * d - b * c
                mahalanobis_squared = (
                    d * residual_u * residual_u
                    - (b + c) * residual_u * residual_v
                    + a * residual_v * residual_v
                ) / determinant
            else:
                mahalanobis_squared = (
                    residual_u * residual_u + residual_v * residual_v
                )
            confidence = correspondence.reliability * (
                1.0 - correspondence.outlier_probability
            )
            squared_pixel_errors.append(
                residual_u * residual_u + residual_v * residual_v
            )
            mahalanobis_errors.append(math.sqrt(max(0.0, mahalanobis_squared)))
            weights.append(confidence)
    if not weights:
        return ContinuousTimeEvaluation(
            objective=math.inf,
            weighted_reprojection_rmse_px=None,
            mean_mahalanobis_error=None,
            valid_correspondence_count=0,
            skipped_correspondence_count=skipped,
        )
    weight_array: FloatArray = np.asarray(weights, dtype=float)
    mahalanobis_array: FloatArray = np.asarray(
        mahalanobis_errors, dtype=float
    )
    pixel_error_array: FloatArray = np.asarray(
        squared_pixel_errors, dtype=float
    )
    robust = np.log1p(
        (mahalanobis_array / settings.cauchy_scale) ** 2
    )
    return ContinuousTimeEvaluation(
        objective=float(np.sum(weight_array * robust) / np.sum(weight_array)),
        weighted_reprojection_rmse_px=math.sqrt(
            float(np.sum(weight_array * pixel_error_array))
            / float(np.sum(weight_array))
        ),
        mean_mahalanobis_error=float(
            np.sum(weight_array * mahalanobis_array) / np.sum(weight_array)
        ),
        valid_correspondence_count=len(weights),
        skipped_correspondence_count=skipped,
    )


def _time_sensitivity(
    captures: Sequence[ContinuousTimeCameraLidarCapture],
    transform: SE3,
    policy: LidarCaptureTimePolicy,
    time_offset: float,
    settings: ContinuousTimeCameraLidarOptions,
) -> float:
    step = settings.time_sensitivity_probe_sec
    left = _projected_pixels(captures, transform, policy, time_offset - step, settings)
    right = _projected_pixels(captures, transform, policy, time_offset + step, settings)
    common = sorted(set(left).intersection(right))
    if not common:
        return 0.0
    squared = [
        (right[key][0] - left[key][0]) ** 2
        + (right[key][1] - left[key][1]) ** 2
        for key in common
    ]
    return math.sqrt(float(np.mean(squared))) / (2.0 * step)


def _projected_pixels(
    captures: Sequence[ContinuousTimeCameraLidarCapture],
    transform: SE3,
    policy: LidarCaptureTimePolicy,
    time_offset: float,
    settings: ContinuousTimeCameraLidarOptions,
) -> dict[tuple[str, str], tuple[float, float]]:
    resolved_policy = LidarCaptureTimePolicy(
        policy.point_offset_unit,
        policy.stamp_reference,
        time_offset,
    )
    output: dict[tuple[str, str], tuple[float, float]] = {}
    for capture in captures:
        selected = [
            item
            for item in capture.correspondences
            if item.reliability * (1.0 - item.outlier_probability)
            >= settings.minimum_confidence
        ]
        for correspondence in selected:
            point = _deskew_correspondence(
                capture, correspondence, resolved_policy, settings
            )
            point_camera = transform.transform_point(
                point.position_lidar_at_reference_m
            )
            if point_camera[2] <= 1.0e-9:
                continue
            output[(capture.capture_id, point.point_id)] = (
                capture.camera.fx * point_camera[0] / point_camera[2]
                + capture.camera.cx,
                capture.camera.fy * point_camera[1] / point_camera[2]
                + capture.camera.cy,
            )
    return output


def _deskew_correspondence(
    capture: ContinuousTimeCameraLidarCapture,
    correspondence: TimedProbabilisticImageCorrespondence,
    policy: LidarCaptureTimePolicy,
    settings: ContinuousTimeCameraLidarOptions,
) -> DeskewedLidarPoint:
    point_offset = (
        correspondence.capture_offset_sec
        if settings.use_per_point_time
        else 0.0
    )
    reference_time = _camera_row_exposure_time(
        capture, correspondence.image_mean_px[1], settings
    )
    if capture.body_trajectory is not None:
        capture_time = policy.point_reference_time_sec(
            capture.lidar_message_stamp_sec,
            point_offset,
        )
        position_body_capture = capture.transform_body_lidar.transform_point(
            correspondence.point_lidar_m
        )
        position_body_reference = (
            capture.body_trajectory.transform_static_point_between_body_frames(
                position_body_capture,
                source_time_sec=capture_time,
                target_time_sec=reference_time,
            )
        )
        position_lidar_reference = (
            capture.transform_body_lidar.inverse().transform_point(
                position_body_reference
            )
        )
        return DeskewedLidarPoint(
            point_id=correspondence.correspondence_id,
            position_lidar_at_reference_m=position_lidar_reference,
            capture_time_reference_sec=capture_time,
            deskew_delta_sec=reference_time - capture_time,
        )
    result = deskew_lidar_points_to_reference(
        [
            TimedLidarPoint(
                correspondence.correspondence_id,
                correspondence.point_lidar_m,
                point_offset,
            )
        ],
        message_stamp_sec=capture.lidar_message_stamp_sec,
        reference_time_sec=reference_time,
        policy=policy,
        twist=capture.twist,
        transform_body_lidar=capture.transform_body_lidar,
    )
    return result.points[0]


def _camera_row_exposure_time(
    capture: ContinuousTimeCameraLidarCapture,
    image_row_px: float,
    settings: ContinuousTimeCameraLidarOptions,
) -> float:
    if (
        not settings.use_rolling_shutter
        or capture.rolling_shutter_readout_sec == 0.0
    ):
        return capture.camera_exposure_time_sec
    fraction = min(
        1.0,
        max(0.0, image_row_px / float(capture.camera.height - 1)),
    )
    if capture.rolling_shutter_direction == "bottom_to_top":
        fraction = 1.0 - fraction
    return (
        capture.camera_exposure_time_sec
        + fraction * capture.rolling_shutter_readout_sec
    )


def _validate_trajectory_temporal_support(
    captures: Sequence[ContinuousTimeCameraLidarCapture],
    policy: LidarCaptureTimePolicy,
    settings: ContinuousTimeCameraLidarOptions,
) -> None:
    clock_margin = (
        settings.time_offset_bound_sec if settings.estimate_time_offset else 0.0
    )
    for capture in captures:
        trajectory = capture.body_trajectory
        if trajectory is None:
            continue
        required_times = [
            _camera_row_exposure_time(
                capture,
                correspondence.image_mean_px[1],
                settings,
            )
            for correspondence in capture.correspondences
        ]
        for correspondence in capture.correspondences:
            point_offset = (
                correspondence.capture_offset_sec
                if settings.use_per_point_time
                else 0.0
            )
            nominal_time = policy.point_reference_time_sec(
                capture.lidar_message_stamp_sec,
                point_offset,
            )
            required_times.extend(
                (nominal_time - clock_margin, nominal_time + clock_margin)
            )
        if (
            required_times
            and (
                min(required_times) < trajectory.minimum_time_sec
                or max(required_times) > trajectory.maximum_time_sec
            )
        ):
            raise ValueError(
                f"capture {capture.capture_id!r} plus the configured clock "
                "search exceeds the supplied trajectory support "
                f"[{trajectory.minimum_time_sec:.9g}, "
                f"{trajectory.maximum_time_sec:.9g}] seconds"
            )


def _iteration(
    evaluation: int,
    delta: FloatArray,
    initial_time: float,
    objective: float,
    *,
    accepted: bool,
) -> ContinuousTimeIteration:
    return ContinuousTimeIteration(
        evaluation=evaluation,
        delta_rotation_deg_xyz=(
            float(delta[0]),
            float(delta[1]),
            float(delta[2]),
        ),
        delta_translation_m_xyz=(
            float(delta[3]),
            float(delta[4]),
            float(delta[5]),
        ),
        time_offset_sec=initial_time + float(delta[6]),
        objective=objective,
        accepted=accepted,
    )


def _terminal_result(
    status: ContinuousTimeStatus,
    reason: str,
    transform: SE3,
    time_offset: float,
    train: Sequence[ContinuousTimeCameraLidarCapture],
    holdout: Sequence[ContinuousTimeCameraLidarCapture],
    train_evaluation: ContinuousTimeEvaluation,
    holdout_evaluation: ContinuousTimeEvaluation,
    sensitivity: float,
    options: ContinuousTimeCameraLidarOptions,
) -> ContinuousTimeCameraLidarResult:
    return ContinuousTimeCameraLidarResult(
        status=status,
        reason=reason,
        initial_transform_camera_lidar=transform,
        transform_camera_lidar=transform,
        initial_time_offset_sec=time_offset,
        estimated_time_offset_sec=time_offset,
        train_capture_ids=tuple(item.capture_id for item in train),
        holdout_capture_ids=tuple(item.capture_id for item in holdout),
        initial_train_evaluation=train_evaluation,
        final_train_evaluation=train_evaluation,
        initial_holdout_evaluation=holdout_evaluation,
        final_holdout_evaluation=holdout_evaluation,
        time_sensitivity_px_per_sec=sensitivity,
        time_observability_rank=0,
        trajectory_model=(
            "piecewise_se3"
            if any(item.body_trajectory is not None for item in (*train, *holdout))
            else "constant_body_twist"
        ),
        trace=(),
        options=options,
    )
