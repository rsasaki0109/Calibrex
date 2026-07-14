"""Online camera--LiDAR calibration monitoring and tracking.

Independent implementation of Levinson and Thrun, RSS 2013.  The core is
ROS-independent and consumes preprocessed image-edge response fields plus
near-side LiDAR depth discontinuities.  Calibrex adds a shadow holdout stream
that is never used to choose tracking updates.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.evaluation.holdout import split_indices
from calibrex.evaluation.numerical_curvature import (
    NumericalCurvatureEvaluation,
    evaluate_numerical_curvature,
)

FloatArray: TypeAlias = NDArray[np.float64]
LevinsonStatus = Literal["tracked", "monitored", "insufficient_observations"]
CameraProjectionKind = Literal["pinhole", "opencv_fisheye"]
CameraAxes = Literal["optical_z", "forward_x_left_y_up_z"]
DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class LevinsonCameraModel:
    """Calibrated camera projection used by the online edge objective."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    projection: CameraProjectionKind = "pinhole"
    distortion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    axes: CameraAxes = "optical_z"

    def __post_init__(self) -> None:
        if self.width <= 1 or self.height <= 1:
            raise ValueError("camera dimensions must exceed one pixel")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class LevinsonOnlineFrame:
    """One synchronized edge-response image and LiDAR discontinuity cloud."""

    frame_id: str
    edge_response: FloatArray
    lidar_points: FloatArray
    discontinuity_weights: FloatArray
    camera: LevinsonCameraModel

    def __post_init__(self) -> None:
        response = np.asarray(self.edge_response, dtype=float)
        points = np.asarray(self.lidar_points, dtype=float)
        weights = np.asarray(self.discontinuity_weights, dtype=float)
        if response.shape != (self.camera.height, self.camera.width):
            raise ValueError("edge_response shape must match camera model")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("lidar_points must have shape Nx3")
        if weights.shape != (points.shape[0],):
            raise ValueError("discontinuity_weights must have shape N")
        if not np.all(np.isfinite(response)) or np.any(response < 0.0):
            raise ValueError("edge_response must be finite and non-negative")
        if not np.all(np.isfinite(points)):
            raise ValueError("lidar_points must be finite")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("discontinuity_weights must be finite and non-negative")
        object.__setattr__(self, "edge_response", response)
        object.__setattr__(self, "lidar_points", points)
        object.__setattr__(self, "discontinuity_weights", weights)


@dataclass(frozen=True)
class LevinsonThrunOnlineOptions:
    """Window, grid, statistical-monitor, and evidence options."""

    window_size: int = 9
    min_train_frames: int = 9
    min_holdout_frames: int = 3
    holdout_ratio: float = 0.2
    split_seed: int = 0
    tracking_enabled: bool = True
    translation_grid_step_m: float = 0.01
    rotation_grid_step_deg: float = 0.05
    correct_fraction_mean: float = 0.997
    correct_fraction_sigma: float = 0.014
    incorrect_fraction_mean: float = 0.505
    incorrect_fraction_sigma: float = 0.14
    calibrated_probability_threshold: float = 0.5
    min_projected_points: int = 16
    known_bad_translation_m: float = 0.1
    known_bad_rotation_deg: float = 0.25
    known_bad_score_margin: float = 1.0e-6
    curvature_translation_step_m: float = 0.01
    curvature_rotation_step_deg: float = 0.1
    rank_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.min_train_frames <= 0:
            raise ValueError("window and minimum train frame counts must be positive")
        if self.min_holdout_frames < 0:
            raise ValueError("minimum holdout frame count must be non-negative")
        if self.translation_grid_step_m <= 0.0 or self.rotation_grid_step_deg <= 0.0:
            raise ValueError("grid steps must be positive")
        if self.correct_fraction_sigma <= 0.0 or self.incorrect_fraction_sigma <= 0.0:
            raise ValueError("monitor distribution sigmas must be positive")


@dataclass(frozen=True)
class LevinsonObjectiveEvaluation:
    """Raw Eq.3 score and support-normalized companion score."""

    raw_objective: float
    normalized_objective: float
    projected_point_count: int
    total_discontinuity_weight: float
    frame_count: int


@dataclass(frozen=True)
class LevinsonOnlineStep:
    """One rolling train-window monitor and optional tracking decision."""

    frame_id: str
    train_window_frame_ids: tuple[str, ...]
    center_objective: float
    best_objective: float
    worsening_fraction: float
    calibrated_probability: float
    classified_calibrated: bool
    update_applied: bool
    selected_grid_offset: tuple[int, int, int, int, int, int]
    transform_camera_lidar: SE3
    projected_point_count: int


@dataclass(frozen=True)
class LevinsonKnownBadProbe:
    """Held-out response to one signed transform error."""

    dof: Literal["x", "y", "z", "roll", "pitch", "yaw"]
    amount: float
    unit: Literal["m", "deg"]
    normalized_objective: float
    score_drop: float
    detectable: bool


@dataclass(frozen=True)
class LevinsonThrunOnlineResult:
    """Online monitor/tracker result with leakage-free final evidence."""

    status: LevinsonStatus
    reason: str
    initial_transform_camera_lidar: SE3
    final_transform_camera_lidar: SE3
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    timeline: tuple[LevinsonOnlineStep, ...]
    train_evaluation: LevinsonObjectiveEvaluation
    holdout_evaluation: LevinsonObjectiveEvaluation
    probes: tuple[LevinsonKnownBadProbe, ...]
    curvature: NumericalCurvatureEvaluation | None

    def as_dict(self) -> dict[str, object]:
        """Return a schema-safe result with primary-paper provenance."""

        return {
            "status": self.status,
            "reason": self.reason,
            "initial_transform_camera_lidar": self.initial_transform_camera_lidar.as_dict(),
            "final_transform_camera_lidar": self.final_transform_camera_lidar.as_dict(),
            "train_frame_ids": list(self.train_frame_ids),
            "holdout_frame_ids": list(self.holdout_frame_ids),
            "timeline": [
                {
                    **item.__dict__,
                    "transform_camera_lidar": item.transform_camera_lidar.as_dict(),
                }
                for item in self.timeline
            ],
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "known_bad_probes": [item.__dict__ for item in self.probes],
            "curvature": self.curvature.as_dict() if self.curvature is not None else None,
            "method": "levinson_thrun_online_edge_grid/v0.1",
            "paper_doi": "10.15607/RSS.2013.IX.029",
            "paper_url": "https://roboticsproceedings.org/rss09/p29.pdf",
            "frame_convention": "p_camera = R_camera_lidar p_lidar + t_camera_lidar",
            "paper_equations": {
                "image_edge_spilloff": 1,
                "near_depth_discontinuity": 2,
                "rolling_objective": 3,
                "calibrated_probability": 4,
            },
            "grid_contract": "radius-one Cartesian grid: 3^6 = 729 candidates",
            "holdout_specialization": (
                "seeded frame-level shadow holdout never participates in tracking updates"
            ),
            "curvature_claim": "negative-objective numerical Hessian; not covariance",
        }


class LevinsonThrunOnlineSolver:
    """Monitor and greedily track local camera--LiDAR calibration drift."""

    def solve(
        self,
        frames: Sequence[LevinsonOnlineFrame],
        initial_transform: SE3,
        options: LevinsonThrunOnlineOptions | None = None,
    ) -> LevinsonThrunOnlineResult:
        solver_options = options or LevinsonThrunOnlineOptions()
        ordered = list(frames)
        train_indices, holdout_indices = split_indices(
            len(ordered), solver_options.holdout_ratio, seed=solver_options.split_seed
        )
        train_index_set = set(train_indices)
        train = [ordered[index] for index in train_indices]
        holdout = [ordered[index] for index in holdout_indices]
        if (
            len(train) < solver_options.min_train_frames
            or len(holdout) < solver_options.min_holdout_frames
        ):
            return _insufficient_result(initial_transform, train, holdout)

        transform = initial_transform
        train_window: deque[LevinsonOnlineFrame] = deque(maxlen=solver_options.window_size)
        timeline: list[LevinsonOnlineStep] = []
        for index, frame in enumerate(ordered):
            if index not in train_index_set:
                continue
            train_window.append(frame)
            if len(train_window) < solver_options.window_size:
                continue
            step, transform = _evaluate_online_step(
                list(train_window), transform, solver_options
            )
            timeline.append(step)

        train_evaluation = evaluate_levinson_objective(train, transform)
        holdout_evaluation = evaluate_levinson_objective(holdout, transform)
        probes = _known_bad_probes(holdout, transform, solver_options)
        curvature = evaluate_numerical_curvature(
            lambda values: -evaluate_levinson_objective(
                holdout, _transform_from_parameters(values)
            ).normalized_objective,
            _parameters_from_transform(transform),
            (
                solver_options.curvature_translation_step_m,
                solver_options.curvature_translation_step_m,
                solver_options.curvature_translation_step_m,
                math.radians(solver_options.curvature_rotation_step_deg),
                math.radians(solver_options.curvature_rotation_step_deg),
                math.radians(solver_options.curvature_rotation_step_deg),
            ),
            rank_tolerance=solver_options.rank_tolerance,
        )
        return LevinsonThrunOnlineResult(
            status="tracked" if solver_options.tracking_enabled else "monitored",
            reason=(
                "rolling radius-one grid tracking completed"
                if solver_options.tracking_enabled
                else "rolling local-optimum monitoring completed without transform updates"
            ),
            initial_transform_camera_lidar=initial_transform,
            final_transform_camera_lidar=transform,
            train_frame_ids=tuple(item.frame_id for item in train),
            holdout_frame_ids=tuple(item.frame_id for item in holdout),
            timeline=tuple(timeline),
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            probes=probes,
            curvature=curvature,
        )


def levinson_edge_response(
    luminance: FloatArray,
    *,
    alpha: float = 1.0 / 3.0,
    gamma: float = 0.98,
    max_radius: int = 64,
) -> FloatArray:
    """Compute the Eq.1 edge image and bounded inverse-distance spilloff.

    The paper uses an unbounded transform.  ``max_radius`` is an explicit
    finite-image numerical budget; at the default radius the omitted multiplier
    is at most ``gamma**65`` relative to a source edge.
    """

    image = np.asarray(luminance, dtype=float)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError("luminance must be a finite 2D array")
    if not 0.0 <= alpha <= 1.0 or not 0.0 < gamma < 1.0 or max_radius < 0:
        raise ValueError("edge response parameters are invalid")
    edge = _maximum_neighbor_difference(image)
    spread = edge.copy()
    frontier = edge.copy()
    for _ in range(max_radius):
        frontier = gamma * _maximum_3x3(frontier)
        spread = np.maximum(spread, frontier)
    return alpha * edge + (1.0 - alpha) * spread


def extract_near_depth_discontinuities(
    organized_points: FloatArray,
    *,
    minimum_jump_m: float = 0.3,
    exponent: float = 0.5,
) -> tuple[FloatArray, FloatArray]:
    """Apply Eq.2 independently to every ordered LiDAR beam.

    ``organized_points`` has shape ``(beam, sample, 3)``.  Endpoints are
    excluded because both neighbors are required.
    """

    points = np.asarray(organized_points, dtype=float)
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 3:
        raise ValueError("organized_points must have shape BxNx3 with N >= 3")
    if minimum_jump_m < 0.0 or exponent <= 0.0:
        raise ValueError("depth discontinuity parameters are invalid")
    ranges = np.linalg.norm(points, axis=2)
    center = ranges[:, 1:-1]
    jumps = np.maximum.reduce(
        (ranges[:, :-2] - center, ranges[:, 2:] - center, np.zeros_like(center))
    )
    selected = jumps >= minimum_jump_m
    return points[:, 1:-1, :][selected], np.power(jumps[selected], exponent)


def evaluate_levinson_objective(
    frames: Sequence[LevinsonOnlineFrame], transform_camera_lidar: SE3
) -> LevinsonObjectiveEvaluation:
    """Evaluate Eq.3 and a support-normalized companion score."""

    raw = 0.0
    total_weight = 0.0
    projected_count = 0
    for frame in frames:
        response, weights = _project_response(frame, transform_camera_lidar)
        raw += float(response @ weights)
        total_weight += float(np.sum(weights))
        projected_count += int(weights.size)
    normalized = raw / total_weight if total_weight > 0.0 else 0.0
    return LevinsonObjectiveEvaluation(
        raw_objective=raw,
        normalized_objective=normalized,
        projected_point_count=projected_count,
        total_discontinuity_weight=total_weight,
        frame_count=len(frames),
    )


def calibrated_probability(
    worsening_fraction: float, options: LevinsonThrunOnlineOptions | None = None
) -> float:
    """Evaluate the paper's Eq.4 statistical calibration test."""

    solver_options = options or LevinsonThrunOnlineOptions()
    x = min(1.0, max(0.0, float(worsening_fraction)))
    correct = math.exp(
        -0.5
        * ((x - solver_options.correct_fraction_mean) / solver_options.correct_fraction_sigma)
        ** 2
    )
    incorrect = math.exp(
        -0.5
        * (
            (x - solver_options.incorrect_fraction_mean)
            / solver_options.incorrect_fraction_sigma
        )
        ** 2
    )
    denominator = correct + incorrect
    return correct / denominator if denominator > 0.0 else 0.0


def _evaluate_online_step(
    window: Sequence[LevinsonOnlineFrame],
    transform: SE3,
    options: LevinsonThrunOnlineOptions,
) -> tuple[LevinsonOnlineStep, SE3]:
    center = np.asarray(_parameters_from_transform(transform), dtype=float)
    translation_step = options.translation_grid_step_m
    rotation_step = math.radians(options.rotation_grid_step_deg)
    steps = np.asarray(
        (
            translation_step,
            translation_step,
            translation_step,
            rotation_step,
            rotation_step,
            rotation_step,
        )
    )
    center_evaluation = evaluate_levinson_objective(window, transform)
    candidates: list[tuple[float, tuple[int, int, int, int, int, int], SE3]] = []
    worse_count = 0
    for raw_offset in itertools.product((-1, 0, 1), repeat=6):
        offset = (
            int(raw_offset[0]),
            int(raw_offset[1]),
            int(raw_offset[2]),
            int(raw_offset[3]),
            int(raw_offset[4]),
            int(raw_offset[5]),
        )
        if offset == (0, 0, 0, 0, 0, 0):
            score = center_evaluation.raw_objective
            candidate_transform = transform
        else:
            candidate_transform = _transform_from_parameters(center + np.asarray(offset) * steps)
            score = evaluate_levinson_objective(window, candidate_transform).raw_objective
            worse_count += int(score < center_evaluation.raw_objective)
        candidates.append((score, offset, candidate_transform))
    best_score, best_offset, best_transform = max(candidates, key=lambda item: item[0])
    fraction = worse_count / 728.0
    probability = calibrated_probability(fraction, options)
    has_support = center_evaluation.projected_point_count >= options.min_projected_points
    apply_update = (
        options.tracking_enabled
        and has_support
        and best_score > center_evaluation.raw_objective
        and best_offset != (0, 0, 0, 0, 0, 0)
    )
    output_transform = best_transform if apply_update else transform
    return (
        LevinsonOnlineStep(
            frame_id=window[-1].frame_id,
            train_window_frame_ids=tuple(item.frame_id for item in window),
            center_objective=center_evaluation.raw_objective,
            best_objective=best_score,
            worsening_fraction=fraction,
            calibrated_probability=probability,
            classified_calibrated=(
                has_support and probability >= options.calibrated_probability_threshold
            ),
            update_applied=apply_update,
            selected_grid_offset=best_offset,
            transform_camera_lidar=output_transform,
            projected_point_count=center_evaluation.projected_point_count,
        ),
        output_transform,
    )


def _known_bad_probes(
    holdout: Sequence[LevinsonOnlineFrame],
    transform: SE3,
    options: LevinsonThrunOnlineOptions,
) -> tuple[LevinsonKnownBadProbe, ...]:
    center = np.asarray(_parameters_from_transform(transform), dtype=float)
    baseline = evaluate_levinson_objective(holdout, transform).normalized_objective
    probes: list[LevinsonKnownBadProbe] = []
    for index, dof in enumerate(DOF_NAMES):
        amount = (
            options.known_bad_translation_m
            if index < 3
            else math.radians(options.known_bad_rotation_deg)
        )
        for sign in (-1.0, 1.0):
            candidate = center.copy()
            candidate[index] += sign * amount
            score = evaluate_levinson_objective(
                holdout, _transform_from_parameters(candidate)
            ).normalized_objective
            drop = baseline - score
            probes.append(
                LevinsonKnownBadProbe(
                    dof=dof,  # type: ignore[arg-type]
                    amount=sign * (amount if index < 3 else options.known_bad_rotation_deg),
                    unit="m" if index < 3 else "deg",
                    normalized_objective=score,
                    score_drop=drop,
                    detectable=drop > options.known_bad_score_margin,
                )
            )
    return tuple(probes)


def _project_response(
    frame: LevinsonOnlineFrame, transform_camera_lidar: SE3
) -> tuple[FloatArray, FloatArray]:
    rotation = _rotation_matrix(transform_camera_lidar)
    translated = (
        frame.lidar_points @ rotation.T
        + np.asarray(transform_camera_lidar.translation_m, dtype=float)
    )
    if frame.camera.axes == "forward_x_left_y_up_z":
        points = np.column_stack((-translated[:, 1], -translated[:, 2], translated[:, 0]))
    else:
        points = translated
    positive = points[:, 2] > 1.0e-6
    points = points[positive]
    weights = frame.discontinuity_weights[positive]
    if not points.size:
        return np.empty(0), np.empty(0)
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    if frame.camera.projection == "opencv_fisheye":
        radius = np.sqrt(x * x + y * y)
        theta = np.arctan(radius)
        k1, k2, k3, k4 = frame.camera.distortion
        theta2 = theta * theta
        distorted = theta * (
            1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
        )
        scale = np.divide(distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
        x *= scale
        y *= scale
    u = frame.camera.fx * x + frame.camera.cx
    v = frame.camera.fy * y + frame.camera.cy
    inside = (
        (u >= 0.0)
        & (v >= 0.0)
        & (u < frame.camera.width - 1)
        & (v < frame.camera.height - 1)
    )
    u, v, weights = u[inside], v[inside], weights[inside]
    if not u.size:
        return np.empty(0), np.empty(0)
    left = np.floor(u).astype(int)
    top = np.floor(v).astype(int)
    du, dv = u - left, v - top
    image = frame.edge_response
    response = (
        (1.0 - du) * (1.0 - dv) * image[top, left]
        + du * (1.0 - dv) * image[top, left + 1]
        + (1.0 - du) * dv * image[top + 1, left]
        + du * dv * image[top + 1, left + 1]
    )
    return response, weights


def _maximum_neighbor_difference(image: FloatArray) -> FloatArray:
    padded = np.pad(image, 1, mode="edge")
    result = np.zeros_like(image)
    height, width = image.shape
    for row_shift in range(3):
        for column_shift in range(3):
            if row_shift == 1 and column_shift == 1:
                continue
            neighbor = padded[
                row_shift : row_shift + height, column_shift : column_shift + width
            ]
            result = np.maximum(result, np.abs(image - neighbor))
    return result


def _maximum_3x3(image: FloatArray) -> FloatArray:
    padded = np.pad(image, 1, mode="edge")
    height, width = image.shape
    return np.maximum.reduce(
        [
            padded[row : row + height, column : column + width]
            for row in range(3)
            for column in range(3)
        ]
    )


def _transform_from_parameters(parameters: Sequence[float]) -> SE3:
    x, y, z, roll, pitch, yaw = (float(value) for value in parameters)
    rotation = _euler_xyz_matrix(roll, pitch, yaw)
    return SE3((x, y, z), quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)))


def _parameters_from_transform(transform: SE3) -> tuple[float, float, float, float, float, float]:
    rotation = _rotation_matrix(transform)
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return (*transform.translation_m, roll, pitch, yaw)


def _euler_xyz_matrix(roll: float, pitch: float, yaw: float) -> FloatArray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=float,
    )


def _rotation_matrix(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=float,
    )


def _insufficient_result(
    initial: SE3,
    train: Sequence[LevinsonOnlineFrame],
    holdout: Sequence[LevinsonOnlineFrame],
) -> LevinsonThrunOnlineResult:
    empty = LevinsonObjectiveEvaluation(0.0, 0.0, 0, 0.0, 0)
    return LevinsonThrunOnlineResult(
        status="insufficient_observations",
        reason="train or shadow-holdout frame count is below the declared minimum",
        initial_transform_camera_lidar=initial,
        final_transform_camera_lidar=initial,
        train_frame_ids=tuple(item.frame_id for item in train),
        holdout_frame_ids=tuple(item.frame_id for item in holdout),
        timeline=(),
        train_evaluation=empty,
        holdout_evaluation=empty,
        probes=(),
        curvature=None,
    )
