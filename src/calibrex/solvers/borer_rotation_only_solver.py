"""Bounded derivative-free SO(3) optimization for the Borer D2D objective."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthEvaluation,
    DepthToDepthObservation,
    DepthToDepthOptions,
    evaluate_depth_to_depth_mi,
    resolve_depth_to_depth_options,
)

FloatArray: TypeAlias = NDArray[np.float64]
RotationObjective = Literal["mutual_information", "normalized_mutual_information"]
RotationOnlyStatus = Literal["converged", "max_evaluations", "insufficient_observations"]


@dataclass(frozen=True)
class BorerRotationOnlyOptions:
    """Deterministic bounded pattern-search settings in local XYZ Euler space."""

    bound_deg: float = 20.0
    initial_step_deg: float = 4.0
    minimum_step_deg: float = 0.05
    max_evaluations: int = 400
    improvement_tolerance: float = 1.0e-9
    objective: RotationObjective = "mutual_information"
    d2d: DepthToDepthOptions = field(default_factory=DepthToDepthOptions)

    def __post_init__(self) -> None:
        if self.bound_deg <= 0.0:
            raise ValueError("bound_deg must be positive")
        if not 0.0 < self.minimum_step_deg <= self.initial_step_deg:
            raise ValueError("rotation step sizes are inconsistent")
        if self.initial_step_deg > self.bound_deg:
            raise ValueError("initial_step_deg must not exceed bound_deg")
        if self.max_evaluations < 7:
            raise ValueError("max_evaluations must be at least 7")
        if self.improvement_tolerance < 0.0:
            raise ValueError("improvement_tolerance must be non-negative")


@dataclass(frozen=True)
class BorerRotationEvaluation:
    """One evaluated local rotation candidate."""

    evaluation: int
    delta_rotation_deg_xyz: tuple[float, float, float]
    objective: float
    mutual_information: float
    normalized_mutual_information: float
    evaluated_frame_count: int
    accepted: bool


@dataclass(frozen=True)
class BorerRotationOnlyResult:
    """Bounded rotation-only optimization result with a complete candidate trace."""

    status: RotationOnlyStatus
    initial_transform_camera_lidar: SE3
    transform_camera_lidar: SE3
    best_delta_rotation_deg_xyz: tuple[float, float, float]
    initial_evaluation: DepthToDepthEvaluation
    final_evaluation: DepthToDepthEvaluation
    trace: tuple[BorerRotationEvaluation, ...]
    evaluation_count: int
    final_step_deg: float
    stopping_reason: str
    method: str = "bounded_so3_pattern_search/v0.1"
    primary_source: str = "https://arxiv.org/abs/2311.01905"

    def as_dict(self) -> dict[str, object]:
        """Return a schema-safe candidate trace and stopping record."""

        return {
            "method": self.method,
            "primary_source": self.primary_source,
            "status": self.status,
            "initial_transform_camera_lidar": (
                self.initial_transform_camera_lidar.as_dict()
            ),
            "transform_camera_lidar": self.transform_camera_lidar.as_dict(),
            "best_delta_rotation_deg_xyz": list(
                self.best_delta_rotation_deg_xyz
            ),
            "initial_evaluation": self.initial_evaluation.as_dict(),
            "final_evaluation": self.final_evaluation.as_dict(),
            "trace": [
                {
                    "evaluation": item.evaluation,
                    "delta_rotation_deg_xyz": list(
                        item.delta_rotation_deg_xyz
                    ),
                    "objective": item.objective,
                    "mutual_information": item.mutual_information,
                    "normalized_mutual_information": (
                        item.normalized_mutual_information
                    ),
                    "evaluated_frame_count": item.evaluated_frame_count,
                    "accepted": item.accepted,
                }
                for item in self.trace
            ],
            "evaluation_count": self.evaluation_count,
            "final_step_deg": self.final_step_deg,
            "stopping_reason": self.stopping_reason,
        }


class BorerRotationOnlySolver:
    """Maximize average per-frame D2D MI while keeping translation fixed."""

    def solve(
        self,
        observations: tuple[DepthToDepthObservation, ...]
        | list[DepthToDepthObservation],
        initial_transform_camera_lidar: SE3,
        options: BorerRotationOnlyOptions | None = None,
    ) -> BorerRotationOnlyResult:
        """Run deterministic bounded local-coordinate pattern search."""

        settings = options or BorerRotationOnlyOptions()
        initial_evaluation = evaluate_depth_to_depth_mi(
            observations,
            initial_transform_camera_lidar,
            settings.d2d,
        )
        if initial_evaluation.evaluated_frame_count == 0:
            return BorerRotationOnlyResult(
                status="insufficient_observations",
                initial_transform_camera_lidar=initial_transform_camera_lidar,
                transform_camera_lidar=initial_transform_camera_lidar,
                best_delta_rotation_deg_xyz=(0.0, 0.0, 0.0),
                initial_evaluation=initial_evaluation,
                final_evaluation=initial_evaluation,
                trace=(),
                evaluation_count=1,
                final_step_deg=settings.initial_step_deg,
                stopping_reason="no frame met min_visible_points",
            )
        settings = replace(
            settings,
            d2d=resolve_depth_to_depth_options(observations, settings.d2d),
        )
        best_delta: FloatArray = np.zeros(3, dtype=float)
        best_evaluation = initial_evaluation
        best_score = _score(best_evaluation, settings.objective)
        trace = [
            _trace_item(
                evaluation=1,
                delta=best_delta,
                result=best_evaluation,
                score=best_score,
                accepted=True,
            )
        ]
        evaluation_count = 1
        step = settings.initial_step_deg
        bound = settings.bound_deg
        while (
            step >= settings.minimum_step_deg
            and evaluation_count < settings.max_evaluations
        ):
            candidates = []
            for axis in range(3):
                for direction in (-1.0, 1.0):
                    delta = best_delta.copy()
                    delta[axis] = float(
                        np.clip(delta[axis] + direction * step, -bound, bound)
                    )
                    if np.array_equal(delta, best_delta):
                        continue
                    transform = apply_local_euler_delta(
                        initial_transform_camera_lidar,
                        delta,
                    )
                    result = evaluate_depth_to_depth_mi(
                        observations,
                        transform,
                        settings.d2d,
                    )
                    score = _score(result, settings.objective)
                    evaluation_count += 1
                    candidates.append((score, tuple(delta.tolist()), delta, result))
                    if evaluation_count >= settings.max_evaluations:
                        break
                if evaluation_count >= settings.max_evaluations:
                    break
            if not candidates:
                break
            candidate_score, _tie_break, candidate_delta, candidate_evaluation = max(
                candidates,
                key=lambda item: (item[0], tuple(-value for value in item[1])),
            )
            accepted = (
                candidate_score > best_score + settings.improvement_tolerance
            )
            for score, _tie_break, delta, result in candidates:
                trace.append(
                    _trace_item(
                        evaluation=len(trace) + 1,
                        delta=delta,
                        result=result,
                        score=score,
                        accepted=accepted
                        and np.array_equal(delta, candidate_delta),
                    )
                )
            if accepted:
                best_delta = candidate_delta
                best_evaluation = candidate_evaluation
                best_score = candidate_score
            else:
                step *= 0.5
        converged = step < settings.minimum_step_deg
        status: RotationOnlyStatus = (
            "converged" if converged else "max_evaluations"
        )
        stopping_reason = (
            "minimum rotation step reached"
            if converged
            else "maximum objective evaluations reached"
        )
        return BorerRotationOnlyResult(
            status=status,
            initial_transform_camera_lidar=initial_transform_camera_lidar,
            transform_camera_lidar=apply_local_euler_delta(
                initial_transform_camera_lidar,
                best_delta,
            ),
            best_delta_rotation_deg_xyz=tuple(best_delta.tolist()),
            initial_evaluation=initial_evaluation,
            final_evaluation=best_evaluation,
            trace=tuple(trace),
            evaluation_count=evaluation_count,
            final_step_deg=step,
            stopping_reason=stopping_reason,
        )


def fibonacci_sphere_rotation_perturbations(
    *,
    count: int,
    magnitude_deg: float,
) -> tuple[tuple[float, float, float], ...]:
    """Generate the exact deterministic Fibonacci-sphere perturbation family."""

    if count < 1:
        raise ValueError("count must be positive")
    if magnitude_deg <= 0.0:
        raise ValueError("magnitude_deg must be positive")
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        azimuth = index * golden_angle
        points.append(
            (
                magnitude_deg * radius * math.cos(azimuth),
                magnitude_deg * radius * math.sin(azimuth),
                magnitude_deg * z,
            )
        )
    return tuple(points)


def _trace_item(
    *,
    evaluation: int,
    delta: FloatArray,
    result: DepthToDepthEvaluation,
    score: float,
    accepted: bool,
) -> BorerRotationEvaluation:
    return BorerRotationEvaluation(
        evaluation=evaluation,
        delta_rotation_deg_xyz=tuple(delta.tolist()),
        objective=score,
        mutual_information=result.mutual_information,
        normalized_mutual_information=result.normalized_mutual_information,
        evaluated_frame_count=result.evaluated_frame_count,
        accepted=accepted,
    )


def _score(
    evaluation: DepthToDepthEvaluation,
    objective: RotationObjective,
) -> float:
    if evaluation.evaluated_frame_count == 0:
        return -math.inf
    if objective == "normalized_mutual_information":
        return evaluation.normalized_mutual_information
    return evaluation.mutual_information


def apply_local_euler_delta(
    initial: SE3,
    delta_deg_xyz: FloatArray | tuple[float, float, float] | list[float],
) -> SE3:
    """Left-compose an XYZ Euler perturbation while preserving translation."""

    delta_array: FloatArray = np.asarray(delta_deg_xyz, dtype=float)
    if delta_array.shape != (3,) or not np.all(np.isfinite(delta_array)):
        raise ValueError("delta_deg_xyz must contain three finite values")
    delta = np.radians(delta_array)
    delta_rotation = _euler_xyz_matrix(float(delta[0]), float(delta[1]), float(delta[2]))
    initial_rotation = _quaternion_rotation_matrix(initial.rotation_quat_xyzw)
    rotation = delta_rotation @ initial_rotation
    return SE3(
        initial.translation_m,
        quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)),
    )


def _euler_xyz_matrix(roll: float, pitch: float, yaw: float) -> FloatArray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _quaternion_rotation_matrix(
    quaternion_xyzw: tuple[float, float, float, float],
) -> FloatArray:
    x, y, z, w = quaternion_xyzw
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )
