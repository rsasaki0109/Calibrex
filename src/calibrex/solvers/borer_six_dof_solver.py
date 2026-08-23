"""Bounded derivative-free SE(3) optimization for the Borer D2D objective."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthPairProjector,
    DepthToDepthEvaluation,
    DepthToDepthObservation,
    DepthToDepthOptions,
    evaluate_depth_to_depth_mi,
    project_depth_pairs,
    resolve_depth_to_depth_options,
)
from calibrex.solvers.borer_rotation_only_solver import apply_local_euler_delta

FloatArray: TypeAlias = NDArray[np.float64]
SixDofObjective = Literal["mutual_information", "normalized_mutual_information"]
SixDofStatus = Literal["converged", "max_evaluations", "insufficient_observations"]


@dataclass(frozen=True)
class BorerSixDofOptions:
    """Deterministic bounded pattern-search settings in local SE(3)."""

    rotation_bound_deg: float = 2.0
    translation_bound_m: float = 1.0
    initial_rotation_step_deg: float = 0.25
    initial_translation_step_m: float = 0.10
    minimum_rotation_step_deg: float = 0.01
    minimum_translation_step_m: float = 0.005
    max_evaluations: int = 800
    improvement_tolerance: float = 1.0e-9
    objective: SixDofObjective = "mutual_information"
    d2d: DepthToDepthOptions = field(default_factory=DepthToDepthOptions)
    projector: DepthPairProjector = field(
        default=project_depth_pairs,
        repr=False,
        compare=False,
    )
    projection_backend: str = "calibrex.numpy_depth_pair_projector/v0.2"

    def __post_init__(self) -> None:
        if self.rotation_bound_deg <= 0.0 or self.translation_bound_m <= 0.0:
            raise ValueError("SE(3) bounds must be positive")
        if not (
            0.0
            < self.minimum_rotation_step_deg
            <= self.initial_rotation_step_deg
            <= self.rotation_bound_deg
        ):
            raise ValueError("rotation step sizes and bound are inconsistent")
        if not (
            0.0
            < self.minimum_translation_step_m
            <= self.initial_translation_step_m
            <= self.translation_bound_m
        ):
            raise ValueError("translation step sizes and bound are inconsistent")
        if self.max_evaluations < 13:
            raise ValueError("max_evaluations must be at least 13")
        if self.improvement_tolerance < 0.0:
            raise ValueError("improvement_tolerance must be non-negative")
        if not self.projection_backend.strip():
            raise ValueError("projection_backend must be non-empty")


@dataclass(frozen=True)
class BorerSixDofEvaluation:
    """One evaluated local SE(3) candidate."""

    evaluation: int
    delta_rotation_deg_xyz: tuple[float, float, float]
    delta_translation_m_xyz: tuple[float, float, float]
    objective: float
    mutual_information: float
    normalized_mutual_information: float
    evaluated_frame_count: int
    accepted: bool


@dataclass(frozen=True)
class BorerSixDofResult:
    """Bounded six-DoF result with a complete deterministic trace."""

    status: SixDofStatus
    initial_transform_camera_lidar: SE3
    transform_camera_lidar: SE3
    best_delta_rotation_deg_xyz: tuple[float, float, float]
    best_delta_translation_m_xyz: tuple[float, float, float]
    initial_evaluation: DepthToDepthEvaluation
    final_evaluation: DepthToDepthEvaluation
    trace: tuple[BorerSixDofEvaluation, ...]
    evaluation_count: int
    final_rotation_step_deg: float
    final_translation_step_m: float
    stopping_reason: str
    projection_backend: str = "calibrex.numpy_depth_pair_projector/v0.2"
    method: str = "bounded_se3_pattern_search/v0.1"
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
            "best_delta_translation_m_xyz": list(
                self.best_delta_translation_m_xyz
            ),
            "initial_evaluation": self.initial_evaluation.as_dict(),
            "final_evaluation": self.final_evaluation.as_dict(),
            "trace": [
                {
                    "evaluation": item.evaluation,
                    "delta_rotation_deg_xyz": list(
                        item.delta_rotation_deg_xyz
                    ),
                    "delta_translation_m_xyz": list(
                        item.delta_translation_m_xyz
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
            "final_rotation_step_deg": self.final_rotation_step_deg,
            "final_translation_step_m": self.final_translation_step_m,
            "stopping_reason": self.stopping_reason,
            "projection_backend": self.projection_backend,
        }


class BorerSixDofSolver:
    """Maximize average per-frame D2D MI over bounded local SE(3)."""

    def solve(
        self,
        observations: tuple[DepthToDepthObservation, ...]
        | list[DepthToDepthObservation],
        initial_transform_camera_lidar: SE3,
        options: BorerSixDofOptions | None = None,
    ) -> BorerSixDofResult:
        """Run deterministic coordinate pattern search."""

        settings = options or BorerSixDofOptions()
        initial_evaluation = evaluate_depth_to_depth_mi(
            observations,
            initial_transform_camera_lidar,
            settings.d2d,
            projector=settings.projector,
        )
        if initial_evaluation.evaluated_frame_count == 0:
            return _insufficient_result(
                initial_transform_camera_lidar,
                initial_evaluation,
                settings,
            )
        settings = replace(
            settings,
            d2d=resolve_depth_to_depth_options(observations, settings.d2d),
        )
        best_delta: FloatArray = np.zeros(6, dtype=float)
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
        rotation_step = settings.initial_rotation_step_deg
        translation_step = settings.initial_translation_step_m
        while (
            (
                rotation_step >= settings.minimum_rotation_step_deg
                or translation_step >= settings.minimum_translation_step_m
            )
            and evaluation_count < settings.max_evaluations
        ):
            candidates = []
            for axis in range(6):
                step = rotation_step if axis < 3 else translation_step
                minimum = (
                    settings.minimum_rotation_step_deg
                    if axis < 3
                    else settings.minimum_translation_step_m
                )
                if step < minimum:
                    continue
                bound = (
                    settings.rotation_bound_deg
                    if axis < 3
                    else settings.translation_bound_m
                )
                for direction in (-1.0, 1.0):
                    delta = best_delta.copy()
                    delta[axis] = float(
                        np.clip(delta[axis] + direction * step, -bound, bound)
                    )
                    if np.array_equal(delta, best_delta):
                        continue
                    result = evaluate_depth_to_depth_mi(
                        observations,
                        apply_local_se3_delta(
                            initial_transform_camera_lidar,
                            delta[:3],
                            delta[3:],
                        ),
                        settings.d2d,
                        projector=settings.projector,
                    )
                    score = _score(result, settings.objective)
                    evaluation_count += 1
                    candidates.append(
                        (score, tuple(delta.tolist()), delta, result)
                    )
                    if evaluation_count >= settings.max_evaluations:
                        break
                if evaluation_count >= settings.max_evaluations:
                    break
            if not candidates:
                break
            candidate_score, _tie_break, candidate_delta, candidate_result = max(
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
                best_evaluation = candidate_result
                best_score = candidate_score
            else:
                rotation_step *= 0.5
                translation_step *= 0.5

        converged = (
            rotation_step < settings.minimum_rotation_step_deg
            and translation_step < settings.minimum_translation_step_m
        )
        return BorerSixDofResult(
            status="converged" if converged else "max_evaluations",
            initial_transform_camera_lidar=initial_transform_camera_lidar,
            transform_camera_lidar=apply_local_se3_delta(
                initial_transform_camera_lidar,
                best_delta[:3],
                best_delta[3:],
            ),
            best_delta_rotation_deg_xyz=tuple(best_delta[:3].tolist()),
            best_delta_translation_m_xyz=tuple(best_delta[3:].tolist()),
            initial_evaluation=initial_evaluation,
            final_evaluation=best_evaluation,
            trace=tuple(trace),
            evaluation_count=evaluation_count,
            final_rotation_step_deg=rotation_step,
            final_translation_step_m=translation_step,
            stopping_reason=(
                "minimum SE(3) steps reached"
                if converged
                else "maximum objective evaluations reached"
            ),
            projection_backend=settings.projection_backend,
        )


def apply_local_se3_delta(
    initial: SE3,
    delta_rotation_deg_xyz: FloatArray
    | tuple[float, float, float]
    | list[float],
    delta_translation_m_xyz: FloatArray
    | tuple[float, float, float]
    | list[float],
) -> SE3:
    """Apply local Euler rotation and camera-frame translation deltas."""

    translation: FloatArray = np.asarray(delta_translation_m_xyz, dtype=float)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("delta_translation_m_xyz must contain three finite values")
    rotated = apply_local_euler_delta(initial, delta_rotation_deg_xyz)
    return SE3(
        translation_m=(
            initial.translation_m[0] + float(translation[0]),
            initial.translation_m[1] + float(translation[1]),
            initial.translation_m[2] + float(translation[2]),
        ),
        rotation_quat_xyzw=rotated.rotation_quat_xyzw,
    )


def _trace_item(
    *,
    evaluation: int,
    delta: FloatArray,
    result: DepthToDepthEvaluation,
    score: float,
    accepted: bool,
) -> BorerSixDofEvaluation:
    return BorerSixDofEvaluation(
        evaluation=evaluation,
        delta_rotation_deg_xyz=tuple(delta[:3].tolist()),
        delta_translation_m_xyz=tuple(delta[3:].tolist()),
        objective=score,
        mutual_information=result.mutual_information,
        normalized_mutual_information=result.normalized_mutual_information,
        evaluated_frame_count=result.evaluated_frame_count,
        accepted=accepted,
    )


def _score(
    evaluation: DepthToDepthEvaluation,
    objective: SixDofObjective,
) -> float:
    if evaluation.evaluated_frame_count == 0:
        return -math.inf
    if objective == "normalized_mutual_information":
        return evaluation.normalized_mutual_information
    return evaluation.mutual_information


def _insufficient_result(
    initial: SE3,
    evaluation: DepthToDepthEvaluation,
    settings: BorerSixDofOptions,
) -> BorerSixDofResult:
    return BorerSixDofResult(
        status="insufficient_observations",
        initial_transform_camera_lidar=initial,
        transform_camera_lidar=initial,
        best_delta_rotation_deg_xyz=(0.0, 0.0, 0.0),
        best_delta_translation_m_xyz=(0.0, 0.0, 0.0),
        initial_evaluation=evaluation,
        final_evaluation=evaluation,
        trace=(),
        evaluation_count=1,
        final_rotation_step_deg=settings.initial_rotation_step_deg,
        final_translation_step_m=settings.initial_translation_step_m,
        stopping_reason="no frame met min_visible_points",
        projection_backend=settings.projection_backend,
    )
