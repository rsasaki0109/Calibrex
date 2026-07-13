"""Radar lever-arm and clock-offset calibration from ego-velocity kinematics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import QuaternionXYZW, Vector3, rotate_vector_xyzw
from calibrex.evaluation.holdout import split_indices

RadarSpatiotemporalStatus = Literal[
    "converged", "insufficient_measurements", "degenerate_motion", "offset_at_boundary"
]


@dataclass(frozen=True)
class RadarVelocityMeasurement:
    """One timestamped scan-wise ego velocity expressed in the Radar frame."""

    measurement_id: str
    timestamp_sec: float
    velocity_radar_mps: Vector3
    weight: float = 1.0


@dataclass(frozen=True)
class ReferenceKinematicSample:
    """Reference-body linear and angular velocity at one reference timestamp."""

    timestamp_sec: float
    linear_velocity_body_mps: Vector3
    angular_velocity_body_radps: Vector3


@dataclass(frozen=True)
class RadarSpatiotemporalLeverArmOptions:
    max_abs_time_offset_sec: float = 0.1
    time_offset_step_sec: float = 0.001
    holdout_ratio: float = 0.2
    split_seed: int = 0
    min_train_measurements: int = 12
    rank_tolerance: float = 1.0e-8
    max_condition_number: float = 1.0e6
    known_bad_translation_m: float = 0.1
    known_bad_time_offset_sec: float = 0.02
    known_bad_margin_mps: float = 0.02


@dataclass(frozen=True)
class RadarSpatiotemporalProbe:
    parameter: Literal["x", "y", "z", "time"]
    amount: float
    unit: Literal["m", "sec"]
    holdout_rmse_mps: float | None
    delta_mps: float | None
    detectable: bool | None


@dataclass(frozen=True)
class RadarSpatiotemporalLeverArmResult:
    status: RadarSpatiotemporalStatus
    reason: str
    translation_body_radar_m: Vector3 | None
    time_offset_sec: float | None
    train_measurement_ids: tuple[str, ...]
    holdout_measurement_ids: tuple[str, ...]
    train_rmse_mps: float | None
    holdout_rmse_mps: float | None
    lever_arm_singular_values: tuple[float, float, float] | None
    lever_arm_rank: int
    lever_arm_condition_number: float | None
    time_objective_curvature_mps2_per_sec2: float | None
    probes: tuple[RadarSpatiotemporalProbe, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "translation_body_radar_m": (
                list(self.translation_body_radar_m)
                if self.translation_body_radar_m is not None
                else None
            ),
            "time_offset_sec": self.time_offset_sec,
            "train_measurement_ids": list(self.train_measurement_ids),
            "holdout_measurement_ids": list(self.holdout_measurement_ids),
            "train_rmse_mps": self.train_rmse_mps,
            "holdout_rmse_mps": self.holdout_rmse_mps,
            "lever_arm_singular_values": self.lever_arm_singular_values,
            "lever_arm_rank": self.lever_arm_rank,
            "lever_arm_condition_number": self.lever_arm_condition_number,
            "time_objective_curvature_mps2_per_sec2": (self.time_objective_curvature_mps2_per_sec2),
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "method": "profiled_velocity_lever_arm_and_clock_offset/v0.1",
            "measurement_model": (
                "R_body_radar v_radar(t) = v_body(t+dt) + omega_body(t+dt) cross t_body_radar"
            ),
            "time_convention": "radar_time + dt_radar = reference_time",
            "estimated_dofs": ["translation_x", "translation_y", "translation_z", "time"],
            "fixed_dofs": ["rotation_body_radar"],
        }


class RadarSpatiotemporalLeverArmSolver:
    """Profile a linear lever arm over a bounded Radar clock-offset grid."""

    def solve(
        self,
        measurements: Sequence[RadarVelocityMeasurement],
        reference_samples: Sequence[ReferenceKinematicSample],
        rotation_body_radar_xyzw: QuaternionXYZW,
        options: RadarSpatiotemporalLeverArmOptions | None = None,
    ) -> RadarSpatiotemporalLeverArmResult:
        solver_options = options or RadarSpatiotemporalLeverArmOptions()
        _validate_options(solver_options)
        reference = sorted(reference_samples, key=lambda item: item.timestamp_sec)
        eligible = sorted(
            (
                item
                for item in measurements
                if item.weight > 0.0 and _has_full_offset_support(item, reference, solver_options)
            ),
            key=lambda item: item.measurement_id,
        )
        train_indices, holdout_indices = split_indices(
            len(eligible), solver_options.holdout_ratio, solver_options.split_seed
        )
        train = [eligible[index] for index in train_indices]
        holdout = [eligible[index] for index in holdout_indices]
        train_ids = tuple(item.measurement_id for item in train)
        holdout_ids = tuple(item.measurement_id for item in holdout)
        if len(train) < solver_options.min_train_measurements or len(reference) < 2:
            return _empty("insufficient_measurements", train_ids, holdout_ids)

        offsets = _offset_grid(solver_options)
        candidates = [
            _profile_candidate(train, reference, rotation_body_radar_xyzw, offset)
            for offset in offsets
        ]
        valid = [(index, candidate) for index, candidate in enumerate(candidates) if candidate]
        if not valid:
            return _empty("degenerate_motion", train_ids, holdout_ids)
        best_index, best = min(valid, key=lambda item: item[1][1])
        translation_array, train_mse, singular_values = best
        rank = int(sum(value > solver_options.rank_tolerance for value in singular_values))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > solver_options.rank_tolerance
            else None
        )
        if rank < 3 or condition is None or condition > solver_options.max_condition_number:
            return _empty(
                "degenerate_motion",
                train_ids,
                holdout_ids,
                singular_values=_array3(singular_values),
                rank=rank,
                condition=condition,
            )
        offset = offsets[best_index]
        translation = _array3(translation_array)
        holdout_rmse = radar_spatiotemporal_rmse(
            holdout, reference, rotation_body_radar_xyzw, translation, offset
        )
        train_rmse = math.sqrt(train_mse)
        curvature = _time_curvature(candidates, best_index, solver_options.time_offset_step_sec)
        probes = _known_bad_probes(
            holdout,
            reference,
            rotation_body_radar_xyzw,
            translation,
            offset,
            holdout_rmse,
            solver_options,
        )
        at_boundary = best_index in {0, len(offsets) - 1}
        return RadarSpatiotemporalLeverArmResult(
            status="offset_at_boundary" if at_boundary else "converged",
            reason=(
                "best clock offset lies on the declared search boundary"
                if at_boundary
                else "profiled lever arm and clock offset converged"
            ),
            translation_body_radar_m=translation,
            time_offset_sec=offset,
            train_measurement_ids=train_ids,
            holdout_measurement_ids=holdout_ids,
            train_rmse_mps=train_rmse,
            holdout_rmse_mps=holdout_rmse,
            lever_arm_singular_values=_array3(singular_values),
            lever_arm_rank=rank,
            lever_arm_condition_number=condition,
            time_objective_curvature_mps2_per_sec2=curvature,
            probes=probes,
        )


def radar_spatiotemporal_rmse(
    measurements: Sequence[RadarVelocityMeasurement],
    reference_samples: Sequence[ReferenceKinematicSample],
    rotation_body_radar_xyzw: QuaternionXYZW,
    translation_body_radar_m: Vector3,
    time_offset_sec: float,
) -> float | None:
    """Evaluate a candidate using the unchanged timestamped measurements."""

    squared: list[float] = []
    for measurement in measurements:
        kinematic = _interpolate(reference_samples, measurement.timestamp_sec + time_offset_sec)
        if kinematic is None:
            continue
        observed = rotate_vector_xyzw(rotation_body_radar_xyzw, measurement.velocity_radar_mps)
        predicted = _add(
            kinematic.linear_velocity_body_mps,
            _cross(kinematic.angular_velocity_body_radps, translation_body_radar_m),
        )
        squared.append(measurement.weight * _squared_distance(observed, predicted))
    return math.sqrt(sum(squared) / len(squared)) if squared else None


ProfileCandidate = tuple[NDArray[np.float64], float, NDArray[np.float64]]


def _profile_candidate(
    measurements: Sequence[RadarVelocityMeasurement],
    reference: Sequence[ReferenceKinematicSample],
    rotation: QuaternionXYZW,
    offset: float,
) -> ProfileCandidate | None:
    rows: list[list[float]] = []
    values: list[float] = []
    for measurement in measurements:
        sample = _interpolate(reference, measurement.timestamp_sec + offset)
        if sample is None:
            return None
        observed = rotate_vector_xyzw(rotation, measurement.velocity_radar_mps)
        rhs = _subtract(observed, sample.linear_velocity_body_mps)
        root_weight = math.sqrt(measurement.weight)
        skew = _skew(sample.angular_velocity_body_radps)
        for row, value in zip(skew, rhs, strict=True):
            rows.append([root_weight * item for item in row])
            values.append(root_weight * value)
    matrix = np.asarray(rows, dtype=np.float64)
    vector = np.asarray(values, dtype=np.float64)
    translation, _residuals, _rank, singular_values = np.linalg.lstsq(matrix, vector, rcond=None)
    residual = matrix @ translation - vector
    mse = float(np.dot(residual, residual) / len(measurements))
    return translation, mse, singular_values


def _known_bad_probes(
    holdout: Sequence[RadarVelocityMeasurement],
    reference: Sequence[ReferenceKinematicSample],
    rotation: QuaternionXYZW,
    translation: Vector3,
    offset: float,
    baseline: float | None,
    options: RadarSpatiotemporalLeverArmOptions,
) -> tuple[RadarSpatiotemporalProbe, ...]:
    specs: list[tuple[Literal["x", "y", "z", "time"], float, Literal["m", "sec"]]] = []
    for parameter in ("x", "y", "z"):
        specs.extend(
            (parameter, sign * options.known_bad_translation_m, "m") for sign in (-1.0, 1.0)
        )
    specs.extend(("time", sign * options.known_bad_time_offset_sec, "sec") for sign in (-1.0, 1.0))
    probes: list[RadarSpatiotemporalProbe] = []
    for parameter, amount, unit in specs:
        candidate_translation = list(translation)
        candidate_offset = offset
        if parameter == "time":
            candidate_offset += amount
        else:
            candidate_translation[("x", "y", "z").index(parameter)] += amount
        rmse = radar_spatiotemporal_rmse(
            holdout,
            reference,
            rotation,
            (candidate_translation[0], candidate_translation[1], candidate_translation[2]),
            candidate_offset,
        )
        delta = rmse - baseline if rmse is not None and baseline is not None else None
        probes.append(
            RadarSpatiotemporalProbe(
                parameter,
                amount,
                unit,
                rmse,
                delta,
                delta > options.known_bad_margin_mps if delta is not None else None,
            )
        )
    return tuple(probes)


def _interpolate(
    samples: Sequence[ReferenceKinematicSample], timestamp: float
) -> ReferenceKinematicSample | None:
    if not samples or timestamp < samples[0].timestamp_sec or timestamp > samples[-1].timestamp_sec:
        return None
    for left, right in pairwise(samples):
        if left.timestamp_sec <= timestamp <= right.timestamp_sec:
            duration = right.timestamp_sec - left.timestamp_sec
            ratio = 0.0 if duration == 0.0 else (timestamp - left.timestamp_sec) / duration
            return ReferenceKinematicSample(
                timestamp,
                _lerp(left.linear_velocity_body_mps, right.linear_velocity_body_mps, ratio),
                _lerp(left.angular_velocity_body_radps, right.angular_velocity_body_radps, ratio),
            )
    return samples[-1] if timestamp == samples[-1].timestamp_sec else None


def _time_curvature(
    candidates: Sequence[ProfileCandidate | None], index: int, step: float
) -> float | None:
    if index == 0 or index + 1 >= len(candidates):
        return None
    left, center, right = candidates[index - 1], candidates[index], candidates[index + 1]
    if left is None or center is None or right is None:
        return None
    return max(0.0, (left[1] - 2.0 * center[1] + right[1]) / (step * step))


def _offset_grid(options: RadarSpatiotemporalLeverArmOptions) -> list[float]:
    count = round(2.0 * options.max_abs_time_offset_sec / options.time_offset_step_sec)
    return [
        -options.max_abs_time_offset_sec + index * options.time_offset_step_sec
        for index in range(count + 1)
    ]


def _has_full_offset_support(
    measurement: RadarVelocityMeasurement,
    reference: Sequence[ReferenceKinematicSample],
    options: RadarSpatiotemporalLeverArmOptions,
) -> bool:
    return bool(reference) and (
        measurement.timestamp_sec - options.max_abs_time_offset_sec >= reference[0].timestamp_sec
        and measurement.timestamp_sec + options.max_abs_time_offset_sec
        <= reference[-1].timestamp_sec
    )


def _validate_options(options: RadarSpatiotemporalLeverArmOptions) -> None:
    if options.max_abs_time_offset_sec <= 0.0 or options.time_offset_step_sec <= 0.0:
        raise ValueError("time-offset range and step must be positive")
    if options.time_offset_step_sec > 2.0 * options.max_abs_time_offset_sec:
        raise ValueError("time-offset step exceeds the search interval")


def _skew(value: Vector3) -> tuple[Vector3, Vector3, Vector3]:
    x, y, z = value
    return ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _lerp(left: Vector3, right: Vector3, ratio: float) -> Vector3:
    return (
        left[0] + ratio * (right[0] - left[0]),
        left[1] + ratio * (right[1] - left[1]),
        left[2] + ratio * (right[2] - left[2]),
    )


def _array3(values: NDArray[np.float64]) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _squared_distance(left: Vector3, right: Vector3) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _empty(
    status: Literal["insufficient_measurements", "degenerate_motion"],
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    *,
    singular_values: tuple[float, float, float] | None = None,
    rank: int = 0,
    condition: float | None = None,
) -> RadarSpatiotemporalLeverArmResult:
    return RadarSpatiotemporalLeverArmResult(
        status=status,
        reason=(
            "not enough measurements with full clock-offset interpolation support"
            if status == "insufficient_measurements"
            else "angular velocity does not observe all lever-arm directions"
        ),
        translation_body_radar_m=None,
        time_offset_sec=None,
        train_measurement_ids=train_ids,
        holdout_measurement_ids=holdout_ids,
        train_rmse_mps=None,
        holdout_rmse_mps=None,
        lever_arm_singular_values=singular_values,
        lever_arm_rank=rank,
        lever_arm_condition_number=condition,
        time_objective_curvature_mps2_per_sec2=None,
        probes=(),
    )
