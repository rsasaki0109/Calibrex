"""Radar lever-arm and clock-offset calibration from ego-velocity kinematics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
    joint_rank_tolerance: float = 1.0e-8
    max_joint_condition_number: float = 1.0e6
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
    joint_scaled_singular_values: tuple[float, float, float, float] | None
    joint_rank: int
    joint_condition_number: float | None
    time_translation_subspace_coupling: float | None
    weak_joint_direction: tuple[float, float, float, float] | None
    probes: tuple[RadarSpatiotemporalProbe, ...]
    solver_options: RadarSpatiotemporalLeverArmOptions

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
            "joint_scaled_singular_values": self.joint_scaled_singular_values,
            "joint_rank": self.joint_rank,
            "joint_condition_number": self.joint_condition_number,
            "time_translation_subspace_coupling": (self.time_translation_subspace_coupling),
            "weak_joint_direction": self.weak_joint_direction,
            "joint_parameter_order": ["translation_x", "translation_y", "translation_z", "time"],
            "joint_parameter_scales": {
                "translation_m": self.solver_options.known_bad_translation_m,
                "time_sec": self.solver_options.known_bad_time_offset_sec,
            },
            "known_bad_probes": [probe.__dict__ for probe in self.probes],
            "solver_options": asdict(self.solver_options),
            "method": "profiled_velocity_lever_arm_and_clock_offset/v0.2",
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
            return _empty("insufficient_measurements", train_ids, holdout_ids, solver_options)

        offsets = _offset_grid(solver_options)
        candidates = [
            _profile_candidate(train, reference, rotation_body_radar_xyzw, offset)
            for offset in offsets
        ]
        valid = [(index, candidate) for index, candidate in enumerate(candidates) if candidate]
        if not valid:
            return _empty("degenerate_motion", train_ids, holdout_ids, solver_options)
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
                solver_options,
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
        joint = _joint_observability(
            train,
            reference,
            translation,
            offset,
            solver_options,
        )
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
        jointly_degenerate = (
            joint.rank < 4
            or joint.condition_number is None
            or joint.condition_number > solver_options.max_joint_condition_number
        )
        return RadarSpatiotemporalLeverArmResult(
            status=(
                "degenerate_motion"
                if jointly_degenerate
                else "offset_at_boundary"
                if at_boundary
                else "converged"
            ),
            reason=(
                "joint lever-arm/time Jacobian is rank deficient or ill-conditioned"
                if jointly_degenerate
                else "best clock offset lies on the declared search boundary"
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
            joint_scaled_singular_values=joint.singular_values,
            joint_rank=joint.rank,
            joint_condition_number=joint.condition_number,
            time_translation_subspace_coupling=joint.time_translation_coupling,
            weak_joint_direction=joint.weak_direction,
            probes=probes,
            solver_options=solver_options,
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


@dataclass(frozen=True)
class _JointObservability:
    singular_values: tuple[float, float, float, float]
    rank: int
    condition_number: float | None
    time_translation_coupling: float | None
    weak_direction: tuple[float, float, float, float]


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


def _joint_observability(
    measurements: Sequence[RadarVelocityMeasurement],
    reference: Sequence[ReferenceKinematicSample],
    translation: Vector3,
    offset: float,
    options: RadarSpatiotemporalLeverArmOptions,
) -> _JointObservability:
    """Diagnose the coupled lever-arm/time tangent system in probe-scaled units."""

    rows: list[list[float]] = []
    derivative_step = min(
        options.time_offset_step_sec,
        options.known_bad_time_offset_sec,
    )
    for measurement in measurements:
        timestamp = measurement.timestamp_sec + offset
        center = _interpolate(reference, timestamp)
        before = _interpolate(reference, timestamp - derivative_step)
        after = _interpolate(reference, timestamp + derivative_step)
        if center is None or before is None or after is None:
            continue
        predicted_before = _predicted_velocity(before, translation)
        predicted_after = _predicted_velocity(after, translation)
        time_derivative = tuple(
            (right - left) / (2.0 * derivative_step)
            for left, right in zip(predicted_before, predicted_after, strict=True)
        )
        root_weight = math.sqrt(measurement.weight)
        translation_jacobian = _skew(center.angular_velocity_body_radps)
        for row_index in range(3):
            rows.append(
                [
                    *(
                        root_weight * options.known_bad_translation_m * value
                        for value in translation_jacobian[row_index]
                    ),
                    root_weight * options.known_bad_time_offset_sec * time_derivative[row_index],
                ]
            )
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape[0] < 4:
        return _JointObservability((0.0, 0.0, 0.0, 0.0), 0, None, None, (0.0,) * 4)
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    values = (
        float(singular_values[0]),
        float(singular_values[1]),
        float(singular_values[2]),
        float(singular_values[3]),
    )
    rank = int(sum(value > options.joint_rank_tolerance for value in singular_values))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > options.joint_rank_tolerance
        else None
    )
    translation_columns = matrix[:, :3]
    time_column = matrix[:, 3]
    time_norm = float(np.linalg.norm(time_column))
    coupling: float | None = None
    if time_norm > options.joint_rank_tolerance:
        coefficients, _residuals, _rank, _spectrum = np.linalg.lstsq(
            translation_columns, time_column, rcond=None
        )
        projection = translation_columns @ coefficients
        coupling = min(1.0, float(np.linalg.norm(projection) / time_norm))
    weak = (float(vh[-1, 0]), float(vh[-1, 1]), float(vh[-1, 2]), float(vh[-1, 3]))
    return _JointObservability(values, rank, condition, coupling, weak)


def _predicted_velocity(sample: ReferenceKinematicSample, translation: Vector3) -> Vector3:
    return _add(
        sample.linear_velocity_body_mps,
        _cross(sample.angular_velocity_body_radps, translation),
    )


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
    if options.known_bad_translation_m <= 0.0 or options.known_bad_time_offset_sec <= 0.0:
        raise ValueError("known-bad perturbation scales must be positive")
    if options.joint_rank_tolerance <= 0.0 or options.max_joint_condition_number <= 1.0:
        raise ValueError("joint observability thresholds must be positive")


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
    options: RadarSpatiotemporalLeverArmOptions,
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
        joint_scaled_singular_values=None,
        joint_rank=0,
        joint_condition_number=None,
        time_translation_subspace_coupling=None,
        weak_joint_direction=None,
        probes=(),
        solver_options=options,
    )
