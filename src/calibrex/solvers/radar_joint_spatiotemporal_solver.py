"""Joint Radar rotation, lever-arm, and clock-offset velocity calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Literal

from calibrex.core.geometry import SE3, Vector3, rotate_vector_xyzw
from calibrex.graph.joint_factors import se3_from_tangent
from calibrex.graph.joint_optimization import (
    BackendNeutralJointOptimizer,
    JointOptimizerOptions,
    JointOptimizerResult,
    JointParameterBlock,
    JointResidualBlock,
    ParameterValues,
)
from calibrex.solvers.radar_spatiotemporal_lever_arm_solver import (
    RadarVelocityMeasurement,
    ReferenceKinematicSample,
)

RadarJointSpatiotemporalStatus = Literal[
    "converged", "max_iterations", "insufficient_measurements", "degenerate_motion"
]
RadarJointProbeParameter = Literal["x", "y", "z", "roll", "pitch", "yaw", "time"]
_EXTRINSIC_BLOCK = "radar_extrinsic_correction"
_TIME_BLOCK = "radar_time_latent"


@dataclass(frozen=True)
class RadarJointSpatiotemporalOptions:
    max_iterations: int = 80
    convergence_tolerance: float = 1.0e-9
    initial_damping: float = 1.0e-3
    huber_delta_mps: float = 0.3
    holdout_ratio: float = 0.2
    split_seed: int = 0
    min_train_measurements: int = 12
    max_initial_residual_mps: float = 3.0
    max_abs_time_offset_sec: float = 0.1
    initial_time_offset_sec: float = 0.0
    rank_tolerance: float = 1.0e-8
    max_condition_number: float = 1.0e10
    known_bad_translation_m: float = 0.1
    known_bad_rotation_deg: float = 5.0
    known_bad_time_offset_sec: float = 0.02
    known_bad_margin_mps: float = 0.02


@dataclass(frozen=True)
class RadarJointSpatiotemporalProbe:
    parameter: RadarJointProbeParameter
    amount: float
    unit: Literal["m", "deg", "sec"]
    holdout_rmse_mps: float | None
    delta_mps: float | None
    detectable: bool | None


@dataclass(frozen=True)
class RadarJointSpatiotemporalResult:
    status: RadarJointSpatiotemporalStatus
    reason: str
    input_measurement_count: int
    eligible_measurement_count: int
    rejected_initial_residual_ids: tuple[str, ...]
    initial_transform_body_radar: SE3
    transform_body_radar: SE3 | None
    time_offset_sec: float | None
    train_measurement_ids: tuple[str, ...]
    holdout_measurement_ids: tuple[str, ...]
    train_rmse_mps: float | None
    holdout_rmse_mps: float | None
    initial_holdout_rmse_mps: float | None
    holdout_rmse_improvement_mps: float | None
    holdout_rmse_improvement_fraction: float | None
    information_singular_values: tuple[float, ...]
    information_rank: int
    information_rank_threshold: float | None
    information_condition_number: float | None
    weak_parameter_blocks: tuple[str, ...]
    probes: tuple[RadarJointSpatiotemporalProbe, ...]
    optimizer: JointOptimizerResult | None
    solver_options: RadarJointSpatiotemporalOptions

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "input_measurement_count": self.input_measurement_count,
            "eligible_measurement_count": self.eligible_measurement_count,
            "rejected_initial_residual_ids": list(self.rejected_initial_residual_ids),
            "initial_transform_body_radar": self.initial_transform_body_radar.as_dict(),
            "transform_body_radar": (
                self.transform_body_radar.as_dict() if self.transform_body_radar else None
            ),
            "time_offset_sec": self.time_offset_sec,
            "train_measurement_ids": list(self.train_measurement_ids),
            "holdout_measurement_ids": list(self.holdout_measurement_ids),
            "train_rmse_mps": self.train_rmse_mps,
            "holdout_rmse_mps": self.holdout_rmse_mps,
            "initial_holdout_rmse_mps": self.initial_holdout_rmse_mps,
            "holdout_rmse_improvement_mps": self.holdout_rmse_improvement_mps,
            "holdout_rmse_improvement_fraction": self.holdout_rmse_improvement_fraction,
            "information_singular_values": self.information_singular_values,
            "information_rank": self.information_rank,
            "information_rank_threshold": self.information_rank_threshold,
            "information_condition_number": self.information_condition_number,
            "weak_parameter_blocks": list(self.weak_parameter_blocks),
            "known_bad_probes": [asdict(probe) for probe in self.probes],
            "optimizer": self.optimizer.as_dict() if self.optimizer else None,
            "solver_options": asdict(self.solver_options),
            "method": "wise_joint_fixed_reference_velocity_lm/v0.1",
            "measurement_model": (
                "R_body_radar v_radar(t) = v_body(t+dt) + omega_body(t+dt) cross t_body_radar"
            ),
            "time_convention": "radar_time + dt_radar = reference_time",
            "estimated_dofs": ["x", "y", "z", "roll", "pitch", "yaw", "time"],
            "time_parameterization": "max_abs_time_offset_sec * tanh(latent)",
            "primary_reference": {
                "title": (
                    "Spatiotemporal Calibration of 3D Millimetre-Wavelength Radar-Camera Pairs"
                ),
                "authors": ["Emmett Wise", "Qilong Cheng", "Jonathan Kelly"],
                "venue": "IEEE Transactions on Robotics 39(6), 2023",
                "doi": "10.1109/TRO.2023.3311680",
                "arxiv": "2211.01871",
            },
            "specialization": (
                "reference trajectory is fixed and linearly interpolated; "
                "the paper's camera trajectory and scale states are not estimated"
            ),
            "implementation": "independent NumPy/backend-neutral implementation",
        }


class RadarJointSpatiotemporalSolver:
    """Jointly refine a Radar SE(3) extrinsic and bounded time offset."""

    def solve(
        self,
        measurements: Sequence[RadarVelocityMeasurement],
        reference_samples: Sequence[ReferenceKinematicSample],
        initial_transform_body_radar: SE3,
        options: RadarJointSpatiotemporalOptions | None = None,
    ) -> RadarJointSpatiotemporalResult:
        opts = options or RadarJointSpatiotemporalOptions()
        _validate_options(opts)
        _validate_inputs(measurements, reference_samples)
        identifiers = [item.measurement_id for item in measurements]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Radar joint measurement IDs must be unique")
        reference = sorted(reference_samples, key=lambda item: item.timestamp_sec)
        supported = sorted(
            (
                item
                for item in measurements
                if item.weight > 0.0 and _has_full_time_support(item, reference, opts)
            ),
            key=lambda item: item.measurement_id,
        )
        initial_residuals = {
            item.measurement_id: _measurement_error(
                item,
                reference,
                initial_transform_body_radar,
                opts.initial_time_offset_sec,
            )
            for item in supported
        }
        rejected_ids = tuple(
            item.measurement_id
            for item in supported
            if initial_residuals[item.measurement_id] > opts.max_initial_residual_mps
        )
        rejected_set = set(rejected_ids)
        eligible = [item for item in supported if item.measurement_id not in rejected_set]
        initial_latent = _encode_time(opts.initial_time_offset_sec, opts)
        blocks = (
            JointParameterBlock(
                _EXTRINSIC_BLOCK,
                (0.0,) * 6,
                finite_difference_steps=(1.0e-5,) * 3 + (1.0e-6,) * 3,
            ),
            JointParameterBlock(_TIME_BLOCK, (initial_latent,), finite_difference_steps=(1.0e-5,)),
        )
        factors = tuple(
            _make_factor(item, reference, initial_transform_body_radar, opts) for item in eligible
        )
        optimizer = BackendNeutralJointOptimizer().solve(
            blocks,
            factors,
            JointOptimizerOptions(
                max_iterations=opts.max_iterations,
                convergence_tolerance=opts.convergence_tolerance,
                initial_damping=opts.initial_damping,
                huber_delta=opts.huber_delta_mps,
                holdout_ratio=opts.holdout_ratio,
                split_seed=opts.split_seed,
                minimum_train_factors=opts.min_train_measurements,
                rank_tolerance=opts.rank_tolerance,
                max_condition_number=opts.max_condition_number,
                known_bad_margin=opts.known_bad_margin_mps,
            ),
        )
        if optimizer.status == "insufficient_factors":
            return _from_optimizer(
                "insufficient_measurements",
                optimizer.reason,
                None,
                None,
                (),
                optimizer,
                opts,
                len(measurements),
                len(eligible),
                rejected_ids,
                initial_transform_body_radar,
                None,
            )
        values = optimizer.optimized_values
        transform = se3_from_tangent(values[_EXTRINSIC_BLOCK]).compose(initial_transform_body_radar)
        offset = _decode_time(values[_TIME_BLOCK][0], opts)
        holdout_ids = set(optimizer.holdout_factor_ids)
        holdout = [item for item in eligible if item.measurement_id in holdout_ids]
        baseline = radar_joint_spatiotemporal_rmse(holdout, reference, transform, offset)
        initial_holdout_rmse = _weighted_radar_joint_spatiotemporal_rmse(
            holdout,
            reference,
            initial_transform_body_radar,
            opts.initial_time_offset_sec,
        )
        probes = _known_bad_probes(holdout, reference, transform, offset, baseline, opts)
        status: RadarJointSpatiotemporalStatus = (
            "degenerate_motion"
            if optimizer.status == "degenerate"
            else "converged"
            if optimizer.status == "converged"
            else "max_iterations"
        )
        return _from_optimizer(
            status,
            optimizer.reason,
            transform,
            offset,
            probes,
            optimizer,
            opts,
            len(measurements),
            len(eligible),
            rejected_ids,
            initial_transform_body_radar,
            initial_holdout_rmse,
        )


def radar_joint_spatiotemporal_rmse(
    measurements: Sequence[RadarVelocityMeasurement],
    reference_samples: Sequence[ReferenceKinematicSample],
    transform_body_radar: SE3,
    time_offset_sec: float,
) -> float | None:
    if not measurements:
        return None
    reference = sorted(reference_samples, key=lambda item: item.timestamp_sec)
    squared: list[float] = []
    for item in measurements:
        kinematic = _interpolate(reference, item.timestamp_sec + time_offset_sec)
        if kinematic is None:
            return None
        predicted = rotate_vector_xyzw(
            transform_body_radar.rotation_quat_xyzw, item.velocity_radar_mps
        )
        expected = _add(kinematic[0], _cross(kinematic[1], transform_body_radar.translation_m))
        squared.extend((predicted[index] - expected[index]) ** 2 for index in range(3))
    return math.sqrt(sum(squared) / len(squared))


def _measurement_error(
    measurement: RadarVelocityMeasurement,
    reference: Sequence[ReferenceKinematicSample],
    transform: SE3,
    offset: float,
) -> float:
    kinematic = _interpolate(reference, measurement.timestamp_sec + offset)
    if kinematic is None:
        return math.inf
    predicted = rotate_vector_xyzw(transform.rotation_quat_xyzw, measurement.velocity_radar_mps)
    expected = _add(kinematic[0], _cross(kinematic[1], transform.translation_m))
    return math.sqrt(sum((predicted[index] - expected[index]) ** 2 for index in range(3)))


def _weighted_radar_joint_spatiotemporal_rmse(
    measurements: Sequence[RadarVelocityMeasurement],
    reference: Sequence[ReferenceKinematicSample],
    transform: SE3,
    offset: float,
) -> float | None:
    if not measurements:
        return None
    squared_norm = 0.0
    for measurement in measurements:
        error = _measurement_error(measurement, reference, transform, offset)
        if not math.isfinite(error):
            return None
        squared_norm += measurement.weight * error * error
    return math.sqrt(squared_norm / (3 * len(measurements)))


def _make_factor(
    measurement: RadarVelocityMeasurement,
    reference: Sequence[ReferenceKinematicSample],
    initial: SE3,
    options: RadarJointSpatiotemporalOptions,
) -> JointResidualBlock:
    def evaluator(values: ParameterValues) -> tuple[float, float, float]:
        transform = se3_from_tangent(values[_EXTRINSIC_BLOCK]).compose(initial)
        offset = _decode_time(values[_TIME_BLOCK][0], options)
        kinematic = _interpolate(reference, measurement.timestamp_sec + offset)
        if kinematic is None:
            raise ValueError("bounded Radar time offset lost interpolation support")
        predicted = rotate_vector_xyzw(transform.rotation_quat_xyzw, measurement.velocity_radar_mps)
        expected = _add(kinematic[0], _cross(kinematic[1], transform.translation_m))
        return (
            predicted[0] - expected[0],
            predicted[1] - expected[1],
            predicted[2] - expected[2],
        )

    return JointResidualBlock(
        measurement.measurement_id,
        measurement.measurement_id,
        (_EXTRINSIC_BLOCK, _TIME_BLOCK),
        evaluator,
        weight=measurement.weight,
        family="radar_joint_spatiotemporal_velocity",
    )


def _known_bad_probes(
    holdout: Sequence[RadarVelocityMeasurement],
    reference: Sequence[ReferenceKinematicSample],
    transform: SE3,
    offset: float,
    baseline: float | None,
    options: RadarJointSpatiotemporalOptions,
) -> tuple[RadarJointSpatiotemporalProbe, ...]:
    probes: list[RadarJointSpatiotemporalProbe] = []
    parameters: tuple[tuple[RadarJointProbeParameter, int, float, Literal["m", "deg"]], ...] = (
        ("x", 0, options.known_bad_translation_m, "m"),
        ("y", 1, options.known_bad_translation_m, "m"),
        ("z", 2, options.known_bad_translation_m, "m"),
        ("roll", 3, math.radians(options.known_bad_rotation_deg), "deg"),
        ("pitch", 4, math.radians(options.known_bad_rotation_deg), "deg"),
        ("yaw", 5, math.radians(options.known_bad_rotation_deg), "deg"),
    )
    for parameter, index, step, unit in parameters:
        for sign in (-1.0, 1.0):
            tangent = [0.0] * 6
            tangent[index] = sign * step
            candidate = se3_from_tangent(tangent).compose(transform)
            rmse = radar_joint_spatiotemporal_rmse(holdout, reference, candidate, offset)
            amount = sign * (options.known_bad_rotation_deg if unit == "deg" else step)
            probes.append(_probe(parameter, amount, unit, rmse, baseline, options))
    for sign in (-1.0, 1.0):
        candidate_offset = offset + sign * options.known_bad_time_offset_sec
        rmse = radar_joint_spatiotemporal_rmse(holdout, reference, transform, candidate_offset)
        probes.append(
            _probe("time", sign * options.known_bad_time_offset_sec, "sec", rmse, baseline, options)
        )
    return tuple(probes)


def _probe(
    parameter: RadarJointProbeParameter,
    amount: float,
    unit: Literal["m", "deg", "sec"],
    rmse: float | None,
    baseline: float | None,
    options: RadarJointSpatiotemporalOptions,
) -> RadarJointSpatiotemporalProbe:
    delta = rmse - baseline if rmse is not None and baseline is not None else None
    return RadarJointSpatiotemporalProbe(
        parameter,
        amount,
        unit,
        rmse,
        delta,
        delta > options.known_bad_margin_mps if delta is not None else None,
    )


def _from_optimizer(
    status: RadarJointSpatiotemporalStatus,
    reason: str,
    transform: SE3 | None,
    offset: float | None,
    probes: tuple[RadarJointSpatiotemporalProbe, ...],
    optimizer: JointOptimizerResult,
    options: RadarJointSpatiotemporalOptions,
    input_count: int,
    eligible_count: int,
    rejected_ids: tuple[str, ...],
    initial_transform: SE3,
    initial_holdout_rmse: float | None,
) -> RadarJointSpatiotemporalResult:
    improvement = (
        initial_holdout_rmse - optimizer.holdout_rmse
        if initial_holdout_rmse is not None and optimizer.holdout_rmse is not None
        else None
    )
    improvement_fraction = (
        improvement / initial_holdout_rmse
        if improvement is not None
        and initial_holdout_rmse is not None
        and initial_holdout_rmse > 1.0e-15
        else None
    )
    return RadarJointSpatiotemporalResult(
        status=status,
        reason=reason,
        input_measurement_count=input_count,
        eligible_measurement_count=eligible_count,
        rejected_initial_residual_ids=rejected_ids,
        initial_transform_body_radar=initial_transform,
        transform_body_radar=transform,
        time_offset_sec=offset,
        train_measurement_ids=optimizer.train_factor_ids,
        holdout_measurement_ids=optimizer.holdout_factor_ids,
        train_rmse_mps=optimizer.train_rmse,
        holdout_rmse_mps=optimizer.holdout_rmse,
        initial_holdout_rmse_mps=initial_holdout_rmse,
        holdout_rmse_improvement_mps=improvement,
        holdout_rmse_improvement_fraction=improvement_fraction,
        information_singular_values=optimizer.information_singular_values,
        information_rank=optimizer.information_rank,
        information_rank_threshold=optimizer.information_rank_threshold,
        information_condition_number=optimizer.condition_number,
        weak_parameter_blocks=optimizer.weak_parameter_blocks,
        probes=probes,
        optimizer=optimizer,
        solver_options=options,
    )


def _has_full_time_support(
    measurement: RadarVelocityMeasurement,
    reference: Sequence[ReferenceKinematicSample],
    options: RadarJointSpatiotemporalOptions,
) -> bool:
    return bool(reference) and (
        reference[0].timestamp_sec <= measurement.timestamp_sec - options.max_abs_time_offset_sec
        and measurement.timestamp_sec + options.max_abs_time_offset_sec
        <= reference[-1].timestamp_sec
    )


def _interpolate(
    samples: Sequence[ReferenceKinematicSample], timestamp: float
) -> tuple[Vector3, Vector3] | None:
    for left, right in pairwise(samples):
        if left.timestamp_sec <= timestamp <= right.timestamp_sec:
            span = right.timestamp_sec - left.timestamp_sec
            if span <= 0.0:
                continue
            alpha = (timestamp - left.timestamp_sec) / span
            return (
                _lerp(left.linear_velocity_body_mps, right.linear_velocity_body_mps, alpha),
                _lerp(left.angular_velocity_body_radps, right.angular_velocity_body_radps, alpha),
            )
    return None


def _lerp(left: Vector3, right: Vector3, alpha: float) -> Vector3:
    return tuple(left[index] + alpha * (right[index] - left[index]) for index in range(3))  # type: ignore[return-value]


def _encode_time(offset: float, options: RadarJointSpatiotemporalOptions) -> float:
    return math.atanh(offset / options.max_abs_time_offset_sec)


def _decode_time(latent: float, options: RadarJointSpatiotemporalOptions) -> float:
    return options.max_abs_time_offset_sec * math.tanh(latent)


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _validate_options(options: RadarJointSpatiotemporalOptions) -> None:
    positive = (
        options.max_iterations,
        options.convergence_tolerance,
        options.initial_damping,
        options.huber_delta_mps,
        options.min_train_measurements,
        options.max_initial_residual_mps,
        options.max_abs_time_offset_sec,
        options.rank_tolerance,
        options.max_condition_number,
        options.known_bad_translation_m,
        options.known_bad_rotation_deg,
        options.known_bad_time_offset_sec,
    )
    if any(not math.isfinite(float(value)) or value <= 0 for value in positive):
        raise ValueError("Radar joint positive options must be finite and positive")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in [0, 1)")
    if options.max_condition_number <= 1.0:
        raise ValueError("max_condition_number must be greater than one")
    if not math.isfinite(options.initial_time_offset_sec):
        raise ValueError("initial time offset must be finite")
    if abs(options.initial_time_offset_sec) >= options.max_abs_time_offset_sec:
        raise ValueError("initial time offset must lie strictly inside its bound")
    if options.known_bad_time_offset_sec >= options.max_abs_time_offset_sec:
        raise ValueError("known-bad time offset must be smaller than the time bound")
    if not math.isfinite(options.known_bad_margin_mps) or options.known_bad_margin_mps < 0.0:
        raise ValueError("known_bad_margin_mps must be non-negative")


def _validate_inputs(
    measurements: Sequence[RadarVelocityMeasurement],
    reference_samples: Sequence[ReferenceKinematicSample],
) -> None:
    for measurement in measurements:
        measurement_values = (
            measurement.timestamp_sec,
            measurement.weight,
            *measurement.velocity_radar_mps,
        )
        if not all(math.isfinite(value) for value in measurement_values):
            raise ValueError("Radar joint measurements must be finite")
    timestamps = [sample.timestamp_sec for sample in reference_samples]
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("Radar joint reference timestamps must be unique")
    for sample in reference_samples:
        reference_values = (
            sample.timestamp_sec,
            *sample.linear_velocity_body_mps,
            *sample.angular_velocity_body_radps,
        )
        if not all(math.isfinite(value) for value in reference_values):
            raise ValueError("Radar joint reference samples must be finite")
