"""Targetless 3D Radar-to-trajectory rotation from paired ego velocities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import QuaternionXYZW, Vector3
from calibrex.evaluation.holdout import split_indices

RadarTrajectoryRotationStatus = Literal[
    "converged", "max_iterations", "insufficient_pairs", "degenerate_motion"
]


@dataclass(frozen=True)
class RadarTrajectoryRotationPair:
    """A Radar velocity and the corresponding lever-arm-compensated body velocity."""

    measurement_id: str
    velocity_radar_mps: Vector3
    velocity_at_radar_origin_body_mps: Vector3
    weight: float = 1.0


@dataclass(frozen=True)
class RadarTrajectoryRotationOptions:
    max_iterations: int = 30
    convergence_tolerance_rad: float = 1.0e-10
    huber_delta_mps: float = 0.5
    min_pair_count: int = 8
    min_speed_mps: float = 0.5
    rank_tolerance: float = 1.0e-8
    max_condition_number: float = 1.0e6
    holdout_ratio: float = 0.2
    split_seed: int = 0
    known_bad_rotation_deg: float = 5.0
    known_bad_margin_mps: float = 0.05


@dataclass(frozen=True)
class RadarRotationProbe:
    axis: Literal["roll", "pitch", "yaw"]
    amount_deg: float
    holdout_rmse_mps: float | None
    delta_mps: float | None
    detectable: bool | None


@dataclass(frozen=True)
class RadarTrajectoryRotationResult:
    status: RadarTrajectoryRotationStatus
    reason: str
    rotation_body_radar_xyzw: QuaternionXYZW | None
    train_measurement_ids: tuple[str, ...]
    holdout_measurement_ids: tuple[str, ...]
    train_rmse_mps: float | None
    holdout_rmse_mps: float | None
    information_singular_values: tuple[float, float, float] | None
    information_rank: int
    information_condition_number: float | None
    inlier_count: int
    iterations: int
    probes: tuple[RadarRotationProbe, ...]
    solver_options: RadarTrajectoryRotationOptions

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "rotation_body_radar_xyzw": (
                list(self.rotation_body_radar_xyzw)
                if self.rotation_body_radar_xyzw is not None
                else None
            ),
            "train_measurement_ids": list(self.train_measurement_ids),
            "holdout_measurement_ids": list(self.holdout_measurement_ids),
            "train_rmse_mps": self.train_rmse_mps,
            "holdout_rmse_mps": self.holdout_rmse_mps,
            "information_singular_values": self.information_singular_values,
            "information_rank": self.information_rank,
            "information_condition_number": self.information_condition_number,
            "inlier_count": self.inlier_count,
            "iterations": self.iterations,
            "known_bad_probes": [asdict(probe) for probe in self.probes],
            "solver_options": asdict(self.solver_options),
            "method": "wise_velocity_alignment_so3_huber/v0.1",
            "measurement_model": "R_body_radar v_radar = v_body + omega_body cross t_body_radar",
            "estimated_dofs": ["roll", "pitch", "yaw"],
            "fixed_inputs": ["translation_body_radar", "time_offset"],
            "primary_reference": {
                "title": "A Continuous-Time Approach for 3D Radar-to-Camera Extrinsic Calibration",
                "authors": [
                    "Emmett Wise",
                    "Juraj Persic",
                    "Christopher Grebe",
                    "Ivan Petrovic",
                    "Jonathan Kelly",
                ],
                "venue": "IEEE ICRA 2021",
                "arxiv": "2103.07505",
            },
        }


class RadarTrajectoryRotationSolver:
    """Estimate ``R_body_radar`` by robust weighted Wahba alignment."""

    def solve(
        self,
        pairs: Sequence[RadarTrajectoryRotationPair],
        options: RadarTrajectoryRotationOptions | None = None,
    ) -> RadarTrajectoryRotationResult:
        opts = options or RadarTrajectoryRotationOptions()
        _validate_options(opts)
        identifiers = [pair.measurement_id for pair in pairs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Radar rotation measurement IDs must be unique")
        eligible = sorted(
            (
                pair
                for pair in pairs
                if pair.weight > 0.0
                and _norm(pair.velocity_radar_mps) >= opts.min_speed_mps
                and _norm(pair.velocity_at_radar_origin_body_mps) >= opts.min_speed_mps
            ),
            key=lambda pair: pair.measurement_id,
        )
        train_indices, holdout_indices = split_indices(
            len(eligible), opts.holdout_ratio, opts.split_seed
        )
        train = [eligible[index] for index in train_indices]
        holdout = [eligible[index] for index in holdout_indices]
        train_ids = tuple(pair.measurement_id for pair in train)
        holdout_ids = tuple(eligible[index].measurement_id for index in holdout_indices)
        if len(train) < opts.min_pair_count:
            return _empty("insufficient_pairs", train_ids, holdout_ids, opts)

        robust: NDArray[np.float64] = np.ones(len(train), dtype=np.float64)
        rotation: NDArray[np.float64] = np.eye(3, dtype=np.float64)
        status: RadarTrajectoryRotationStatus = "max_iterations"
        iterations = 0
        for iteration in range(opts.max_iterations):
            solved = _weighted_rotation(train, robust)
            if solved is None:
                spectrum, rank, condition = _information_diagnostics(train, rotation, robust, opts)
                return _empty(
                    "degenerate_motion",
                    train_ids,
                    holdout_ids,
                    opts,
                    spectrum=spectrum,
                    rank=rank,
                    condition=condition,
                )
            step = _rotation_angle(solved @ rotation.T)
            rotation = solved
            residuals = np.asarray([_error(pair, rotation) for pair in train])
            robust = np.asarray(
                [
                    1.0 if value <= opts.huber_delta_mps else opts.huber_delta_mps / value
                    for value in residuals
                ]
            )
            iterations = iteration + 1
            if step <= opts.convergence_tolerance_rad:
                status = "converged"
                break

        spectrum_tuple, rank, condition = _information_diagnostics(train, rotation, robust, opts)
        if rank < 3 or condition is None or condition > opts.max_condition_number:
            return _empty(
                "degenerate_motion",
                train_ids,
                holdout_ids,
                opts,
                spectrum=spectrum_tuple,
                rank=rank,
                condition=condition,
            )
        train_rmse = radar_trajectory_rotation_rmse(train, rotation)
        holdout_rmse = radar_trajectory_rotation_rmse(holdout, rotation)
        probes = _probes(holdout, rotation, holdout_rmse, opts)
        return RadarTrajectoryRotationResult(
            status=status,
            reason="3D velocity alignment converged"
            if status == "converged"
            else "maximum IRLS iterations reached",
            rotation_body_radar_xyzw=_matrix_to_quaternion(rotation),
            train_measurement_ids=train_ids,
            holdout_measurement_ids=holdout_ids,
            train_rmse_mps=train_rmse,
            holdout_rmse_mps=holdout_rmse,
            information_singular_values=spectrum_tuple,
            information_rank=rank,
            information_condition_number=condition,
            inlier_count=sum(_error(pair, rotation) <= opts.huber_delta_mps for pair in train),
            iterations=iterations,
            probes=probes,
            solver_options=opts,
        )


def radar_trajectory_rotation_rmse(
    pairs: Sequence[RadarTrajectoryRotationPair],
    rotation_body_radar: NDArray[np.float64],
) -> float | None:
    """Evaluate a candidate rotation on unchanged velocity pairs."""

    if not pairs:
        return None
    squared = [_error(pair, rotation_body_radar) ** 2 for pair in pairs]
    return math.sqrt(sum(squared) / len(squared))


def _weighted_rotation(
    pairs: Sequence[RadarTrajectoryRotationPair], robust: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    covariance: NDArray[np.float64] = np.zeros((3, 3), dtype=np.float64)
    for pair, robust_weight in zip(pairs, robust, strict=True):
        source: NDArray[np.float64] = np.asarray(pair.velocity_radar_mps, dtype=np.float64)
        target: NDArray[np.float64] = np.asarray(
            pair.velocity_at_radar_origin_body_mps, dtype=np.float64
        )
        covariance += pair.weight * robust_weight * np.outer(target, source)
    left, spectrum, right_t = np.linalg.svd(covariance)
    if spectrum[0] < 1.0e-12 or int(np.sum(spectrum > 1.0e-10 * spectrum[0])) < 2:
        return None
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    return np.asarray(left @ correction @ right_t, dtype=np.float64)


def _rotation_information(
    pairs: Sequence[RadarTrajectoryRotationPair],
    rotation: NDArray[np.float64],
    robust: NDArray[np.float64],
) -> NDArray[np.float64]:
    information: NDArray[np.float64] = np.zeros((3, 3), dtype=np.float64)
    for pair, robust_weight in zip(pairs, robust, strict=True):
        vector = rotation @ np.asarray(pair.velocity_radar_mps, dtype=np.float64)
        information += (
            pair.weight
            * robust_weight
            * (float(vector @ vector) * np.eye(3) - np.outer(vector, vector))
        )
    return information


def _information_diagnostics(
    pairs: Sequence[RadarTrajectoryRotationPair],
    rotation: NDArray[np.float64],
    robust: NDArray[np.float64],
    options: RadarTrajectoryRotationOptions,
) -> tuple[tuple[float, float, float], int, float | None]:
    spectrum = np.linalg.svd(_rotation_information(pairs, rotation, robust), compute_uv=False)
    threshold = options.rank_tolerance * max(1.0, float(spectrum[0]))
    rank = int(np.sum(spectrum > threshold))
    condition = float(spectrum[0] / spectrum[-1]) if spectrum[-1] > threshold else None
    return _array3(spectrum), rank, condition


def _error(pair: RadarTrajectoryRotationPair, rotation: NDArray[np.float64]) -> float:
    delta = rotation @ np.asarray(pair.velocity_radar_mps) - np.asarray(
        pair.velocity_at_radar_origin_body_mps
    )
    return float(np.linalg.norm(delta))


def _probes(
    holdout: Sequence[RadarTrajectoryRotationPair],
    rotation: NDArray[np.float64],
    baseline: float | None,
    options: RadarTrajectoryRotationOptions,
) -> tuple[RadarRotationProbe, ...]:
    output: list[RadarRotationProbe] = []
    axes: tuple[tuple[Literal["roll", "pitch", "yaw"], int], ...] = (
        ("roll", 0),
        ("pitch", 1),
        ("yaw", 2),
    )
    for axis, axis_index in axes:
        for sign in (-1.0, 1.0):
            amount = sign * options.known_bad_rotation_deg
            perturbed = _axis_rotation(axis_index, math.radians(amount)) @ rotation
            rmse = radar_trajectory_rotation_rmse(holdout, perturbed)
            delta = rmse - baseline if rmse is not None and baseline is not None else None
            output.append(
                RadarRotationProbe(
                    axis,
                    amount,
                    rmse,
                    delta,
                    delta > options.known_bad_margin_mps if delta is not None else None,
                )
            )
    return tuple(output)


def _axis_rotation(axis: int, angle: float) -> NDArray[np.float64]:
    vector = np.zeros(3)
    vector[axis] = 1.0
    skew = np.array(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _matrix_to_quaternion(matrix: NDArray[np.float64]) -> QuaternionXYZW:
    # Stable branch conversion, followed by a deterministic positive-w sign.
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        x, y, z, w = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        j, k = (index + 1) % 3, (index + 2) % 3
        scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[index, index] - matrix[j, j] - matrix[k, k]))
        values = [0.0, 0.0, 0.0]
        values[index] = 0.25 * scale
        values[j] = (matrix[j, index] + matrix[index, j]) / scale
        values[k] = (matrix[k, index] + matrix[index, k]) / scale
        w = (matrix[k, j] - matrix[j, k]) / scale
        x, y, z = values
    quaternion = (float(x), float(y), float(z), float(w))
    if quaternion[3] < 0.0:
        return (-quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])
    return quaternion


def _rotation_angle(matrix: NDArray[np.float64]) -> float:
    cosine = max(-1.0, min(1.0, (float(np.trace(matrix)) - 1.0) / 2.0))
    return 0.0 if cosine >= 1.0 - 1.0e-14 else math.acos(cosine)


def _norm(vector: Vector3) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _array3(values: NDArray[np.float64]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _validate_options(options: RadarTrajectoryRotationOptions) -> None:
    if options.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if options.convergence_tolerance_rad <= 0.0:
        raise ValueError("convergence_tolerance_rad must be positive")
    if options.huber_delta_mps <= 0.0:
        raise ValueError("huber_delta_mps must be positive")
    if options.min_pair_count < 2:
        raise ValueError("min_pair_count must be at least two")
    if options.min_speed_mps < 0.0:
        raise ValueError("min_speed_mps must be non-negative")
    if (
        not math.isfinite(options.rank_tolerance)
        or not math.isfinite(options.max_condition_number)
        or options.rank_tolerance <= 0.0
        or options.max_condition_number <= 1.0
    ):
        raise ValueError("observability tolerances must be positive and finite")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in [0, 1)")
    if options.known_bad_rotation_deg <= 0.0 or options.known_bad_margin_mps < 0.0:
        raise ValueError("known-bad probe amount must be positive and margin non-negative")


def _empty(
    status: Literal["insufficient_pairs", "degenerate_motion"],
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    options: RadarTrajectoryRotationOptions,
    *,
    spectrum: tuple[float, float, float] | None = None,
    rank: int = 0,
    condition: float | None = None,
) -> RadarTrajectoryRotationResult:
    return RadarTrajectoryRotationResult(
        status=status,
        reason="not enough train velocity pairs"
        if status == "insufficient_pairs"
        else "velocity excitation does not constrain all rotation axes",
        rotation_body_radar_xyzw=None,
        train_measurement_ids=train_ids,
        holdout_measurement_ids=holdout_ids,
        train_rmse_mps=None,
        holdout_rmse_mps=None,
        information_singular_values=spectrum,
        information_rank=rank,
        information_condition_number=condition,
        inlier_count=0,
        iterations=0,
        probes=(),
        solver_options=options,
    )
