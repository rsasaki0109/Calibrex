"""Horn quaternion point-only Camera-LiDAR extrinsic calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, Vector3
from calibrex.evaluation.holdout import split_indices

FloatArray: TypeAlias = NDArray[np.float64]
HornPointStatus = Literal[
    "converged", "max_iterations", "insufficient_observations", "degenerate_geometry"
]
HornPointDof = Literal["x", "y", "z", "roll", "pitch", "yaw"]


@dataclass(frozen=True)
class HornPointObservation:
    """One corresponding 3D target point in Camera and LiDAR frames."""

    frame_id: str
    camera_point_m: Vector3
    lidar_point_m: Vector3
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")
        object.__setattr__(self, "camera_point_m", _finite_vector(self.camera_point_m))
        object.__setattr__(self, "lidar_point_m", _finite_vector(self.lidar_point_m))
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("point weight must be finite and non-negative")


@dataclass(frozen=True)
class HornPointSolverOptions:
    min_train_observations: int = 3
    holdout_ratio: float = 0.2
    split_seed: int = 0
    huber_delta_m: float = 0.05
    max_iterations: int = 20
    convergence_tolerance: float = 1.0e-10
    rank_tolerance: float = 1.0e-7
    max_condition_number: float = 1.0e6
    known_bad_rotation_deg: float = 5.0
    known_bad_translation_m: float = 0.05
    known_bad_margin_m: float = 0.005


@dataclass(frozen=True)
class HornPointEvaluation:
    point_rmse_m: float | None


@dataclass(frozen=True)
class HornPointProbe:
    dof: HornPointDof
    amount: float
    unit: Literal["m", "deg"]
    holdout_rmse_m: float | None
    delta_m: float | None
    detectable: bool | None


@dataclass(frozen=True)
class HornPointLidarCameraResult:
    status: HornPointStatus
    reason: str
    transform_camera_lidar: SE3 | None
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    centered_lidar_singular_values: tuple[float, float, float] | None
    centered_geometry_rank: int
    centered_geometry_condition_number: float | None
    quaternion_objective_eigenvalues: tuple[float, float, float, float] | None
    quaternion_maximum_eigengap: float | None
    quaternion_normalized_eigengap: float | None
    joint_singular_values: tuple[float, float, float, float, float, float] | None
    joint_rank: int
    joint_condition_number: float | None
    weak_joint_direction: tuple[float, float, float, float, float, float] | None
    rms_scale_ratio_camera_over_lidar: float | None
    train_evaluation: HornPointEvaluation
    holdout_evaluation: HornPointEvaluation
    probes: tuple[HornPointProbe, ...]
    iterations: int
    solver_options: HornPointSolverOptions

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "transform_camera_lidar": (
                self.transform_camera_lidar.as_dict()
                if self.transform_camera_lidar is not None
                else None
            ),
            "train_frame_ids": list(self.train_frame_ids),
            "holdout_frame_ids": list(self.holdout_frame_ids),
            "centered_lidar_singular_values": self.centered_lidar_singular_values,
            "centered_geometry_rank": self.centered_geometry_rank,
            "centered_geometry_condition_number": self.centered_geometry_condition_number,
            "quaternion_objective_eigenvalues": self.quaternion_objective_eigenvalues,
            "quaternion_maximum_eigengap": self.quaternion_maximum_eigengap,
            "quaternion_normalized_eigengap": self.quaternion_normalized_eigengap,
            "joint_singular_values": self.joint_singular_values,
            "joint_rank": self.joint_rank,
            "joint_condition_number": self.joint_condition_number,
            "weak_joint_direction": self.weak_joint_direction,
            "joint_parameter_order": ["rx", "ry", "rz", "tx", "ty", "tz"],
            "rms_scale_ratio_camera_over_lidar": self.rms_scale_ratio_camera_over_lidar,
            "scale_policy": "rigid extrinsic scale fixed to one; ratio is diagnostic only",
            "train_evaluation": asdict(self.train_evaluation),
            "holdout_evaluation": asdict(self.holdout_evaluation),
            "known_bad_probes": [asdict(probe) for probe in self.probes],
            "iterations": self.iterations,
            "solver_options": asdict(self.solver_options),
            "method": "horn_unit_quaternion_point_alignment_irls/v0.1",
            "paper": {
                "title": "Closed-form solution of absolute orientation using unit quaternions",
                "author": "Berthold K. P. Horn",
                "journal": "Journal of the Optical Society of America A 4(4), 1987",
                "doi": "10.1364/JOSAA.4.000629",
            },
            "frame_convention": "p_camera = R_camera_lidar p_lidar + t_camera_lidar",
            "input_features": "corresponding 3D target centres only; normals are not consumed",
            "implementation": "independent NumPy implementation; no external code copied",
        }


class HornPointLidarCameraSolver:
    """Fit a rigid Camera-from-LiDAR transform with Horn's quaternion eigenproblem."""

    def solve(
        self,
        observations: Sequence[HornPointObservation],
        options: HornPointSolverOptions | None = None,
    ) -> HornPointLidarCameraResult:
        opts = options or HornPointSolverOptions()
        _validate_options(opts)
        identifiers = [item.frame_id for item in observations]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Horn point frame IDs must be unique")
        usable = sorted(
            (item for item in observations if item.weight > 0.0),
            key=lambda item: item.frame_id,
        )
        train_indices, holdout_indices = split_indices(
            len(usable), opts.holdout_ratio, opts.split_seed
        )
        train = [usable[index] for index in train_indices]
        holdout = [usable[index] for index in holdout_indices]
        train_ids = tuple(item.frame_id for item in train)
        holdout_ids = tuple(item.frame_id for item in holdout)
        if len(train) < opts.min_train_observations:
            return _empty("insufficient_observations", train_ids, holdout_ids, opts)

        robust: FloatArray = np.ones(len(train), dtype=np.float64)
        transform: SE3 | None = None
        diagnostics: _FitDiagnostics | None = None
        status: HornPointStatus = "max_iterations"
        iterations = 0
        for iteration in range(1, opts.max_iterations + 1):
            candidate, diagnostics = _weighted_horn(train, robust, opts)
            if candidate is None:
                return _empty(
                    "degenerate_geometry",
                    train_ids,
                    holdout_ids,
                    opts,
                    diagnostics=diagnostics,
                )
            iterations = iteration
            if transform is not None and _transform_delta(candidate, transform) <= (
                opts.convergence_tolerance
            ):
                transform = candidate
                status = "converged"
                break
            transform = candidate
            residuals = np.asarray(
                [_point_error(item, transform) for item in train], dtype=np.float64
            )
            if iteration < opts.max_iterations:
                robust = np.minimum(
                    1.0,
                    opts.huber_delta_m / np.maximum(residuals, 1.0e-15),
                )

        if transform is None or diagnostics is None:
            return _empty("degenerate_geometry", train_ids, holdout_ids, opts)
        joint = _joint_observability(train, transform, robust, opts)
        train_eval = evaluate_horn_point_observations(train, transform)
        holdout_eval = evaluate_horn_point_observations(holdout, transform)
        if (
            diagnostics.geometry_rank < 2
            or joint.rank < 6
            or joint.condition_number is None
            or joint.condition_number > opts.max_condition_number
        ):
            return HornPointLidarCameraResult(
                "degenerate_geometry",
                "point geometry does not constrain a well-conditioned rigid transform",
                transform,
                train_ids,
                holdout_ids,
                diagnostics.geometry_spectrum,
                diagnostics.geometry_rank,
                diagnostics.geometry_condition,
                diagnostics.objective_eigenvalues,
                diagnostics.eigengap,
                diagnostics.normalized_eigengap,
                joint.singular_values,
                joint.rank,
                joint.condition_number,
                joint.weak_direction,
                diagnostics.scale_ratio,
                train_eval,
                holdout_eval,
                (),
                iterations,
                opts,
            )
        return HornPointLidarCameraResult(
            status,
            "Horn quaternion alignment converged"
            if status == "converged"
            else "Horn IRLS reached the iteration limit",
            transform,
            train_ids,
            holdout_ids,
            diagnostics.geometry_spectrum,
            diagnostics.geometry_rank,
            diagnostics.geometry_condition,
            diagnostics.objective_eigenvalues,
            diagnostics.eigengap,
            diagnostics.normalized_eigengap,
            joint.singular_values,
            joint.rank,
            joint.condition_number,
            joint.weak_direction,
            diagnostics.scale_ratio,
            train_eval,
            holdout_eval,
            _known_bad_probes(holdout, transform, holdout_eval, opts),
            iterations,
            opts,
        )


def evaluate_horn_point_observations(
    observations: Sequence[HornPointObservation], transform: SE3
) -> HornPointEvaluation:
    if not observations:
        return HornPointEvaluation(None)
    squared = [_point_error(item, transform) ** 2 for item in observations]
    return HornPointEvaluation(math.sqrt(sum(squared) / len(squared)))


@dataclass(frozen=True)
class _FitDiagnostics:
    geometry_spectrum: tuple[float, float, float]
    geometry_rank: int
    geometry_condition: float | None
    objective_eigenvalues: tuple[float, float, float, float]
    eigengap: float
    normalized_eigengap: float
    scale_ratio: float | None


@dataclass(frozen=True)
class _JointDiagnostics:
    singular_values: tuple[float, float, float, float, float, float]
    rank: int
    condition_number: float | None
    weak_direction: tuple[float, float, float, float, float, float]


def _weighted_horn(
    observations: Sequence[HornPointObservation],
    robust: FloatArray,
    options: HornPointSolverOptions,
) -> tuple[SE3 | None, _FitDiagnostics]:
    weights: FloatArray = np.asarray(
        [item.weight * value for item, value in zip(observations, robust, strict=True)],
        dtype=np.float64,
    )
    total = float(np.sum(weights))
    camera: FloatArray = np.asarray([item.camera_point_m for item in observations])
    lidar: FloatArray = np.asarray([item.lidar_point_m for item in observations])
    camera_mean = np.sum(camera * weights[:, None], axis=0) / total
    lidar_mean = np.sum(lidar * weights[:, None], axis=0) / total
    camera_centered = camera - camera_mean
    lidar_centered = lidar - lidar_mean
    weighted_lidar = lidar_centered * np.sqrt(weights)[:, None]
    geometry_values = np.linalg.svd(weighted_lidar, compute_uv=False)
    padded = np.pad(geometry_values, (0, max(0, 3 - len(geometry_values))))[:3]
    threshold = options.rank_tolerance * max(float(padded[0]), 1.0e-15)
    geometry_rank = int(np.count_nonzero(padded > threshold))
    geometry_condition = float(padded[0] / padded[geometry_rank - 1]) if geometry_rank > 0 else None
    covariance: FloatArray = np.zeros((3, 3), dtype=np.float64)
    for index in range(len(observations)):
        covariance += weights[index] * np.outer(lidar_centered[index], camera_centered[index])
    objective = _horn_objective(covariance)
    eigenvalues, eigenvectors = np.linalg.eigh(objective)
    eigengap = float(eigenvalues[-1] - eigenvalues[-2])
    normalized_gap = eigengap / max(float(np.max(np.abs(eigenvalues))), 1.0e-15)
    camera_energy = float(np.sum(weights * np.sum(camera_centered**2, axis=1)))
    lidar_energy = float(np.sum(weights * np.sum(lidar_centered**2, axis=1)))
    scale_ratio = math.sqrt(camera_energy / lidar_energy) if lidar_energy > 0.0 else None
    objective_eigenvalues = (
        float(eigenvalues[0]),
        float(eigenvalues[1]),
        float(eigenvalues[2]),
        float(eigenvalues[3]),
    )
    diagnostics = _FitDiagnostics(
        (float(padded[0]), float(padded[1]), float(padded[2])),
        geometry_rank,
        geometry_condition,
        objective_eigenvalues,
        eigengap,
        normalized_gap,
        scale_ratio,
    )
    if geometry_rank < 2 or normalized_gap <= options.rank_tolerance:
        return None, diagnostics
    quaternion_wxyz = eigenvectors[:, -1]
    quaternion_wxyz /= np.linalg.norm(quaternion_wxyz)
    if quaternion_wxyz[0] < 0.0:
        quaternion_wxyz *= -1.0
    quaternion = (
        float(quaternion_wxyz[1]),
        float(quaternion_wxyz[2]),
        float(quaternion_wxyz[3]),
        float(quaternion_wxyz[0]),
    )
    rotation = _rotation_matrix(SE3((0.0, 0.0, 0.0), quaternion))
    translation = camera_mean - rotation @ lidar_mean
    return (
        SE3(
            (float(translation[0]), float(translation[1]), float(translation[2])),
            quaternion,
        ),
        diagnostics,
    )


def _horn_objective(matrix: FloatArray) -> FloatArray:
    sxx, sxy, sxz = matrix[0]
    syx, syy, syz = matrix[1]
    szx, szy, szz = matrix[2]
    return np.asarray(
        [
            [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
            [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
            [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
            [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
        ],
        dtype=np.float64,
    )


def _joint_observability(
    observations: Sequence[HornPointObservation],
    transform: SE3,
    robust: FloatArray,
    options: HornPointSolverOptions,
) -> _JointDiagnostics:
    rotation = _rotation_matrix(transform)
    rows: list[FloatArray] = []
    for item, robust_weight in zip(observations, robust, strict=True):
        root = math.sqrt(item.weight * float(robust_weight))
        point = rotation @ np.asarray(item.lidar_point_m)
        rows.extend(root * row for row in np.hstack((-_skew(point), np.eye(3))))
    matrix: FloatArray = np.asarray(rows)
    _u, spectrum, vh = np.linalg.svd(matrix, full_matrices=False)
    threshold = options.rank_tolerance * max(float(spectrum[0]), 1.0e-15)
    rank = int(np.count_nonzero(spectrum > threshold))
    condition = float(spectrum[0] / spectrum[-1]) if spectrum[-1] > threshold else None
    singular_values = (
        float(spectrum[0]),
        float(spectrum[1]),
        float(spectrum[2]),
        float(spectrum[3]),
        float(spectrum[4]),
        float(spectrum[5]),
    )
    weak_direction = (
        float(vh[-1, 0]),
        float(vh[-1, 1]),
        float(vh[-1, 2]),
        float(vh[-1, 3]),
        float(vh[-1, 4]),
        float(vh[-1, 5]),
    )
    return _JointDiagnostics(
        singular_values,
        rank,
        condition,
        weak_direction,
    )


def _known_bad_probes(
    holdout: Sequence[HornPointObservation],
    transform: SE3,
    baseline: HornPointEvaluation,
    options: HornPointSolverOptions,
) -> tuple[HornPointProbe, ...]:
    probes: list[HornPointProbe] = []
    dofs: tuple[HornPointDof, ...] = ("x", "y", "z", "roll", "pitch", "yaw")
    for dof in dofs:
        magnitude = (
            options.known_bad_translation_m
            if dof in {"x", "y", "z"}
            else options.known_bad_rotation_deg
        )
        unit: Literal["m", "deg"] = "m" if dof in {"x", "y", "z"} else "deg"
        for amount in (-magnitude, magnitude):
            evaluation = evaluate_horn_point_observations(holdout, _perturb(transform, dof, amount))
            delta = (
                evaluation.point_rmse_m - baseline.point_rmse_m
                if evaluation.point_rmse_m is not None and baseline.point_rmse_m is not None
                else None
            )
            probes.append(
                HornPointProbe(
                    dof,
                    amount,
                    unit,
                    evaluation.point_rmse_m,
                    delta,
                    delta > options.known_bad_margin_m if delta is not None else None,
                )
            )
    return tuple(probes)


def _perturb(transform: SE3, dof: HornPointDof, amount: float) -> SE3:
    if dof in {"x", "y", "z"}:
        translation = [0.0, 0.0, 0.0]
        translation[("x", "y", "z").index(dof)] = amount
        delta = SE3(
            (translation[0], translation[1], translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    else:
        axis = {
            "roll": (1.0, 0.0, 0.0),
            "pitch": (0.0, 1.0, 0.0),
            "yaw": (0.0, 0.0, 1.0),
        }[dof]
        angle = math.radians(amount)
        sine = math.sin(angle / 2.0)
        delta = SE3(
            (0.0, 0.0, 0.0),
            (axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(angle / 2.0)),
        )
    return delta.compose(transform)


def _point_error(observation: HornPointObservation, transform: SE3) -> float:
    predicted = transform.transform_point(observation.lidar_point_m)
    return math.dist(predicted, observation.camera_point_m)


def _rotation_matrix(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _transform_delta(left: SE3, right: SE3) -> float:
    relative = left.inverse().compose(right)
    angle = 2.0 * math.acos(min(1.0, abs(relative.rotation_quat_xyzw[3])))
    return math.sqrt(sum(value * value for value in relative.translation_m) + angle * angle)


def _finite_vector(vector: Vector3) -> Vector3:
    values = (float(vector[0]), float(vector[1]), float(vector[2]))
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("point coordinates must contain three finite values")
    return values


def _validate_options(options: HornPointSolverOptions) -> None:
    positive = (
        options.min_train_observations,
        options.huber_delta_m,
        options.max_iterations,
        options.convergence_tolerance,
        options.rank_tolerance,
        options.max_condition_number,
        options.known_bad_rotation_deg,
        options.known_bad_translation_m,
        options.known_bad_margin_m,
    )
    if any(not math.isfinite(float(value)) or value <= 0 for value in positive):
        raise ValueError("Horn point positive options must be finite and positive")
    if not 0.0 <= options.holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be in [0, 1)")
    if options.min_train_observations < 3:
        raise ValueError("min_train_observations must be at least three")
    if options.max_condition_number <= 1.0:
        raise ValueError("max_condition_number must be greater than one")


def _empty(
    status: Literal["insufficient_observations", "degenerate_geometry"],
    train_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    options: HornPointSolverOptions,
    *,
    diagnostics: _FitDiagnostics | None = None,
) -> HornPointLidarCameraResult:
    return HornPointLidarCameraResult(
        status,
        "not enough train point correspondences"
        if status == "insufficient_observations"
        else "centred point geometry is collinear or quaternion maximum is ambiguous",
        None,
        train_ids,
        holdout_ids,
        diagnostics.geometry_spectrum if diagnostics else None,
        diagnostics.geometry_rank if diagnostics else 0,
        diagnostics.geometry_condition if diagnostics else None,
        diagnostics.objective_eigenvalues if diagnostics else None,
        diagnostics.eigengap if diagnostics else None,
        diagnostics.normalized_eigengap if diagnostics else None,
        None,
        0,
        None,
        None,
        diagnostics.scale_ratio if diagnostics else None,
        HornPointEvaluation(None),
        HornPointEvaluation(None),
        (),
        0,
        options,
    )
