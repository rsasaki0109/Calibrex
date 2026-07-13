"""Targetless camera--LiDAR calibration by mutual-information maximization.

This is an independent implementation of Pandey et al. (AAAI 2012).  The
ROS-independent core consumes luminance images, LiDAR points, and calibrated
camera models.  Its finite binned Gaussian KDE is a documented numerical
specialization of the continuous Parzen estimate in the paper.
"""

from __future__ import annotations

import math
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
PandeyStatus = Literal[
    "converged", "max_iterations", "insufficient_observations", "numerical_failure"
]
CameraProjectionKind = Literal["pinhole", "opencv_fisheye"]
CameraAxes = Literal["optical_z", "forward_x_left_y_up_z"]
DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class MutualInformationCameraModel:
    """Calibrated pinhole or OpenCV-fisheye camera model."""

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
            raise ValueError("camera image dimensions must exceed one pixel")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class MutualInformationObservation:
    """One synchronized luminance image and reflective LiDAR point cloud."""

    frame_id: str
    luminance: FloatArray
    lidar_points: FloatArray
    reflectivity: FloatArray
    camera: MutualInformationCameraModel

    def __post_init__(self) -> None:
        image = np.asarray(self.luminance, dtype=float)
        points = np.asarray(self.lidar_points, dtype=float)
        reflectivity = np.asarray(self.reflectivity, dtype=float)
        if image.shape != (self.camera.height, self.camera.width):
            raise ValueError("luminance shape must match the camera model")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("lidar_points must have shape Nx3")
        if reflectivity.shape != (points.shape[0],):
            raise ValueError("reflectivity must have shape N")
        if not np.all(np.isfinite(image)) or not np.all(np.isfinite(points)):
            raise ValueError("image and LiDAR points must be finite")
        if not np.all(np.isfinite(reflectivity)):
            raise ValueError("reflectivity must be finite")
        object.__setattr__(self, "luminance", image)
        object.__setattr__(self, "lidar_points", points)
        object.__setattr__(self, "reflectivity", reflectivity)


@dataclass(frozen=True)
class PandeyMutualInformationOptions:
    """Deterministic optimization and evidence settings."""

    min_train_observations: int = 2
    holdout_ratio: float = 0.2
    split_seed: int = 0
    histogram_bins: int = 32
    min_projected_points: int = 64
    max_iterations: int = 40
    convergence_tolerance: float = 1.0e-5
    gradient_steps: tuple[float, float, float, float, float, float] = (
        0.005,
        0.005,
        0.005,
        math.radians(0.1),
        math.radians(0.1),
        math.radians(0.1),
    )
    curvature_steps: tuple[float, float, float, float, float, float] = (
        0.01,
        0.01,
        0.01,
        math.radians(0.25),
        math.radians(0.25),
        math.radians(0.25),
    )
    initial_step_size: float = 0.02
    min_step_size: float = 1.0e-5
    max_step_size: float = 0.1
    max_backtracks: int = 8
    known_bad_translation_m: float = 0.05
    known_bad_rotation_deg: float = 5.0
    rank_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.histogram_bins < 8 or self.min_projected_points < 4:
            raise ValueError("histogram_bins and min_projected_points are too small")
        if len(self.gradient_steps) != 6 or len(self.curvature_steps) != 6:
            raise ValueError("gradient_steps and curvature_steps must have six values")
        if any(value <= 0.0 for value in self.gradient_steps + self.curvature_steps):
            raise ValueError("finite-difference steps must be positive")


@dataclass(frozen=True)
class MutualInformationEvaluation:
    """Aggregate projected-pair score for a fixed transform."""

    raw_mutual_information: float
    normalized_mutual_information: float
    projected_point_count: int
    observation_count: int


@dataclass(frozen=True)
class PandeyMutualInformationIteration:
    """One accepted numerical-gradient ascent update."""

    iteration: int
    raw_mutual_information: float
    gradient_norm: float
    step_size: float
    parameter_delta_norm: float
    backtrack_count: int


@dataclass(frozen=True)
class PandeyMutualInformationProbe:
    """Score response to a signed known-bad transform perturbation."""

    dof: Literal["x", "y", "z", "roll", "pitch", "yaw"]
    amount: float
    unit: Literal["m", "deg"]
    normalized_mutual_information: float
    score_drop: float
    detectable: bool


@dataclass(frozen=True)
class PandeyMutualInformationResult:
    """Transform estimate plus independent validation and observability evidence."""

    status: PandeyStatus
    reason: str
    transform_camera_lidar: SE3 | None
    train_frame_ids: tuple[str, ...]
    holdout_frame_ids: tuple[str, ...]
    train_evaluation: MutualInformationEvaluation
    holdout_evaluation: MutualInformationEvaluation
    iterations: tuple[PandeyMutualInformationIteration, ...]
    probes: tuple[PandeyMutualInformationProbe, ...]
    curvature: NumericalCurvatureEvaluation | None

    def as_dict(self) -> dict[str, object]:
        """Return a schema-safe payload with primary-paper provenance."""

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
            "train_evaluation": self.train_evaluation.__dict__,
            "holdout_evaluation": self.holdout_evaluation.__dict__,
            "iterations": [item.__dict__ for item in self.iterations],
            "known_bad_probes": [item.__dict__ for item in self.probes],
            "curvature": self.curvature.as_dict() if self.curvature is not None else None,
            "method": "pandey_mutual_information_bb_ascent/v0.1",
            "paper_doi": "10.1609/aaai.v26i1.8379",
            "paper_url": "https://robots.engin.umich.edu/publications/gpandey-2012a.pdf",
            "frame_convention": "p_camera = R_camera_lidar p_lidar + t_camera_lidar",
            "objective": "raw mutual information; equations 1--8",
            "optimizer": "numerical gradient and Barzilai--Borwein ascent; equations 9--11",
            "kde_specialization": (
                "finite 2D histogram convolved with covariance-derived Gaussian kernel"
            ),
            "numerical_safeguards": "step clipping and monotonic backtracking",
            "curvature_claim": "central-difference objective Hessian; not covariance or CRLB",
        }


class PandeyMutualInformationSolver:
    """Estimate ``T_camera_lidar`` by maximizing image/reflectivity MI."""

    def solve(
        self,
        observations: Sequence[MutualInformationObservation],
        initial_transform: SE3,
        options: PandeyMutualInformationOptions | None = None,
    ) -> PandeyMutualInformationResult:
        solver_options = options or PandeyMutualInformationOptions()
        usable = sorted(observations, key=lambda item: item.frame_id)
        train_indices, holdout_indices = split_indices(
            len(usable), solver_options.holdout_ratio, seed=solver_options.split_seed
        )
        train = [usable[index] for index in train_indices]
        holdout = [usable[index] for index in holdout_indices]
        if len(train) < solver_options.min_train_observations:
            return _empty_result(train, holdout)

        parameters = np.asarray(_parameters_from_transform(initial_transform), dtype=float)
        score = evaluate_mutual_information(
            train, _transform_from_parameters(parameters), solver_options
        ).raw_mutual_information
        history: list[PandeyMutualInformationIteration] = []
        previous_parameters: FloatArray | None = None
        previous_gradient: FloatArray | None = None
        status: PandeyStatus = "max_iterations"
        reason = "maximum iteration budget reached"
        for iteration in range(solver_options.max_iterations):
            gradient = _numerical_gradient(train, parameters, solver_options)
            gradient_norm = float(np.linalg.norm(gradient))
            if not math.isfinite(gradient_norm):
                status = "numerical_failure"
                reason = "mutual-information numerical gradient became non-finite"
                break
            if gradient_norm <= 1.0e-14:
                status = "converged"
                reason = "mutual-information numerical gradient reached zero"
                break
            step_size = solver_options.initial_step_size
            if previous_parameters is not None and previous_gradient is not None:
                parameter_delta = parameters - previous_parameters
                gradient_delta = gradient - previous_gradient
                denominator = float(parameter_delta @ gradient_delta)
                if abs(denominator) > 1.0e-14:
                    step_size = abs(float(parameter_delta @ parameter_delta) / denominator)
            step_size = min(
                solver_options.max_step_size,
                max(solver_options.min_step_size, step_size),
            )
            direction = gradient / gradient_norm
            candidate = parameters + step_size * direction
            candidate_score = _raw_score(train, candidate, solver_options)
            backtracks = 0
            while candidate_score < score and backtracks < solver_options.max_backtracks:
                step_size *= 0.5
                candidate = parameters + step_size * direction
                candidate_score = _raw_score(train, candidate, solver_options)
                backtracks += 1
            if candidate_score < score or not math.isfinite(candidate_score):
                status = "converged"
                reason = "no improving step remained after monotonic backtracking"
                break
            delta_norm = float(np.linalg.norm(candidate - parameters))
            history.append(
                PandeyMutualInformationIteration(
                    iteration=iteration + 1,
                    raw_mutual_information=candidate_score,
                    gradient_norm=gradient_norm,
                    step_size=step_size,
                    parameter_delta_norm=delta_norm,
                    backtrack_count=backtracks,
                )
            )
            previous_parameters = parameters.copy()
            previous_gradient = gradient.copy()
            parameters = candidate
            score = candidate_score
            if delta_norm <= solver_options.convergence_tolerance:
                status = "converged"
                reason = "parameter update reached the convergence tolerance"
                break

        transform = _transform_from_parameters(parameters)
        train_evaluation = evaluate_mutual_information(train, transform, solver_options)
        validation = holdout if holdout else train
        holdout_evaluation = evaluate_mutual_information(holdout, transform, solver_options)
        probes = _known_bad_probes(validation, parameters, solver_options)
        curvature = evaluate_numerical_curvature(
            lambda values: -_raw_score(validation, np.asarray(values), solver_options),
            parameters,
            solver_options.curvature_steps,
            rank_tolerance=solver_options.rank_tolerance,
        )
        return PandeyMutualInformationResult(
            status=status,
            reason=reason,
            transform_camera_lidar=transform,
            train_frame_ids=tuple(item.frame_id for item in train),
            holdout_frame_ids=tuple(item.frame_id for item in holdout),
            train_evaluation=train_evaluation,
            holdout_evaluation=holdout_evaluation,
            iterations=tuple(history),
            probes=probes,
            curvature=curvature,
        )


def evaluate_mutual_information(
    observations: Sequence[MutualInformationObservation],
    transform_camera_lidar: SE3,
    options: PandeyMutualInformationOptions | None = None,
) -> MutualInformationEvaluation:
    """Evaluate aggregate Pandey reflectivity/luminance mutual information."""

    solver_options = options or PandeyMutualInformationOptions()
    luminance_parts: list[FloatArray] = []
    reflectivity_parts: list[FloatArray] = []
    for observation in observations:
        luminance, reflectivity = _project_samples(observation, transform_camera_lidar)
        if luminance.size:
            luminance_parts.append(luminance)
            reflectivity_parts.append(reflectivity)
    count = sum(part.size for part in luminance_parts)
    if count < solver_options.min_projected_points:
        return MutualInformationEvaluation(0.0, 0.0, count, len(observations))
    luminance = np.concatenate(luminance_parts)
    reflectivity = np.concatenate(reflectivity_parts)
    raw, normalized = _mutual_information_kde(
        reflectivity, luminance, solver_options.histogram_bins
    )
    return MutualInformationEvaluation(raw, normalized, count, len(observations))


def _project_samples(
    observation: MutualInformationObservation, transform_camera_lidar: SE3
) -> tuple[FloatArray, FloatArray]:
    rotation = _rotation_matrix(transform_camera_lidar)
    translation = np.asarray(transform_camera_lidar.translation_m)
    transformed_points = observation.lidar_points @ rotation.T + translation
    if observation.camera.axes == "forward_x_left_y_up_z":
        camera_points = np.column_stack(
            (-transformed_points[:, 1], -transformed_points[:, 2], transformed_points[:, 0])
        )
    else:
        camera_points = transformed_points
    z = camera_points[:, 2]
    valid = z > 1.0e-6
    points = camera_points[valid]
    reflectivity = observation.reflectivity[valid]
    if not points.size:
        return np.empty(0), np.empty(0)
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    if observation.camera.projection == "opencv_fisheye":
        radius = np.sqrt(x * x + y * y)
        theta = np.arctan(radius)
        k1, k2, k3, k4 = observation.camera.distortion
        theta2 = theta * theta
        theta_distorted = theta * (
            1.0 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))
        )
        scale = np.divide(theta_distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
        x *= scale
        y *= scale
    u = observation.camera.fx * x + observation.camera.cx
    v = observation.camera.fy * y + observation.camera.cy
    inside = (
        (u >= 0.0)
        & (v >= 0.0)
        & (u < observation.camera.width - 1)
        & (v < observation.camera.height - 1)
    )
    u = u[inside]
    v = v[inside]
    reflectivity = reflectivity[inside]
    if not u.size:
        return np.empty(0), np.empty(0)
    left = np.floor(u).astype(int)
    top = np.floor(v).astype(int)
    du = u - left
    dv = v - top
    image = observation.luminance
    luminance = (
        (1.0 - du) * (1.0 - dv) * image[top, left]
        + du * (1.0 - dv) * image[top, left + 1]
        + (1.0 - du) * dv * image[top + 1, left]
        + du * dv * image[top + 1, left + 1]
    )
    return luminance, reflectivity


def _mutual_information_kde(
    reflectivity: FloatArray, luminance: FloatArray, bins: int
) -> tuple[float, float]:
    paired = np.column_stack((_unit_interval(reflectivity), _unit_interval(luminance)))
    histogram = _soft_histogram2d(paired, bins)
    covariance = np.cov(paired, rowvar=False)
    bandwidth = covariance * max(float(paired.shape[0]) ** (-1.0 / 3.0), 1.0e-4)
    bandwidth_bins = bandwidth * float(bins * bins)
    smoothed = _convolve_gaussian(histogram, bandwidth_bins)
    probability = smoothed / max(float(np.sum(smoothed)), 1.0)
    marginal_x = np.sum(probability, axis=1)
    marginal_y = np.sum(probability, axis=0)
    expected = marginal_x[:, None] * marginal_y[None, :]
    valid = (probability > 0.0) & (expected > 0.0)
    mutual_information = float(
        np.sum(probability[valid] * np.log(probability[valid] / expected[valid]))
    )
    entropy_x = _entropy(marginal_x)
    entropy_y = _entropy(marginal_y)
    denominator = math.sqrt(max(entropy_x * entropy_y, 0.0))
    normalized = mutual_information / denominator if denominator > 1.0e-12 else 0.0
    return mutual_information, normalized


def _soft_histogram2d(paired: FloatArray, bins: int) -> FloatArray:
    """Deposit each sample into four adjacent bins for a smooth finite KDE."""

    coordinates = np.clip(paired, 0.0, 1.0) * float(bins - 1)
    lower = np.floor(coordinates).astype(int)
    upper = np.minimum(lower + 1, bins - 1)
    fraction = coordinates - lower
    histogram = np.zeros((bins, bins), dtype=float)
    for choose_x, choose_y in ((0, 0), (0, 1), (1, 0), (1, 1)):
        indices_x = upper[:, 0] if choose_x else lower[:, 0]
        indices_y = upper[:, 1] if choose_y else lower[:, 1]
        weights_x = fraction[:, 0] if choose_x else 1.0 - fraction[:, 0]
        weights_y = fraction[:, 1] if choose_y else 1.0 - fraction[:, 1]
        np.add.at(histogram, (indices_x, indices_y), weights_x * weights_y)
    return histogram


def _convolve_gaussian(histogram: FloatArray, covariance: FloatArray) -> FloatArray:
    covariance = np.asarray(covariance, dtype=float) + np.eye(2) * 0.25
    eigenvalues = np.linalg.eigvalsh(covariance)
    radius = min(
        7,
        max(1, math.ceil(3.0 * math.sqrt(max(float(eigenvalues[-1]), 0.25)))),
    )
    offsets = np.arange(-radius, radius + 1, dtype=float)
    xx, yy = np.meshgrid(offsets, offsets, indexing="ij")
    coordinates = np.stack((xx, yy), axis=-1)
    inverse = np.linalg.pinv(covariance)
    exponent = np.einsum("...i,ij,...j->...", coordinates, inverse, coordinates)
    kernel = np.exp(-0.5 * exponent)
    kernel /= np.sum(kernel)
    padded = np.pad(histogram, radius)
    result = np.zeros_like(histogram)
    for row in range(histogram.shape[0]):
        for column in range(histogram.shape[1]):
            window = padded[row : row + kernel.shape[0], column : column + kernel.shape[1]]
            result[row, column] = float(np.sum(window * kernel))
    return result


def _unit_interval(values: FloatArray) -> FloatArray:
    low, high = np.percentile(values, (1.0, 99.0))
    if high - low <= 1.0e-12:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _entropy(probability: FloatArray) -> float:
    positive = probability[probability > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _numerical_gradient(
    observations: Sequence[MutualInformationObservation],
    parameters: FloatArray,
    options: PandeyMutualInformationOptions,
) -> FloatArray:
    gradient = np.zeros(6, dtype=float)
    for index, increment in enumerate(options.gradient_steps):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += increment
        minus[index] -= increment
        gradient[index] = (
            _raw_score(observations, plus, options) - _raw_score(observations, minus, options)
        ) / (2.0 * increment)
    return gradient


def _raw_score(
    observations: Sequence[MutualInformationObservation],
    parameters: FloatArray,
    options: PandeyMutualInformationOptions,
) -> float:
    return evaluate_mutual_information(
        observations, _transform_from_parameters(parameters), options
    ).raw_mutual_information


def _known_bad_probes(
    observations: Sequence[MutualInformationObservation],
    center: FloatArray,
    options: PandeyMutualInformationOptions,
) -> tuple[PandeyMutualInformationProbe, ...]:
    baseline = evaluate_mutual_information(
        observations, _transform_from_parameters(center), options
    ).normalized_mutual_information
    probes: list[PandeyMutualInformationProbe] = []
    for index, name in enumerate(DOF_NAMES):
        amount = (
            options.known_bad_translation_m
            if index < 3
            else math.radians(options.known_bad_rotation_deg)
        )
        for sign in (-1.0, 1.0):
            candidate = center.copy()
            candidate[index] += sign * amount
            score = evaluate_mutual_information(
                observations, _transform_from_parameters(candidate), options
            ).normalized_mutual_information
            drop = baseline - score
            probes.append(
                PandeyMutualInformationProbe(
                    dof=name,  # type: ignore[arg-type]
                    amount=sign * (amount if index < 3 else options.known_bad_rotation_deg),
                    unit="m" if index < 3 else "deg",
                    normalized_mutual_information=score,
                    score_drop=drop,
                    detectable=drop > 0.0,
                )
            )
    return tuple(probes)


def _transform_from_parameters(parameters: Sequence[float]) -> SE3:
    x, y, z, roll, pitch, yaw = (float(value) for value in parameters)
    rotation = _euler_xyz_matrix(roll, pitch, yaw)
    return SE3(
        (x, y, z),
        quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)),
    )


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


def _empty_result(
    train: Sequence[MutualInformationObservation],
    holdout: Sequence[MutualInformationObservation],
) -> PandeyMutualInformationResult:
    empty = MutualInformationEvaluation(0.0, 0.0, 0, 0)
    return PandeyMutualInformationResult(
        status="insufficient_observations",
        reason="too few synchronized observations remain in the training partition",
        transform_camera_lidar=None,
        train_frame_ids=tuple(item.frame_id for item in train),
        holdout_frame_ids=tuple(item.frame_id for item in holdout),
        train_evaluation=empty,
        holdout_evaluation=empty,
        iterations=(),
        probes=(),
        curvature=None,
    )
