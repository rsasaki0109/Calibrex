"""Native LiDAR rig point-to-plane factor primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import cos, isclose, sin, sqrt

from calibrex.core.geometry import SE3, Vector3

_DEFAULT_DOF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class LidarPointToPlaneObservation:
    """One point-to-plane correspondence for a fixed-rig LiDAR factor."""

    point_lidar_m: Vector3
    plane_point_world_m: Vector3
    plane_normal_world: Vector3
    t_world_ego: SE3 = field(default_factory=SE3.identity)
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_lidar_m", _vector3(self.point_lidar_m))
        object.__setattr__(
            self,
            "plane_point_world_m",
            _vector3(self.plane_point_world_m),
        )
        object.__setattr__(
            self,
            "plane_normal_world",
            _normalize3(self.plane_normal_world),
        )
        if self.weight < 0.0:
            msg = "observation weight must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class LidarRigPointToPlaneLinearization:
    """Residual/Jacobian linearization for `LidarRigPointToPlaneFactor`."""

    variable: str
    dof_names: tuple[str, str, str, str, str, str]
    residuals_m: list[float]
    jacobian: list[list[float]]
    hessian: list[list[float]]
    gradient: list[float]


@dataclass(frozen=True)
class LidarRigPointToPlaneEvaluation:
    """Numerical diagnostics for a LiDAR rig point-to-plane factor."""

    variable: str
    residual_count: int
    rmse_m: float | None
    rank: int
    condition_number_estimate: float | None
    diagonal_information: dict[str, float]
    weak_directions: list[str]
    hessian: list[list[float]]


class LidarRigPointToPlaneFactor:
    """Point-to-plane factor for a fixed-mounted LiDAR extrinsic.

    The factor evaluates residuals of the form:

    `r = n_world^T (T_world_ego * T_ego_lidar(xi) * p_lidar - q_world)`.

    Corrections use a left perturbation convention:
    `T_ego_lidar(xi) = Exp(xi) * T_ego_lidar_initial`, with the DoF order
    `[x, y, z, roll, pitch, yaw]`.
    """

    def __init__(
        self,
        *,
        variable: str,
        t_ego_lidar: SE3,
        observations: Sequence[LidarPointToPlaneObservation],
        sensor: str = "lidar0",
    ) -> None:
        self.variable = variable
        self.t_ego_lidar = t_ego_lidar
        self.observations = tuple(observations)
        self.sensor = sensor
        self.dof_names: tuple[str, str, str, str, str, str] = (
            f"{_DEFAULT_DOF_NAMES[0]}_{sensor}",
            f"{_DEFAULT_DOF_NAMES[1]}_{sensor}",
            f"{_DEFAULT_DOF_NAMES[2]}_{sensor}",
            f"{_DEFAULT_DOF_NAMES[3]}_{sensor}",
            f"{_DEFAULT_DOF_NAMES[4]}_{sensor}",
            f"{_DEFAULT_DOF_NAMES[5]}_{sensor}",
        )

    def residuals(self, correction: Sequence[float] | None = None) -> list[float]:
        """Return weighted point-to-plane residuals in meters."""

        t_ego_lidar = self._corrected_transform(correction)
        residuals: list[float] = []
        for observation in self.observations:
            t_world_lidar = observation.t_world_ego.compose(t_ego_lidar)
            point_world = t_world_lidar.transform_point(observation.point_lidar_m)
            delta = _sub3(point_world, observation.plane_point_world_m)
            residual = _dot3(observation.plane_normal_world, delta)
            residuals.append(sqrt(observation.weight) * residual)
        return residuals

    def linearize(
        self,
        correction: Sequence[float] | None = None,
        *,
        translation_step_m: float = 1.0e-4,
        rotation_step_rad: float = 1.0e-5,
    ) -> LidarRigPointToPlaneLinearization:
        """Return residuals, numerical Jacobian, Hessian, and gradient."""

        base = _correction_vector(correction)
        residuals = self.residuals(base)
        jacobian: list[list[float]] = []
        steps = (
            translation_step_m,
            translation_step_m,
            translation_step_m,
            rotation_step_rad,
            rotation_step_rad,
            rotation_step_rad,
        )
        for column, step in enumerate(steps):
            plus = list(base)
            minus = list(base)
            plus[column] += step
            minus[column] -= step
            plus_residuals = self.residuals(plus)
            minus_residuals = self.residuals(minus)
            jacobian.append(
                [
                    (plus_residuals[index] - minus_residuals[index]) / (2.0 * step)
                    for index in range(len(residuals))
                ]
            )
        jacobian_rows = _transpose(jacobian)
        hessian = _jtj(jacobian_rows)
        gradient = _jtr(jacobian_rows, residuals)
        return LidarRigPointToPlaneLinearization(
            variable=self.variable,
            dof_names=self.dof_names,
            residuals_m=residuals,
            jacobian=jacobian_rows,
            hessian=hessian,
            gradient=gradient,
        )

    def evaluate(
        self,
        correction: Sequence[float] | None = None,
        *,
        weak_information_threshold: float = 1.0e-8,
    ) -> LidarRigPointToPlaneEvaluation:
        """Return factor-level residual and observability diagnostics."""

        linearization = self.linearize(correction)
        residuals = linearization.residuals_m
        hessian = linearization.hessian
        diagonal = {
            name: hessian[index][index] for index, name in enumerate(linearization.dof_names)
        }
        weak_directions = [
            name for name, value in diagonal.items() if value < weak_information_threshold
        ]
        return LidarRigPointToPlaneEvaluation(
            variable=self.variable,
            residual_count=len(residuals),
            rmse_m=_rmse(residuals),
            rank=_matrix_rank(hessian),
            condition_number_estimate=_diagonal_condition_number(diagonal.values()),
            diagonal_information=diagonal,
            weak_directions=weak_directions,
            hessian=hessian,
        )

    def corrected_transform(self, correction: Sequence[float] | None = None) -> SE3:
        """Return `Exp(correction) * T_ego_lidar_initial` using the factor contract."""

        return self._corrected_transform(correction)

    def _corrected_transform(self, correction: Sequence[float] | None) -> SE3:
        correction_vector = _correction_vector(correction)
        delta = _small_se3(correction_vector)
        return delta.compose(self.t_ego_lidar)


def _correction_vector(correction: Sequence[float] | None) -> tuple[float, ...]:
    if correction is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = tuple(float(value) for value in correction)
    if len(values) != 6:
        msg = "LiDAR rig point-to-plane correction must have 6 values"
        raise ValueError(msg)
    return values


def _small_se3(correction: Sequence[float]) -> SE3:
    tx, ty, tz, rx, ry, rz = correction
    return SE3((tx, ty, tz), _rotation_vector_to_quaternion(rx, ry, rz))


def _rotation_vector_to_quaternion(
    rx: float,
    ry: float,
    rz: float,
) -> tuple[float, float, float, float]:
    angle = sqrt(rx * rx + ry * ry + rz * rz)
    if isclose(angle, 0.0):
        return (0.0, 0.0, 0.0, 1.0)
    half_angle = 0.5 * angle
    scale = sin(half_angle) / angle
    return (rx * scale, ry * scale, rz * scale, cos(half_angle))


def _vector3(values: Iterable[float]) -> Vector3:
    data = tuple(float(value) for value in values)
    if len(data) != 3:
        msg = "expected exactly 3 values"
        raise ValueError(msg)
    return (data[0], data[1], data[2])


def _normalize3(values: Iterable[float]) -> Vector3:
    x, y, z = _vector3(values)
    norm = sqrt(x * x + y * y + z * z)
    if isclose(norm, 0.0):
        msg = "plane normal must be non-zero"
        raise ValueError(msg)
    return (x / norm, y / norm, z / norm)


def _sub3(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _dot3(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _transpose(columns: list[list[float]]) -> list[list[float]]:
    if not columns:
        return []
    row_count = len(columns[0])
    return [[column[row] for column in columns] for row in range(row_count)]


def _jtj(jacobian: list[list[float]]) -> list[list[float]]:
    size = 6
    hessian = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in jacobian:
        for left in range(size):
            for right in range(size):
                hessian[left][right] += row[left] * row[right]
    return hessian


def _jtr(jacobian: list[list[float]], residuals: list[float]) -> list[float]:
    gradient = [0.0 for _ in range(6)]
    for row, residual in zip(jacobian, residuals, strict=True):
        for column in range(6):
            gradient[column] += row[column] * residual
    return gradient


def _rmse(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sqrt(sum(value * value for value in values) / len(values))


def _matrix_rank(matrix: list[list[float]]) -> int:
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    row_count = len(rows)
    column_count = len(rows[0])
    tolerance = max(1.0, max(abs(value) for row in rows for value in row)) * 1.0e-9
    rank = 0
    for column in range(column_count):
        pivot = max(range(rank, row_count), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) <= tolerance:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(row_count):
            if row == rank:
                continue
            scale = rows[row][column]
            rows[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(rows[row], rows[rank], strict=True)
            ]
        rank += 1
        if rank == row_count:
            break
    return rank


def _diagonal_condition_number(values: Iterable[float]) -> float | None:
    positive_values = [value for value in values if value > 0.0]
    if not positive_values:
        return None
    minimum = min(positive_values)
    if isclose(minimum, 0.0):
        return None
    return max(positive_values) / minimum
