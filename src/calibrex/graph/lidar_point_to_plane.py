"""Native LiDAR rig point-to-plane factor primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from math import cos, isclose, isfinite, sin, sqrt

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, Vector3, rotate_vector_xyzw

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


_NATIVE_OBSERVATION = LidarPointToPlaneObservation
_NATIVE_OBSERVATION_INIT = LidarPointToPlaneObservation.__post_init__
_NATIVE_OBSERVATION_CONSTRUCTOR = LidarPointToPlaneObservation.__init__


class _PreparedObservationBuilder:
    """Reuse immutable native plane conversions and exact normalizations."""

    def __init__(self, points: Sequence[Vector3]) -> None:
        # Bound retained identity keys/references; unsupported points use the constructor.
        captured = points[:65536] if type(points) in (list, tuple) else ()
        self._points = {id(point): point for point in captured
                        if type(point) is tuple and len(point) == 3
                        and all(type(value) is float for value in point)}
        self._planes: dict[tuple[int, int], tuple[Vector3, Vector3, Vector3, Vector3]] = {}

    def build(
        self, point: Vector3, centroid: Vector3, normal: Vector3, pose: SE3,
    ) -> LidarPointToPlaneObservation:
        constructor = _NATIVE_OBSERVATION
        if (
            self._points.get(id(point)) is not point or type(pose) is not SE3
            or LidarPointToPlaneObservation is not _NATIVE_OBSERVATION
            or LidarPointToPlaneObservation.__post_init__ is not _NATIVE_OBSERVATION_INIT
            or LidarPointToPlaneObservation.__init__ is not _NATIVE_OBSERVATION_CONSTRUCTOR
        ):
            return constructor(point, centroid, normal, pose)
        key = (id(centroid), id(normal))
        cached = self._planes.get(key)
        if cached is None or cached[0] is not centroid or cached[1] is not normal:
            if not all(type(vector) is tuple and len(vector) == 3
                       and all(type(value) is float and isfinite(value) for value in vector)
                       for vector in (centroid, normal)):
                return constructor(point, centroid, normal, pose)
            cached = (centroid, normal, _vector3(centroid), _normalize3(normal))
            if len(self._planes) >= 1024:
                self._planes.pop(next(iter(self._planes)))
            self._planes[key] = cached
        observation = object.__new__(constructor)
        object.__setattr__(observation, "point_lidar_m", (point[0], point[1], point[2]))
        object.__setattr__(observation, "plane_point_world_m", cached[2])
        object.__setattr__(observation, "plane_normal_world", cached[3])
        object.__setattr__(observation, "t_world_ego", pose)
        object.__setattr__(observation, "weight", 1.0)
        return observation


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
    normalized_condition_number_estimate: float | None
    normalization_length_m: float
    diagonal_information: dict[str, float]
    normalized_diagonal_information: dict[str, float]
    weak_directions: list[str]
    hessian: list[list[float]]
    normalized_hessian: list[list[float]]
    diagnostic_kind: str = "local_curvature_proxy"


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
        _array_kernel: Callable[[_ObservationArrays, SE3], list[float]] | None = None,
        _array_linearizer: Callable[
            [_ObservationArrays, Sequence[SE3], Sequence[float]],
            tuple[NDArray[np.float64], NDArray[np.float64]],
        ] | None = None,
        _array_normal_kernel: Callable[
            [NDArray[np.float64], NDArray[np.float64], bool, float],
            tuple[list[list[float]], list[float]],
        ] | None = None,
    ) -> None:
        self.variable = variable
        self.t_ego_lidar = t_ego_lidar
        self.observations = tuple(observations)
        self._array_observations: tuple[
            tuple[LidarPointToPlaneObservation, ...], _ObservationArrays | None
        ] | None = None
        self.sensor = sensor
        self._array_kernel = _array_kernel
        self._captured_array_kernel = _array_kernel
        self._array_linearizer = _array_linearizer
        self._array_normal_kernel = _array_normal_kernel
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
        if len(self.observations) >= 64:
            if (
                self._array_observations is None
                or self._array_observations[0] is not self.observations
            ):
                self._array_observations = (
                    self.observations, _prepare_observation_arrays(self.observations)
                )
            arrays = self._array_observations[1]
            if arrays is not None and all(
                np.isfinite((*t_ego_lidar.translation_m, *t_ego_lidar.rotation_quat_xyzw))
            ):
                return (
                    self._array_kernel(arrays, t_ego_lidar) if self._array_kernel is not None
                    else _array_residuals(arrays, t_ego_lidar)
                )
        residuals: list[float] = []
        for observation in self.observations:
            t_world_lidar = observation.t_world_ego.compose(t_ego_lidar)
            point_world = t_world_lidar.transform_point(observation.point_lidar_m)
            delta = _sub3(point_world, observation.plane_point_world_m)
            residual = _dot3(observation.plane_normal_world, delta)
            residuals.append(sqrt(observation.weight) * residual)
        return residuals

    def _retain_observations(
        self, observations: tuple[LidarPointToPlaneObservation, ...], indices: list[int],
    ) -> None:
        original, captured = self.observations, self._array_observations
        self.observations = observations
        self._array_observations = None
        if (
            len(observations) < 64 or captured is None or captured[0] is not original
            or captured[1] is None or not all(type(o) is LidarPointToPlaneObservation
                and type(o.t_world_ego) is SE3 and type(o.weight) in (float, int) for o in original)
        ):
            return
        arrays = captured[1]
        positions = np.asarray(indices, dtype=np.int64)
        selected = [array[positions] for array in (
            arrays.points, arrays.plane_points, arrays.normals, arrays.poses, arrays.sqrt_weights,
        )]
        for array in selected:
            array.flags.writeable = False
        bits = selected[3].view(np.uint64)
        shared = observations[0].t_world_ego if np.all(bits == bits[0]) else None
        self._array_observations = (observations, _ObservationArrays(
            selected[0], selected[1], selected[2], selected[3], selected[4], shared,
        ))

    def _compiled_linearization(
        self, correction: Sequence[float] | None,
        translation_step_m: float = 1.0e-4, rotation_step_rad: float = 1.0e-5,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        if (
            self._array_linearizer is None or self._array_normal_kernel is None
            or len(self.observations) < 64 or type(self) is not LidarRigPointToPlaneFactor
            or self._array_kernel is not self._captured_array_kernel
            or any(name in self.__dict__
                   for name in ("residuals", "linearize", "_corrected_transform"))
            or type(self).residuals is not _NATIVE_RESIDUALS
            or type(self).linearize is not _NATIVE_LINEARIZE
            or type(self)._corrected_transform is not _NATIVE_CORRECTION
        ):
            return None
        if self._array_observations is None or self._array_observations[0] is not self.observations:
            self._array_observations = (
                self.observations, _prepare_observation_arrays(self.observations),
            )
        arrays = self._array_observations[1]
        if arrays is None:
            return None
        base = _correction_vector(correction)
        steps = (translation_step_m,)*3 + (rotation_step_rad,)*3
        transforms = [self._corrected_transform(base)]
        for column, step in enumerate(steps):
            plus, minus = list(base), list(base)
            plus[column] += step
            minus[column] -= step
            transforms.extend((self._corrected_transform(plus), self._corrected_transform(minus)))
        if not all(np.isfinite((*t.translation_m, *t.rotation_quat_xyzw)).all()
                   for t in transforms):
            return None
        residuals, jacobian = self._array_linearizer(arrays, transforms, steps)
        if not np.isfinite(residuals).all() or not np.isfinite(jacobian).all():
            return None
        return residuals, jacobian

    def _compiled_system(
        self, correction: Sequence[float], *, robust_loss: str, huber_delta_m: float,
    ) -> tuple[list[float], list[list[float]], list[float]] | None:
        if not isfinite(huber_delta_m):
            return None
        dense = self._compiled_linearization(correction)
        if dense is None or self._array_normal_kernel is None:
            return None
        residuals, jacobian = dense
        hessian, gradient = self._array_normal_kernel(
            jacobian, residuals, robust_loss != "none", huber_delta_m,
        )
        return residuals.tolist(), hessian, gradient

    def linearize(
        self,
        correction: Sequence[float] | None = None,
        *,
        translation_step_m: float = 1.0e-4,
        rotation_step_rad: float = 1.0e-5,
    ) -> LidarRigPointToPlaneLinearization:
        """Return residuals, numerical Jacobian, Hessian, and gradient."""

        dense = self._compiled_linearization(correction, translation_step_m, rotation_step_rad)
        if dense is not None and self._array_normal_kernel is not None:
            residuals_array, jacobian_array = dense
            hessian, gradient = self._array_normal_kernel(
                jacobian_array, residuals_array, False, 0.0,
            )
            return LidarRigPointToPlaneLinearization(
                self.variable, self.dof_names, residuals_array.tolist(), jacobian_array.tolist(),
                hessian, gradient,
            )

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
        if len(jacobian_rows) >= 64:
            rows = np.asarray(jacobian_rows, dtype=np.float64)
            values = np.asarray(residuals, dtype=np.float64)
            if np.all(np.isfinite(rows)) and np.all(np.isfinite(values)):
                hessian, gradient = _ordered_normal_equations(rows, values)
            else:
                hessian, gradient = _jtj(jacobian_rows), _jtr(jacobian_rows, residuals)
        else:
            hessian, gradient = _jtj(jacobian_rows), _jtr(jacobian_rows, residuals)
        return LidarRigPointToPlaneLinearization(
            variable=self.variable,
            dof_names=self.dof_names,
            residuals_m=residuals,
            jacobian=jacobian_rows,
            hessian=hessian,
            gradient=gradient,
        )

    def analytic_jacobian_at_zero(self) -> list[list[float]]:
        """Return analytic Jacobian rows for the zero left-SE(3) correction.

        The columns match `[x, y, z, roll, pitch, yaw]`. This is intentionally
        scoped to the zero correction used by the current update contract; the
        general rotation-vector derivative away from zero stays numerical.
        """

        rows: list[list[float]] = []
        axes = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        for observation in self.observations:
            sqrt_weight = sqrt(observation.weight)
            point_ego = self.t_ego_lidar.transform_point(observation.point_lidar_m)
            translation_columns = [
                sqrt_weight
                * _dot3(
                    observation.plane_normal_world,
                    rotate_vector_xyzw(observation.t_world_ego.rotation_quat_xyzw, axis),
                )
                for axis in axes
            ]
            rotation_columns = [
                sqrt_weight
                * _dot3(
                    observation.plane_normal_world,
                    rotate_vector_xyzw(
                        observation.t_world_ego.rotation_quat_xyzw,
                        _cross3(axis, point_ego),
                    ),
                )
                for axis in axes
            ]
            rows.append([*translation_columns, *rotation_columns])
        return rows

    def evaluate(
        self,
        correction: Sequence[float] | None = None,
        *,
        weak_information_threshold: float = 1.0e-8,
        normalization_length_m: float | None = None,
    ) -> LidarRigPointToPlaneEvaluation:
        """Return factor-level residual and observability diagnostics."""

        linearization = self.linearize(correction)
        residuals = linearization.residuals_m
        hessian = linearization.hessian
        scale_length = normalization_length_m or _normalization_length_m(self.observations)
        normalized_hessian = _normalize_hessian(hessian, scale_length)
        diagonal = {
            name: hessian[index][index] for index, name in enumerate(linearization.dof_names)
        }
        normalized_diagonal = {
            name: normalized_hessian[index][index]
            for index, name in enumerate(linearization.dof_names)
        }
        weak_directions = [
            name
            for name, value in normalized_diagonal.items()
            if value < weak_information_threshold
        ]
        return LidarRigPointToPlaneEvaluation(
            variable=self.variable,
            residual_count=len(residuals),
            rmse_m=_rmse(residuals),
            rank=_matrix_rank(hessian),
            condition_number_estimate=_diagonal_condition_number(diagonal.values()),
            normalized_condition_number_estimate=_diagonal_condition_number(
                normalized_diagonal.values()
            ),
            normalization_length_m=scale_length,
            diagonal_information=diagonal,
            normalized_diagonal_information=normalized_diagonal,
            weak_directions=weak_directions,
            hessian=hessian,
            normalized_hessian=normalized_hessian,
        )

    def corrected_transform(self, correction: Sequence[float] | None = None) -> SE3:
        """Return `Exp(correction) * T_ego_lidar_initial` using the factor contract."""

        return self._corrected_transform(correction)

    def _corrected_transform(self, correction: Sequence[float] | None) -> SE3:
        correction_vector = _correction_vector(correction)
        delta = _small_se3(correction_vector)
        return delta.compose(self.t_ego_lidar)


_NATIVE_RESIDUALS = LidarRigPointToPlaneFactor.residuals
_NATIVE_LINEARIZE = LidarRigPointToPlaneFactor.linearize
_NATIVE_CORRECTION = LidarRigPointToPlaneFactor._corrected_transform


@dataclass(frozen=True)
class _ObservationArrays:
    points: NDArray[np.float64]
    plane_points: NDArray[np.float64]
    normals: NDArray[np.float64]
    poses: NDArray[np.float64]
    sqrt_weights: NDArray[np.float64]
    shared_pose: SE3 | None


def _prepare_observation_arrays(
    observations: Sequence[LidarPointToPlaneObservation],
) -> _ObservationArrays | None:
    arrays = [
        np.asarray([observation.point_lidar_m for observation in observations], dtype=np.float64),
        np.asarray([
            observation.plane_point_world_m for observation in observations
        ], dtype=np.float64),
        np.asarray([
            observation.plane_normal_world for observation in observations
        ], dtype=np.float64),
        np.asarray([
            (*observation.t_world_ego.translation_m, *observation.t_world_ego.rotation_quat_xyzw)
            for observation in observations
        ], dtype=np.float64),
        np.asarray([observation.weight for observation in observations], dtype=np.float64),
    ]
    if not all(np.all(np.isfinite(array)) for array in arrays):
        return None
    arrays[-1] = np.sqrt(arrays[-1])
    for array in arrays:
        array.flags.writeable = False
    poses = arrays[3].view(np.uint64)
    shared_pose = observations[0].t_world_ego if np.all(poses == poses[0]) else None
    return _ObservationArrays(arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], shared_pose)


def _raw_array_product(
    left: NDArray[np.float64], right: NDArray[np.float64],
) -> NDArray[np.float64]:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), axis=-1)


def _normalize_array_quaternions(quaternions: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z, w = np.moveaxis(quaternions, -1, 0)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    return quaternions / norm[..., None]


def _rotate_array_points(
    quaternions: NDArray[np.float64], points: NDArray[np.float64],
) -> NDArray[np.float64]:
    # Keep both normalizations and both Hamilton products from rotate_vector_xyzw.
    # A rotation matrix changes rounding and hence numerical finite differences.
    normalized = _normalize_array_quaternions(quaternions)
    conjugate = _normalize_array_quaternions(normalized).copy()
    conjugate[..., :3] *= -1.0
    point_quaternions = np.concatenate((
        points, np.zeros((*points.shape[:-1], 1), dtype=np.float64),
    ), axis=-1)
    return _raw_array_product(
        _raw_array_product(normalized, point_quaternions), conjugate,
    )[..., :3]


def _array_residuals(arrays: _ObservationArrays, transform: SE3) -> list[float]:
    translation = np.asarray(transform.translation_m, dtype=np.float64)
    rotation = np.asarray(transform.rotation_quat_xyzw, dtype=np.float64)
    residuals: list[float] = []
    shared = arrays.shared_pose.compose(transform) if arrays.shared_pose is not None else None
    # Bound temporary arrays independently of the observation population.
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, len(arrays.points), 4096):
            stop = start + 4096
            if shared is not None:
                world_translation = np.asarray(shared.translation_m, dtype=np.float64)
                world_rotation = np.asarray(shared.rotation_quat_xyzw, dtype=np.float64)
            else:
                poses = arrays.poses[start:stop]
                world_translation = poses[:, :3] + _rotate_array_points(poses[:, 3:], translation)
                # quaternion_multiply_xyzw and SE3.__post_init__ each normalize.
                world_rotation = _normalize_array_quaternions(_normalize_array_quaternions(
                    _raw_array_product(poses[:, 3:], rotation)
                ))
            world_points = (
                _rotate_array_points(world_rotation, arrays.points[start:stop])
                + world_translation
            )
            delta = world_points - arrays.plane_points[start:stop]
            normals = arrays.normals[start:stop]
            values = (
                normals[:, 0] * delta[:, 0] + normals[:, 1] * delta[:, 1]
                + normals[:, 2] * delta[:, 2]
            )
            residuals.extend((arrays.sqrt_weights[start:stop] * values).tolist())
    return residuals


def _array_world_points(
    points: NDArray[np.float64], poses: NDArray[np.float64], transform: SE3,
) -> NDArray[np.float64]:
    """Match each pose.compose(transform).transform_point with ordered products."""

    translation = np.asarray(transform.translation_m, dtype=np.float64)
    rotation = np.asarray(transform.rotation_quat_xyzw, dtype=np.float64)
    result = np.empty_like(points)
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, len(points), 4096):
            stop = start + 4096
            chunk = poses[start:stop]
            world_translation = chunk[:, :3] + _rotate_array_points(chunk[:, 3:], translation)
            world_rotation = _normalize_array_quaternions(_normalize_array_quaternions(
                _raw_array_product(chunk[:, 3:], rotation),
            ))
            result[start:stop] = (
                _rotate_array_points(world_rotation, points[start:stop]) + world_translation
            )
    return result


def _ordered_normal_equations(
    rows: NDArray[np.float64], residuals: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> tuple[list[list[float]], list[float]]:
    """Accumulate six-column products in the original row and multiply order."""

    hessian = np.zeros((6, 6), dtype=np.float64)
    gradient = np.zeros(6, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, len(rows), 4096):
            stop = start + 4096
            chunk = rows[start:stop]
            weighted = chunk if weights is None else weights[start:stop, None] * chunk
            products = weighted[:, :, None] * chunk[:, None, :]
            hessian = np.add.accumulate(np.concatenate((
                hessian[None, :, :], products,
            )), axis=0)[-1]
            products_gradient = weighted * residuals[start:stop, None]
            gradient = np.add.accumulate(np.concatenate((
                gradient[None, :], products_gradient,
            )), axis=0)[-1]
    return hessian.tolist(), gradient.tolist()


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


def _cross3(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


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


def _normalization_length_m(
    observations: Sequence[LidarPointToPlaneObservation],
) -> float:
    if not observations:
        return 1.0
    mean_squared_radius = sum(
        _dot3(observation.point_lidar_m, observation.point_lidar_m)
        for observation in observations
    ) / len(observations)
    return max(1.0, sqrt(mean_squared_radius))


def _normalize_hessian(hessian: list[list[float]], length_m: float) -> list[list[float]]:
    scales = (length_m, length_m, length_m, 1.0, 1.0, 1.0)
    return [
        [
            value * scales[row_index] * scales[column_index]
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(hessian)
    ]


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
