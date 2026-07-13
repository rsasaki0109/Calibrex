"""Concrete multi-sensor residual blocks for the backend-neutral joint graph."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from calibrex.core.geometry import (
    SE3,
    QuaternionXYZW,
    Vector3,
    quaternion_conjugate_xyzw,
    rotate_vector_xyzw,
)
from calibrex.graph.joint_optimization import (
    JointResidualBlock,
    ParameterValues,
)


@dataclass(frozen=True)
class JointPointToPlaneMeasurement:
    """One world-plane observation coupling a body pose and sensor extrinsic."""

    measurement_id: str
    observation_group: str
    point_sensor_m: Vector3
    plane_point_world_m: Vector3
    plane_normal_world: Vector3
    transform_world_body_initial: SE3
    transform_body_sensor_initial: SE3
    weight: float = 1.0


@dataclass(frozen=True)
class JointDepthPointToPlaneMeasurement:
    """One depth ray/world-plane observation with shared scale and bias."""

    measurement_id: str
    observation_group: str
    normalized_ray_sensor: Vector3
    nominal_depth_m: float
    plane_point_world_m: Vector3
    plane_normal_world: Vector3
    transform_world_body_initial: SE3
    transform_body_sensor_initial: SE3
    weight: float = 1.0


@dataclass(frozen=True)
class JointTrilinearDepthPointToPlaneMeasurement:
    """Depth observation with a shared trilinear ray-depth displacement field."""

    measurement_id: str
    observation_group: str
    normalized_ray_sensor: Vector3
    nominal_depth_m: float
    lattice_weights: tuple[float, ...]
    plane_point_world_m: Vector3
    plane_normal_world: Vector3
    transform_world_body_initial: SE3
    transform_body_sensor_initial: SE3
    weight: float = 1.0


@dataclass(frozen=True)
class JointRadarDopplerMeasurement:
    """One static-target Doppler constraint coupling velocity/extrinsic/time."""

    measurement_id: str
    observation_group: str
    line_of_sight_radar: Vector3
    measured_radial_velocity_mps: float
    linear_acceleration_body_mps2: Vector3 = (0.0, 0.0, 0.0)
    angular_velocity_body_radps: Vector3 = (0.0, 0.0, 0.0)
    rotation_body_radar_initial: QuaternionXYZW = (0.0, 0.0, 0.0, 1.0)
    translation_body_radar_initial_m: Vector3 = (0.0, 0.0, 0.0)
    weight: float = 1.0


def make_joint_point_to_plane_factor(
    measurement: JointPointToPlaneMeasurement,
    *,
    pose_block: str,
    extrinsic_block: str,
) -> JointResidualBlock:
    """Create `nᵀ(T_world_body T_body_sensor p-q)` as a joint factor."""

    normal = _normalize(measurement.plane_normal_world)

    def evaluator(values: ParameterValues) -> tuple[float]:
        pose_delta = se3_from_tangent(values[pose_block])
        extrinsic_delta = se3_from_tangent(values[extrinsic_block])
        transform_world_sensor = pose_delta.compose(
            measurement.transform_world_body_initial
        ).compose(extrinsic_delta.compose(measurement.transform_body_sensor_initial))
        point_world = transform_world_sensor.transform_point(measurement.point_sensor_m)
        difference = _subtract(point_world, measurement.plane_point_world_m)
        return (_dot(normal, difference),)

    return JointResidualBlock(
        factor_id=measurement.measurement_id,
        observation_group=measurement.observation_group,
        variable_names=(pose_block, extrinsic_block),
        evaluator=evaluator,
        weight=measurement.weight,
        family="lidar_point_to_plane",
    )


def make_joint_depth_point_to_plane_factor(
    measurement: JointDepthPointToPlaneMeasurement,
    *,
    pose_block: str,
    extrinsic_block: str,
    depth_calibration_block: str,
) -> JointResidualBlock:
    """Create a pose/extrinsic/depth factor with `z'=exp(s)z+b`."""

    normal = _normalize(measurement.plane_normal_world)
    ray = measurement.normalized_ray_sensor
    if measurement.nominal_depth_m <= 0.0 or abs(ray[2] - 1.0) > 1.0e-9:
        raise ValueError("joint depth factor requires positive depth and ray z=1")

    def evaluator(values: ParameterValues) -> tuple[float]:
        depth_values = tuple(values[depth_calibration_block])
        if len(depth_values) != 2:
            raise ValueError("joint depth calibration block requires log-scale and bias")
        corrected_depth = math.exp(depth_values[0]) * measurement.nominal_depth_m + depth_values[1]
        point_sensor = _scale(ray, corrected_depth)
        pose_delta = se3_from_tangent(values[pose_block])
        extrinsic_delta = se3_from_tangent(values[extrinsic_block])
        transform_world_sensor = pose_delta.compose(
            measurement.transform_world_body_initial
        ).compose(extrinsic_delta.compose(measurement.transform_body_sensor_initial))
        point_world = transform_world_sensor.transform_point(point_sensor)
        return (_dot(normal, _subtract(point_world, measurement.plane_point_world_m)),)

    return JointResidualBlock(
        factor_id=measurement.measurement_id,
        observation_group=measurement.observation_group,
        variable_names=(pose_block, extrinsic_block, depth_calibration_block),
        evaluator=evaluator,
        weight=measurement.weight,
        family="rgbd_depth_point_to_plane",
    )


def make_joint_trilinear_depth_point_to_plane_factor(
    measurement: JointTrilinearDepthPointToPlaneMeasurement,
    *,
    pose_block: str,
    extrinsic_block: str,
    depth_lattice_block: str,
) -> JointResidualBlock:
    """Create a shared-lattice depth factor with `z'=z+Σγ_l δz_l`."""

    normal = _normalize(measurement.plane_normal_world)
    ray = measurement.normalized_ray_sensor
    weights = measurement.lattice_weights
    if measurement.nominal_depth_m <= 0.0 or abs(ray[2] - 1.0) > 1.0e-9:
        raise ValueError("joint trilinear depth factor requires positive depth and ray z=1")
    if not weights or any(weight < 0.0 for weight in weights):
        raise ValueError("joint trilinear depth factor requires nonnegative lattice weights")
    if abs(sum(weights) - 1.0) > 1.0e-9:
        raise ValueError("joint trilinear depth lattice weights must sum to one")

    def evaluator(values: ParameterValues) -> tuple[float]:
        offsets = values[depth_lattice_block]
        if len(offsets) != len(weights):
            raise ValueError("joint trilinear depth lattice block dimension mismatch")
        displacement = sum(weight * offset for weight, offset in zip(weights, offsets, strict=True))
        corrected_depth = measurement.nominal_depth_m + displacement
        point_sensor = _scale(ray, corrected_depth)
        pose_delta = se3_from_tangent(values[pose_block])
        extrinsic_delta = se3_from_tangent(values[extrinsic_block])
        transform_world_sensor = pose_delta.compose(
            measurement.transform_world_body_initial
        ).compose(extrinsic_delta.compose(measurement.transform_body_sensor_initial))
        point_world = transform_world_sensor.transform_point(point_sensor)
        return (_dot(normal, _subtract(point_world, measurement.plane_point_world_m)),)

    return JointResidualBlock(
        factor_id=measurement.measurement_id,
        observation_group=measurement.observation_group,
        variable_names=(pose_block, extrinsic_block, depth_lattice_block),
        evaluator=evaluator,
        weight=measurement.weight,
        family="rgbd_trilinear_depth_point_to_plane",
    )


def make_joint_lattice_smoothness_factor(
    *,
    factor_id: str,
    observation_group: str,
    block: str,
    shape: tuple[int, int, int],
    sigma_m: float,
) -> JointResidualBlock:
    """Penalize neighboring scalar lattice-offset differences."""

    nx, ny, nz = shape
    if min(shape) < 2 or sigma_m <= 0.0:
        raise ValueError("lattice shape dimensions and sigma must be at least two/positive")
    edges: list[tuple[int, int]] = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                current = _lattice_index(ix, iy, iz, shape)
                if ix + 1 < nx:
                    edges.append((current, _lattice_index(ix + 1, iy, iz, shape)))
                if iy + 1 < ny:
                    edges.append((current, _lattice_index(ix, iy + 1, iz, shape)))
                if iz + 1 < nz:
                    edges.append((current, _lattice_index(ix, iy, iz + 1, shape)))

    def evaluator(values: ParameterValues) -> tuple[float, ...]:
        offsets = values[block]
        if len(offsets) != nx * ny * nz:
            raise ValueError("joint lattice smoothness block dimension mismatch")
        return tuple((offsets[right] - offsets[left]) / sigma_m for left, right in edges)

    return JointResidualBlock(
        factor_id=factor_id,
        observation_group=observation_group,
        variable_names=(block,),
        evaluator=evaluator,
        family="trilinear_lattice_smoothness",
        split_policy="train_only",
    )


def trilinear_lattice_weights(
    point: Vector3,
    *,
    minimum: Vector3,
    maximum: Vector3,
    shape: tuple[int, int, int],
) -> tuple[float, ...]:
    """Return dense control weights for a bounded regular 3D lattice."""

    if min(shape) < 2 or any(upper <= lower for lower, upper in zip(minimum, maximum, strict=True)):
        raise ValueError("lattice bounds must increase and shape dimensions must be >= 2")
    cells: list[int] = []
    fractions: list[float] = []
    for coordinate, lower, upper, count in zip(point, minimum, maximum, shape, strict=True):
        if coordinate < lower or coordinate > upper:
            raise ValueError("point lies outside trilinear lattice bounds")
        scaled = (coordinate - lower) * (count - 1) / (upper - lower)
        cell = min(math.floor(scaled), count - 2)
        cells.append(cell)
        fractions.append(scaled - cell)
    weights = [0.0] * math.prod(shape)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                weight = (
                    (fractions[0] if dx else 1.0 - fractions[0])
                    * (fractions[1] if dy else 1.0 - fractions[1])
                    * (fractions[2] if dz else 1.0 - fractions[2])
                )
                index = _lattice_index(cells[0] + dx, cells[1] + dy, cells[2] + dz, shape)
                weights[index] = weight
    return tuple(weights)


def make_joint_radar_doppler_factor(
    measurement: JointRadarDopplerMeasurement,
    *,
    velocity_block: str,
    extrinsic_block: str,
    time_offset_block: str,
) -> JointResidualBlock:
    """Create a Radar Doppler factor with lever-arm and clock-offset terms."""

    line_of_sight = _normalize(measurement.line_of_sight_radar)

    def evaluator(values: ParameterValues) -> tuple[float]:
        velocity = _vector3(values[velocity_block])
        extrinsic_delta = se3_from_tangent(values[extrinsic_block])
        initial = SE3(
            measurement.translation_body_radar_initial_m,
            measurement.rotation_body_radar_initial,
        )
        transform_body_radar = extrinsic_delta.compose(initial)
        offset = _scalar(values[time_offset_block])
        body_velocity = _add(
            velocity,
            _scale(measurement.linear_acceleration_body_mps2, offset),
        )
        radar_origin_velocity = _add(
            body_velocity,
            _cross(
                measurement.angular_velocity_body_radps,
                transform_body_radar.translation_m,
            ),
        )
        velocity_radar = rotate_vector_xyzw(
            quaternion_conjugate_xyzw(transform_body_radar.rotation_quat_xyzw),
            radar_origin_velocity,
        )
        predicted = -_dot(line_of_sight, velocity_radar)
        return (measurement.measured_radial_velocity_mps - predicted,)

    return JointResidualBlock(
        factor_id=measurement.measurement_id,
        observation_group=measurement.observation_group,
        variable_names=(velocity_block, extrinsic_block, time_offset_block),
        evaluator=evaluator,
        weight=measurement.weight,
        family="radar_doppler_spatiotemporal",
    )


def make_joint_prior_factor(
    *,
    factor_id: str,
    observation_group: str,
    block: str,
    target: Sequence[float],
    sigma: Sequence[float],
) -> JointResidualBlock:
    """Create a diagonal Gaussian tangent prior as ordinary residuals."""

    targets = tuple(float(value) for value in target)
    sigmas = tuple(float(value) for value in sigma)
    if not targets or len(targets) != len(sigmas) or any(value <= 0.0 for value in sigmas):
        raise ValueError(
            "joint prior target/sigma dimensions must match and sigma must be positive"
        )

    def evaluator(values: ParameterValues) -> tuple[float, ...]:
        current = values[block]
        if len(current) != len(targets):
            raise ValueError("joint prior block dimension mismatch")
        return tuple(
            (value - expected) / scale
            for value, expected, scale in zip(current, targets, sigmas, strict=True)
        )

    return JointResidualBlock(
        factor_id,
        observation_group,
        (block,),
        evaluator,
        family="diagonal_prior",
        split_policy="train_only",
    )


def se3_from_tangent(values: Sequence[float]) -> SE3:
    """Decode `[x,y,z,roll,pitch,yaw]` as a left SE(3) tangent update."""

    vector = tuple(float(value) for value in values)
    if len(vector) != 6:
        raise ValueError("SE3 tangent block requires six values")
    tx, ty, tz, rx, ry, rz = vector
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1.0e-15:
        quaternion = (0.0, 0.0, 0.0, 1.0)
    else:
        scale = math.sin(0.5 * angle) / angle
        quaternion = (rx * scale, ry * scale, rz * scale, math.cos(0.5 * angle))
    return SE3((tx, ty, tz), quaternion)


def _vector3(values: Sequence[float]) -> Vector3:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError("expected three values")
    return (vector[0], vector[1], vector[2])


def _scalar(values: Sequence[float]) -> float:
    if len(values) != 1:
        raise ValueError("expected one scalar value")
    return float(values[0])


def _normalize(value: Vector3) -> Vector3:
    norm = math.sqrt(_dot(value, value))
    if norm < 1.0e-15:
        raise ValueError("direction vector must be non-zero")
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def _dot(left: Vector3, right: Vector3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _scale(value: Vector3, amount: float) -> Vector3:
    return (value[0] * amount, value[1] * amount, value[2] * amount)


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _lattice_index(ix: int, iy: int, iz: int, shape: tuple[int, int, int]) -> int:
    nx, ny, _nz = shape
    return ix + nx * (iy + ny * iz)
