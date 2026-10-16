"""Ordered array evaluation of pure, factory-owned point-plane factors."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from types import CellType, CodeType, FunctionType
from typing import TYPE_CHECKING, TypeAlias
from weakref import ReferenceType, ref

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import (
    SE3,
    Vector3,
    normalize_quaternion_xyzw,
    quaternion_conjugate_xyzw,
)
from calibrex.graph._evaluation_scope import _PREPARATION_CACHE
from calibrex.graph.joint_optimization import (
    _BUILTIN_RESIDUALS_METHOD,
    JointOptimizerOptions,
    JointParameterBlock,
    JointResidualBlock,
    ParameterValues,
    ResidualEvaluator,
    SquareRootInformation,
)

if TYPE_CHECKING:
    from calibrex.graph.joint_factors import JointPointToPlaneMeasurement

_MIN_POINTS = 64
_MAX_POINTS = 65536
_MAX_GROUPS = 128
_CHUNK_ROWS = 4096
_PLAN_ENTRIES = 2
_PLAN_FACTOR_LIMIT = 65536
_ATTRIBUTE = "_calibrex_joint_builtin"
_SE3_POINT_METHOD = SE3.transform_point
_FloatArray: TypeAlias = NDArray[np.float64]
_IndexArray: TypeAlias = NDArray[np.intp]


@dataclass(frozen=True)
class _Owner:
    reference: ReferenceType[object]
    code: CodeType
    closure: tuple[object, ...]


@dataclass(frozen=True)
class _PointBinding:
    owner: _Owner
    measurement: JointPointToPlaneMeasurement
    normal: Vector3
    pose: str
    extrinsic: str


@dataclass(frozen=True)
class _PriorBinding:
    owner: _Owner
    block: str
    dimension: int


def _owner(evaluator: ResidualEvaluator) -> _Owner:
    assert isinstance(evaluator, FunctionType)
    return _Owner(
        ref(evaluator),
        evaluator.__code__,
        tuple(cell.cell_contents for cell in evaluator.__closure__ or ()),
    )


def _bind_point_plane(
    evaluator: ResidualEvaluator,
    measurement: JointPointToPlaneMeasurement,
    normal: Vector3,
    pose: str,
    extrinsic: str,
) -> None:
    setattr(
        evaluator,
        _ATTRIBUTE,
        _PointBinding(_owner(evaluator), measurement, normal, pose, extrinsic),
    )


def _bind_prior(evaluator: ResidualEvaluator, block: str, dimension: int) -> None:
    setattr(evaluator, _ATTRIBUTE, _PriorBinding(_owner(evaluator), block, dimension))


def _binding(factor: JointResidualBlock) -> _PointBinding | _PriorBinding | None:
    if type(factor) is not JointResidualBlock or type(factor.evaluator) is not FunctionType:
        return None
    value = getattr(factor.evaluator, _ATTRIBUTE, None)
    if not isinstance(value, (_PointBinding, _PriorBinding)):
        return None
    owner = value.owner
    closure = factor.evaluator.__closure__ or ()
    try:
        closed_values = tuple(cell.cell_contents for cell in closure)
    except ValueError:
        return None
    if (
        owner.reference() is not factor.evaluator
        or owner.code is not factor.evaluator.__code__
        or len(closure) != len(owner.closure)
        or any(
            value is not original
            for value, original in zip(closed_values, owner.closure, strict=True)
        )
    ):
        return None
    return value


def _ordinary(value: object) -> bool:
    return type(value) is float and math.isfinite(value) and abs(value) <= 1.0e100


def _vector(value: object, dimension: int) -> bool:
    return type(value) is tuple and len(value) == dimension and all(_ordinary(v) for v in value)


def _values_supported(values: ParameterValues) -> bool:
    return type(values) is dict and all(
        type(name) is str and type(vector) is tuple and all(_ordinary(v) for v in vector)
        for name, vector in values.items()
    )


def _transform_points(points: _FloatArray, transform: SE3) -> _FloatArray:
    # Two Hamilton products retain the scalar core's zero terms and rounding.
    x, y, z = points.T
    qx, qy, qz, qw = normalize_quaternion_xyzw(transform.rotation_quat_xyzw)
    cx, cy, cz, cw = quaternion_conjugate_xyzw((qx, qy, qz, qw))
    ax = qw * x + qx * 0.0 + qy * z - qz * y
    ay = qw * y - qx * z + qy * 0.0 + qz * x
    az = qw * z + qx * y - qy * x + qz * 0.0
    aw = qw * 0.0 - qx * x - qy * y - qz * z
    tx, ty, tz = transform.translation_m
    return np.column_stack(
        (
            aw * cx + ax * cw + ay * cz - az * cy + tx,
            aw * cy - ax * cz + ay * cw + az * cx + ty,
            aw * cz + ax * cy - ay * cx + az * cw + tz,
        )
    )


@dataclass(frozen=True)
class _PointGroup:
    binding: _PointBinding
    rows: _IndexArray
    points: _FloatArray
    planes: _FloatArray
    normals: _FloatArray
    sqrt_weights: _FloatArray


@dataclass(frozen=True)
class _PointPlanePlan:
    size: int
    groups: tuple[_PointGroup, ...]
    priors: tuple[tuple[int, int, JointResidualBlock], ...]

    def accepts(self, values: ParameterValues) -> bool:
        return all(
            name in values and len(values[name]) == 6
            for group in self.groups
            for name in (group.binding.pose, group.binding.extrinsic)
        ) and all(
            factor.variable_names[0] in values
            and len(values[factor.variable_names[0]]) == dimension
            for _, dimension, factor in self.priors
        )

    def evaluate(self, values: ParameterValues, active: str | None = None) -> _FloatArray:
        from calibrex.graph.joint_factors import se3_from_tangent

        result = np.zeros(self.size, dtype=np.float64)
        for group in self.groups:
            binding = group.binding
            if active is not None and active not in (binding.pose, binding.extrinsic):
                continue
            measurement = binding.measurement
            pose_delta = se3_from_tangent(values[binding.pose])
            extrinsic_delta = se3_from_tangent(values[binding.extrinsic])
            transform = pose_delta.compose(measurement.transform_world_body_initial).compose(
                extrinsic_delta.compose(measurement.transform_body_sensor_initial)
            )
            with np.errstate(all="ignore"):
                for start in range(0, len(group.rows), _CHUNK_ROWS):
                    stop = start + _CHUNK_ROWS
                    difference = (
                        _transform_points(group.points[start:stop], transform)
                        - group.planes[start:stop]
                    )
                    normal = group.normals[start:stop]
                    raw = (
                        normal[:, 0] * difference[:, 0]
                        + normal[:, 1] * difference[:, 1]
                        + normal[:, 2] * difference[:, 2]
                    )
                    result[group.rows[start:stop]] = group.sqrt_weights[start:stop] * raw
        for row, dimension, factor in self.priors:
            if active is None or active in factor.variable_names:
                residuals = factor.residuals(values)
                if len(residuals) != dimension:
                    raise ValueError("joint factor residual dimension changed during linearization")
                result[row : row + dimension] = residuals
        if not np.isfinite(result).all():
            raise ValueError("joint factor residuals must be non-empty and finite")
        return result

    def affected(self, name: str) -> NDArray[np.bool_]:
        mask = np.zeros(self.size, dtype=np.bool_)
        for group in self.groups:
            if name in (group.binding.pose, group.binding.extrinsic):
                mask[group.rows] = True
        for row, dimension, factor in self.priors:
            if name in factor.variable_names:
                mask[row : row + dimension] = True
        return mask

    def linearize(
        self,
        values: dict[str, tuple[float, ...]],
        blocks: Sequence[JointParameterBlock],
        layout: dict[str, slice],
        options: JointOptimizerOptions,
    ) -> tuple[_FloatArray, _FloatArray]:
        from calibrex.graph.joint_optimization import _perturb

        baseline = self.evaluate(values)
        jacobian = np.zeros((self.size, sum(s.stop - s.start for s in layout.values())))
        block_by_name = {block.name: block for block in blocks}
        for name, block_slice in layout.items():
            block = block_by_name[name]
            affected = self.affected(name)
            for index in range(block.dimension):
                step = (
                    block.finite_difference_steps[index]
                    if block.finite_difference_steps
                    else options.default_finite_difference_step
                )
                plus = self.evaluate(_perturb(values, name, index, step), name)
                minus = self.evaluate(_perturb(values, name, index, -step), name)
                jacobian[affected, block_slice.start + index] = (
                    plus[affected] - minus[affected]
                ) / (2.0 * step)
        return baseline, jacobian


@dataclass(frozen=True)
class _PointState:
    measurement: JointPointToPlaneMeasurement
    measurement_type: type[object]
    point: Vector3
    plane: Vector3
    world: SE3
    sensor: SE3
    world_translation: Vector3
    world_rotation: tuple[float, float, float, float]
    sensor_translation: Vector3
    sensor_rotation: tuple[float, float, float, float]
    normal: Vector3
    pose: str
    extrinsic: str

    def matches(self, binding: _PointBinding) -> bool:
        measurement = self.measurement
        return (
            binding.measurement is measurement
            and type(measurement) is self.measurement_type
            and measurement.point_sensor_m is self.point
            and measurement.plane_point_world_m is self.plane
            and measurement.transform_world_body_initial is self.world
            and measurement.transform_body_sensor_initial is self.sensor
            and type(self.world) is SE3
            and type(self.sensor) is SE3
            and self.world.translation_m is self.world_translation
            and self.world.rotation_quat_xyzw is self.world_rotation
            and self.sensor.translation_m is self.sensor_translation
            and self.sensor.rotation_quat_xyzw is self.sensor_rotation
            and binding.normal is self.normal
            and binding.pose is self.pose
            and binding.extrinsic is self.extrinsic
        )


@dataclass(frozen=True)
class _FactorState:
    factor: JointResidualBlock
    evaluator: FunctionType
    binding: _PointBinding | _PriorBinding
    owner: _Owner
    reference: ReferenceType[object]
    code: CodeType
    cells: tuple[CellType, ...]
    closed: tuple[object, ...]
    weight: float
    names: tuple[str, ...]
    information: SquareRootInformation
    point: _PointState | None
    prior: tuple[str, int] | None

    def matches(self, factor: JointResidualBlock) -> bool:
        if (
            factor is not self.factor
            or type(factor) is not JointResidualBlock
            or factor.evaluator is not self.evaluator
            or factor.weight is not self.weight
            or factor.variable_names is not self.names
            or factor.sqrt_information is not self.information
        ):
            return False
        evaluator, binding, owner = self.evaluator, self.binding, self.owner
        if (
            getattr(evaluator, _ATTRIBUTE, None) is not binding
            or evaluator.__code__ is not self.code
            or (evaluator.__closure__ or ()) is not self.cells
            or binding.owner is not owner
            or owner.reference is not self.reference
            or owner.code is not self.code
            or owner.closure is not self.closed
        ):
            return False
        cells, closed = self.cells, self.closed
        try:
            if len(cells) == 6:
                # Native point factories share this closure layout. Avoid an
                # allocated generator for every point on every LM evaluation.
                if (
                    cells[0].cell_contents is not closed[0]
                    or cells[1].cell_contents is not closed[1]
                    or cells[2].cell_contents is not closed[2]
                    or cells[3].cell_contents is not closed[3]
                    or cells[4].cell_contents is not closed[4]
                    or cells[5].cell_contents is not closed[5]
                ):
                    return False
            elif any(
                cell.cell_contents is not value for cell, value in zip(cells, closed, strict=True)
            ):
                return False
        except ValueError:
            return False
        if self.point is not None and type(binding) is _PointBinding:
            return self.point.matches(binding)
        return (
            type(binding) is _PriorBinding
            and self.prior is not None
            and binding.block is self.prior[0]
            and binding.dimension is self.prior[1]
        )


def _factor_state(factor: JointResidualBlock) -> _FactorState | None:
    binding = _binding(factor)
    if type(binding) not in (_PointBinding, _PriorBinding):
        return None
    assert isinstance(binding, (_PointBinding, _PriorBinding))
    evaluator = factor.evaluator
    assert isinstance(evaluator, FunctionType)
    point = None
    prior = None
    if isinstance(binding, _PointBinding):
        measurement = binding.measurement
        world, sensor = (
            measurement.transform_world_body_initial,
            measurement.transform_body_sensor_initial,
        )
        if not (
            _vector(world.translation_m, 3)
            and _vector(world.rotation_quat_xyzw, 4)
            and _vector(sensor.translation_m, 3)
            and _vector(sensor.rotation_quat_xyzw, 4)
        ):
            # Frozen SE3 normally owns plain tuples. A caller can deliberately
            # bypass freezing; mutable/custom leaves must be prepared afresh.
            return None
        point = _PointState(
            measurement,
            type(measurement),
            measurement.point_sensor_m,
            measurement.plane_point_world_m,
            world,
            sensor,
            world.translation_m,
            world.rotation_quat_xyzw,
            sensor.translation_m,
            sensor.rotation_quat_xyzw,
            binding.normal,
            binding.pose,
            binding.extrinsic,
        )
    else:
        prior = (binding.block, binding.dimension)
    owner = binding.owner
    return _FactorState(
        factor,
        evaluator,
        binding,
        owner,
        owner.reference,
        owner.code,
        evaluator.__closure__ or (),
        owner.closure,
        factor.weight,
        factor.variable_names,
        factor.sqrt_information,
        point,
        prior,
    )


@dataclass(frozen=True)
class _PreparedPlan:
    factors: Sequence[JointResidualBlock]
    states: tuple[_FactorState, ...]
    plan: _PointPlanePlan
    limits: tuple[int, int, int]

    def matches(self, factors: Sequence[JointResidualBlock], values: ParameterValues) -> bool:
        return (
            factors is self.factors
            and len(factors) == len(self.states)
            and self.limits == (_MIN_POINTS, _MAX_POINTS, _MAX_GROUPS)
            and all(
                state.matches(factor) for state, factor in zip(self.states, factors, strict=True)
            )
            and self.plan.accepts(values)
        )


def _prepare_point_plane_plan(
    factors: Sequence[JointResidualBlock], values: ParameterValues
) -> _PointPlanePlan | None:
    from calibrex.graph.joint_factors import (
        _SE3_FROM_TANGENT_IMPLEMENTATION,
        se3_from_tangent,
    )

    if (
        len(factors) > _MAX_POINTS + 64
        or JointResidualBlock.residuals is not _BUILTIN_RESIDUALS_METHOD
        or SE3.transform_point is not _SE3_POINT_METHOD
        or se3_from_tangent is not _SE3_FROM_TANGENT_IMPLEMENTATION
        or not _values_supported(values)
    ):
        return None
    cache = _PREPARATION_CACHE.get()
    key = id(factors)
    cached = cache.get(key) if cache is not None else None
    if isinstance(cached, _PreparedPlan):
        if cached.matches(factors, values):
            assert cache is not None
            cache.move_to_end(key)
            return cached.plan
        assert cache is not None
        del cache[key]
    plan = _build_point_plane_plan(factors, values)
    if (
        plan is not None
        and cache is not None
        and type(factors) in (list, tuple)
        and len(factors) <= _PLAN_FACTOR_LIMIT
    ):
        states = tuple(_factor_state(factor) for factor in factors)
        if all(state is not None for state in states):
            prepared = _PreparedPlan(
                factors,
                tuple(state for state in states if state is not None),
                plan,
                (_MIN_POINTS, _MAX_POINTS, _MAX_GROUPS),
            )
            while cache and (
                len(cache) >= _PLAN_ENTRIES
                or sum(
                    len(value.factors)
                    for value in cache.values()
                    if isinstance(value, _PreparedPlan)
                )
                + len(factors)
                > _PLAN_FACTOR_LIMIT
            ):
                cache.popitem(last=False)
            cache[key] = prepared
    return plan


def _build_point_plane_plan(
    factors: Sequence[JointResidualBlock], values: ParameterValues
) -> _PointPlanePlan | None:
    from calibrex.graph.joint_factors import JointPointToPlaneMeasurement

    pending: dict[tuple[str, str, bytes], list[tuple[int, _PointBinding, float]]] = {}
    priors: list[tuple[int, int, JointResidualBlock]] = []
    cursor, point_count = 0, 0
    for factor in factors:
        binding = _binding(factor)
        if (
            binding is None
            or not _ordinary(factor.weight)
            or factor.weight <= 0.0
            or type(factor.variable_names) is not tuple
            or not all(type(name) is str for name in factor.variable_names)
            or type(factor.sqrt_information) is not tuple
        ):
            return None
        if isinstance(binding, _PriorBinding):
            if (
                type(binding.block) is not str
                or factor.variable_names != (binding.block,)
                or not 0 < binding.dimension <= 64
                or len(priors) >= 64
                or binding.block not in values
                or len(values[binding.block]) != binding.dimension
                or any(
                    type(row) is not tuple or not all(_ordinary(v) for v in row)
                    for row in factor.sqrt_information
                )
            ):
                return None
            priors.append((cursor, binding.dimension, factor))
            cursor += binding.dimension
            continue
        measurement = binding.measurement
        if (
            type(measurement) is not JointPointToPlaneMeasurement
            or type(measurement.transform_world_body_initial) is not SE3
            or type(measurement.transform_body_sensor_initial) is not SE3
            or type(binding.pose) is not str
            or type(binding.extrinsic) is not str
            or factor.variable_names != (binding.pose, binding.extrinsic)
            or binding.pose not in values
            or binding.extrinsic not in values
            or len(values[binding.pose]) != 6
            or len(values[binding.extrinsic]) != 6
            or factor.sqrt_information
            or not _vector(measurement.point_sensor_m, 3)
            or not _vector(measurement.plane_point_world_m, 3)
            or not _vector(binding.normal, 3)
        ):
            return None
        initial = (
            *measurement.transform_world_body_initial.translation_m,
            *measurement.transform_world_body_initial.rotation_quat_xyzw,
            *measurement.transform_body_sensor_initial.translation_m,
            *measurement.transform_body_sensor_initial.rotation_quat_xyzw,
        )
        key = (binding.pose, binding.extrinsic, struct.pack("<14d", *initial))
        if key not in pending:
            if len(pending) >= _MAX_GROUPS or not all(_ordinary(v) for v in initial):
                return None
            pending[key] = []
        pending[key].append((cursor, binding, math.sqrt(factor.weight)))
        cursor += 1
        point_count += 1
    if not _MIN_POINTS <= point_count <= _MAX_POINTS:
        return None
    groups = []
    for rows in pending.values():
        group = _PointGroup(
            rows[0][1],
            np.asarray([row for row, _, _ in rows], dtype=np.intp),
            np.asarray([b.measurement.point_sensor_m for _, b, _ in rows], dtype=np.float64),
            np.asarray([b.measurement.plane_point_world_m for _, b, _ in rows], dtype=np.float64),
            np.asarray([b.normal for _, b, _ in rows], dtype=np.float64),
            np.asarray([weight for _, _, weight in rows], dtype=np.float64),
        )
        for array in (group.rows, group.points, group.planes, group.normals, group.sqrt_weights):
            array.flags.writeable = False
        groups.append(group)
    return _PointPlanePlan(cursor, tuple(groups), tuple(priors))


def _linearization_supported(
    blocks: Sequence[JointParameterBlock], options: JointOptimizerOptions
) -> bool:
    return (
        type(options) is JointOptimizerOptions
        and _ordinary(options.default_finite_difference_step)
        and options.default_finite_difference_step > 0.0
        and all(
            type(block) is JointParameterBlock
            and type(block.finite_difference_steps) is tuple
            and all(_ordinary(step) and step > 0.0 for step in block.finite_difference_steps)
            for block in blocks
        )
        and sum(block.dimension for block in blocks) <= 64
    )
