"""Ordered RGB-D geometry arrays, retaining the factories' scalar lattice sums."""

from __future__ import annotations

import math
import struct
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import FunctionType
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, Vector3
from calibrex.graph._evaluation_scope import _RGBD_PREPARATION_CACHE
from calibrex.graph._ordered_sums import _ordered_float_sums
from calibrex.graph._point_plane_batches import (
    _OwnedPointFactors,
    _Owner,
    _owner,
    _transform_points,
)
from calibrex.graph.joint_optimization import (
    _BUILTIN_RESIDUALS_METHOD,
    JointOptimizerOptions,
    JointParameterBlock,
    JointResidualBlock,
    ParameterValues,
    ResidualEvaluator,
)

if TYPE_CHECKING:
    from calibrex.graph.joint_factors import (
        JointDepthPointToPlaneMeasurement,
        JointTrilinearDepthPointToPlaneMeasurement,
        JointTrilinearXYZPairMeasurement,
        JointTrilinearXYZPointToPlaneMeasurement,
    )

_Kind = Literal["depth", "trilinear", "centered", "xyz", "pair"]
_FloatArray: TypeAlias = NDArray[np.float64]
_MIN_POINTS = 64
_MAX_POINTS = 65536
_MAX_GROUPS = 128
_CHUNK_ROWS = 4096
_PLAN_ENTRIES = 2
_PLAN_FACTOR_LIMIT = 65536
_ATTRIBUTE = "_calibrex_rgbd_builtin"
_POINT_METHOD = SE3.transform_point
_COMPOSE_METHOD = SE3.compose


@dataclass(frozen=True)
class _RGBDBinding:
    owner: _Owner
    kind: _Kind
    measurement: (
        JointDepthPointToPlaneMeasurement
        | JointTrilinearDepthPointToPlaneMeasurement
        | JointTrilinearXYZPointToPlaneMeasurement
        | JointTrilinearXYZPairMeasurement
    )
    normal: Vector3
    names: tuple[str, ...]
    ray: Vector3 | None = None
    weights: tuple[float, ...] = ()
    target_weights: tuple[float, ...] = ()
    original: tuple[object, ...] = ()


@dataclass(frozen=True)
class _ScalarBinding:
    owner: _Owner
    block: str
    parameter_dimension: int
    residual_dimension: int
    original: tuple[object, ...]


def _bind_rgbd(
    evaluator: ResidualEvaluator,
    kind: _Kind,
    measurement: (
        JointDepthPointToPlaneMeasurement
        | JointTrilinearDepthPointToPlaneMeasurement
        | JointTrilinearXYZPointToPlaneMeasurement
        | JointTrilinearXYZPairMeasurement
    ),
    normal: Vector3,
    names: tuple[str, ...],
    *,
    ray: Vector3 | None = None,
    weights: tuple[float, ...] = (),
    target_weights: tuple[float, ...] = (),
) -> None:
    setattr(
        evaluator,
        _ATTRIBUTE,
        _RGBDBinding(
            _owner(evaluator),
            kind,
            measurement,
            normal,
            names,
            ray,
            weights,
            target_weights,
            (kind, measurement, normal, names, ray, weights, target_weights),
        ),
    )


def _bind_rgbd_scalar(
    evaluator: ResidualEvaluator, block: str, parameter_dimension: int, residual_dimension: int
) -> None:
    setattr(
        evaluator,
        _ATTRIBUTE,
        _ScalarBinding(
            _owner(evaluator),
            block,
            parameter_dimension,
            residual_dimension,
            (block, parameter_dimension, residual_dimension),
        ),
    )


def _rgbd_binding(factor: JointResidualBlock) -> _RGBDBinding | _ScalarBinding | None:
    if type(factor) is not JointResidualBlock or type(factor.evaluator) is not FunctionType:
        return None
    binding = getattr(factor.evaluator, _ATTRIBUTE, None)
    if type(binding) not in (_RGBDBinding, _ScalarBinding):
        return None
    assert isinstance(binding, (_RGBDBinding, _ScalarBinding))
    fields = (
        (
            binding.kind,
            binding.measurement,
            binding.normal,
            binding.names,
            binding.ray,
            binding.weights,
            binding.target_weights,
        )
        if isinstance(binding, _RGBDBinding)
        else (binding.block, binding.parameter_dimension, binding.residual_dimension)
    )
    if len(fields) != len(binding.original) or any(
        a is not b for a, b in zip(fields, binding.original, strict=True)
    ):
        return None
    owner = binding.owner
    try:
        closed = tuple(cell.cell_contents for cell in factor.evaluator.__closure__ or ())
    except ValueError:
        return None
    if (
        owner.reference() is not factor.evaluator
        or owner.code is not factor.evaluator.__code__
        or len(closed) != len(owner.closure)
        or any(a is not b for a, b in zip(closed, owner.closure, strict=True))
    ):
        return None
    return binding


def _ordinary(value: object) -> bool:
    return type(value) is float and math.isfinite(value) and abs(value) <= 1.0e12


def _vector(value: object, dimension: int) -> bool:
    return type(value) is tuple and len(value) == dimension and all(_ordinary(v) for v in value)


def _frame(frame: SE3) -> bool:
    return (
        type(frame) is SE3
        and _vector(frame.translation_m, 3)
        and _vector(frame.rotation_quat_xyzw, 4)
    )


@dataclass(frozen=True)
class _FrameKey:
    world: SE3
    sensor: SE3
    leaves: tuple[tuple[float, ...], ...]
    key: bytes


def _frame_key(
    world: SE3, sensor: SE3, cache: dict[tuple[int, int], _FrameKey]
) -> bytes | None:
    """Validate shared immutable frames once within this plan construction."""
    identity = (id(world), id(sensor))
    cached = cache.get(identity)
    if (
        cached is not None
        and type(world) is SE3
        and type(sensor) is SE3
        and world is cached.world
        and sensor is cached.sensor
        and "translation_m" not in vars(SE3)
        and "rotation_quat_xyzw" not in vars(SE3)
        and world.translation_m is cached.leaves[0]
        and world.rotation_quat_xyzw is cached.leaves[1]
        and sensor.translation_m is cached.leaves[2]
        and sensor.rotation_quat_xyzw is cached.leaves[3]
    ):
        return cached.key
    if not _frame(world) or not _frame(sensor):
        return None
    leaves = (
        world.translation_m, world.rotation_quat_xyzw,
        sensor.translation_m, sensor.rotation_quat_xyzw,
    )
    key = struct.pack("<14d", *(value for leaf in leaves for value in leaf))
    if len(cache) < _MAX_GROUPS or identity in cache:
        cache[identity] = _FrameKey(world, sensor, leaves, key)
    return key


def _plain_closed(value: object, depth: int = 0) -> bool:
    """Regularizers may only close over ordinary numeric, side-effect-free data."""
    if depth > 6:
        return False
    if type(value) in (str, bool) or value is None:
        return True
    if type(value) is int:
        return abs(value) <= 1_000_000_000_000
    if type(value) is float:
        return _ordinary(value)
    if type(value) in (list, tuple):
        sequence = cast(Sequence[object], value)
        return len(sequence) <= 8192 and all(_plain_closed(v, depth + 1) for v in sequence)
    if type(value) is np.ndarray:
        array = cast(_FloatArray, value)
        return (
            array.dtype == np.float64
            and array.size <= 65536
            and bool(np.isfinite(array).all())
            and bool((np.abs(array) <= 1.0e12).all())
        )
    return False


@dataclass(frozen=True)
class _Geometry:
    point: Vector3
    nominal: float
    weights: tuple[float, ...]
    target: Vector3
    target_weights: tuple[float, ...]


@dataclass
class _Group:
    kind: _Kind
    names: tuple[str, ...]
    world: SE3
    sensor: SE3
    rows: NDArray[np.intp]
    sensor_points: _FloatArray
    depths: _FloatArray
    weights: _FloatArray
    target_points: _FloatArray
    target_weights: _FloatArray
    planes: _FloatArray
    normals: _FloatArray
    sqrt_weights: _FloatArray
    # Two complete point-array states per group, local to one graph evaluation.
    points: OrderedDict[
        tuple[int, ...], tuple[tuple[object, ...], _FloatArray, _FloatArray | None]
    ] = field(default_factory=OrderedDict)

    def calibrated_points(self, values: ParameterValues) -> tuple[_FloatArray, _FloatArray | None]:
        parameters = tuple(values[name] for name in self.names[2:])
        key = tuple(id(vector) for vector in parameters)
        cached = self.points.get(key)
        if cached is not None:
            assert all(a is b for a, b in zip(parameters, cached[0], strict=True))
            self.points.move_to_end(key)
            return cached[1], cached[2]
        target = None
        if self.kind in ("xyz", "pair"):
            offsets = parameters[0]
            points = np.column_stack(
                [
                    self.sensor_points[:, axis] + _weighted_sums(self.weights, offsets[axis::3])
                    for axis in range(3)
                ]
            )
            if self.kind == "pair":
                target = np.column_stack(
                    [
                        self.target_points[:, axis]
                        + _weighted_sums(self.target_weights, offsets[axis::3])
                        for axis in range(3)
                    ]
                )
        elif self.kind == "depth":
            log_scale, depth_bias = parameters[0]
            scale = math.exp(log_scale)
            points = self.sensor_points * (scale * self.depths + depth_bias)[:, None]
        else:
            offsets = parameters[-1]
            bias = parameters[0][0] if self.kind == "centered" else None
            nominal = self.depths if bias is None else self.depths + bias
            depth = nominal + _weighted_sums(self.weights, offsets)
            points = self.sensor_points * depth[:, None]
        points.flags.writeable = False
        if target is not None:
            target.flags.writeable = False
        self.points[key] = (parameters, points, target)
        if len(self.points) > 2:
            self.points.popitem(last=False)
        return points, target


def _weighted_sums(weights: _FloatArray, offsets: tuple[float, ...]) -> _FloatArray:
    # Batch independent products and retain the original CPython sum order.
    products = weights * np.asarray(offsets, dtype=np.float64)
    ordered = _ordered_float_sums(products)
    if ordered is not None:
        return ordered
    return np.asarray([sum(row) for row in products.tolist()], dtype=np.float64)


@dataclass(frozen=True)
class _RGBDPlan:
    size: int
    groups: tuple[_Group, ...]
    scalars: tuple[tuple[int, int, JointResidualBlock], ...]
    dimensions: tuple[tuple[str, int], ...]
    depth_scales: tuple[str, ...]

    def accepts(self, values: ParameterValues) -> bool:
        return all(
            name in values and len(values[name]) == dimension for name, dimension in self.dimensions
        ) and all(abs(values[name][0]) <= 50.0 for name in self.depth_scales)

    def evaluate(self, values: ParameterValues, active: str | None = None) -> _FloatArray:
        from calibrex.graph.joint_factors import se3_from_tangent

        result = np.zeros(self.size, dtype=np.float64)
        for group in self.groups:
            if active is not None and active not in group.names:
                continue
            points, target = group.calibrated_points(values)
            first = se3_from_tangent(values[group.names[0]]).compose(group.world)
            if group.kind == "pair":
                second = se3_from_tangent(values[group.names[1]]).compose(group.sensor)
                assert target is not None
            else:
                second = se3_from_tangent(values[group.names[1]]).compose(group.sensor)
                first = first.compose(second)
            with np.errstate(all="ignore"):
                for start in range(0, len(group.rows), _CHUNK_ROWS):
                    stop = start + _CHUNK_ROWS
                    point_world = _transform_points(points[start:stop], first)
                    difference = point_world - (
                        _transform_points(target[start:stop], second)
                        if group.kind == "pair" and target is not None
                        else group.planes[start:stop]
                    )
                    normal = group.normals[start:stop]
                    raw = (
                        normal[:, 0] * difference[:, 0]
                        + normal[:, 1] * difference[:, 1]
                        + normal[:, 2] * difference[:, 2]
                    )
                    result[group.rows[start:stop]] = group.sqrt_weights[start:stop] * raw
        for row, dimension, factor in self.scalars:
            if active is None or active in factor.variable_names:
                residuals = factor.residuals(values)
                if len(residuals) != dimension:
                    raise ValueError("joint factor residual dimension changed during linearization")
                result[row : row + dimension] = residuals
        if not np.isfinite(result).all():
            raise ValueError("joint factor residuals must be non-empty and finite")
        return result

    def linearize(
        self,
        values: dict[str, tuple[float, ...]],
        blocks: Sequence[JointParameterBlock],
        layout: dict[str, slice],
        options: JointOptimizerOptions,
    ) -> tuple[_FloatArray, _FloatArray]:
        from calibrex.graph._rgbd_field_columns import _field_columns
        from calibrex.graph.joint_optimization import _perturb

        baseline = self.evaluate(values)
        jacobian = np.zeros((self.size, sum(s.stop - s.start for s in layout.values())))
        by_name = {block.name: block for block in blocks}
        for name, column in layout.items():
            block = by_name[name]
            if (
                block.dimension >= 8
                and any(name in group.names[2:] for group in self.groups)
                and not any(name in group.names[:2] for group in self.groups)
            ):
                _field_columns(self, values, name, block, options, jacobian[:, column])
                continue
            affected = np.zeros(self.size, dtype=np.bool_)
            for group in self.groups:
                if name in group.names:
                    affected[group.rows] = True
            for row, dimension, factor in self.scalars:
                if name in factor.variable_names:
                    affected[row : row + dimension] = True
            for index in range(block.dimension):
                step = (
                    block.finite_difference_steps[index]
                    if block.finite_difference_steps
                    else options.default_finite_difference_step
                )
                plus = self.evaluate(_perturb(values, name, index, step), name)
                minus = self.evaluate(_perturb(values, name, index, -step), name)
                jacobian[affected, column.start + index] = (plus[affected] - minus[affected]) / (
                    2.0 * step
                )
        return baseline, jacobian


def _rgbd_linearization_supported(
    blocks: Sequence[JointParameterBlock], options: JointOptimizerOptions
) -> bool:
    return (
        type(options) is JointOptimizerOptions
        and _ordinary(options.default_finite_difference_step)
        and options.default_finite_difference_step > 0.0
        and all(
            type(block) is JointParameterBlock
            and type(block.finite_difference_steps) is tuple
            and all(_ordinary(v) and v > 0.0 for v in block.finite_difference_steps)
            for block in blocks
        )
        and sum(block.dimension for block in blocks) <= 256
    )


def _prepare_rgbd_plan(
    factors: Sequence[JointResidualBlock], values: ParameterValues
) -> _RGBDPlan | None:
    from calibrex.graph.joint_factors import (
        _RGBD_POINT_FUNCTION_IMPLEMENTATION,
        _RGBD_SCALE_FUNCTION_IMPLEMENTATION,
        _SE3_FROM_TANGENT_IMPLEMENTATION,
        _calibrated_xyz_lattice_point,
        _scale,
        se3_from_tangent,
    )

    if (
        not _MIN_POINTS <= len(factors) <= _MAX_POINTS + 64
        or type(values) is not dict
        or not all(
            type(k) is str and type(v) is tuple and all(_ordinary(x) for x in v)
            for k, v in values.items()
        )
        or JointResidualBlock.residuals is not _BUILTIN_RESIDUALS_METHOD
        or SE3.transform_point is not _POINT_METHOD
        or SE3.compose is not _COMPOSE_METHOD
        or se3_from_tangent is not _SE3_FROM_TANGENT_IMPLEMENTATION
        or _calibrated_xyz_lattice_point is not _RGBD_POINT_FUNCTION_IMPLEMENTATION
        or _scale is not _RGBD_SCALE_FUNCTION_IMPLEMENTATION
    ):
        cache = _RGBD_PREPARATION_CACHE.get()
        if cache is not None:
            cache.clear()
        return None
    cache = _RGBD_PREPARATION_CACHE.get()
    key = id(factors)
    cached = cache.get(key) if cache is not None else None
    if isinstance(cached, _PreparedRGBDPlan):
        if (
            factors is cached.factors
            and cached.limits == (_MIN_POINTS, _MAX_POINTS, _MAX_GROUPS)
            and cached.plan.accepts(values)
        ):
            assert cache is not None
            cache.move_to_end(key)
            return cached.plan
        assert cache is not None
        del cache[key]
    plan = _build_rgbd_plan(factors, values)
    if (
        plan is not None
        and cache is not None
        and type(factors) is _OwnedPointFactors
        and len(factors) <= _PLAN_FACTOR_LIMIT
    ):
        # Generic callers are always prepared afresh. Only a native adapter's
        # synchronous, private factor sequence owns these immutable snapshots.
        for group in plan.groups:
            group.world = _frozen_frame(group.world)
            group.sensor = _frozen_frame(group.sensor)
        while cache and (
            len(cache) >= _PLAN_ENTRIES
            or sum(
                len(value.factors)
                for value in cache.values()
                if isinstance(value, _PreparedRGBDPlan)
            )
            + len(factors)
            > _PLAN_FACTOR_LIMIT
        ):
            cache.popitem(last=False)
        cache[key] = _PreparedRGBDPlan(factors, plan, (_MIN_POINTS, _MAX_POINTS, _MAX_GROUPS))
    return plan


@dataclass(frozen=True)
class _PreparedRGBDPlan:
    factors: Sequence[JointResidualBlock]
    plan: _RGBDPlan
    limits: tuple[int, int, int]


def _frozen_frame(frame: SE3) -> SE3:
    # Constructing an SE3 normally normalizes its quaternion. Preserve the
    # already validated bits, including quaternions rounded by composition.
    result = object.__new__(SE3)
    object.__setattr__(result, "translation_m", frame.translation_m)
    object.__setattr__(result, "rotation_quat_xyzw", frame.rotation_quat_xyzw)
    return result


def _build_rgbd_plan(
    factors: Sequence[JointResidualBlock], values: ParameterValues
) -> _RGBDPlan | None:
    from calibrex.graph.joint_factors import (
        JointDepthPointToPlaneMeasurement,
        JointTrilinearDepthPointToPlaneMeasurement,
        JointTrilinearXYZPairMeasurement,
        JointTrilinearXYZPointToPlaneMeasurement,
    )

    pending: dict[
        tuple[_Kind, tuple[str, ...], bytes], list[tuple[int, _RGBDBinding, _Geometry, float]]
    ] = {}
    scalars: list[tuple[int, int, JointResidualBlock]] = []
    frames: dict[tuple[_Kind, tuple[str, ...], bytes], tuple[SE3, SE3]] = {}
    frame_keys: dict[tuple[int, int], _FrameKey] = {}
    cursor, point_count = 0, 0
    for factor in factors:
        if (
            type(factor) is not JointResidualBlock
            or not _ordinary(factor.weight)
            or factor.weight <= 0.0
            or type(factor.variable_names) is not tuple
            or not all(type(name) is str for name in factor.variable_names)
            or type(factor.sqrt_information) is not tuple
        ):
            return None
        binding = _rgbd_binding(factor)
        if binding is None:
            return None
        scalar = binding if isinstance(binding, _ScalarBinding) else None
        if scalar is not None:
            if (
                len(scalars) >= 64
                or type(scalar.parameter_dimension) is not int
                or type(scalar.residual_dimension) is not int
                or not 0 < scalar.parameter_dimension <= 256
                or factor.variable_names != (scalar.block,)
                or scalar.block not in values
                or len(values[scalar.block]) != scalar.parameter_dimension
                or not 0 < scalar.residual_dimension <= 4096
                or len(factor.sqrt_information) not in (0, scalar.residual_dimension)
                or not all(_plain_closed(value) for value in scalar.owner.closure)
                or any(
                    not _vector(row, scalar.residual_dimension) for row in factor.sqrt_information
                )
            ):
                return None
            scalars.append((cursor, scalar.residual_dimension, factor))
            cursor += scalar.residual_dimension
            continue
        assert isinstance(binding, _RGBDBinding)
        if (
            factor.variable_names != binding.names
            or factor.sqrt_information
            or not _vector(binding.normal, 3)
            or any(type(name) is not str or name not in values for name in binding.names)
            or any(len(values[name]) != 6 for name in binding.names[:2])
        ):
            return None
        m = binding.measurement
        if binding.kind == "pair":
            if type(m) is not JointTrilinearXYZPairMeasurement:
                return None
            world, sensor = m.transform_world_source_initial, m.transform_world_target_initial
            geometry = _Geometry(
                m.source_point_sensor_m,
                0.0,
                binding.weights,
                m.target_point_sensor_m,
                binding.target_weights,
            )
            plane = (0.0, 0.0, 0.0)
        else:
            if type(m) not in (
                JointDepthPointToPlaneMeasurement,
                JointTrilinearDepthPointToPlaneMeasurement,
                JointTrilinearXYZPointToPlaneMeasurement,
            ):
                return None
            assert not isinstance(m, JointTrilinearXYZPairMeasurement)
            world, sensor = m.transform_world_body_initial, m.transform_body_sensor_initial
            plane = m.plane_point_world_m
            if binding.kind == "xyz":
                if type(m) is not JointTrilinearXYZPointToPlaneMeasurement:
                    return None
                geometry = _Geometry(m.point_sensor_m, 0.0, binding.weights, (0.0, 0.0, 0.0), ())
            else:
                if not isinstance(
                    m,
                    (JointDepthPointToPlaneMeasurement, JointTrilinearDepthPointToPlaneMeasurement),
                ):
                    return None
                if binding.ray is None:
                    return None
                geometry = _Geometry(
                    binding.ray, m.nominal_depth_m, binding.weights, (0.0, 0.0, 0.0), ()
                )
        initial_key = _frame_key(world, sensor, frame_keys)
        if (
            initial_key is None
            or not _vector(plane, 3)
            or not _vector(geometry.point, 3)
            or not _vector(geometry.target, 3)
            or not _ordinary(geometry.nominal)
            or not _vector(geometry.weights, len(geometry.weights))
            or not _vector(geometry.target_weights, len(geometry.target_weights))
        ):
            return None
        calibration = binding.names[2:]
        if binding.kind == "depth":
            if (
                len(calibration) != 1
                or len(values[calibration[0]]) != 2
                or abs(values[calibration[0]][0]) > 50.0
            ):
                return None
        else:
            count = len(geometry.weights)
            if (
                not 0 < count <= 64
                or not calibration
                or len(values[calibration[-1]])
                != count * (3 if binding.kind in ("xyz", "pair") else 1)
                or (
                    binding.kind == "centered"
                    and (len(calibration) != 2 or len(values[calibration[0]]) != 1)
                )
                or (binding.kind == "pair" and len(geometry.target_weights) != count)
            ):
                return None
        key = (binding.kind, binding.names, initial_key)
        if key not in pending:
            if len(pending) >= _MAX_GROUPS:
                return None
            pending[key] = []
            frames[key] = (world, sensor)
        pending[key].append((cursor, binding, geometry, math.sqrt(factor.weight)))
        cursor += 1
        point_count += 1
    if not _MIN_POINTS <= point_count <= _MAX_POINTS:
        return None
    groups = []
    for key, rows in pending.items():
        kind, names, _ = key
        world, sensor = frames[key]
        planes = [
            (
                (0.0, 0.0, 0.0)
                if kind == "pair"
                else cast("JointDepthPointToPlaneMeasurement", b.measurement).plane_point_world_m
            )
            for _, b, _, _ in rows
        ]
        group = _Group(
            kind,
            names,
            world,
            sensor,
            np.asarray([r for r, _, _, _ in rows], dtype=np.intp),
            np.asarray([g.point for _, _, g, _ in rows], dtype=np.float64),
            np.asarray([g.nominal for _, _, g, _ in rows], dtype=np.float64),
            np.asarray([g.weights for _, _, g, _ in rows], dtype=np.float64),
            np.asarray([g.target for _, _, g, _ in rows], dtype=np.float64),
            np.asarray([g.target_weights for _, _, g, _ in rows], dtype=np.float64),
            np.asarray(planes, dtype=np.float64),
            np.asarray([b.normal for _, b, _, _ in rows], dtype=np.float64),
            np.asarray([w for _, _, _, w in rows], dtype=np.float64),
        )
        for array in (
            group.rows,
            group.sensor_points,
            group.depths,
            group.weights,
            group.target_points,
            group.target_weights,
            group.planes,
            group.normals,
            group.sqrt_weights,
        ):
            array.flags.writeable = False
        groups.append(group)
    dimensions = dict.fromkeys(name for group in groups for name in group.names)
    dimensions.update(dict.fromkeys(factor.variable_names[0] for _, _, factor in scalars))
    return _RGBDPlan(
        cursor,
        tuple(groups),
        tuple(scalars),
        tuple((name, len(values[name])) for name in dimensions),
        tuple(group.names[2] for group in groups if group.kind == "depth"),
    )
