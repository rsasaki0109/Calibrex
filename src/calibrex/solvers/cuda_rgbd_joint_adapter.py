"""Optional FP64 native RGB-D execution with bounded, solve-local device data."""

from __future__ import annotations

import hashlib
import math
import sys
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, TypeAlias

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.graph._point_plane_batches import (
    normalize_quaternion_xyzw,
    quaternion_conjugate_xyzw,
)
from calibrex.graph._rgbd_batches import _Group, _RGBDPlan
from calibrex.graph.joint_factors import se3_from_tangent
from calibrex.graph.joint_optimization import (
    JointOptimizerOptions,
    JointParameterBlock,
    ParameterValues,
    _perturb,
)
from calibrex.solvers._cuda_rgbd_joint_kernels import SOURCE

_FloatArray: TypeAlias = NDArray[np.float64]
_ARRAYS = (
    "sensor_points",
    "depths",
    "weights",
    "target_points",
    "target_weights",
    "planes",
    "normals",
    "sqrt_weights",
)
_KINDS = {"xyz": 0, "pair": 1, "depth": 2, "trilinear": 3, "centered": 4}
_CACHE_BYTES = 64 * 1024 * 1024
_SCRATCH_BYTES = 4 * 1024 * 1024
_COLUMN_ELEMENTS = 131072


def _identity() -> str:
    return (
        f"calibrex.cuda_rgbd_joint/v0.1;cupy={cp.__version__};fp64;fmad=false;"
        f"sum=cpython{sys.version_info.major}.{sys.version_info.minor};"
        f"kernel_sha256={hashlib.sha256(SOURCE.encode()).hexdigest()}"
    )


def _frame(transform: SE3) -> _FloatArray:
    q = normalize_quaternion_xyzw(transform.rotation_quat_xyzw)
    return np.asarray(
        (*transform.translation_m, *q, *quaternion_conjugate_xyzw(q)), dtype=np.float64
    )


def _parameters(group: _Group, values: ParameterValues) -> tuple[float, ...]:
    parameters = tuple(values[name] for name in group.names[2:])
    if group.kind == "depth":
        return math.exp(parameters[0][0]), parameters[0][1]
    if group.kind == "centered":
        return parameters[0][0], *parameters[1]
    return parameters[0]


def _transforms(group: _Group, values: ParameterValues) -> tuple[_FloatArray, _FloatArray]:
    first = se3_from_tangent(values[group.names[0]]).compose(group.world)
    second = se3_from_tangent(values[group.names[1]]).compose(group.sensor)
    if group.kind != "pair":
        first = first.compose(second)
    return _frame(first), _frame(second)


class _DeviceGroup:
    def __init__(self, group: _Group) -> None:
        self.group = group
        self.static = {name: cp.asarray(getattr(group, name)) for name in _ARRAYS}
        self.points: OrderedDict[
            tuple[int, ...], tuple[tuple[tuple[float, ...], ...], Any, Any]
        ] = OrderedDict()

    def calibrated(self, backend: _CudaRGBDBackend, values: ParameterValues) -> tuple[Any, Any]:
        group = self.group
        parameters = tuple(values[name] for name in group.names[2:])
        key = tuple(id(vector) for vector in parameters)
        cached = self.points.get(key)
        if cached is not None:
            assert all(a is b for a, b in zip(parameters, cached[0], strict=True))
            self.points.move_to_end(key)
            return cached[1], cached[2]
        if len(self.points) >= 2:
            self.points.popitem(last=False)
        packed = cp.asarray(np.asarray([_parameters(group, values)], dtype=np.float64))
        n = len(group.rows)

        def one(points: str, weights: str) -> Any:
            result = cp.empty((n, 3), dtype=cp.float64)
            backend.calibrate_kernel(
                ((n + 255) // 256,),
                (256,),
                (
                    np.int32(_KINDS[group.kind]),
                    np.int32(n),
                    np.int32(self.static[weights].shape[1]),
                    np.int32(packed.shape[1]),
                    self.static[points],
                    self.static["depths"],
                    self.static[weights],
                    packed,
                    np.int32(backend.compensated),
                    result,
                ),
            )
            backend.launches += 1
            return result

        points = one("sensor_points", "weights")
        target = one("target_points", "target_weights") if group.kind == "pair" else None
        self.points[key] = (parameters, points, target)
        return points, target


class _DevicePlan:
    def __init__(self, plan: _RGBDPlan, size: int) -> None:
        self.plan = plan
        self.size = size
        self.groups = tuple(_DeviceGroup(group) for group in plan.groups)


class _CudaRGBDBackend:
    def __init__(self, *, cache_bytes: int = _CACHE_BYTES) -> None:
        if sys.implementation.name != "cpython" or sys.version_info[:2] not in (
            (3, 10),
            (3, 11),
            (3, 12),
        ):
            raise ValueError("CUDA RGB-D ordered sums require CPython 3.10, 3.11 or 3.12")
        if cache_bytes < 0:
            raise ValueError("CUDA RGB-D cache bytes must be nonnegative")
        self.budget = cache_bytes
        self.compensated = sys.version_info[:2] == (3, 12)
        self.plans: OrderedDict[int, _DevicePlan] = OrderedDict()
        self.peak_bytes = self.launches = self.linearizations = self.evaluations = 0
        self.fallbacks = self.max_column_elements = 0
        self.field_kernel = cp.RawKernel(SOURCE, "fields", options=("--fmad=false",))
        self.pose_kernel = cp.RawKernel(SOURCE, "poses", options=("--fmad=false",))
        self.calibrate_kernel = cp.RawKernel(SOURCE, "calibrate", options=("--fmad=false",))
        self.evaluate_kernel = cp.RawKernel(SOURCE, "evaluate", options=("--fmad=false",))

    def accepts(self, plan: _RGBDPlan) -> bool:
        key = id(plan)
        if key in self.plans:
            assert self.plans[key].plan is plan
            self.plans.move_to_end(key)
            return True
        size = sum(
            sum(int(getattr(group, name).nbytes) for name in _ARRAYS)
            + 2 * len(group.rows) * 3 * 8 * (2 if group.kind == "pair" else 1)
            for group in plan.groups
        )
        if size + _SCRATCH_BYTES > self.budget:
            self.fallbacks += 1
            return False
        while self.plans and (
            len(self.plans) >= 2
            or sum(value.size for value in self.plans.values()) + size + _SCRATCH_BYTES
            > self.budget
        ):
            self.plans.popitem(last=False)
        self.plans[key] = _DevicePlan(plan, size)
        self.peak_bytes = max(
            self.peak_bytes, sum(value.size for value in self.plans.values()) + _SCRATCH_BYTES
        )
        return True

    def close(self) -> None:
        self.plans.clear()

    def report(self) -> dict[str, object]:
        return {
            "implementation": _identity(),
            "cache_budget_bytes": self.budget,
            "peak_owned_bytes_bound": self.peak_bytes,
            "max_column_elements": self.max_column_elements,
            "linearizations": self.linearizations,
            "evaluations": self.evaluations,
            "kernel_launches": self.launches,
            "budget_fallbacks": self.fallbacks,
        }

    def evaluate(
        self, plan: _RGBDPlan, values: ParameterValues, active: str | None = None
    ) -> _FloatArray:
        device = self.plans[id(plan)]
        result = np.zeros(plan.size, dtype=np.float64)
        for packed in device.groups:
            group = packed.group
            if active is not None and active not in group.names:
                continue
            points, target = packed.calibrated(self, values)
            first, second = _transforms(group, values)
            n = len(group.rows)
            output = cp.empty(n, dtype=cp.float64)
            self.evaluate_kernel(
                ((n + 255) // 256,),
                (256,),
                (
                    np.int32(group.kind == "pair"),
                    np.int32(n),
                    points,
                    target if target is not None else packed.static["target_points"],
                    packed.static["planes"],
                    packed.static["normals"],
                    packed.static["sqrt_weights"],
                    cp.asarray(first),
                    cp.asarray(second),
                    output,
                ),
            )
            self.launches += 1
            result[group.rows] = output.get()
        for row, dimension, factor in plan.scalars:
            if active is None or active in factor.variable_names:
                residual = factor.residuals(values)
                if len(residual) != dimension:
                    raise ValueError("joint factor residual dimension changed during linearization")
                result[row : row + dimension] = residual
        if not np.isfinite(result).all():
            raise ValueError("joint factor residuals must be non-empty and finite")
        self.evaluations += 1
        return result

    def linearize(
        self,
        plan: _RGBDPlan,
        values: dict[str, tuple[float, ...]],
        blocks: Sequence[JointParameterBlock],
        layout: dict[str, slice],
        options: JointOptimizerOptions,
    ) -> tuple[_FloatArray, _FloatArray]:
        baseline = self.evaluate(plan, values)
        jacobian = np.zeros((plan.size, sum(s.stop - s.start for s in layout.values())))
        by_name = {block.name: block for block in blocks}
        for name, column in layout.items():
            block = by_name[name]
            steps = np.asarray(
                block.finite_difference_steps
                or (options.default_finite_difference_step,) * block.dimension,
                dtype=np.float64,
            )
            probes = [
                _perturb(values, name, index, sign * float(step))
                for index, step in enumerate(steps)
                for sign in (1.0, -1.0)
            ]
            field = (
                block.dimension >= 8
                and any(name in g.names[2:] for g in plan.groups)
                and not any(name in g.names[:2] for g in plan.groups)
            )
            pose = any(name in g.names[:2] for g in plan.groups) and not any(
                name in g.names[2:] for g in plan.groups
            )
            if field or pose:
                for packed in self.plans[id(plan)].groups:
                    if name in packed.group.names:
                        self._columns(packed, values, probes, steps, field, jacobian[:, column])
                self._scalar_columns(plan, name, probes, steps, jacobian[:, column])
            else:
                affected = np.zeros(plan.size, dtype=np.bool_)
                for group in plan.groups:
                    if name in group.names:
                        affected[group.rows] = True
                for row, dimension, factor in plan.scalars:
                    if name in factor.variable_names:
                        affected[row : row + dimension] = True
                for index, step in enumerate(steps):
                    plus = self.evaluate(plan, probes[2 * index], name)
                    minus = self.evaluate(plan, probes[2 * index + 1], name)
                    jacobian[affected, column.start + index] = (
                        plus[affected] - minus[affected]
                    ) / (2.0 * step)
        self.linearizations += 1
        return baseline, jacobian

    @staticmethod
    def _scalar_columns(
        plan: _RGBDPlan,
        name: str,
        probes: Sequence[ParameterValues],
        steps: _FloatArray,
        output: _FloatArray,
    ) -> None:
        for row, dimension, factor in plan.scalars:
            if name not in factor.variable_names:
                continue
            for index, step in enumerate(steps):
                plus = np.asarray(factor.residuals(probes[2 * index]), dtype=np.float64)
                minus = np.asarray(factor.residuals(probes[2 * index + 1]), dtype=np.float64)
                if len(plus) != dimension or len(minus) != dimension:
                    raise ValueError("joint factor residual dimension changed during linearization")
                output[row : row + dimension, index] = (plus - minus) / (2.0 * step)

    def _columns(
        self,
        packed: _DeviceGroup,
        values: ParameterValues,
        probes: Sequence[ParameterValues],
        steps: _FloatArray,
        field: bool,
        output: _FloatArray,
    ) -> None:
        group = packed.group
        device_steps = cp.asarray(steps)
        if field:
            parameters = cp.asarray(np.asarray([_parameters(group, p) for p in probes]))
            first, second = (cp.asarray(frame) for frame in _transforms(group, values))
            points = target = None
        else:
            points, target = packed.calibrated(self, values)
            frames = [_transforms(group, probe) for probe in probes]
            first = cp.asarray(np.asarray([pair[0] for pair in frames]))
            second = cp.asarray(np.asarray([pair[1] for pair in frames]))
            parameters = None
        width = max(1, _COLUMN_ELEMENTS // len(steps))
        for start in range(0, len(group.rows), width):
            selected = slice(start, start + width)
            n = len(group.rows[selected])
            device_output = cp.empty((n, len(steps)), dtype=cp.float64)
            invalid = cp.zeros(1, dtype=cp.int32)
            if field:
                assert parameters is not None
                arrays = [packed.static[name][selected] for name in _ARRAYS]
                self.field_kernel(
                    ((n * len(steps) + 255) // 256,),
                    (256,),
                    (
                        np.int32(_KINDS[group.kind]),
                        np.int32(n),
                        np.int32(len(steps)),
                        np.int32(group.weights.shape[1]),
                        np.int32(parameters.shape[1]),
                        *arrays,
                        parameters,
                        first,
                        second,
                        device_steps,
                        np.int32(self.compensated),
                        device_output,
                        invalid,
                    ),
                )
            else:
                assert points is not None
                self.pose_kernel(
                    ((n * len(steps) + 255) // 256,),
                    (256,),
                    (
                        np.int32(group.kind == "pair"),
                        np.int32(n),
                        np.int32(len(steps)),
                        points[selected],
                        target[selected]
                        if target is not None
                        else packed.static["target_points"][selected],
                        packed.static["planes"][selected],
                        packed.static["normals"][selected],
                        packed.static["sqrt_weights"][selected],
                        first,
                        second,
                        device_steps,
                        device_output,
                        invalid,
                    ),
                )
            result = device_output.get()
            if not np.isfinite(result).all() and int(invalid.get()[0]):
                raise ValueError("joint factor residuals must be non-empty and finite")
            self.launches += 1
            self.max_column_elements = max(self.max_column_elements, result.size)
            output[group.rows[selected]] = result
