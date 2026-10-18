"""Bounded, ordered finite differences for native RGB-D calibration fields."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.graph._ordered_sums import _ordered_float_sums
from calibrex.graph._point_plane_batches import _transform_points
from calibrex.graph.joint_factors import se3_from_tangent
from calibrex.graph.joint_optimization import JointOptimizerOptions, JointParameterBlock, _perturb

if TYPE_CHECKING:
    from calibrex.graph._rgbd_batches import _Group, _RGBDPlan

_FloatArray: TypeAlias = NDArray[np.float64]
_PROBE_ROWS = 16384
_PRODUCT_ELEMENTS = 131072
_SUM_VERSION = sys.version_info[:2] if sys.implementation.name == "cpython" else None


def _weighted_probes(weights: _FloatArray, offsets: _FloatArray) -> _FloatArray:
    products = weights[None, :, :] * offsets[:, None, :]
    ordered = _ordered_float_sums(products, version=_SUM_VERSION, minimum_rows=512)
    if ordered is not None:
        return ordered
    rows = products.reshape((-1, weights.shape[1])).tolist()
    return np.asarray([sum(row) for row in rows], dtype=np.float64).reshape(products.shape[:2])


def _unique_weighted_probes(weights: _FloatArray, offsets: _FloatArray) -> _FloatArray:
    unique: list[_FloatArray] = []
    positions: list[int] = []
    known: dict[bytes, int] = {}
    for row in offsets:
        # Byte keys keep signed zero distinct. XYZ probes change only one axis;
        # other axes need the same ordered sum only once per point chunk.
        key = row.tobytes()
        position = known.get(key)
        if position is None:
            position = len(unique)
            known[key] = position
            unique.append(row)
        positions.append(position)
    return _weighted_probes(weights, np.asarray(unique, dtype=np.float64))[positions]


def _probe_points(
    group: _Group, probes: Sequence[dict[str, tuple[float, ...]]], selected: slice
) -> tuple[_FloatArray, _FloatArray | None]:
    parameters = [tuple(values[name] for name in group.names[2:]) for values in probes]
    source = group.sensor_points[selected]
    weights = group.weights[selected]
    target = None
    if group.kind in ("xyz", "pair"):
        offsets = np.asarray([parameter[0] for parameter in parameters], dtype=np.float64)
        result = np.stack(
            [
                source[None, :, axis] + _unique_weighted_probes(weights, offsets[:, axis::3])
                for axis in range(3)
            ],
            axis=-1,
        )
        if group.kind == "pair":
            target = np.stack(
                [
                    group.target_points[None, selected, axis]
                    + _unique_weighted_probes(group.target_weights[selected], offsets[:, axis::3])
                    for axis in range(3)
                ],
                axis=-1,
            )
    elif group.kind == "depth":
        depth = np.asarray(
            [
                math.exp(parameter[0][0]) * group.depths[selected] + parameter[0][1]
                for parameter in parameters
            ],
            dtype=np.float64,
        )
        result = source[None, :, :] * depth[:, :, None]
    else:
        offsets = np.asarray([parameter[-1] for parameter in parameters], dtype=np.float64)
        nominal = group.depths[None, selected]
        if group.kind == "centered":
            nominal = nominal + np.asarray([parameter[0][0] for parameter in parameters])[:, None]
        depth = nominal + _weighted_probes(weights, offsets)
        result = source[None, :, :] * depth[:, :, None]
    return result, target


def _field_columns(
    plan: _RGBDPlan,
    values: dict[str, tuple[float, ...]],
    name: str,
    block: JointParameterBlock,
    options: JointOptimizerOptions,
    output: _FloatArray,
) -> None:
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
    for group in plan.groups:
        if name not in group.names:
            continue
        first = se3_from_tangent(values[group.names[0]]).compose(group.world)
        second = se3_from_tangent(values[group.names[1]]).compose(group.sensor)
        if group.kind != "pair":
            first = first.compose(second)
        controls = max(1, group.weights.shape[1], group.target_weights.shape[1])
        width = max(
            1, min(_PROBE_ROWS // len(probes), _PRODUCT_ELEMENTS // (len(probes) * controls))
        )
        with np.errstate(all="ignore"):
            for start in range(0, len(group.rows), width):
                selected = slice(start, start + width)
                source, target = _probe_points(group, probes, selected)
                shape = source.shape
                world = _transform_points(source.reshape((-1, 3)), first).reshape(shape)
                difference = world - (
                    _transform_points(target.reshape((-1, 3)), second).reshape(shape)
                    if target is not None else group.planes[None, selected, :]
                )
                normal = group.normals[None, selected, :]
                raw = (
                    normal[..., 0] * difference[..., 0] + normal[..., 1] * difference[..., 1]
                    + normal[..., 2] * difference[..., 2]
                )
                residuals = group.sqrt_weights[None, selected] * raw
                if not np.isfinite(residuals).all():
                    raise ValueError("joint factor residuals must be non-empty and finite")
                output[group.rows[selected]] = (
                    (residuals[0::2] - residuals[1::2]) / (2.0 * steps[:, None])
                ).T
    for row, dimension, factor in plan.scalars:
        if name not in factor.variable_names:
            continue
        for index, step in enumerate(steps):
            plus = np.asarray(factor.residuals(probes[2 * index]), dtype=np.float64)
            minus = np.asarray(factor.residuals(probes[2 * index + 1]), dtype=np.float64)
            if len(plus) != dimension or len(minus) != dimension:
                raise ValueError("joint factor residual dimension changed during linearization")
            output[row : row + dimension, index] = (plus - minus) / (2.0 * step)
