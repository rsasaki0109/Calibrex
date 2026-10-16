"""Owned geometry arrays for the native TUM adapter's repeated diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.graph._point_plane_batches import _transform_points
from calibrex.graph.joint_factors import se3_from_tangent
from calibrex.graph.joint_optimization import ParameterValues

_FloatArray: TypeAlias = NDArray[np.float64]


def _weighted_sums(weights: _FloatArray, offsets: tuple[float, ...]) -> _FloatArray:
    # Keep Python's sum on ordinary floats. Its rounding differs between the
    # supported Python versions and cannot be replaced by a BLAS dot product.
    products = weights * np.asarray(offsets, dtype=np.float64)
    return np.asarray([sum(row) for row in products.tolist()], dtype=np.float64)


@dataclass(frozen=True)
class _XYZSamples:
    points: _FloatArray
    weights: _FloatArray

    def calibrated(self, offsets: tuple[float, ...]) -> _FloatArray:
        if len(offsets) != 3 * self.weights.shape[1]:
            raise ValueError("pair full-XYZ field dimension does not match lattice")
        result = np.column_stack(
            [
                self.points[:, axis] + _weighted_sums(self.weights, offsets[axis::3])
                for axis in range(3)
            ]
        )
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class _DepthPointGroup:
    rows: NDArray[np.intp]
    rays: _FloatArray
    depths: _FloatArray
    weights: _FloatArray
    pose_block: str
    initial_pose: SE3


@dataclass(frozen=True)
class _SpatialPoints:
    size: int
    groups: tuple[_DepthPointGroup, ...]

    def world_points(
        self,
        values: ParameterValues,
        *,
        extrinsic: str,
        bias: str,
        lattice: str,
    ) -> _FloatArray:
        result = np.empty((self.size, 3), dtype=np.float64)
        correction = se3_from_tangent(values[extrinsic])
        for group in self.groups:
            # Addition order is (nominal + bias) + weighted field, as in the
            # scalar adapter; pre-adding the field would change the objective.
            depth = group.depths + values[bias][0] + _weighted_sums(group.weights, values[lattice])
            world_sensor = (
                se3_from_tangent(values[group.pose_block])
                .compose(group.initial_pose)
                .compose(correction)
            )
            result[group.rows] = _transform_points(group.rays * depth[:, None], world_sensor)
        return result
