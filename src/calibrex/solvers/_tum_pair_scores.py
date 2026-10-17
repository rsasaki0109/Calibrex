"""Reuse native matching geometry for the immediately following probe score."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.graph import joint_factors as f
from calibrex.graph._probe_scores import _SCORES, _capture_scores
from calibrex.graph._rgbd_batches import (
    _frame_key,
    _FrameKey,
    _rgbd_binding,
    _RGBDBinding,
    _vector,
)
from calibrex.graph.joint_optimization import JointResidualBlock, ParameterValues

if TYPE_CHECKING:
    from calibrex.solvers import native_tum_joint_slac_solver as a


def _math_supported() -> bool:
    from calibrex.solvers import native_tum_joint_slac_solver as a

    return a._pair_probe_math_supported()


class _PairProbeScores:
    def __init__(
        self, problem: a._TUMXYZPairProblem, context: a._TUMXYZPairReassociationContext,
        values: ParameterValues,
    ) -> None:
        self.problem = problem
        self.context = context
        scope = _SCORES.get()
        self.active = (
            scope is not None and bool(scope.population) and context.native_score_geometry
            and type(values) is dict and _math_supported()
            and all(type(k) is str and type(v) is tuple
                    and all(type(x) is float and math.isfinite(x) and abs(x) <= 1e12 for x in v)
                    for k, v in values.items())
        )
        self.residuals: dict[str, float] = {}
        self.population = scope.population if scope is not None else frozenset()
        self.frames: dict[tuple[int, int], _FrameKey] = {}
        self.owners: list[tuple[JointResidualBlock, _RGBDBinding, tuple[object, ...]]] = []

    def add_group(
        self, source: int, target: int, source_initial: SE3, target_initial: SE3,
        target_pose: SE3, source_world: NDArray[np.float64],
        geometry: a._TUMXYZPairGeometry, rows: list[tuple[int, int, JointResidualBlock]],
    ) -> None:
        from calibrex.solvers import native_tum_joint_slac_solver as a

        if not self.active or not rows:
            return
        expected_frame = _frame_key(source_initial, target_initial, self.frames)
        if (id(source_initial), id(target_initial)) not in self.frames:
            self.active = False
            return
        normals = []
        for position, target_row, factor in rows:
            binding = _rgbd_binding(factor)
            if not isinstance(binding, _RGBDBinding):
                self.active = False
                return
            m = binding.measurement
            association = self.context.associations[source][position]
            if (
                type(m) is not f.JointTrilinearXYZPairMeasurement or binding.kind != "pair"
                or type(factor.weight) is not float or factor.weight != 1.0
                or factor.sqrt_information or not _vector(binding.normal, 3)
                or factor.variable_names != binding.names
                or binding.names != (
                    a._pose_block(source, self.problem.query_frames[source]),
                    a._pose_block(target, self.problem.query_frames[target]), a._XYZ_LATTICE_BLOCK,
                )
                or m.source_point_sensor_m
                is not self.problem.sampled_points[source][association.source_point_index]
                or m.target_point_sensor_m is not self.problem.sampled_points[target][target_row]
                or binding.weights
                is not self.context.weights[source][association.source_point_index]
                or binding.target_weights is not self.context.weights[target][target_row]
                or _frame_key(m.transform_world_source_initial, m.transform_world_target_initial,
                              self.frames) != expected_frame or expected_frame is None
            ):
                self.active = False
                return
            normals.append(binding.normal)
            if (id(m.transform_world_source_initial), id(m.transform_world_target_initial)) \
                    not in self.frames:
                self.active = False
                return
            # Detect edits to an otherwise frozen native measurement before use.
            self.owners.append((factor, binding, (
                factor.factor_id, m.source_point_sensor_m, m.target_point_sensor_m,
                m.source_lattice_weights, m.target_lattice_weights,
                m.transform_world_source_initial, m.transform_world_target_initial,
            )))
        targets = a._PAIR_POINT_IMPLEMENTATION(
            geometry.calibrated[target][[row for _, row, _ in rows]], target_pose,
        )
        difference = source_world[[position for position, _, _ in rows]] - targets
        normal = np.asarray(normals, dtype=np.float64)
        raw = 1.0 * (normal[:, 0] * difference[:, 0] + normal[:, 1] * difference[:, 1]
                     + normal[:, 2] * difference[:, 2])
        if not np.isfinite(raw).all() or len(self.residuals) + len(rows) > 65536:
            self.active = False
            return
        self.residuals.update((factor.factor_id, value)
                              for (_, _, factor), value in zip(rows, raw.tolist(), strict=True))

    def valid(self) -> bool:
        if not self.active or not _math_supported():
            return False
        for frame in self.frames.values():
            if (frame.world.translation_m is not frame.leaves[0]
                or frame.world.rotation_quat_xyzw is not frame.leaves[1]
                or frame.sensor.translation_m is not frame.leaves[2]
                or frame.sensor.rotation_quat_xyzw is not frame.leaves[3]):
                return False
        for factor, binding, expected in self.owners:
            m = binding.measurement
            if type(m) is not f.JointTrilinearXYZPairMeasurement:
                return False
            current = (
                factor.factor_id, m.source_point_sensor_m, m.target_point_sensor_m,
                m.source_lattice_weights, m.target_lattice_weights,
                m.transform_world_source_initial, m.transform_world_target_initial,
            )
            if (_rgbd_binding(factor) is not binding or factor.weight != 1.0
                or factor.sqrt_information or factor.variable_names != binding.names
                or any(x is not y for x, y in zip(current, expected, strict=True))):
                return False
        return True

    def capture(self, state: object, values: ParameterValues) -> None:
        if self.active:
            _capture_scores(state, values, self.residuals, self.valid)
