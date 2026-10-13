"""Exact finite-difference and solver parity for ordered array residuals."""

import dataclasses
import math
import random
import struct

import pytest

from calibrex.core.geometry import SE3
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
)


class ScalarFactor(LidarRigPointToPlaneFactor):
    def residuals(self, correction=None):
        transform = self._corrected_transform(correction)
        values = []
        for observation in self.observations:
            world = observation.t_world_ego.compose(transform).transform_point(
                observation.point_lidar_m
            )
            delta = tuple(world[i] - observation.plane_point_world_m[i] for i in range(3))
            normal = observation.plane_normal_world
            values.append(math.sqrt(observation.weight) * (
                normal[0] * delta[0] + normal[1] * delta[1] + normal[2] * delta[2]
            ))
        return values


def _factors(count, moving=True):
    rng = random.Random(371)
    observations = [
        LidarPointToPlaneObservation(
            point_lidar_m=tuple(rng.uniform(-100, 100) for _ in range(3)),
            plane_point_world_m=tuple(rng.uniform(-10, 10) for _ in range(3)),
            plane_normal_world=tuple(rng.uniform(-1, 1) for _ in range(3)),
            t_world_ego=(SE3(
                tuple(rng.uniform(-3, 3) for _ in range(3)),
                tuple(rng.uniform(-1, 1) for _ in range(4)),
            ) if moving else SE3.identity()),
            weight=(0.0 if index % 19 == 0 else rng.uniform(0.01, 3.0)),
        )
        for index in range(count)
    ]
    kwargs = {"variable": "T_ego_lidar", "t_ego_lidar": SE3(
        (0.2, -0.4, 0.1), (0.1, -0.3, 0.5, 0.8)
    ), "observations": observations}
    return LidarRigPointToPlaneFactor(**kwargs), ScalarFactor(**kwargs)


def _bits(values):
    return b"".join(struct.pack("!d", value) for value in values)


@pytest.mark.parametrize("count", [0, 1, 63, 64, 65, 4095, 4096, 4097])
@pytest.mark.parametrize("moving", [False, True])
def test_residuals_match_scalar_bits_across_threshold_and_chunks(count, moving):
    factor, scalar = _factors(count, moving)
    for correction in (None, [0.0] * 6, [0.3, -0.2, 0.4, 0.17, -0.31, 0.23]):
        assert _bits(factor.residuals(correction)) == _bits(scalar.residuals(correction))


@pytest.mark.parametrize("moving", [False, True])
def test_numerical_linearization_and_all_diagnostics_match_exactly(moving):
    factor, scalar = _factors(128, moving)
    for correction in (None, [0.1, -0.2, 0.3, 0.01, -0.02, 0.03]):
        assert factor.linearize(correction) == scalar.linearize(correction)
        assert factor.evaluate(correction) == scalar.evaluate(correction)


def test_complete_solver_history_matches_scalar():
    factor, scalar = _factors(128)
    options = FixedTrajectorySe3SolverOptions(max_iterations=8)
    solver = FixedTrajectorySe3ExtrinsicSolver()
    assert solver.solve(factor, options) == solver.solve(scalar, options)


def test_cached_observation_arrays_follow_replacement_and_transform_changes():
    factor, scalar = _factors(65)
    assert factor.residuals() == scalar.residuals()
    cached = factor._array_observations
    assert cached is not None and cached[1] is not None
    for field in dataclasses.fields(cached[1]):
        if field.name != "shared_pose":
            assert not getattr(cached[1], field.name).flags.writeable
    factor.observations = scalar.observations = tuple(reversed(factor.observations))
    factor.t_ego_lidar = scalar.t_ego_lidar = SE3((1.0, 2.0, 3.0), (0.2, 0.3, 0.4, 0.5))
    assert _bits(factor.residuals()) == _bits(scalar.residuals())
    assert factor._array_observations is not cached


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_observations_keep_scalar_path(value):
    factor, scalar = _factors(65)
    observations = (*factor.observations[:-1], dataclasses.replace(
        factor.observations[-1], point_lidar_m=(value, 0.0, 1.0)
    ))
    factor.observations = scalar.observations = observations
    actual, expected = factor.residuals(), scalar.residuals()
    assert _bits(actual[:-1]) == _bits(expected[:-1])
    assert math.isnan(actual[-1]) and math.isnan(expected[-1])
    assert factor._array_observations[1] is None


def test_signed_zeros_and_empty_population():
    observations = [LidarPointToPlaneObservation(
        point_lidar_m=(-0.0, 0.0, -0.0), plane_point_world_m=(0.0, -0.0, 0.0),
        plane_normal_world=(1.0, -0.0, 0.0), weight=-0.0,
        t_world_ego=SE3((-0.0, 0.0, -0.0), (-0.0, 0.0, -0.0, 1.0)),
    )] * 64
    kwargs = {"variable": "T_ego_lidar", "t_ego_lidar": SE3.identity(),
              "observations": observations}
    assert _bits(LidarRigPointToPlaneFactor(**kwargs).residuals()) == _bits(
        ScalarFactor(**kwargs).residuals()
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_corrected_translation_keeps_scalar_path(value):
    factor, scalar = _factors(65)
    actual = factor.residuals([value, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected = scalar.residuals([value, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert all(math.isnan(item) for item in actual)
    assert all(math.isnan(item) for item in expected)


def test_large_finite_coordinates_keep_scalar_overflow_semantics():
    factor, scalar = _factors(65)
    observations = tuple(dataclasses.replace(
        observation, point_lidar_m=(1.0e308, -1.0e308, 1.0e308),
        plane_point_world_m=(-1.0e308, 1.0e308, -1.0e308), weight=1.0e308,
    ) for observation in factor.observations)
    factor.observations = scalar.observations = observations
    for actual, expected in zip(factor.residuals(), scalar.residuals(), strict=True):
        assert actual == expected or (math.isnan(actual) and math.isnan(expected))


def test_repeated_nonidentity_pose_keeps_all_scalar_normalizations():
    factor, scalar = _factors(65)
    pose = SE3((0.7, -0.4, 0.2), (0.13, -0.24, 0.19, 0.97))
    factor.observations = scalar.observations = tuple(
        dataclasses.replace(observation, t_world_ego=pose) for observation in factor.observations
    )
    assert _bits(factor.residuals([0.1, 0.2, -0.3, 0.04, -0.05, 0.06])) == _bits(
        scalar.residuals([0.1, 0.2, -0.3, 0.04, -0.05, 0.06])
    )
    assert factor._array_observations[1].shared_pose is pose
