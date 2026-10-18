from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.test_radar_joint_spatiotemporal_solver import _measurements, _reference

from calibrex.core.geometry import SE3, Vector3, rotate_vector_xyzw
from calibrex.graph.joint_optimization import JointResidualBlock, ParameterValues
from calibrex.solvers import radar_joint_spatiotemporal_solver as solver
from calibrex.solvers.radar_spatiotemporal_lever_arm_solver import (
    RadarVelocityMeasurement,
    ReferenceKinematicSample,
)


def _uncached_factor(
    measurement: RadarVelocityMeasurement,
    reference: Sequence[ReferenceKinematicSample],
    initial: SE3,
    options: solver.RadarJointSpatiotemporalOptions,
) -> JointResidualBlock:
    # Original scalar evaluation, including its exact operation order.
    def evaluator(values: ParameterValues) -> Vector3:
        transform = solver.se3_from_tangent(values[solver._EXTRINSIC_BLOCK]).compose(initial)
        offset = solver._decode_time(values[solver._TIME_BLOCK][0], options)
        kinematic = solver._interpolate(reference, measurement.timestamp_sec + offset)
        if kinematic is None:
            raise ValueError("bounded Radar time offset lost interpolation support")
        predicted = rotate_vector_xyzw(transform.rotation_quat_xyzw, measurement.velocity_radar_mps)
        expected = solver._add(kinematic[0], solver._cross(kinematic[1], transform.translation_m))
        return (
            predicted[0] - expected[0],
            predicted[1] - expected[1],
            predicted[2] - expected[2],
        )

    return JointResidualBlock(
        measurement.measurement_id,
        measurement.measurement_id,
        (solver._EXTRINSIC_BLOCK, solver._TIME_BLOCK),
        evaluator,
        weight=measurement.weight,
        family="radar_joint_spatiotemporal_velocity",
    )


def _values(tangent: tuple[float, ...], latent: float = 0.0) -> ParameterValues:
    return {solver._EXTRINSIC_BLOCK: tangent, solver._TIME_BLOCK: (latent,)}


def test_cached_factor_preserves_every_residual_bit_across_evictions() -> None:
    rng = np.random.default_rng(412)
    initial = SE3((0.42, -0.18, 0.27), (0.01, -0.04, 0.14, 0.99))
    reference = solver._ordered_reference(_reference())
    measurement = RadarVelocityMeasurement("test", 2.0, (2.1, -0.3, 0.8), weight=1.07)
    options = solver.RadarJointSpatiotemporalOptions()
    cached = solver._make_factor(
        measurement, reference, options, solver._transform_decoder(initial)
    )
    original = _uncached_factor(measurement, reference, initial, options)
    states = [
        _values(tuple(float(v) for v in rng.normal(size=6)), float(v)) for v in rng.normal(size=40)
    ]
    states += [_values((zero,) * 6, zero) for zero in (0.0, -0.0)]
    for state in [*states, *reversed(states), *states]:
        assert struct.pack("3d", *cached.residuals(state)) == struct.pack(
            "3d", *original.residuals(state)
        )


@pytest.mark.parametrize("nonidentity", [False, True])
def test_complete_weighted_joint_solve_preserves_history_and_probes(
    monkeypatch: pytest.MonkeyPatch, nonidentity: bool
) -> None:
    truth = SE3((0.32, -0.18, 0.11), (0.061917, -0.037150, 0.049534, 0.996153))
    initial = SE3((0.1, -0.05, 0.02), (0.03, -0.01, 0.02, 0.998)) if nonidentity else SE3.identity()
    reference = _reference()
    measurements = [
        replace(item, weight=0.5 + i % 7)
        for i, item in enumerate(_measurements(reference, truth, 0.02))
    ]
    measurements[4] = replace(measurements[4], velocity_radar_mps=(13.0, -12.0, 6.0))
    options = solver.RadarJointSpatiotemporalOptions(initial_time_offset_sec=-0.01)
    cached = solver.RadarJointSpatiotemporalSolver().solve(
        measurements, reference, initial, options
    )

    def original_factor(
        measurement: RadarVelocityMeasurement,
        samples: Sequence[ReferenceKinematicSample],
        opts: solver.RadarJointSpatiotemporalOptions,
        _decoder: Callable[[tuple[float, ...]], solver._FactorTransform],
    ) -> JointResidualBlock:
        return _uncached_factor(measurement, samples, initial, opts)

    monkeypatch.setattr(solver, "_make_factor", original_factor)
    original = solver.RadarJointSpatiotemporalSolver().solve(
        measurements, reference, initial, options
    )
    assert cached.as_dict() == original.as_dict()
    assert cached.status == "converged" and len(cached.probes) == 14
    assert cached.rejected_initial_residual_ids == (measurements[4].measurement_id,)
    assert cached.optimizer is not None and cached.optimizer.history


def test_transform_cache_is_bounded_and_preserves_signed_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = solver.se3_from_tangent
    decodes: list[bytes] = []

    def counted(values: Sequence[float]) -> SE3:
        decodes.append(struct.pack("6d", *values))
        return original(values)

    monkeypatch.setattr(solver, "se3_from_tangent", counted)
    decoder = solver._transform_decoder(SE3.identity())
    positive = (0.0,) * 6
    negative = (-0.0,) * 6
    decoder(positive)
    decoder(negative)
    for _ in range(10):
        decoder(positive)
        decoder(negative)
    assert decodes == [struct.pack("6d", *positive), struct.pack("6d", *negative)]
    for value in range(1, 33):
        decoder((float(value),) + (0.0,) * 5)
    decoder(positive)
    assert len(decodes) == 35  # The oldest entry was evicted after 32 distinct newer states.


def test_interpolation_reuse_is_local_to_measurement_and_keeps_two_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = solver._ordered_reference(_reference())
    decoder = solver._transform_decoder(SE3.identity())
    options = solver.RadarJointSpatiotemporalOptions()
    factors = [
        solver._make_factor(
            RadarVelocityMeasurement(str(i), 2.0 + i, (1.0, 2.0, 3.0)), reference, options, decoder
        )
        for i in range(2)
    ]
    original = solver._interpolate
    queries: list[float] = []

    def counted(
        samples: Sequence[ReferenceKinematicSample], timestamp: float
    ) -> tuple[Vector3, Vector3] | None:
        queries.append(timestamp)
        return original(samples, timestamp)

    monkeypatch.setattr(solver, "_interpolate", counted)
    for factor in factors:
        for i in range(12):
            for latent in (-1.0e-5, 1.0e-5):
                factor.residuals(_values((float(i),) + (0.0,) * 5, latent))
    assert len(queries) == 4
    # A third clock displaces the least recent entry in each two-entry cache.
    factors[0].residuals(_values((0.0,) * 6, 0.0))
    factors[0].residuals(_values((0.0,) * 6, -1.0e-5))
    assert len(queries) == 6


def test_independent_decoders_capture_their_own_initial_transform() -> None:
    tangent = (0.12, -0.23, 0.34, 0.2, -0.1, 0.3)
    first = solver._transform_decoder(SE3.identity())
    initial = SE3((1.0, 2.0, 3.0), (0.2, -0.3, 0.1, 0.9))
    second = solver._transform_decoder(initial)
    assert first(tangent).transform == solver.se3_from_tangent(tangent).compose(SE3.identity())
    assert second(tangent).transform == solver.se3_from_tangent(tangent).compose(initial)
    assert first(tangent).transform != second(tangent).transform


def test_cached_factor_still_rejects_unsupported_clock() -> None:
    factor = solver._make_factor(
        RadarVelocityMeasurement("unsupported", -1.0, (1.0, 2.0, 3.0)),
        solver._ordered_reference(_reference()),
        solver.RadarJointSpatiotemporalOptions(),
        solver._transform_decoder(SE3.identity()),
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="lost interpolation support"):
            factor.residuals(_values((0.0,) * 6, math.inf))
