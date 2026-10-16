from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from tests.unit.test_radar_joint_spatiotemporal_solver import _measurements, _reference

from calibrex.core.geometry import SE3
from calibrex.solvers import radar_joint_spatiotemporal_solver as solver
from calibrex.solvers.radar_spatiotemporal_lever_arm_solver import ReferenceKinematicSample


@pytest.mark.parametrize(
    "times",
    [
        (),
        (0.0,),
        (0.0, 0.1, 0.7, 2.0),
        (0.0, 0.0, 0.1, 0.7, 2.0),
        (0.0, 0.1, 0.1, 0.7, 2.0),
        (0.0, 0.1, 0.7, 2.0, 2.0),
        (0.0, 0.0, 0.0),
        (1.0e9, 1.0e9 + 0.1, 1.0e9 + 0.7, 1.0e9 + 2.0),
    ],
)
def test_reference_interpolation_preserves_float_bits(times: tuple[float, ...]) -> None:
    rng = np.random.default_rng(57)
    samples = [
        ReferenceKinematicSample(t, tuple(rng.normal(size=3)), tuple(rng.normal(size=3)))
        for t in times
    ]
    input_samples = list(reversed(samples))
    prepared = solver._ordered_reference(input_samples)
    # Stable sorting must retain input order within duplicate-clock groups.
    original = sorted(input_samples, key=lambda item: item.timestamp_sec)
    queries = [math.nan, math.inf, -math.inf, -0.0, 2.0]
    for timestamp in times:
        queries += [
            float(np.nextafter(timestamp, -math.inf)),
            timestamp,
            float(np.nextafter(timestamp, math.inf)),
        ]
    queries += [(left + right) / 2 for left, right in pairwise(times)]
    for timestamp in queries:
        linear = solver._interpolate(original, timestamp)
        indexed = solver._interpolate(prepared, timestamp)
        assert (linear is None) == (indexed is None)
        if linear is not None and indexed is not None:
            assert struct.pack("6d", *linear[0], *linear[1]) == struct.pack(
                "6d", *indexed[0], *indexed[1]
            )


def test_complete_joint_solver_and_history_match_linear_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth = SE3.from_lists([0.32, -0.18, 0.11], [0.061917, -0.037150, 0.049534, 0.996153])
    reference = _reference()
    measurements = [
        replace(item, weight=0.5 + i % 7)
        for i, item in enumerate(_measurements(reference, truth, 0.02))
    ]
    # Exercise the original initial-residual rejection as well as all probes.
    measurements[4] = replace(measurements[4], velocity_radar_mps=(13.0, -12.0, 6.0))
    indexed = solver.RadarJointSpatiotemporalSolver().solve(
        measurements, list(reversed(reference)), SE3.identity()
    )

    def linear(samples: Sequence[ReferenceKinematicSample]) -> list[ReferenceKinematicSample]:
        return sorted(samples, key=lambda item: item.timestamp_sec)

    monkeypatch.setattr(solver, "_ordered_reference", linear)
    previous = solver.RadarJointSpatiotemporalSolver().solve(
        measurements, list(reversed(reference)), SE3.identity()
    )
    assert indexed.as_dict() == previous.as_dict()
    assert indexed.status == "converged"
    assert indexed.rejected_initial_residual_ids == (measurements[4].measurement_id,)
    assert len(indexed.probes) == 14
    assert indexed.optimizer is not None and indexed.optimizer.history


@pytest.mark.parametrize("timestamp", [math.nan, math.inf, -math.inf])
def test_nonfinite_reference_clocks_keep_linear_semantics(timestamp: float) -> None:
    reference = _reference()
    reference[30] = replace(reference[30], timestamp_sec=timestamp)
    prepared = solver._ordered_reference(reference)
    assert not isinstance(prepared, solver._IndexedReference)
    assert prepared == sorted(reference, key=lambda item: item.timestamp_sec)


def test_public_rmse_duplicate_clocks_keep_the_original_interval() -> None:
    reference = _reference()
    reference.insert(0, replace(reference[0], linear_velocity_body_mps=(9.0, 7.0, 5.0)))
    measurement = solver.RadarVelocityMeasurement("first", 0.0, (2.0, 1.0, 0.0))
    expected = solver._interpolate(reference, 0.0)
    assert expected is not None
    error = sum((measurement.velocity_radar_mps[i] - expected[0][i]) ** 2 for i in range(3))
    assert solver.radar_joint_spatiotemporal_rmse(
        [measurement], reference, SE3.identity(), 0.0
    ) == math.sqrt(error / 3)


def test_single_query_uses_logarithmic_reference_access(monkeypatch: pytest.MonkeyPatch) -> None:
    count = 8192
    reference = solver._IndexedReference(
        ReferenceKinematicSample(float(i), (1.0, 2.0, 3.0), (0.1, 0.2, 0.3)) for i in range(count)
    )
    original = solver._TIMESTAMP
    reads: list[float] = []

    def timestamp(sample: ReferenceKinematicSample) -> float:
        reads.append(sample.timestamp_sec)
        return original(sample)

    monkeypatch.setattr(solver, "_TIMESTAMP", timestamp)
    assert solver._interpolate(reference, count - 1.25) is not None
    assert 0 < len(reads) <= math.ceil(math.log2(count)) + 1
