from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from tests.unit.test_radar_spatiotemporal_lever_arm_solver import _synthetic_problem

from calibrex.solvers import radar_spatiotemporal_lever_arm_solver as solver


@pytest.mark.parametrize(
    "times",
    [
        (0.0, 0.1, 0.7, 2.0),
        (0.0, 0.0, 0.1, 0.7, 2.0),
        (0.0, 0.1, 0.1, 0.7, 2.0),
        (0.0, 0.1, 0.7, 2.0, 2.0),
        (0.0, 0.0, 0.0),
        (1.0e9, 1.0e9 + 0.1, 1.0e9 + 0.7, 1.0e9 + 2.0),
    ],
)
def test_indexed_profiles_preserve_interval_and_float_bits(times: tuple[float, ...]) -> None:
    rng = np.random.default_rng(43)
    reference = [
        solver.ReferenceKinematicSample(
            timestamp, tuple(rng.normal(size=3)), tuple(rng.normal(size=3))
        )
        for timestamp in times
    ]
    queries = [times[0] - 0.1, times[-1] + 0.1]
    for timestamp in times:
        queries += [
            float(np.nextafter(timestamp, -math.inf)),
            timestamp,
            float(np.nextafter(timestamp, math.inf)),
        ]
    queries += [(left + right) / 2.0 for left, right in pairwise(times)]
    for query in queries:
        measurements = [solver.RadarVelocityMeasurement("test", query, (0.2, -0.7, 1.1), 1.7)]
        previous = solver._profile_candidate(measurements, reference, (0.0, 0.0, 0.0, 1.0), 0.0)
        indexed = solver._profile_candidate(
            measurements,
            reference,
            (0.0, 0.0, 0.0, 1.0),
            0.0,
            reference_timestamps=times,
        )
        assert (previous is None) == (indexed is None)
        if previous is not None and indexed is not None:
            assert previous[0].tobytes() == indexed[0].tobytes()
            assert previous[1].hex() == indexed[1].hex()
            assert previous[2].tobytes() == indexed[2].tobytes()


def test_complete_solver_matches_linear_reference_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    measurements, reference, rotation = _synthetic_problem()
    # Include unequal weights and duplicate reference knots with different values.
    measurements = [
        replace(item, weight=0.5 + index % 7) for index, item in enumerate(measurements)
    ]
    reference.insert(111, replace(reference[110], linear_velocity_body_mps=(4.0, -2.0, 0.3)))
    options = solver.RadarSpatiotemporalLeverArmOptions(
        max_abs_time_offset_sec=0.05, holdout_ratio=0.25, split_seed=19
    )
    indexed = solver.RadarSpatiotemporalLeverArmSolver().solve(
        measurements, list(reversed(reference)), rotation, options
    )
    original = solver._profile_candidate

    def linear(*args: object, **kwargs: object) -> solver.ProfileCandidate | None:
        kwargs.pop("reference_timestamps", None)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(solver, "_profile_candidate", linear)
    previous = solver.RadarSpatiotemporalLeverArmSolver().solve(
        measurements, list(reversed(reference)), rotation, options
    )
    assert indexed.as_dict() == previous.as_dict()
    assert len(indexed.probes) == 8


def test_each_profile_query_reads_at_most_two_reference_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurements, reference, rotation = _synthetic_problem()
    original = solver._interpolate
    lengths: list[int] = []

    def record(samples: list[solver.ReferenceKinematicSample], timestamp: float):
        lengths.append(len(samples))
        return original(samples, timestamp)

    monkeypatch.setattr(solver, "_interpolate", record)
    result = solver._profile_candidate(
        measurements,
        reference,
        rotation,
        0.023,
        reference_timestamps=tuple(item.timestamp_sec for item in reference),
    )
    assert result is not None
    assert len(lengths) == len(measurements)
    assert set(lengths) == {2}


@pytest.mark.parametrize("timestamp", [math.nan, math.inf, -math.inf])
def test_nonfinite_reference_clocks_keep_the_linear_path(
    monkeypatch: pytest.MonkeyPatch, timestamp: float
) -> None:
    measurements, reference, rotation = _synthetic_problem()
    reference[30] = replace(reference[30], timestamp_sec=timestamp)
    original = solver._profile_candidate
    indexes: list[object] = []

    def record(*args: object, **kwargs: object) -> solver.ProfileCandidate | None:
        indexes.append(kwargs.get("reference_timestamps"))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(solver, "_profile_candidate", record)
    solver.RadarSpatiotemporalLeverArmSolver().solve(measurements, reference, rotation)
    assert len(indexes) == 201
    assert all(index is None for index in indexes)
