from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from calibrex.solvers import radar_ego_velocity_solver as solver


def observations(count: int, *, planar: bool = False) -> list[solver.RadarDopplerObservation]:
    rng = np.random.default_rng(918)
    directions = rng.normal(size=(count, 3))
    if planar:
        directions[:, 2] = 0.0
    truth = (8.0, -1.5, 0.7)
    result = []
    for index, direction in enumerate(directions):
        x, y, z = (float(value) for value in direction)
        norm = math.sqrt(x * x + y * y + z * z)
        doppler = -(x * truth[0] + y * truth[1] + z * truth[2]) / norm
        doppler += 0.01 * math.sin(index) + (4.0 if index % 17 == 0 else 0.0)
        result.append(solver.RadarDopplerObservation((x, y, z), doppler, 1.0 + (index % 5) / 10.0))
    return result


def bits(value: Any) -> bytes:
    return np.asarray(value, dtype=np.float64).tobytes()


@pytest.mark.parametrize("count", [64, 65, 4095, 4096, 4097])
def test_normal_equations_keep_scalar_accumulation_bits_across_chunks(count: int) -> None:
    rows = [solver._normalized_row(item) for item in observations(count)]
    prepared = solver._prepare_rows(rows)
    assert prepared is not None
    for robust in (None, np.linspace(0.01, 1.0, count)):
        expected_hessian, expected_gradient = solver._normal_equations(rows, robust)
        hessian, gradient = prepared.normal_equations(robust)
        assert bits(hessian) == bits(expected_hessian)
        assert bits(gradient) == bits(expected_gradient)


def test_residuals_and_huber_weights_keep_scalar_bits_and_signed_zeros() -> None:
    rows = [solver._normalized_row(item) for item in observations(129)]
    rows[:4] = [((0.0, -0.0, 1.0), -0.0, 1.0)] * 4
    prepared = solver._prepare_rows(rows)
    assert prepared is not None
    for velocity in ((0.0, -0.0, 0.0), (8.0, -1.5, 0.7), (-3.0, 2.0, -1.0)):
        expected = [solver._residual(row, velocity) for row in rows]
        actual = prepared.residuals(velocity)
        assert bits(actual) == bits(expected)
        for delta in (0.0, -1.0, 0.2, math.inf):
            assert bits(prepared.weights(actual, delta)) == bits(
                [solver._huber_weight(value, delta) for value in expected]
            )
    for robust in (None, np.ones(len(rows))):
        hessian, gradient = prepared.normal_equations(robust)
        expected_hessian, expected_gradient = solver._normal_equations(rows, robust)
        assert bits(hessian) == bits(expected_hessian)
        assert bits(gradient) == bits(expected_gradient)


@pytest.mark.parametrize(
    "count,planar,iterations",
    [
        (63, False, 20),
        (64, False, 20),
        (512, False, 20),
        (129, True, 20),
        (128, False, 1),
        (128, False, 0),
    ],
)
def test_every_irls_system_and_complete_result_match_scalar(
    count: int,
    planar: bool,
    iterations: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = observations(count, planar=planar)
    # Filtering keeps the original order and normalization precedes preparation.
    samples.insert(2, solver.RadarDopplerObservation((0.0, 0.0, 0.0), 1.0, 0.0))
    options = solver.RadarEgoVelocitySolverOptions(max_iterations=iterations)
    original_solve = solver._solve_3x3
    original_eigen = solver._symmetric_eigenvalues_3x3
    histories: list[list[bytes]] = []
    for scalar in (False, True):
        history: list[bytes] = []

        def solve(matrix: Any, rhs: Any, history: list[bytes] = history) -> Any:
            value = original_solve(matrix, rhs)
            history.extend((bits(matrix), bits(rhs), bits(value)))
            return value

        def eigen(matrix: Any, history: list[bytes] = history) -> Any:
            history.append(bits(matrix))
            return original_eigen(matrix)

        with monkeypatch.context() as patch:
            patch.setattr(solver, "_solve_3x3", solve)
            patch.setattr(solver, "_symmetric_eigenvalues_3x3", eigen)
            if scalar:
                patch.setattr(solver, "_prepare_rows", lambda rows: None)
            result = solver.RadarEgoVelocitySolver().solve(samples, options)
        history.append(json.dumps(result.as_dict(), allow_nan=False, sort_keys=True).encode())
        histories.append(history)
    assert histories[0] == histories[1]
    if not planar and iterations == 20:
        assert result.velocity_radar_mps is not None
        assert math.dist(result.velocity_radar_mps, (8.0, -1.5, 0.7)) < 0.05


@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_chunk_size_only_bounds_storage(
    chunk: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [solver._normalized_row(item) for item in observations(130)]
    prepared = solver._prepare_rows(rows)
    assert prepared is not None
    monkeypatch.setattr(solver, "_BATCH_ROWS", chunk)
    allocations = []
    original = np.empty

    def empty(shape: Any, *args: Any, **kwargs: Any) -> Any:
        allocations.append(shape)
        return original(shape, *args, **kwargs)

    monkeypatch.setattr(np, "empty", empty)
    actual = prepared.normal_equations(None)
    expected = solver._normal_equations(rows, None)
    assert bits(actual[0]) == bits(expected[0]) and bits(actual[1]) == bits(expected[1])
    assert all(shape[0] <= chunk + 1 and shape[1] == 12 for shape in allocations)


def test_small_unusual_and_extreme_rows_keep_the_scalar_path() -> None:
    rows = [solver._normalized_row(item) for item in observations(64)]
    assert solver._prepare_rows(rows[:63]) is None
    for target in (float("inf"), float("nan"), 1e101, np.float32(1.0)):
        changed = rows.copy()
        changed[0] = (changed[0][0], target, 1.0)
        assert solver._prepare_rows(changed) is None
    with pytest.raises(ValueError, match="non-zero"):
        solver.RadarEgoVelocitySolver().solve(
            [solver.RadarDopplerObservation((0.0, 0.0, 0.0), 1.0)] * 64
        )


def test_new_fit_observes_input_and_option_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = observations(128)
    first = solver.RadarEgoVelocitySolver().solve(samples)
    samples[3] = replace(samples[3], measured_radial_velocity_mps=2.0, weight=4.0)
    options = solver.RadarEgoVelocitySolverOptions(huber_delta_mps=0.4)
    changed = solver.RadarEgoVelocitySolver().solve(samples, options)
    monkeypatch.setattr(solver, "_prepare_rows", lambda rows: None)
    scalar = solver.RadarEgoVelocitySolver().solve(samples, options)
    assert changed.as_dict() == scalar.as_dict()
    assert changed.as_dict() != first.as_dict()
