"""Verify prepared spline terms against the independent public evaluator."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from calibrex.core.continuous_time_bspline import (
    BSplineStates,
    CubicBSplineTrajectory,
    so3_exp_batch,
)
from calibrex.core.continuous_time_bspline_derivatives import _rotation_control_derivatives
from calibrex.core.continuous_time_bspline_fit import _gather


@pytest.mark.parametrize("epoch", [0.0, 1.76e9])
def test_prepared_states_and_derivatives_match_reference(epoch: float) -> None:
    rng = np.random.default_rng(442)
    rotations = np.stack([np.eye(3)] * 13)
    for slot in range(1, len(rotations)):
        rotations[slot] = rotations[slot - 1] @ so3_exp_batch(rng.normal(0, 0.3, (1, 3)))[0]
    path = CubicBSplineTrajectory(epoch, 0.2, rotations, rng.normal(size=(13, 3)))
    # Repeated, unsorted samples and exact knots must retain their row order.
    times = epoch + np.array([0, 0.2, 1.6, 0.39, 0.2, 2.0, 0.400001, 0.399999])
    gathered = _gather(path, times, derivatives=True)
    expected = path.evaluate(times)
    for field in fields(BSplineStates):
        assert np.array_equal(getattr(gathered.states(), field.name), getattr(expected, field.name))
    for order in (0, 1, 2):
        states = gathered.states(order)
        assert np.array_equal(states.rotation_world_body, expected.rotation_world_body)
        assert np.array_equal(states.position_world_m, expected.position_world_m)
        if order > 0:
            assert np.array_equal(
                states.angular_velocity_body_rad_s, expected.angular_velocity_body_rad_s
            )
        reference = _rotation_control_derivatives(
            gathered.rotations, gathered.fraction, gathered.spacing, order=order
        )
        shared = _rotation_control_derivatives(
            gathered.rotations, gathered.fraction, gathered.spacing, order=order,
            relatives=gathered.relatives, deltas=gathered.deltas,
            steps=gathered.steps, log_jacobians=gathered.log_jacobians,
            basis_terms=(gathered.basis, gathered.d_basis, gathered.dd_basis),
        )
        for field in fields(reference):
            assert np.array_equal(getattr(reference, field.name), getattr(shared, field.name))


def test_unused_near_pi_edge_is_not_evaluated_and_control_edits_are_visible() -> None:
    rotations = np.stack([np.eye(3)] * 7)
    rotations[-1] = so3_exp_batch(np.array([[np.pi - 1e-7, 0, 0]]))[0]
    path = CubicBSplineTrajectory(0, 0.1, rotations, np.zeros((7, 3)))
    times = np.array([0.01, 0.09])
    assert np.array_equal(_gather(path, times).states().rotation_world_body,
                          path.evaluate(times).rotation_world_body)
    path.rotations[1] = so3_exp_batch(np.array([[0.1, -0.2, 0.3]]))[0]
    path.positions[1] = [1, 2, 3]
    actual = _gather(path, times).states()
    expected = path.evaluate(times)
    assert np.array_equal(actual.rotation_world_body, expected.rotation_world_body)
    assert np.array_equal(actual.position_world_m, expected.position_world_m)
