"""Feasible IMU clock updates and one-sided stationarity at screw knots."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.core import continuous_time_sparse as sparse
from calibrex.core.clock_domain import ClockOffsetDomain
from calibrex.core.geometry import SE3


def _gyro(name: str, start: float, end: float) -> sparse.TrajectoryImuPreintegrationMeasurement:
    return sparse.TrajectoryImuPreintegrationMeasurement(
        name,
        (
            sparse.TrajectoryImuGyroSample(start, (0.0, 0.0, 0.0)),
            sparse.TrajectoryImuGyroSample(end, (0.0, 0.0, 0.0)),
        ),
    )


def _problem(
    *, offset: float = 0.0, plateau: bool = False
) -> sparse.ContinuousTimeTrajectoryFitProblem:
    times = (0.0, 0.5, 1.0, 1.5, 2.0)
    angles = (0.0, 0.0, 0.4, 0.4, 0.4) if plateau else (0.0, 0.2, 0.4, 0.6, 0.6)
    knots = tuple(
        SE3((0.0, 0.0, 0.0), tuple(Rotation.from_rotvec([0.0, 0.0, a]).as_quat())) for a in angles
    )
    return sparse.ContinuousTimeTrajectoryFitProblem(
        times,
        knots,
        pose_measurements=tuple(
            sparse.TrajectoryPoseMeasurement(str(i), t, k, 1.0e6)
            for i, (t, k) in enumerate(zip(times, knots, strict=True))
        ),
        imu_preintegration_measurements=(_gyro("middle", 0.5 + offset, 1.5 + offset),),
        initial_imu_clock_offset_sec=offset,
        estimate_imu_clock_offset=True,
    )


def test_domain_is_intersection_not_union_and_derivatives_stay_inside() -> None:
    domain = ClockOffsetDomain(0.0, 2.0, 0.2, 1.8)
    assert domain.lower == pytest.approx(-0.2) and domain.upper == 0.2
    for clock in (domain.lower, 0.0, domain.upper):
        step = domain.difference_step(clock)
        assert step != 0 and domain.contains(clock + step)
    assert domain.difference_step(domain.lower) > 0
    assert domain.difference_step(domain.upper) < 0
    assert not domain.contains(0.2001) and not domain.contains(float("nan"))


def test_narrow_and_large_epoch_domains_use_actual_representable_increments() -> None:
    for domain in (
        ClockOffsetDomain(0.0, 1.0, 1.0e-9, 1.0 - 1.0e-9),
        ClockOffsetDomain(0.0, 2.0, 1.7e9 + 0.2, 1.7e9 + 1.8),
    ):
        clock = (domain.lower + domain.upper) / 2
        step = domain.difference_step(clock)
        assert step != 0 and domain.contains(clock + step)
        assert (clock + step) - clock == step


@pytest.mark.parametrize("last", [2.0, 2.1])
def test_clock_estimation_refuses_no_width(last: float) -> None:
    with pytest.raises(ValueError, match="positive-width"):
        ClockOffsetDomain(0.0, 2.0, 0.0, last)


def test_bad_clock_proposal_is_not_evaluated_and_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    original = sparse._objective
    observed = []
    domain = ClockOffsetDomain(0.0, 2.0, 0.5, 1.5)

    def objective(*args, **kwargs):
        clock = args[9]
        assert domain.contains(clock)
        observed.append(clock)
        return original(*args, **kwargs)

    def outside(matrix, rhs):
        step = np.zeros_like(rhs)
        step[-1] = 100.0
        return step

    monkeypatch.setattr(sparse, "_objective", objective)
    monkeypatch.setattr(sparse, "spsolve", outside)
    result = sparse.fit_continuous_trajectory(
        problem,
        sparse.ContinuousTimeTrajectoryFitOptions(
            initial_damping=1.0e12,
            convergence_gradient_norm=1.0e-12,
        ),
    )
    assert observed == [0.0]
    assert result.imu_clock_offset_sec == 0.0 and result.knot_poses == problem.initial_knot_poses
    assert result.status == "stalled"
    assert result.diagnostics.termination_reason == "clock_domain_limit"
    assert result.diagnostics.clock_domain.rejected_steps == 1


def test_feasible_clock_fit_can_resume_after_a_rejected_proposal(monkeypatch) -> None:
    original = sparse.spsolve
    calls = 0

    def solve(matrix, rhs):
        nonlocal calls
        calls += 1
        step = original(matrix, rhs)
        if calls == 1:
            step[-1] = 100.0
        return step

    monkeypatch.setattr(sparse, "spsolve", solve)
    result = sparse.fit_continuous_trajectory(
        _problem(offset=0.03),
        sparse.ContinuousTimeTrajectoryFitOptions(
            max_iterations=5, convergence_gradient_norm=1.0e-12
        ),
    )
    assert calls > 1
    assert result.diagnostics.clock_domain.rejected_steps >= 1
    assert 0.03 - 0.5 <= result.imu_clock_offset_sec <= 0.03 + 0.5


def test_global_upper_bound_uses_one_inward_side_for_every_gyro_factor(monkeypatch) -> None:
    problem = _problem(offset=0.25)
    first = _gyro("first", 0.25, 0.75)
    problem = replace(
        problem, imu_preintegration_measurements=(first, *problem.imu_preintegration_measurements)
    )
    original = sparse._linearize_imu_preintegration
    sides = []

    def linearize(*args, **kwargs):
        sides.append(kwargs.get("clock_side"))
        return original(*args, **kwargs)

    monkeypatch.setattr(sparse, "_linearize_imu_preintegration", linearize)
    # A finite clock-free proposal keeps the initial boundary state for inspection.
    monkeypatch.setattr(sparse, "spsolve", lambda matrix, rhs: np.zeros_like(rhs))
    result = sparse.fit_continuous_trajectory(
        problem,
        sparse.ContinuousTimeTrajectoryFitOptions(
            max_iterations=1, convergence_gradient_norm=1.0e-12
        ),
    )
    assert sides[:2] == ["right", "right"]
    assert result.diagnostics.clock_domain.derivative_step_sec < 0
    # Independently check the second factor against the same negative-clock side.
    measurement = problem.imu_preintegration_measurements[1]
    knots = list(problem.initial_knot_poses)
    factor = original(
        knots, problem.knot_timestamps, measurement, np.zeros(3), 0.25, clock_side="right"
    )
    epsilon = 1.0e-7
    shifted = original(knots, problem.knot_timestamps, measurement, np.zeros(3), 0.25 - epsilon)
    np.testing.assert_allclose(
        factor.jacobian_clock[:, 0], (shifted.residual - factor.residual) / (-epsilon), atol=1.0e-7
    )


def test_one_zero_clock_side_at_a_knot_cannot_hide_opposite_descent() -> None:
    # Positive-clock endpoints have zero slopes, hence a zero column.
    # Negative-clock endpoints see .8 and 0 rad/s: negative steps reduce residual.
    result = sparse.fit_continuous_trajectory(
        _problem(plateau=True),
        sparse.ContinuousTimeTrajectoryFitOptions(
            convergence_gradient_norm=1.0e-6,
        ),
    )
    diagnostics = result.diagnostics
    assert diagnostics.normalized_gradient_norm < diagnostics.gradient_tolerance
    assert diagnostics.clock_domain.opposite_gradient_norm > diagnostics.gradient_tolerance
    assert not diagnostics.stationary and result.status == "stalled"
    assert diagnostics.termination_reason == "stationarity_not_confirmed"


def test_all_imu_families_contribute_to_domain() -> None:
    p = _problem()
    p = replace(
        p,
        imu_lever_arm_measurements=(
            sparse.TrajectoryImuLeverArmMeasurement("accel", 0.1, (0.0, 0.0, 0.0)),
        ),
    )
    result = sparse.fit_continuous_trajectory(
        p, sparse.ContinuousTimeTrajectoryFitOptions(max_iterations=1)
    )
    assert result.diagnostics.clock_domain.upper_sec == 0.1
    assert result.diagnostics.clock_domain.lower_sec == -0.5


def test_fixed_clock_remains_supported_when_offset_cannot_be_estimated() -> None:
    problem = replace(_problem(), imu_preintegration_measurements=(_gyro("full", 0.0, 2.0),))
    with pytest.raises(ValueError, match="positive-width"):
        sparse.fit_continuous_trajectory(problem)
    result = sparse.fit_continuous_trajectory(replace(problem, estimate_imu_clock_offset=False))
    assert result.diagnostics.clock_domain is None


def test_accelerometer_clock_probes_obey_the_shared_gyro_domain(monkeypatch) -> None:
    problem = replace(
        _problem(offset=0.25),
        imu_preintegration_measurements=(_gyro("first", 0.25, 0.75),),
        imu_lever_arm_measurements=(
            sparse.TrajectoryImuLeverArmMeasurement("accel", 1.25, (0.0, 0.0, 0.0)),
        ),
    )
    domain = ClockOffsetDomain(0.0, 2.0, 0.25, 1.25)
    original = sparse._evaluate_lever_arm
    probed = []

    def evaluate(*args, **kwargs):
        clock = args[4]
        assert domain.contains(clock)
        probed.append(clock)
        return original(*args, **kwargs)

    monkeypatch.setattr(sparse, "_evaluate_lever_arm", evaluate)
    monkeypatch.setattr(sparse, "spsolve", lambda matrix, rhs: np.zeros_like(rhs))
    sparse.fit_continuous_trajectory(
        problem,
        sparse.ContinuousTimeTrajectoryFitOptions(
            max_iterations=1, convergence_gradient_norm=1.0e-12
        ),
    )
    assert any(clock < 0.25 for clock in probed)
    assert all(clock <= 0.25 for clock in probed)
