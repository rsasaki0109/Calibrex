"""Independent weighted least-squares checks for the native screw fitter."""

from __future__ import annotations

import numpy as np
import pytest

from calibrex.core import continuous_time_sparse as sparse
from calibrex.core.continuous_time_marginalization import KnotMarginalizationPrior
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_log

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
_ANCHORS = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _score(problem: sparse.ContinuousTimeTrajectoryFitProblem) -> float:
    """Evaluate exactly the state in the problem, with no solver update."""

    return sparse._objective(
        problem,
        [(0, item.point_body_m) for item in problem.point_measurements],
        [(0, item) for item in problem.point_to_plane_measurements],
        [(0, item.pose_world_body) for item in problem.pose_measurements],
        list(problem.imu_preintegration_measurements),
        list(problem.imu_lever_arm_measurements),
        list(problem.initial_knot_poses),
        np.asarray(problem.initial_gyro_bias_rad_s),
        np.asarray(problem.initial_lever_arm_body_m),
        problem.initial_imu_clock_offset_sec,
        np.asarray(problem.initial_accel_bias_body_m_s2),
        np.asarray(problem.initial_gravity_world_m_s2),
        np.asarray(problem.initial_gyro_scale),
        np.asarray(problem.initial_accel_scale),
    )


@pytest.mark.parametrize("weight", [0.25, 1.0, 3.0])
@pytest.mark.parametrize("family", ["point", "plane", "pose", "gyro", "accel"])
def test_each_factor_uses_inverse_sigma_and_keeps_information_prior_units(
    family: str, weight: float
) -> None:
    # All residual norms below follow directly from static translations or a
    # constant z gyro over one second; no production residual computes the oracle.
    knot = SE3((0.3, -0.2, 0.1), _IDENTITY_QUAT)
    points = ()
    planes = ()
    poses = ()
    gyro = ()
    accel = ()
    if family == "point":
        points = (
            sparse.TrajectoryPointMeasurement(
                "point", 0.5, (0.0, 0.0, 0.0), (1.0, -2.0, 0.5), weight
            ),
        )
        squared_error = 0.7**2 + 1.8**2 + 0.4**2
    elif family == "plane":
        planes = (
            sparse.TrajectoryPointToPlaneMeasurement(
                "plane", 0.5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, 2.0), weight
            ),
        )
        squared_error = 0.4**2
    elif family == "pose":
        poses = (
            sparse.TrajectoryPoseMeasurement(
                "pose", 0.5, SE3((0.6, -0.4, 0.3), _IDENTITY_QUAT), weight
            ),
        )
        squared_error = 4.0 * (0.3**2 + 0.2**2 + 0.2**2)
    elif family == "gyro":
        gyro = (
            sparse.TrajectoryImuPreintegrationMeasurement(
                "gyro",
                (
                    sparse.TrajectoryImuGyroSample(0.0, (0.0, 0.0, 0.2)),
                    sparse.TrajectoryImuGyroSample(1.0, (0.0, 0.0, 0.2)),
                ),
                weight,
            ),
        )
        squared_error = 0.2**2
    else:
        accel = (
            sparse.TrajectoryImuLeverArmMeasurement("accel", 0.5, (1.0, -2.0, 0.5), weight),
        )
        squared_error = 1.0**2 + 2.0**2 + 0.5**2
    problem = sparse.ContinuousTimeTrajectoryFitProblem(
        (0.0, 1.0),
        (knot, knot),
        point_measurements=points,
        point_to_plane_measurements=planes,
        pose_measurements=poses,
        imu_preintegration_measurements=gyro,
        imu_lever_arm_measurements=accel,
        knot_marginalization_priors=(
            KnotMarginalizationPrior(0, SE3.identity(), np.diag([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])),
        ),
    )
    # Prior.information is already information, unlike a measurement's 1/sigma.
    prior_error = 2.0 * 0.3**2 + 3.0 * 0.2**2 + 5.0 * 0.1**2
    assert _score(problem) == pytest.approx(weight**2 * squared_error + prior_error, abs=1e-12)


def test_objective_directional_derivative_matches_weighted_normal_model() -> None:
    def problem_at(x: float) -> sparse.ContinuousTimeTrajectoryFitProblem:
        knot = SE3((x, 0.0, 0.0), _IDENTITY_QUAT)
        return sparse.ContinuousTimeTrajectoryFitProblem(
            (0.0, 1.0),
            (knot, knot),
            point_measurements=(
                sparse.TrajectoryPointMeasurement("a", 0.5, (0.0,) * 3, (0.0,) * 3, 1.0),
                sparse.TrajectoryPointMeasurement("b", 0.5, (0.0,) * 3, (2.0, 0.0, 0.0), 3.0),
            ),
        )

    epsilon = 1e-6
    numeric = (_score(problem_at(1.0 + epsilon)) - _score(problem_at(1.0 - epsilon))) / (
        2.0 * epsilon
    )
    # d/dx [x^2 + 9 (x - 2)^2] at x=1 is -16.
    assert numeric == pytest.approx(-16.0, abs=1e-7)
    problem = problem_at(1.0)
    direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    model = 0.0
    for item in problem.point_measurements:
        residual, left, right = sparse._point_residual(
            list(problem.initial_knot_poses),
            problem.knot_timestamps,
            0,
            item.timestamp_sec,
            item.point_body_m,
            item.target_world_m,
        )
        model += 2.0 * float(
            (item.weight * residual) @ (item.weight * ((left + right) @ direction))
        )
    assert model == pytest.approx(numeric, abs=1e-7)


@pytest.mark.parametrize("family", ["point", "plane", "pose"])
def test_solver_reaches_closed_form_weighted_optimum_from_old_loss_minimum(family: str) -> None:
    points = []
    planes = []
    poses = []
    for knot_index, timestamp in enumerate((0.0, 1.0)):
        for target_x, weight in ((0.0, 1.0), (2.0, 3.0)):
            prefix = f"{knot_index}-{target_x}"
            if family == "pose":
                poses.append(
                    sparse.TrajectoryPoseMeasurement(
                        prefix, timestamp, SE3((target_x, 0.0, 0.0), _IDENTITY_QUAT), weight
                    )
                )
                continue
            for index, anchor in enumerate(_ANCHORS):
                target = (anchor[0] + target_x, anchor[1], anchor[2])
                if family == "point":
                    points.append(
                        sparse.TrajectoryPointMeasurement(
                            f"{prefix}-{index}", timestamp, anchor, target, weight
                        )
                    )
                else:
                    for axis in range(3):
                        normal = tuple(float(component == axis) for component in range(3))
                        planes.append(
                            sparse.TrajectoryPointToPlaneMeasurement(
                                f"{prefix}-{index}-{axis}",
                                timestamp, anchor, target, normal, weight,
                            )
                        )
    # x=1.5 minimizes the old, inconsistent x^2 + 3(x-2)^2 acceptance loss.
    initial = SE3((1.5, 0.0, 0.0), _IDENTITY_QUAT)
    result = sparse.fit_continuous_trajectory(
        sparse.ContinuousTimeTrajectoryFitProblem(
            (0.0, 1.0),
            (initial, initial),
            point_measurements=tuple(points),
            point_to_plane_measurements=tuple(planes),
            pose_measurements=tuple(poses),
        )
    )
    # The independent inverse-variance least-squares solution is (0 + 9*2)/10.
    assert result.status == "converged"
    for knot in result.knot_poses:
        np.testing.assert_allclose(knot.translation_m, (1.8, 0.0, 0.0), atol=1e-8)
        np.testing.assert_allclose(se3_log(SE3.identity(), knot)[3:], 0.0, atol=1e-8)
    assert result.final_objective == pytest.approx(2.0 * 4.0 * (1.8**2 + 9.0 * 0.2**2))
