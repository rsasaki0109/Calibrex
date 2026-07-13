import math

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers.dornaika_horaud_nonlinear_robot_world_hand_eye_solver import (
    DornaikaHoraudNonlinearOptions,
    DornaikaHoraudNonlinearSolver,
)
from calibrex.solvers.shah_robot_world_hand_eye_solver import RobotWorldHandEyePosePair


def _pose(
    axis: tuple[float, float, float],
    angle_deg: float,
    translation: tuple[float, float, float],
) -> SE3:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    half = math.radians(angle_deg) / 2.0
    return SE3(
        translation,
        (
            float(vector[0] * math.sin(half)),
            float(vector[1] * math.sin(half)),
            float(vector[2] * math.sin(half)),
            math.cos(half),
        ),
    )


def _problem(
    *, noise_rotation_deg: float = 0.0, noise_translation_m: float = 0.0
) -> tuple[SE3, SE3, list[RobotWorldHandEyePosePair]]:
    transform_x = _pose((1.0, -2.0, 0.5), 23.0, (0.24, -0.16, 0.11))
    transform_z = _pose((-0.5, 1.0, 2.0), -31.0, (-0.3, 0.5, 0.2))
    rng = np.random.default_rng(881)
    pairs = []
    for index in range(40):
        axis = rng.normal(size=3)
        angle = float(rng.uniform(-150.0, 150.0))
        translation = tuple(float(value) for value in rng.uniform(-0.5, 0.5, size=3))
        pose_a = _pose(tuple(float(value) for value in axis), angle, translation)
        pose_b = transform_z.inverse().compose(pose_a).compose(transform_x)
        if noise_rotation_deg or noise_translation_m:
            noise_axis = rng.normal(size=3)
            noise_axis /= np.linalg.norm(noise_axis)
            noise = _pose(
                tuple(float(value) for value in noise_axis),
                float(rng.normal(scale=noise_rotation_deg)),
                tuple(float(value) for value in rng.normal(scale=noise_translation_m, size=3)),
            )
            pose_b = noise.compose(pose_b)
        pairs.append(RobotWorldHandEyePosePair(f"pose-{index:03d}", pose_a, pose_b))
    return transform_x, transform_z, pairs


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    delta = left.inverse().compose(right)
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, abs(delta.rotation_quat_xyzw[3])))))


def test_nonlinear_solver_recovers_truth_and_complete_evidence() -> None:
    truth_x, truth_z, pairs = _problem()

    result = DornaikaHoraudNonlinearSolver().solve(pairs)

    assert result.status == "converged"
    assert result.transform_x is not None
    assert result.transform_z is not None
    assert result.transform_y == result.transform_z
    assert _rotation_error_deg(result.transform_x, truth_x) < 1.0e-5
    assert _rotation_error_deg(result.transform_z, truth_z) < 1.0e-5
    assert (
        np.linalg.norm(
            np.asarray(result.transform_x.translation_m) - np.asarray(truth_x.translation_m)
        )
        < 1.0e-9
    )
    assert (
        np.linalg.norm(
            np.asarray(result.transform_z.translation_m) - np.asarray(truth_z.translation_m)
        )
        < 1.0e-9
    )
    assert result.initial_cost is not None
    assert result.final_cost is not None
    assert result.final_cost <= result.initial_cost + 1.0e-20
    assert result.final_data_residual_rmse is not None
    assert result.final_data_residual_rmse < 1.0e-12
    assert result.data_jacobian_rank == 24
    assert len(result.data_jacobian_singular_values) == 24
    assert result.data_jacobian_condition_number is not None
    assert result.rotation_x_projection_correction_frobenius is not None
    assert result.rotation_x_projection_correction_frobenius < 1.0e-12
    assert len(result.probes) == 24
    assert all(probe.detectable is True for probe in result.probes)
    document = result.as_dict()
    assert document["paper"]["doi"] == "10.1109/70.704233"
    assert document["selected_paper_method"] == "section_III_B_nonlinear_24_parameter"
    assert document["paper_weights"]["mu3_rotation_x_orthogonality"] == 1.0e6


def test_nonlinear_solver_reduces_noisy_simultaneous_objective() -> None:
    _truth_x, _truth_z, pairs = _problem(noise_rotation_deg=0.2, noise_translation_m=0.001)

    result = DornaikaHoraudNonlinearSolver().solve(pairs)

    assert result.status == "converged"
    assert result.initial_cost is not None
    assert result.final_cost is not None
    assert result.final_cost < result.initial_cost
    assert result.accepted_step_count > 0
    assert result.initial_data_residual_rmse is not None
    assert result.final_data_residual_rmse is not None
    assert result.final_data_residual_rmse < result.initial_data_residual_rmse
    assert any(iteration.accepted for iteration in result.iterations)


def test_nonlinear_solver_is_order_invariant() -> None:
    _truth_x, _truth_z, pairs = _problem(noise_rotation_deg=0.1)
    solver = DornaikaHoraudNonlinearSolver()

    first = solver.solve(pairs)
    second = solver.solve(list(reversed(pairs)))

    assert first.status == second.status == "converged"
    assert first.transform_x == second.transform_x
    assert first.transform_z == second.transform_z
    assert first.train_pair_ids == second.train_pair_ids
    assert first.holdout_pair_ids == second.holdout_pair_ids


def test_nonlinear_solver_does_not_emit_unconverged_max_iteration_result() -> None:
    _truth_x, _truth_z, pairs = _problem(noise_rotation_deg=0.5, noise_translation_m=0.003)

    result = DornaikaHoraudNonlinearSolver().solve(
        pairs,
        DornaikaHoraudNonlinearOptions(
            max_iterations=1,
            gradient_tolerance=1.0e-30,
            step_tolerance=1.0e-30,
            relative_cost_tolerance=1.0e-30,
        ),
    )

    assert result.status == "max_iterations"
    assert result.transform_x is None
    assert result.transform_z is None
    assert len(result.iterations) == 1
    assert len(result.data_jacobian_singular_values) == 24


def test_nonlinear_solver_preserves_failed_initializer_diagnostics() -> None:
    pose = _pose((0.0, 0.0, 1.0), 20.0, (0.1, 0.2, 0.3))
    pairs = [RobotWorldHandEyePosePair(f"duplicate-{index}", pose, pose) for index in range(6)]

    result = DornaikaHoraudNonlinearSolver().solve(
        pairs, DornaikaHoraudNonlinearOptions(holdout_ratio=0.0)
    )

    assert result.status == "initialization_failed"
    assert "degenerate_rotation" in result.reason
    assert result.transform_x is None
    assert result.transform_z is None


def test_nonlinear_solver_requires_three_train_absolute_poses() -> None:
    _truth_x, _truth_z, pairs = _problem()

    result = DornaikaHoraudNonlinearSolver().solve(pairs[:2])

    assert result.status == "insufficient_poses"


def test_nonlinear_solver_rejects_nonpaper_objective_weights() -> None:
    _truth_x, _truth_z, pairs = _problem()

    with pytest.raises(ValueError, match="mu1=mu2=1"):
        DornaikaHoraudNonlinearSolver().solve(
            pairs,
            DornaikaHoraudNonlinearOptions(rotation_weight=2.0),
        )
