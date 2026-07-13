import math

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers.horaud_dornaika_nonlinear_hand_eye_solver import (
    HoraudDornaikaNonlinearHandEyeOptions,
    HoraudDornaikaNonlinearHandEyeSolver,
)
from calibrex.solvers.park_martin_hand_eye_solver import HandEyeMotionPair


def _axis_transform(
    axis: tuple[float, float, float],
    angle_deg: float,
    translation: tuple[float, float, float],
) -> SE3:
    half = math.radians(angle_deg) / 2.0
    sine = math.sin(half)
    return SE3(
        translation,
        (axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(half)),
    )


def _motions(
    truth: SE3, count: int = 30, *, noise_seed: int | None = None
) -> list[HandEyeMotionPair]:
    rng = np.random.default_rng(1995)
    noise_rng = np.random.default_rng(noise_seed)
    motions: list[HandEyeMotionPair] = []
    for index in range(count):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(rng.uniform(-75.0, 75.0))
        if abs(angle) < 5.0:
            angle = math.copysign(5.0, angle or 1.0)
        motion_a = _axis_transform(tuple(axis), angle, tuple(rng.uniform(-0.4, 0.4, size=3)))
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        if noise_seed is not None:
            noise_axis = noise_rng.normal(size=3)
            noise_axis /= np.linalg.norm(noise_axis)
            noise = _axis_transform(
                tuple(noise_axis),
                float(noise_rng.normal(0.0, 0.15)),
                tuple(noise_rng.normal(0.0, 0.001, size=3)),
            )
            motion_b = motion_b.compose(noise)
        motions.append(HandEyeMotionPair(f"motion-{index:03d}", motion_a, motion_b))
    return motions


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(
        sum(a * b for a, b in zip(left.rotation_quat_xyzw, right.rotation_quat_xyzw, strict=True))
    )
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def test_recovers_truth_with_physical_observability_holdout_and_controls() -> None:
    truth = SE3((0.27, -0.13, 0.08), (0.08, -0.11, 0.16, 0.978))

    result = HoraudDornaikaNonlinearHandEyeSolver().solve(_motions(truth))

    assert result.status == "converged"
    assert result.transform_x is not None
    assert _rotation_error_deg(result.transform_x, truth) < 1.0e-5
    assert np.linalg.norm(np.asarray(result.transform_x.translation_m) - truth.translation_m) < 1e-9
    assert result.data_jacobian_rank == 6
    assert result.data_jacobian_condition_number is not None
    assert result.data_jacobian_condition_number < 10.0
    assert result.quaternion_unit_error is not None
    assert result.quaternion_unit_error < 1.0e-12
    assert result.holdout_evaluation.translation_closure_rmse_m is not None
    assert result.holdout_evaluation.translation_closure_rmse_m < 1.0e-9
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)


def test_noisy_problem_accepts_only_nonincreasing_lm_steps() -> None:
    truth = SE3((0.27, -0.13, 0.08), (0.08, -0.11, 0.16, 0.978))

    result = HoraudDornaikaNonlinearHandEyeSolver().solve(_motions(truth, noise_seed=17))

    assert result.status == "converged"
    assert result.transform_x is not None
    assert result.initial_cost is not None
    assert result.final_cost is not None
    assert result.final_cost <= result.initial_cost
    assert result.accepted_step_count > 0
    assert all(
        item.trial_cost < item.cost for item in result.iterations if item.accepted
    )
    assert _rotation_error_deg(result.transform_x, truth) < 0.2
    translation_error = np.linalg.norm(
        np.asarray(result.transform_x.translation_m) - truth.translation_m
    )
    assert translation_error < 0.005


def test_single_axis_motion_is_rejected_by_initializer() -> None:
    truth = SE3((0.2, 0.1, -0.1), (0.02, 0.04, -0.06, 0.997))
    motions = []
    for index, angle in enumerate((10.0, 20.0, 30.0, 40.0, 50.0)):
        motion_a = _axis_transform((0.0, 0.0, 1.0), angle, (0.1, 0.0, 0.0))
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        motions.append(HandEyeMotionPair(f"parallel-{index}", motion_a, motion_b))

    result = HoraudDornaikaNonlinearHandEyeSolver().solve(
        motions, HoraudDornaikaNonlinearHandEyeOptions(holdout_ratio=0.0)
    )

    assert result.status == "initialization_failed"
    assert result.transform_x is None


def test_paper_weights_cannot_be_silently_changed() -> None:
    truth = SE3((0.1, -0.2, 0.3), (0.03, -0.06, 0.1, 0.992))

    with pytest.raises(ValueError, match="paper contract"):
        HoraudDornaikaNonlinearHandEyeSolver().solve(
            _motions(truth),
            HoraudDornaikaNonlinearHandEyeOptions(rotation_weight=2.0),
        )


def test_result_records_primary_equations_optimizer_and_split_lineage() -> None:
    truth = SE3((0.1, -0.2, 0.3), (0.03, -0.06, 0.1, 0.992))

    payload = HoraudDornaikaNonlinearHandEyeSolver().solve(_motions(truth)).as_dict()

    paper = payload["paper"]
    assert isinstance(paper, dict)
    assert paper["doi"] == "10.1177/027836499501400301"
    assert paper["equations"] == [27, 28, 30, 32]
    assert payload["method"] == "horaud_dornaika_simultaneous_nonlinear_hand_eye/v0.1"
    assert payload["selected_paper_method"] == "section_5_2_simultaneous_rotation_translation"
    assert len(payload["train_pair_ids"]) == 24
    assert len(payload["holdout_pair_ids"]) == 6
    assert len(payload["known_bad_probes"]) == 12
