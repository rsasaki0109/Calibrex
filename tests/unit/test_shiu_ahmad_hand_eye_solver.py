import math

import numpy as np
import pytest
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.solvers.park_martin_hand_eye_solver import HandEyeMotionPair
from calibrex.solvers.shiu_ahmad_hand_eye_solver import (
    ShiuAhmadHandEyeOptions,
    ShiuAhmadHandEyeSolver,
)


def _axis_transform(
    axis: tuple[float, float, float],
    angle_rad: float,
    translation: tuple[float, float, float],
) -> SE3:
    vector: NDArray[np.float64] = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    half = angle_rad / 2.0
    return SE3(
        translation,
        (
            float(vector[0] * math.sin(half)),
            float(vector[1] * math.sin(half)),
            float(vector[2] * math.sin(half)),
            math.cos(half),
        ),
    )


def _motions(truth: SE3, count: int = 24) -> list[HandEyeMotionPair]:
    rng = np.random.default_rng(88014)
    result: list[HandEyeMotionPair] = []
    for index in range(count):
        axis = rng.normal(size=3)
        motion_a = _axis_transform(
            tuple(axis),
            float(rng.uniform(math.radians(5.0), math.radians(150.0))),
            tuple(rng.uniform(-0.3, 0.3, size=3)),
        )
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        result.append(HandEyeMotionPair(f"motion-{index:03d}", motion_a, motion_b))
    return result


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(
        sum(
            a * b
            for a, b in zip(
                left.rotation_quat_xyzw, right.rotation_quat_xyzw, strict=True
            )
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def test_recovers_paper_section_v_numerical_example() -> None:
    truth = _axis_transform((1.0, 0.0, 0.0), 0.2, (0.01, 0.05, 0.1))
    first_a = _axis_transform((0.0, 0.0, 1.0), 3.0, (0.0, 0.0, 0.0))
    second_a = _axis_transform((0.0, 1.0, 0.0), 1.5, (-0.4, 0.0, 0.4))
    motions = [
        HandEyeMotionPair(
            "paper-A1", first_a, truth.inverse().compose(first_a).compose(truth)
        ),
        HandEyeMotionPair(
            "paper-A2", second_a, truth.inverse().compose(second_a).compose(truth)
        ),
    ]

    result = ShiuAhmadHandEyeSolver().solve(
        motions,
        ShiuAhmadHandEyeOptions(min_train_motions=2, holdout_ratio=0.0),
    )

    assert result.status == "converged"
    assert result.transform_x is not None
    assert result.fit_pair_ids == ("paper-A1", "paper-A2")
    assert result.selected_axis_separation_sine == pytest.approx(1.0)
    assert _rotation_error_deg(result.transform_x, truth) < 1.0e-6
    assert np.linalg.norm(
        np.asarray(result.transform_x.translation_m) - truth.translation_m
    ) < 1.0e-12
    assert result.rotation_rank == 4
    assert result.translation_rank == 3


def test_uses_two_train_motions_and_evaluates_untouched_holdout() -> None:
    truth = _axis_transform((1.0, -0.4, 0.2), 0.37, (0.24, -0.11, 0.07))
    result = ShiuAhmadHandEyeSolver().solve(_motions(truth))

    assert result.status == "converged"
    assert result.transform_x is not None
    assert len(result.fit_pair_ids) == 2
    assert set(result.fit_pair_ids) < set(result.train_pair_ids)
    assert set(result.fit_pair_ids).isdisjoint(result.holdout_pair_ids)
    assert _rotation_error_deg(result.transform_x, truth) < 1.0e-5
    assert result.holdout_evaluation.rotation_closure_rmse_deg is not None
    assert result.holdout_evaluation.rotation_closure_rmse_deg < 1.0e-5
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)


def test_parallel_motion_axes_are_rejected_by_paper_uniqueness_condition() -> None:
    truth = _axis_transform((1.0, 0.0, 0.0), 0.2, (0.2, 0.1, -0.1))
    motions: list[HandEyeMotionPair] = []
    for index, angle in enumerate((0.2, 0.4, 0.6, 0.8)):
        motion_a = _axis_transform((0.0, 0.0, 1.0), angle, (0.1, 0.0, 0.0))
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        motions.append(HandEyeMotionPair(f"parallel-{index}", motion_a, motion_b))

    result = ShiuAhmadHandEyeSolver().solve(
        motions, ShiuAhmadHandEyeOptions(holdout_ratio=0.0)
    )

    assert result.status == "no_unique_motion_pair"
    assert result.transform_x is None
    assert result.selected_axis_separation_sine == pytest.approx(0.0)


def test_zero_and_pi_rotations_are_excluded_as_required_by_theorem_four() -> None:
    truth = _axis_transform((1.0, 0.0, 0.0), 0.2, (0.2, 0.1, -0.1))
    motions: list[HandEyeMotionPair] = []
    for index, (axis, angle) in enumerate(
        (
            ((1.0, 0.0, 0.0), math.radians(0.5)),
            ((0.0, 1.0, 0.0), math.pi - math.radians(0.5)),
            ((0.0, 0.0, 1.0), math.radians(40.0)),
        )
    ):
        motion_a = _axis_transform(axis, angle, (0.0, 0.0, 0.0))
        motions.append(
            HandEyeMotionPair(
                f"motion-{index}",
                motion_a,
                truth.inverse().compose(motion_a).compose(truth),
            )
        )

    result = ShiuAhmadHandEyeSolver().solve(
        motions, ShiuAhmadHandEyeOptions(holdout_ratio=0.0)
    )

    assert result.status == "insufficient_motions"
    assert result.train_pair_ids == ("motion-2",)


def test_result_records_primary_equations_options_and_fidelity_boundary() -> None:
    truth = _axis_transform((1.0, -0.4, 0.2), 0.37, (0.24, -0.11, 0.07))
    payload = ShiuAhmadHandEyeSolver().solve(_motions(truth)).as_dict()

    paper = payload["primary_paper"]
    assert isinstance(paper, dict)
    assert paper["doi"] == "10.1109/70.88014"
    assert paper["equations"] == [37, 38, 39, 42, 44, 45, 46, 47]
    assert payload["method"] == "shiu_ahmad_two_motion_geometric_hand_eye/v0.1"
    assert payload["external_code_executed"] is False
    assert "two-motion estimator is preserved" in str(payload["fidelity_boundary"])
    options = payload["resolved_options"]
    assert isinstance(options, dict)
    assert options["minimum_axis_separation_sine"] == 0.1
