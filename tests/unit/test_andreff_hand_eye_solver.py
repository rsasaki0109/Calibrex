import math

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers.andreff_hand_eye_solver import (
    AndreffHandEyeOptions,
    AndreffHandEyeSolver,
    AndreffKroneckerAccumulator,
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


def _motions(truth: SE3, count: int = 30) -> list[HandEyeMotionPair]:
    rng = np.random.default_rng(902)
    motions: list[HandEyeMotionPair] = []
    for index in range(count):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(rng.uniform(-70.0, 70.0))
        if abs(angle) < 4.0:
            angle = math.copysign(4.0, angle or 1.0)
        motion_a = _axis_transform(tuple(axis), angle, tuple(rng.uniform(-0.4, 0.4, size=3)))
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        motions.append(HandEyeMotionPair(f"motion-{index:03d}", motion_a, motion_b))
    return motions


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(
        sum(a * b for a, b in zip(left.rotation_quat_xyzw, right.rotation_quat_xyzw, strict=True))
    )
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def test_incremental_kronecker_solver_recovers_truth_and_controls() -> None:
    truth = SE3((0.26, -0.15, 0.09), (0.07, -0.10, 0.15, 0.981))

    result = AndreffHandEyeSolver().solve(_motions(truth))

    assert result.status == "converged"
    assert result.transform_x is not None
    assert _rotation_error_deg(result.transform_x, truth) < 1.0e-5
    assert (
        np.linalg.norm(np.asarray(result.transform_x.translation_m) - truth.translation_m) < 1.0e-9
    )
    assert result.rotation_observable_rank == 8
    assert result.translation_rank == 3
    assert result.rotation_minimum_width is not None
    assert result.rotation_minimum_width > 0.1
    assert result.determinant_normalized_orthogonality_error is not None
    assert result.determinant_normalized_orthogonality_error < 1.0e-10
    assert result.so3_projection_correction_frobenius is not None
    assert result.so3_projection_correction_frobenius < 1.0e-10
    assert len(result.incremental_updates) == 24
    assert result.incremental_updates[0].rotation_axis_rank == 1
    assert result.incremental_updates[0].rotation_observable_rank <= 6
    assert result.incremental_updates[-1].rotation_observable_rank == 8
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)


def test_small_nonzero_rotations_are_not_discarded_by_default() -> None:
    truth = SE3((0.2, -0.1, 0.08), (0.04, -0.05, 0.08, 0.994))
    root_two = math.sqrt(2.0)
    axes = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0 / root_two, 1.0 / root_two, 0.0),
        (0.0, 1.0 / root_two, 1.0 / root_two),
        (1.0 / root_two, 0.0, 1.0 / root_two),
    )
    motions = []
    for index, axis in enumerate(axes):
        motion_a = _axis_transform(
            axis,
            0.05 + 0.01 * index,
            (0.1 * (index + 1), (-1.0) ** index * 0.04, 0.03 * index),
        )
        motions.append(
            HandEyeMotionPair(
                f"small-{index}",
                motion_a,
                truth.inverse().compose(motion_a).compose(truth),
            )
        )

    result = AndreffHandEyeSolver().solve(motions, AndreffHandEyeOptions(holdout_ratio=0.0))

    assert result.status == "converged"
    assert result.transform_x is not None
    assert _rotation_error_deg(result.transform_x, truth) < 1.0e-5
    assert (
        np.linalg.norm(np.asarray(result.transform_x.translation_m) - truth.translation_m) < 1.0e-9
    )


def test_two_nonparallel_motions_satisfy_the_paper_minimum() -> None:
    truth = SE3((0.2, -0.1, 0.08), (0.04, -0.05, 0.08, 0.994))
    motions = []
    for index, (axis, angle, translation) in enumerate(
        (
            ((1.0, 0.0, 0.0), 25.0, (0.2, -0.1, 0.05)),
            ((0.0, 1.0, 0.0), -35.0, (-0.1, 0.15, 0.03)),
        )
    ):
        motion_a = _axis_transform(axis, angle, translation)
        motions.append(
            HandEyeMotionPair(
                f"minimum-{index}",
                motion_a,
                truth.inverse().compose(motion_a).compose(truth),
            )
        )

    result = AndreffHandEyeSolver().solve(motions, AndreffHandEyeOptions(holdout_ratio=0.0))

    assert result.status == "converged"
    assert result.transform_x is not None
    assert result.rotation_axis_rank == 2
    assert result.rotation_observable_rank == 8
    assert result.translation_rank == 3
    assert _rotation_error_deg(result.transform_x, truth) < 1.0e-5


def test_pure_translations_report_partial_rotation_but_no_full_calibration() -> None:
    truth = SE3((0.2, -0.1, 0.08), (0.04, -0.05, 0.08, 0.994))
    translations = ((0.2, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.4))
    motions = []
    for index, translation in enumerate(translations):
        motion_a = SE3(translation, (0.0, 0.0, 0.0, 1.0))
        motions.append(
            HandEyeMotionPair(
                f"translation-{index}",
                motion_a,
                truth.inverse().compose(motion_a).compose(truth),
            )
        )

    result = AndreffHandEyeSolver().solve(motions, AndreffHandEyeOptions(holdout_ratio=0.0))

    assert result.status == "degenerate_rotation"
    assert result.rotation_observable_rank == 0
    assert result.translation_rank == 0
    assert result.translation_rotation_rank == 9
    assert result.transform_x is None


def test_accumulator_rejects_duplicate_motion_lineage() -> None:
    truth = SE3((0.1, -0.2, 0.3), (0.03, -0.06, 0.1, 0.992))
    motion = _motions(truth, count=1)[0]
    accumulator = AndreffKroneckerAccumulator()

    first = accumulator.update(motion)

    assert first.motion_count == 1
    assert accumulator.pair_ids == (motion.pair_id,)
    with pytest.raises(ValueError, match="duplicate Andreff motion pair_id"):
        accumulator.update(motion)


def test_result_records_two_stage_online_provenance() -> None:
    truth = SE3((0.1, -0.2, 0.3), (0.03, -0.06, 0.1, 0.992))

    payload = AndreffHandEyeSolver().solve(_motions(truth)).as_dict()

    assert payload["paper_doi"] == "10.1109/IM.1999.805374"
    assert payload["paper_title"] == "On-line Hand-Eye Calibration"
    assert payload["paper_equations"] == [10, 13, 14, 15]
    assert payload["full_12_variable_solution_executed"] is False
    assert payload["rotation_expected_rank"] == 8
    assert len(payload["incremental_updates"]) == 24
