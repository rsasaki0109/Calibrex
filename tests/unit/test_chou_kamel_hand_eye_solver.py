import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.chou_kamel_hand_eye_solver import (
    ChouKamelHandEyeOptions,
    ChouKamelHandEyeSolver,
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
    rng = np.random.default_rng(1991)
    motions: list[HandEyeMotionPair] = []
    for index in range(count):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(rng.uniform(-75.0, 75.0))
        if abs(angle) < 5.0:
            angle = math.copysign(5.0, angle or 1.0)
        motion_a = _axis_transform(tuple(axis), angle, tuple(rng.uniform(-0.4, 0.4, size=3)))
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        motions.append(HandEyeMotionPair(f"motion-{index:03d}", motion_a, motion_b))
    return motions


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(
        sum(a * b for a, b in zip(left.rotation_quat_xyzw, right.rotation_quat_xyzw, strict=True))
    )
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def test_recovers_truth_with_unfitted_holdout_and_falsification() -> None:
    truth = SE3((0.27, -0.13, 0.08), (0.08, -0.11, 0.16, 0.978))

    result = ChouKamelHandEyeSolver().solve(_motions(truth))

    assert result.status == "converged"
    assert result.transform_x is not None
    assert _rotation_error_deg(result.transform_x, truth) < 1.0e-5
    assert np.linalg.norm(np.asarray(result.transform_x.translation_m) - truth.translation_m) < 1e-9
    assert result.rotation_rank == 3
    assert result.rotation_nullspace_gap is not None
    assert result.rotation_nullspace_gap > 0.1
    assert result.rotation_nullspace_residual is not None
    assert result.rotation_nullspace_residual < 1.0e-12
    assert result.translation_rank == 3
    assert result.holdout_evaluation.translation_closure_rmse_m is not None
    assert result.holdout_evaluation.translation_closure_rmse_m < 1.0e-9
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)


def test_one_rotation_axis_exposes_two_dimensional_nullspace() -> None:
    truth = SE3((0.2, 0.1, -0.1), (0.02, 0.04, -0.06, 0.997))
    motions = []
    for index, angle in enumerate((10.0, 20.0, 30.0, 40.0, 50.0)):
        motion_a = _axis_transform((0.0, 0.0, 1.0), angle, (0.1, 0.0, 0.0))
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        motions.append(HandEyeMotionPair(f"parallel-{index}", motion_a, motion_b))

    result = ChouKamelHandEyeSolver().solve(
        motions, ChouKamelHandEyeOptions(holdout_ratio=0.0)
    )

    assert result.status == "degenerate_rotation"
    assert result.rotation_rank == 2
    assert result.rotation_nullspace_gap is not None
    assert result.rotation_nullspace_gap < 1.0e-12


def test_pure_translation_cannot_be_reported_as_six_dof() -> None:
    motions = [
        HandEyeMotionPair(
            f"translation-{index}",
            SE3((float(index), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            SE3((float(index), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        )
        for index in range(6)
    ]

    result = ChouKamelHandEyeSolver().solve(motions)

    assert result.status == "insufficient_motions"
    assert result.transform_x is None


def test_result_records_primary_paper_and_split_lineage() -> None:
    truth = SE3((0.1, -0.2, 0.3), (0.03, -0.06, 0.1, 0.992))

    payload = ChouKamelHandEyeSolver().solve(_motions(truth)).as_dict()

    assert payload["paper_doi"] == "10.1177/027836499101000305"
    assert payload["method"] == "chou_kamel_normalized_quaternion_generalized_inverse/v0.1"
    assert len(payload["train_pair_ids"]) == 24
    assert len(payload["holdout_pair_ids"]) == 6
    assert len(payload["known_bad_probes"]) == 12
