import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.daniilidis_hand_eye_solver import (
    DaniilidisHandEyeOptions,
    DaniilidisHandEyeSolver,
)
from calibrex.solvers.park_martin_hand_eye_solver import HandEyeMotionPair
from calibrex.solvers.tsai_lenz_hand_eye_solver import TsaiLenzHandEyeSolver


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
    rng = np.random.default_rng(117)
    result = []
    for index in range(count):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = float(rng.uniform(-70.0, 70.0))
        if abs(angle) < 5.0:
            angle = math.copysign(5.0, angle or 1.0)
        motion_a = _axis_transform(
            tuple(axis), angle, tuple(rng.uniform(-0.4, 0.4, size=3))
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


def test_recovers_truth_simultaneously_with_holdout_and_controls() -> None:
    truth = SE3((0.31, -0.18, 0.12), (0.09, -0.07, 0.17, 0.978))
    result = DaniilidisHandEyeSolver().solve(_motions(truth))

    assert result.status == "converged"
    assert result.transform_x is not None
    assert _rotation_error_deg(result.transform_x, truth) < 1.0e-5
    assert np.linalg.norm(
        np.asarray(result.transform_x.translation_m) - truth.translation_m
    ) < 1.0e-8
    assert result.linear_rank == 6
    assert result.unit_constraint_error is not None
    assert result.unit_constraint_error < 1.0e-12
    assert result.study_constraint_error is not None
    assert result.study_constraint_error < 1.0e-12
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)


def test_matches_tsai_lenz_on_exact_data_and_split() -> None:
    truth = SE3((0.11, 0.06, -0.15), (-0.05, 0.08, 0.14, 0.985))
    motions = _motions(truth)
    dual = DaniilidisHandEyeSolver().solve(motions)
    tsai = TsaiLenzHandEyeSolver().solve(motions)

    assert dual.transform_x is not None
    assert tsai.transform_x is not None
    assert _rotation_error_deg(dual.transform_x, tsai.transform_x) < 1.0e-5
    assert np.linalg.norm(
        np.asarray(dual.transform_x.translation_m) - tsai.transform_x.translation_m
    ) < 1.0e-8
    assert dual.train_pair_ids == tsai.train_pair_ids
    assert dual.holdout_pair_ids == tsai.holdout_pair_ids


def test_single_axis_motion_is_rejected_as_degenerate() -> None:
    truth = SE3((0.2, 0.1, -0.1), (0.02, 0.04, -0.06, 0.997))
    motions = []
    for index, angle in enumerate((10.0, 20.0, 30.0, 40.0, 50.0)):
        motion_a = _axis_transform((0.0, 0.0, 1.0), angle, (0.1, 0.0, 0.0))
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        motions.append(HandEyeMotionPair(f"parallel-{index}", motion_a, motion_b))

    result = DaniilidisHandEyeSolver().solve(
        motions, DaniilidisHandEyeOptions(holdout_ratio=0.0)
    )

    assert result.status == "degenerate_dual_quaternion_system"
    assert result.linear_rank < 6


def test_result_records_simultaneous_method_provenance() -> None:
    truth = SE3((0.1, -0.2, 0.3), (0.03, -0.06, 0.1, 0.992))
    payload = DaniilidisHandEyeSolver().solve(_motions(truth)).as_dict()

    assert payload["paper_doi"] == "10.1177/02783649922066213"
    assert payload["method"] == "daniilidis_dual_quaternion_nullspace/v0.1"
    assert payload["constraints"] == [
        "real quaternion unit norm",
        "Study real-dual orthogonality",
    ]
