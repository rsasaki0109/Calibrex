import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.park_martin_hand_eye_solver import (
    HandEyeMotionPair,
    ParkMartinHandEyeOptions,
    ParkMartinHandEyeSolver,
)
from calibrex.solvers.tsai_lenz_hand_eye_solver import (
    TsaiLenzHandEyeOptions,
    TsaiLenzHandEyeSolver,
)


def _axis_transform(
    axis: tuple[float, float, float],
    angle_deg: float,
    translation: tuple[float, float, float],
) -> SE3:
    half = math.radians(angle_deg) / 2.0
    return SE3(
        translation,
        (
            axis[0] * math.sin(half),
            axis[1] * math.sin(half),
            axis[2] * math.sin(half),
            math.cos(half),
        ),
    )


def _motions(truth: SE3) -> list[HandEyeMotionPair]:
    rng = np.random.default_rng(91)
    result: list[HandEyeMotionPair] = []
    for index in range(24):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        motion_a = _axis_transform(
            tuple(axis),
            float(rng.uniform(-65.0, 65.0)),
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


def test_recovers_truth_with_holdout_and_known_bad_controls() -> None:
    truth = SE3((0.28, -0.14, 0.09), (0.08, -0.11, 0.16, 0.978))
    result = TsaiLenzHandEyeSolver().solve(_motions(truth))

    assert result.status == "converged"
    assert result.transform_x is not None
    assert _rotation_error_deg(result.transform_x, truth) < 1.0e-5
    assert np.linalg.norm(
        np.asarray(result.transform_x.translation_m) - truth.translation_m
    ) < 1.0e-9
    assert result.holdout_evaluation.rotation_closure_rmse_deg is not None
    assert result.holdout_evaluation.rotation_closure_rmse_deg < 1.0e-5
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)


def test_matches_independent_park_martin_baseline_on_exact_data() -> None:
    truth = SE3((0.18, 0.04, -0.12), (-0.04, 0.09, 0.13, 0.986))
    motions = _motions(truth)
    tsai = TsaiLenzHandEyeSolver().solve(motions)
    park = ParkMartinHandEyeSolver().solve(
        motions, ParkMartinHandEyeOptions(split_seed=0)
    )

    assert tsai.transform_x is not None
    assert park.transform_x is not None
    assert _rotation_error_deg(tsai.transform_x, park.transform_x) < 1.0e-5
    assert np.linalg.norm(
        np.asarray(tsai.transform_x.translation_m) - park.transform_x.translation_m
    ) < 1.0e-9
    assert tsai.train_pair_ids == park.train_pair_ids
    assert tsai.holdout_pair_ids == park.holdout_pair_ids


def test_single_rotation_axis_is_rejected() -> None:
    truth = SE3((0.2, 0.1, -0.1), (0.02, 0.04, -0.06, 0.997))
    motions = []
    for index, angle in enumerate((10.0, 20.0, 30.0, 40.0, 50.0)):
        motion_a = _axis_transform((0.0, 0.0, 1.0), angle, (0.1, 0.0, 0.0))
        motion_b = truth.inverse().compose(motion_a).compose(truth)
        motions.append(HandEyeMotionPair(f"parallel-{index}", motion_a, motion_b))

    result = TsaiLenzHandEyeSolver().solve(
        motions, TsaiLenzHandEyeOptions(holdout_ratio=0.0)
    )

    assert result.status == "degenerate_rotation"
    assert result.rotation_rank < 3


def test_result_records_paper_and_falsification_provenance() -> None:
    truth = SE3((0.1, -0.2, 0.3), (0.03, -0.06, 0.1, 0.992))
    payload = TsaiLenzHandEyeSolver().solve(_motions(truth)).as_dict()

    assert payload["paper_doi"] == "10.1109/70.34770"
    assert payload["equation"] == "A X = X B"
    probes = payload["known_bad_probes"]
    assert isinstance(probes, list)
    assert len(probes) == 12
