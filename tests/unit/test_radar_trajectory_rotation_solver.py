import math

import numpy as np
import pytest

from calibrex.solvers import RadarTrajectoryRotationSolver as PublicSolver
from calibrex.solvers.radar_trajectory_rotation_solver import (
    RadarTrajectoryRotationOptions,
    RadarTrajectoryRotationPair,
    RadarTrajectoryRotationSolver,
)


def _rotation(axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    skew = np.array(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]]
    )
    angle = math.radians(angle_deg)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def _pairs(truth: np.ndarray, *, outlier: bool = False) -> list[RadarTrajectoryRotationPair]:
    directions = (
        (1.0, 0.2, -0.3),
        (0.1, 1.0, 0.4),
        (-0.5, 0.3, 1.0),
        (0.7, -0.9, 0.2),
        (-0.8, -0.1, 0.6),
        (0.2, 0.5, -0.9),
    )
    output = []
    for index in range(30):
        source = (4.0 + index % 7) * np.asarray(directions[index % len(directions)])
        target = truth @ source
        if outlier and index in {4, 17, 25}:
            target = target + np.array((20.0, -15.0, 12.0))
        output.append(
            RadarTrajectoryRotationPair(
                f"scan-{index:03d}",
                tuple(source),
                tuple(target),  # type: ignore[arg-type]
            )
        )
    return output


def _quaternion_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_rotation_solver_recovers_truth_holdout_and_provenance() -> None:
    assert PublicSolver is RadarTrajectoryRotationSolver
    truth = _rotation((0.3, -0.7, 0.6), 37.0)
    result = RadarTrajectoryRotationSolver().solve(
        _pairs(truth), RadarTrajectoryRotationOptions(holdout_ratio=0.25, split_seed=8)
    )

    assert result.status == "converged"
    assert result.rotation_body_radar_xyzw is not None
    estimate = _quaternion_matrix(result.rotation_body_radar_xyzw)
    assert np.linalg.norm(estimate - truth) < 1.0e-10
    assert result.holdout_rmse_mps is not None and result.holdout_rmse_mps < 1.0e-10
    assert result.information_rank == 3
    assert set(result.train_measurement_ids).isdisjoint(result.holdout_measurement_ids)
    assert len(result.probes) == 6 and all(probe.detectable is True for probe in result.probes)
    serialized = result.as_dict()
    assert serialized["method"] == "wise_velocity_alignment_so3_huber/v0.1"
    assert serialized["primary_reference"]["arxiv"] == "2103.07505"  # type: ignore[index]


def test_rotation_solver_is_order_invariant_and_robust_to_outliers() -> None:
    truth = _rotation((-0.4, 0.2, 0.9), 24.0)
    pairs = _pairs(truth, outlier=True)
    options = RadarTrajectoryRotationOptions(holdout_ratio=0.0, huber_delta_mps=0.2)

    forward = RadarTrajectoryRotationSolver().solve(pairs, options)
    reverse = RadarTrajectoryRotationSolver().solve(list(reversed(pairs)), options)

    assert forward.rotation_body_radar_xyzw is not None
    assert reverse.rotation_body_radar_xyzw is not None
    assert np.linalg.norm(_quaternion_matrix(forward.rotation_body_radar_xyzw) - truth) < 0.01
    assert reverse.rotation_body_radar_xyzw == pytest.approx(forward.rotation_body_radar_xyzw)
    assert reverse.train_measurement_ids == forward.train_measurement_ids


def test_rotation_solver_rejects_collinear_motion() -> None:
    pairs = [
        RadarTrajectoryRotationPair(
            f"scan-{index}", (float(index + 1), 0.0, 0.0), (0.0, float(index + 1), 0.0)
        )
        for index in range(12)
    ]

    result = RadarTrajectoryRotationSolver().solve(pairs)

    assert result.status == "degenerate_motion"
    assert result.rotation_body_radar_xyzw is None
    assert result.information_rank == 2
    assert result.information_singular_values is not None


def test_rotation_solver_filters_slow_pairs_before_split() -> None:
    truth = _rotation((0.0, 0.0, 1.0), 15.0)
    pairs = [
        *_pairs(truth),
        RadarTrajectoryRotationPair("slow", (0.01, 0.0, 0.0), (0.01, 0.0, 0.0)),
    ]

    result = RadarTrajectoryRotationSolver().solve(pairs)

    assert "slow" not in result.train_measurement_ids
    assert "slow" not in result.holdout_measurement_ids


def test_rotation_solver_rejects_duplicate_lineage_and_invalid_options() -> None:
    pair = RadarTrajectoryRotationPair("duplicate", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="IDs must be unique"):
        RadarTrajectoryRotationSolver().solve([pair, pair])
    with pytest.raises(ValueError, match="huber_delta_mps"):
        RadarTrajectoryRotationSolver().solve(
            [], RadarTrajectoryRotationOptions(huber_delta_mps=0.0)
        )
