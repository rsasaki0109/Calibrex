import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.robust_point_to_point_icp_solver import (
    IcpPoint,
    RobustPointToPointIcpOptions,
    RobustPointToPointIcpSolver,
)


def _clouds(truth: SE3, count: int = 80) -> tuple[list[IcpPoint], list[IcpPoint]]:
    rng = np.random.default_rng(41)
    target_values = rng.uniform(-1.5, 1.5, size=(count, 3))
    inverse = truth.inverse()
    target = [
        IcpPoint(f"target-{index:03d}", tuple(float(value) for value in point))
        for index, point in enumerate(target_values)
    ]
    source = [
        IcpPoint(f"source-{index:03d}", inverse.transform_point(point))
        for index, point in enumerate(target_values)
    ]
    return source, target


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(
        sum(
            a * b
            for a, b in zip(
                left.rotation_quat_xyzw,
                right.rotation_quat_xyzw,
                strict=True,
            )
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def test_recovers_asymmetric_cloud_and_spatial_holdout() -> None:
    truth = SE3((0.12, -0.08, 0.06), (0.018, -0.026, 0.035, 0.9989))
    source, target = _clouds(truth)
    result = RobustPointToPointIcpSolver().solve(
        source,
        target,
        options=RobustPointToPointIcpOptions(
            correspondence_distance_m=0.5,
            trim_fraction=1.0,
        ),
    )

    assert result.status == "converged"
    assert result.transform_target_source is not None
    assert _rotation_error_deg(result.transform_target_source, truth) < 1.0e-5
    assert (
        np.linalg.norm(
            np.asarray(result.transform_target_source.translation_m)
            - np.asarray(truth.translation_m)
        )
        < 1.0e-9
    )
    assert result.holdout_rmse_m is not None
    assert result.holdout_rmse_m < 1.0e-9
    assert result.information_rank == 6
    assert result.rematching_diagnostics is not None
    assert result.rematching_diagnostics.weak_directions == ()
    assert all(
        value is not None and value > 0.0
        for value in result.rematching_diagnostics.objective_curvatures
    )
    assert result.rematching_diagnostics.minimum_correspondence_jaccard is not None
    assert result.rematching_diagnostics.minimum_correspondence_jaccard > 0.8
    assert len(result.multi_start_trials) == 6
    assert result.symmetry_ambiguous is False


def test_rematching_diagnostics_are_serialized_as_diagnostics_not_covariance() -> None:
    truth = SE3((0.04, -0.03, 0.02), (0.01, -0.015, 0.02, 0.9996))
    source, target = _clouds(truth)
    result = RobustPointToPointIcpSolver().solve(
        source,
        target,
        options=RobustPointToPointIcpOptions(
            correspondence_distance_m=0.4,
            trim_fraction=1.0,
        ),
    )

    payload = result.as_dict()
    diagnostics = payload["rematching_diagnostics"]
    assert isinstance(diagnostics, dict)
    assert len(diagnostics["objective_curvatures"]) == 6
    assert "covariance" not in diagnostics
    assert payload["method"] == "trimmed_mutual_point_to_point_icp/v0.1"


def test_multi_start_detects_repeated_structure_with_equivalent_holdout() -> None:
    rng = np.random.default_rng(73)
    values = rng.uniform(-0.08, 0.08, size=(60, 3))
    source = [
        IcpPoint(f"source-{index:03d}", tuple(point))
        for index, point in enumerate(values)
    ]
    target = [
        IcpPoint(f"target-a-{index:03d}", tuple(point))
        for index, point in enumerate(values)
    ] + [
        IcpPoint(f"target-b-{index:03d}", (point[0] + 1.0, point[1], point[2]))
        for index, point in enumerate(values)
    ]
    result = RobustPointToPointIcpSolver().solve(
        source,
        target,
        options=RobustPointToPointIcpOptions(
            correspondence_distance_m=0.25,
            trim_fraction=1.0,
            holdout_voxel_size_m=0.04,
            multi_start_translation_m=1.0,
        ),
    )

    assert result.status == "converged"
    assert result.symmetry_ambiguous is True
    x_trial = next(item for item in result.multi_start_trials if item.direction == "x")
    assert x_trial.equivalent_holdout is True
    assert x_trial.distinct_solution is True
    assert x_trial.symmetry_ambiguous is True
    assert x_trial.solution_translation_delta_m is not None
    assert x_trial.solution_translation_delta_m > 0.9


def test_trim_and_distance_gate_reject_outliers() -> None:
    truth = SE3((0.08, 0.04, -0.03), (0.01, 0.02, -0.025, 0.9994))
    source, target = _clouds(truth)
    source.extend(IcpPoint(f"outlier-{index}", (10.0 + index, -12.0, 8.0)) for index in range(12))
    result = RobustPointToPointIcpSolver().solve(
        source,
        target,
        options=RobustPointToPointIcpOptions(
            correspondence_distance_m=0.5,
            trim_fraction=0.9,
            holdout_ratio=0.0,
        ),
    )

    assert result.status == "converged"
    assert result.transform_target_source is not None
    assert _rotation_error_deg(result.transform_target_source, truth) < 1.0e-5
    assert result.inlier_fraction < 1.0


def test_planar_cloud_is_conservatively_reported_degenerate() -> None:
    points = [
        IcpPoint(f"point-{x}-{y}", (float(x), float(y), 0.0)) for x in range(4) for y in range(4)
    ]
    result = RobustPointToPointIcpSolver().solve(
        points,
        points,
        options=RobustPointToPointIcpOptions(holdout_ratio=0.0),
    )

    assert result.status == "degenerate_geometry"
    assert result.transform_target_source is None


def test_result_is_invariant_to_input_order_with_stable_ids() -> None:
    truth = SE3((0.05, -0.04, 0.02), (0.01, -0.01, 0.015, 0.9998))
    source, target = _clouds(truth)
    solver = RobustPointToPointIcpSolver()
    options = RobustPointToPointIcpOptions(correspondence_distance_m=0.4)
    first = solver.solve(source, target, options=options)
    second = solver.solve(list(reversed(source)), list(reversed(target)), options=options)

    assert first.status == second.status == "converged"
    assert first.transform_target_source == second.transform_target_source
    assert first.train_source_ids == second.train_source_ids
    assert first.holdout_source_ids == second.holdout_source_ids
