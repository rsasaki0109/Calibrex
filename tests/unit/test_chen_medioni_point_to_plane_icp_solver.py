import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.chen_medioni_point_to_plane_icp_solver import (
    ChenMedioniPointToPlaneIcpSolver,
    ChenMedioniPointToPlaneOptions,
)
from calibrex.solvers.robust_point_to_point_icp_solver import (
    IcpPoint,
    RobustPointToPointIcpOptions,
)


def _three_plane_clouds(truth: SE3) -> tuple[list[IcpPoint], list[IcpPoint]]:
    coordinates = np.linspace(-0.8, 0.8, 9)
    target_values: list[tuple[float, float, float]] = []
    for first in coordinates:
        for second in coordinates:
            target_values.extend(
                (
                    (-0.65, float(first), float(second)),
                    (float(first), 0.55, float(second)),
                    (float(first), float(second), -0.45),
                )
            )
    inverse = truth.inverse()
    target = [
        IcpPoint(f"target-{index:04d}", point)
        for index, point in enumerate(target_values)
    ]
    source = [
        IcpPoint(f"source-{index:04d}", inverse.transform_point(point))
        for index, point in enumerate(target_values)
    ]
    return source, target


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(
        sum(a * b for a, b in zip(left.rotation_quat_xyzw, right.rotation_quat_xyzw, strict=True))
    )
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def test_recovers_three_surface_truth_with_holdout_and_signed_controls() -> None:
    truth = SE3((0.06, -0.04, 0.03), (0.012, -0.018, 0.022, 0.9995))
    source, target = _three_plane_clouds(truth)

    result = ChenMedioniPointToPlaneIcpSolver().solve(
        source,
        target,
        common_options=RobustPointToPointIcpOptions(
            correspondence_distance_m=0.35,
            trim_fraction=1.0,
            mutual_correspondences=False,
            holdout_voxel_size_m=0.25,
            convergence_tolerance_m=1.0e-9,
            effective_diagnostics=False,
            multi_start_diagnostics=False,
        ),
        options=ChenMedioniPointToPlaneOptions(
            normal_neighbor_count=8,
            known_bad_residual_margin_m=0.001,
        ),
    )

    assert result.status == "converged"
    assert result.transform_target_source is not None
    assert _rotation_error_deg(result.transform_target_source, truth) < 1.0e-3
    translation_error = np.linalg.norm(
        np.asarray(result.transform_target_source.translation_m) - truth.translation_m
    )
    assert translation_error < 1.0e-5
    assert result.tangent_rank == 6
    assert result.target_normal_fraction > 0.8
    assert result.train_point_to_plane_rmse_m is not None
    assert result.train_point_to_plane_rmse_m < 1.0e-5
    assert result.holdout_point_to_plane_rmse_m is not None
    assert result.holdout_point_to_plane_rmse_m < 1.0e-5
    assert result.common_evaluation is not None
    assert result.common_evaluation.holdout_rmse_m is not None
    assert result.common_evaluation.holdout_rmse_m < 1.0e-5
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)


def test_single_plane_exposes_point_to_plane_nullspace() -> None:
    points = [
        IcpPoint(f"point-{x}-{y}", (0.2 * x, 0.2 * y, 0.0))
        for x in range(-4, 5)
        for y in range(-4, 5)
    ]

    result = ChenMedioniPointToPlaneIcpSolver().solve(
        points,
        points,
        common_options=RobustPointToPointIcpOptions(
            holdout_ratio=0.0,
            trim_fraction=1.0,
            mutual_correspondences=False,
            effective_diagnostics=False,
            multi_start_diagnostics=False,
        ),
        options=ChenMedioniPointToPlaneOptions(normal_neighbor_count=8),
    )

    assert result.status == "degenerate_tangent_geometry"
    assert result.transform_target_source is None
    assert result.tangent_rank == 3
    assert result.tangent_condition_number == math.inf


def test_normal_gate_rejects_line_like_target_neighborhoods() -> None:
    points = [IcpPoint(f"line-{index}", (0.1 * index, 0.0, 0.0)) for index in range(30)]

    result = ChenMedioniPointToPlaneIcpSolver().solve(
        points,
        points,
        common_options=RobustPointToPointIcpOptions(
            holdout_ratio=0.0,
            mutual_correspondences=False,
            effective_diagnostics=False,
            multi_start_diagnostics=False,
        ),
        options=ChenMedioniPointToPlaneOptions(normal_neighbor_count=8),
    )

    assert result.status == "insufficient_correspondences"
    assert result.valid_target_normal_count == 0
    assert result.target_normal_fraction == 0.0


def test_condition_gate_cannot_be_weakened_by_iterative_convergence() -> None:
    truth = SE3((0.03, -0.02, 0.01), (0.005, -0.008, 0.011, 0.9999))
    source, target = _three_plane_clouds(truth)

    result = ChenMedioniPointToPlaneIcpSolver().solve(
        source,
        target,
        common_options=RobustPointToPointIcpOptions(
            correspondence_distance_m=0.3,
            trim_fraction=1.0,
            mutual_correspondences=False,
            effective_diagnostics=False,
            multi_start_diagnostics=False,
        ),
        options=ChenMedioniPointToPlaneOptions(
            normal_neighbor_count=8,
            max_condition_number=1.0,
        ),
    )

    assert result.status == "degenerate_tangent_geometry"
    assert result.transform_target_source is None
    assert result.tangent_rank == 6
    assert result.tangent_condition_number is not None
    assert result.tangent_condition_number > 1.0


def test_result_records_discrete_specialization_and_primary_equations() -> None:
    truth = SE3((0.03, -0.02, 0.01), (0.005, -0.008, 0.011, 0.9999))
    source, target = _three_plane_clouds(truth)

    result = ChenMedioniPointToPlaneIcpSolver().solve(
        source,
        target,
        common_options=RobustPointToPointIcpOptions(
            correspondence_distance_m=0.3,
            trim_fraction=1.0,
            mutual_correspondences=False,
            effective_diagnostics=False,
            multi_start_diagnostics=False,
        ),
        options=ChenMedioniPointToPlaneOptions(normal_neighbor_count=8),
    )
    payload = result.as_dict()

    paper = payload["paper"]
    assert isinstance(paper, dict)
    assert paper["conference_doi"] == "10.1109/ROBOT.1991.132043"
    assert paper["journal_doi"] == "10.1016/0262-8856(92)90066-C"
    assert paper["equations"] == [10, 11, 12]
    assert payload["method"] == "chen_medioni_discrete_tangent_plane_icp/v0.1"
    assert "approximates" in payload["surface_specialization"]
    assert payload["external_code_executed"] is False
    resolved = payload["resolved_options"]
    assert isinstance(resolved, dict)
    assert resolved["chen_medioni"]["normal_neighbor_count"] == 8
    assert resolved["common_icp"]["correspondence_distance_m"] == 0.3
