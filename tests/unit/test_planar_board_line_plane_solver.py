import math

import numpy as np

from calibrex.core.geometry import SE3, rotate_vector_xyzw
from calibrex.solvers.planar_board_lidar_camera_solver import OrientedPlane
from calibrex.solvers.planar_board_line_plane_solver import (
    BoardBoundaryCorrespondence,
    LinePlaneBoardObservation,
    LinePlaneSolverOptions,
    OrientedLine3D,
    PlanarBoardLinePlaneSolver,
)


def _transform_direction(
    transform: SE3, direction: tuple[float, float, float]
) -> tuple[float, float, float]:
    return rotate_vector_xyzw(transform.rotation_quat_xyzw, direction)


def _capture(transform: SE3, frame_id: str = "board-000") -> LinePlaneBoardObservation:
    lidar_plane = OrientedPlane((0.0, 0.0, 1.0), 0.0)
    camera_normal = _transform_direction(transform, lidar_plane.normal)
    camera_offset = -sum(
        normal * translation
        for normal, translation in zip(camera_normal, transform.translation_m, strict=True)
    )
    lidar_lines = (
        OrientedLine3D((-0.8, -0.4, 0.0), (1.0, 0.0, 0.0)),
        OrientedLine3D((-0.8, -0.4, 0.0), (0.0, 1.0, 0.0)),
    )
    boundaries = tuple(
        BoardBoundaryCorrespondence(
            f"edge-{index}",
            OrientedLine3D(
                transform.transform_point(line.point),
                _transform_direction(transform, line.direction),
            ),
            line,
        )
        for index, line in enumerate(lidar_lines)
    )
    return LinePlaneBoardObservation(
        frame_id,
        OrientedPlane(camera_normal, camera_offset),
        lidar_plane,
        boundaries,
    )


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


def test_one_pose_with_adjacent_edges_recovers_six_dof() -> None:
    truth = SE3((0.35, -0.18, 0.42), (0.09, -0.12, 0.16, 0.976))
    result = PlanarBoardLinePlaneSolver().solve(
        [_capture(truth)], options=LinePlaneSolverOptions(holdout_ratio=0.0)
    )

    assert result.status == "converged"
    assert result.transform_camera_lidar is not None
    assert _rotation_error_deg(result.transform_camera_lidar, truth) < 1.0e-5
    assert (
        np.linalg.norm(
            np.asarray(result.transform_camera_lidar.translation_m)
            - np.asarray(truth.translation_m)
        )
        < 1.0e-9
    )
    assert result.rotation_rank == 3
    assert result.translation_rank == 3
    assert result.train_evaluation.boundary_distance_rmse_m is not None
    assert result.train_evaluation.boundary_distance_rmse_m < 1.0e-9


def test_parallel_boundaries_are_rejected_as_rank_five_geometry() -> None:
    truth = SE3.identity()
    capture = _capture(truth)
    parallel = LinePlaneBoardObservation(
        capture.frame_id,
        capture.camera_plane,
        capture.lidar_plane,
        (
            BoardBoundaryCorrespondence(
                "top",
                OrientedLine3D((0.0, 0.5, 0.0), (1.0, 0.0, 0.0)),
                OrientedLine3D((0.0, 0.5, 0.0), (1.0, 0.0, 0.0)),
            ),
            BoardBoundaryCorrespondence(
                "bottom",
                OrientedLine3D((0.0, -0.5, 0.0), (1.0, 0.0, 0.0)),
                OrientedLine3D((0.0, -0.5, 0.0), (1.0, 0.0, 0.0)),
            ),
        ),
    )
    result = PlanarBoardLinePlaneSolver().solve(
        [parallel], options=LinePlaneSolverOptions(holdout_ratio=0.0)
    )

    assert result.status == "degenerate_geometry"


def test_nearly_parallel_boundaries_fail_edge_angle_gate() -> None:
    truth = SE3.identity()
    capture = _capture(truth)
    angle = math.radians(0.5)
    weak_line = OrientedLine3D((0.0, 0.4, 0.0), (math.cos(angle), math.sin(angle), 0.0))
    weak = LinePlaneBoardObservation(
        capture.frame_id,
        capture.camera_plane,
        capture.lidar_plane,
        (capture.boundaries[0], BoardBoundaryCorrespondence("weak", weak_line, weak_line)),
    )
    result = PlanarBoardLinePlaneSolver().solve(
        [weak], options=LinePlaneSolverOptions(holdout_ratio=0.0, minimum_edge_angle_deg=5.0)
    )

    assert result.status == "degenerate_geometry"


def test_multiple_captures_have_deterministic_holdout() -> None:
    truth = SE3((0.2, 0.1, -0.05), (0.04, 0.03, -0.08, 0.995))
    observations = [_capture(truth, f"board-{index:03d}") for index in range(8)]
    solver = PlanarBoardLinePlaneSolver()
    first = solver.solve(observations)
    second = solver.solve(list(reversed(observations)))

    assert first.status == "converged"
    assert first.train_frame_ids == second.train_frame_ids
    assert first.holdout_frame_ids == second.holdout_frame_ids
    assert first.holdout_evaluation.boundary_distance_rmse_m is not None
    assert first.holdout_evaluation.boundary_distance_rmse_m < 1.0e-9
