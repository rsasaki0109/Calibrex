from math import sqrt

from calibrex.core.geometry import SE3
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers import FixedTrajectorySe3ExtrinsicSolver as PublicSolver
from calibrex.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
)


def test_fixed_trajectory_se3_solver_recovers_synthetic_extrinsic() -> None:
    assert PublicSolver is FixedTrajectorySe3ExtrinsicSolver

    true_correction = (0.08, -0.04, 0.03, 0.03, -0.02, 0.04)
    true_transform = LidarRigPointToPlaneFactor(
        variable="T_base_lidar0",
        t_ego_lidar=SE3.identity(),
        observations=[],
    ).corrected_transform(true_correction)
    observations = [
        LidarPointToPlaneObservation(
            point_lidar_m=point,
            plane_point_world_m=true_transform.transform_point(point),
            plane_normal_world=normal,
        )
        for point, normal in [
            ((1.0, 0.0, 0.0), (1.0, 0.2, 0.1)),
            ((0.0, 1.0, 0.0), (0.1, 1.0, 0.3)),
            ((0.0, 0.0, 1.0), (0.3, 0.1, 1.0)),
            ((1.0, 1.0, 0.0), (0.7, -0.4, 0.2)),
            ((1.0, 0.0, 1.0), (-0.2, 0.4, 0.8)),
            ((0.0, 1.0, 1.0), (0.5, 0.6, -0.3)),
            ((-1.0, 0.5, 0.3), (0.4, 0.9, 0.1)),
            ((0.4, -0.8, 1.2), (0.2, -0.5, 0.9)),
            ((1.3, 0.2, -0.6), (0.8, 0.1, -0.4)),
        ]
    ]
    factor = LidarRigPointToPlaneFactor(
        variable="T_base_lidar0",
        t_ego_lidar=SE3.identity(),
        observations=observations,
    )

    result = FixedTrajectorySe3ExtrinsicSolver().solve(
        factor,
        FixedTrajectorySe3SolverOptions(
            max_iterations=80,
            max_translation_step_m=0.03,
            max_rotation_step_rad=0.02,
            robust_loss="none",
            convergence_tolerance=1.0e-10,
        ),
    )

    assert result.status == "converged"
    assert result.initial_rmse_m is not None
    assert result.final_rmse_m is not None
    assert result.final_rmse_m < 1.0e-6
    assert result.final_rmse_m < result.initial_rmse_m
    assert result.accepted_steps > 0
    translation_error = _translation_error(result.refined_transform, true_transform)
    assert translation_error < 1.0e-5


def test_fixed_trajectory_se3_solver_reports_insufficient_constraints() -> None:
    factor = LidarRigPointToPlaneFactor(
        variable="T_base_lidar0",
        t_ego_lidar=SE3.identity(),
        observations=[
            LidarPointToPlaneObservation(
                point_lidar_m=(0.0, 0.0, 0.0),
                plane_point_world_m=(0.0, 0.0, 0.0),
                plane_normal_world=(0.0, 0.0, 1.0),
                weight=0.0,
            )
        ],
    )

    result = FixedTrajectorySe3ExtrinsicSolver().solve(factor)

    assert result.status == "insufficient_constraints"
    assert result.stop_reason == "normal equations have zero rank"


def _translation_error(left: SE3, right: SE3) -> float:
    return sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(
                left.translation_m,
                right.translation_m,
                strict=True,
            )
        )
    )
