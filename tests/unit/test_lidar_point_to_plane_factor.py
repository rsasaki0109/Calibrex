from calibrex.core.geometry import SE3
from calibrex.evaluation.lidar import lidar_rig_point_to_plane_metrics_from_evaluation
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneFactor,
)


def test_lidar_rig_point_to_plane_factor_residuals_and_jacobian() -> None:
    factor = LidarRigPointToPlaneFactor(
        variable="T_base_lidar0",
        t_ego_lidar=SE3((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        observations=[
            LidarPointToPlaneObservation(
                point_lidar_m=(0.0, 0.0, 0.0),
                plane_point_world_m=(0.0, 0.0, 0.0),
                plane_normal_world=(0.0, 0.0, 2.0),
            ),
            LidarPointToPlaneObservation(
                point_lidar_m=(1.0, 0.0, 0.0),
                plane_point_world_m=(1.0, 0.0, 0.0),
                plane_normal_world=(0.0, 0.0, 1.0),
            ),
        ],
    )

    residuals = factor.residuals()
    linearization = factor.linearize()
    evaluation = factor.evaluate()

    assert residuals == [1.0, 1.0]
    assert round(factor.residuals([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])[0], 6) == 0.0
    assert linearization.dof_names == (
        "x_lidar0",
        "y_lidar0",
        "z_lidar0",
        "roll_lidar0",
        "pitch_lidar0",
        "yaw_lidar0",
    )
    assert len(linearization.jacobian) == 2
    assert len(linearization.jacobian[0]) == 6
    assert round(linearization.jacobian[0][2], 6) == 1.0
    assert round(evaluation.rmse_m or 0.0, 6) == 1.0
    assert evaluation.residual_count == 2
    assert evaluation.rank >= 2
    assert evaluation.diagonal_information["z_lidar0"] > 0.0
    assert evaluation.diagonal_information["pitch_lidar0"] > 0.0
    assert evaluation.diagnostic_kind == "local_curvature_proxy"
    assert evaluation.normalization_length_m >= 1.0
    assert evaluation.normalized_diagonal_information["z_lidar0"] >= (
        evaluation.diagonal_information["z_lidar0"]
    )
    assert evaluation.normalized_condition_number_estimate is not None
    assert len(evaluation.normalized_hessian) == 6
    assert "x_lidar0" in evaluation.weak_directions
    metrics = lidar_rig_point_to_plane_metrics_from_evaluation(evaluation)
    assert metrics["lidar_rig_point_to_plane_residual_count"].value == 2.0
    assert metrics["lidar_rig_point_to_plane_rmse_m"].value == 1.0
    assert metrics["lidar_rig_point_to_plane_normalization_length_m"].value == (
        evaluation.normalization_length_m
    )
    assert "not a covariance" in (
        metrics["lidar_rig_point_to_plane_condition_number"].reason or ""
    )
    assert metrics["lidar_rig_point_to_plane_weak_dof_count"].grade == "warn"


def test_lidar_rig_point_to_plane_analytic_jacobian_matches_numerical_zero() -> None:
    factor = LidarRigPointToPlaneFactor(
        variable="T_base_lidar0",
        t_ego_lidar=SE3((0.2, -0.1, 0.3), (0.0, 0.0, 0.1, 0.995)),
        observations=[
            LidarPointToPlaneObservation(
                point_lidar_m=(1.0, -0.3, 2.0),
                plane_point_world_m=(1.2, -0.4, 2.3),
                plane_normal_world=(0.3, -0.2, 1.0),
                t_world_ego=SE3((0.5, 0.0, -0.2), (0.0, 0.1, 0.0, 0.995)),
                weight=2.0,
            ),
            LidarPointToPlaneObservation(
                point_lidar_m=(-1.0, 0.8, 0.5),
                plane_point_world_m=(-0.8, 0.7, 0.9),
                plane_normal_world=(1.0, 0.4, -0.1),
                t_world_ego=SE3((-0.2, 0.4, 0.0), (0.05, 0.0, -0.1, 0.994)),
                weight=0.5,
            ),
        ],
    )

    numerical = factor.linearize(
        translation_step_m=1.0e-6,
        rotation_step_rad=1.0e-6,
    ).jacobian
    analytic = factor.analytic_jacobian_at_zero()

    assert len(analytic) == len(numerical)
    for analytic_row, numerical_row in zip(analytic, numerical, strict=True):
        assert len(analytic_row) == 6
        for analytic_value, numerical_value in zip(analytic_row, numerical_row, strict=True):
            assert abs(analytic_value - numerical_value) < 1.0e-6


def test_lidar_rig_point_to_plane_factor_validates_observations() -> None:
    try:
        LidarPointToPlaneObservation(
            point_lidar_m=(0.0, 0.0, 0.0),
            plane_point_world_m=(0.0, 0.0, 0.0),
            plane_normal_world=(0.0, 0.0, 0.0),
        )
    except ValueError as exc:
        assert "plane normal" in str(exc)
    else:
        raise AssertionError("zero plane normal should fail")
