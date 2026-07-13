import math

import numpy as np
import pytest

from calibrex.core.geometry import SE3, rotate_vector_xyzw
from calibrex.solvers.point_plane_lidar_camera_solver import (
    PointPlaneLidarCameraSolver,
    PointPlaneObservation,
    PointPlaneSolverOptions,
    evaluate_point_plane_observations,
)


def _observations(
    transform: SE3,
    count: int = 30,
) -> list[PointPlaneObservation]:
    observations = []
    for index in range(count):
        phase = index * 0.37
        lidar_center = (
            1.5 + 0.8 * math.sin(phase),
            -0.4 + 0.7 * math.cos(phase * 0.7),
            0.6 + 0.5 * math.sin(phase * 1.3),
        )
        raw_normal = (
            0.4 + math.sin(phase * 0.9),
            -0.2 + math.cos(phase * 1.1),
            0.7 + math.sin(phase * 0.5),
        )
        norm = math.sqrt(sum(value * value for value in raw_normal))
        lidar_normal = tuple(value / norm for value in raw_normal)
        observations.append(
            PointPlaneObservation(
                frame_id=f"board-{index:03d}",
                camera_center_m=transform.transform_point(lidar_center),
                camera_normal=rotate_vector_xyzw(transform.rotation_quat_xyzw, lidar_normal),
                lidar_center_m=lidar_center,
                lidar_normal=lidar_normal,
            )
        )
    return observations


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    relative = left.inverse().compose(right)
    return math.degrees(2.0 * math.acos(min(1.0, abs(relative.rotation_quat_xyzw[3]))))


def test_point_plane_solver_recovers_truth_holdout_and_known_bad_controls() -> None:
    truth = SE3((0.28, -0.13, 0.41), (0.08, -0.12, 0.17, 0.974))
    result = PointPlaneLidarCameraSolver().solve(
        _observations(truth),
        PointPlaneSolverOptions(holdout_ratio=0.25, split_seed=17),
    )

    assert result.status == "converged"
    assert result.transform_camera_lidar is not None
    assert result.transform_camera_lidar.translation_m == pytest.approx(
        truth.translation_m, abs=1.0e-10
    )
    assert _rotation_error_deg(result.transform_camera_lidar, truth) < 1.0e-6
    assert result.train_evaluation.center_rmse_m is not None
    assert result.train_evaluation.center_rmse_m < 1.0e-12
    assert result.holdout_evaluation.center_rmse_m is not None
    assert result.holdout_evaluation.center_rmse_m < 1.0e-12
    assert result.joint_rank == 6
    assert result.joint_condition_number is not None
    assert set(result.train_frame_ids).isdisjoint(result.holdout_frame_ids)
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)
    assert result.as_dict()["method"] == "verma_center_normal_procrustes_irls/v0.1"


def test_point_plane_solver_rejects_repeated_pose_geometry() -> None:
    observation = PointPlaneObservation(
        "repeated",
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    observations = [
        PointPlaneObservation(
            f"repeated-{index}",
            observation.camera_center_m,
            observation.camera_normal,
            observation.lidar_center_m,
            observation.lidar_normal,
        )
        for index in range(12)
    ]

    result = PointPlaneLidarCameraSolver().solve(observations)

    assert result.status == "degenerate_geometry"
    assert result.joint_rank < 6
    assert result.transform_camera_lidar is not None


def test_point_plane_solver_is_order_invariant_with_one_outlier() -> None:
    truth = SE3((0.2, 0.1, -0.05), (0.04, 0.03, -0.08, 0.995))
    observations = _observations(truth)
    outlier = observations[7]
    observations[7] = PointPlaneObservation(
        outlier.frame_id,
        tuple(np.asarray(outlier.camera_center_m) + np.asarray((0.5, -0.4, 0.3))),
        outlier.camera_normal,
        outlier.lidar_center_m,
        outlier.lidar_normal,
    )
    options = PointPlaneSolverOptions(holdout_ratio=0.2, split_seed=3)

    first = PointPlaneLidarCameraSolver().solve(observations, options)
    second = PointPlaneLidarCameraSolver().solve(list(reversed(observations)), options)

    assert first.train_frame_ids == second.train_frame_ids
    assert first.holdout_frame_ids == second.holdout_frame_ids
    assert outlier.frame_id in first.train_frame_ids
    assert first.transform_camera_lidar is not None
    assert second.transform_camera_lidar is not None
    assert first.transform_camera_lidar.translation_m == pytest.approx(
        second.transform_camera_lidar.translation_m
    )
    assert _rotation_error_deg(first.transform_camera_lidar, truth) < 0.1
    assert (
        np.linalg.norm(
            np.asarray(first.transform_camera_lidar.translation_m) - np.asarray(truth.translation_m)
        )
        < 0.005
    )


def test_point_plane_evaluation_preserves_oriented_normal_sign() -> None:
    observation = PointPlaneObservation(
        "opposite-normal",
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    )

    evaluation = evaluate_point_plane_observations([observation], SE3.identity())

    assert evaluation.center_rmse_m == 0.0
    assert evaluation.normal_rmse_deg == pytest.approx(180.0)


def test_point_plane_observation_rejects_non_finite_centres() -> None:
    with pytest.raises(ValueError, match="finite"):
        PointPlaneObservation(
            "non-finite",
            (math.nan, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )


def test_point_plane_solver_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="holdout ratio"):
        PointPlaneLidarCameraSolver().solve([], PointPlaneSolverOptions(holdout_ratio=1.0))
