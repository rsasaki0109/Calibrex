from calibrex.core.geometry import SE3
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneEvaluation,
    LidarRigPointToPlaneFactor,
)


def test_horizontal_plane_scene_has_expected_weak_lidar_dof() -> None:
    evaluation = _evaluate_observability(
        [
            _observation((float(x), float(y), 0.0), (0.0, 0.0, 1.0))
            for x in [-2, -1, 0, 1, 2]
            for y in [-2, 0, 2]
        ]
    )

    assert evaluation.rank == 3
    assert set(evaluation.weak_directions) == {
        "x_lidar0",
        "y_lidar0",
        "yaw_lidar0",
    }
    assert evaluation.normalized_diagonal_information["z_lidar0"] > 0.0
    assert evaluation.normalized_diagonal_information["roll_lidar0"] > 0.0
    assert evaluation.normalized_diagonal_information["pitch_lidar0"] > 0.0


def test_corridor_scene_keeps_lateral_lidar_translation_weak() -> None:
    observations = [
        _observation((float(x), float(y), 0.0), (0.0, 0.0, 1.0))
        for x in [-2, 0, 2]
        for y in [-2, 0, 2]
    ]
    for y in [-2, 0, 2]:
        for z in [0, 1, 2]:
            observations.append(_observation((-3.0, float(y), float(z)), (1.0, 0.0, 0.0)))
            observations.append(_observation((3.0, float(y), float(z)), (1.0, 0.0, 0.0)))

    evaluation = _evaluate_observability(observations)

    assert evaluation.rank == 5
    assert evaluation.weak_directions == ["y_lidar0"]
    assert evaluation.normalized_condition_number_estimate is not None
    assert evaluation.normalized_condition_number_estimate < 10.0


def test_rich_3d_scene_has_full_lidar_extrinsic_rank() -> None:
    observations: list[LidarPointToPlaneObservation] = []
    for x in [-2, 0, 2]:
        for y in [-2, 0, 2]:
            observations.append(_observation((float(x), float(y), 2.0), (0.0, 0.0, 1.0)))
            observations.append(_observation((float(x), float(y), -2.0), (0.0, 0.0, 1.0)))
    for x in [-2, 0, 2]:
        for z in [-2, 0, 2]:
            observations.append(_observation((float(x), 2.0, float(z)), (0.0, 1.0, 0.0)))
            observations.append(_observation((float(x), -2.0, float(z)), (0.0, 1.0, 0.0)))
    for y in [-2, 0, 2]:
        for z in [-2, 0, 2]:
            observations.append(_observation((2.0, float(y), float(z)), (1.0, 0.0, 0.0)))
            observations.append(_observation((-2.0, float(y), float(z)), (1.0, 0.0, 0.0)))

    evaluation = _evaluate_observability(observations)

    assert evaluation.rank == 6
    assert evaluation.weak_directions == []
    assert evaluation.normalized_condition_number_estimate is not None
    assert evaluation.normalized_condition_number_estimate < 2.0


def _evaluate_observability(
    observations: list[LidarPointToPlaneObservation],
) -> LidarRigPointToPlaneEvaluation:
    factor = LidarRigPointToPlaneFactor(
        variable="T_base_lidar0",
        t_ego_lidar=SE3.identity(),
        observations=observations,
    )
    return factor.evaluate()


def _observation(
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> LidarPointToPlaneObservation:
    return LidarPointToPlaneObservation(
        point_lidar_m=point,
        plane_point_world_m=point,
        plane_normal_world=normal,
    )
