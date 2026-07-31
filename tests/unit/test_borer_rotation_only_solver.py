import math

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
    DepthToDepthOptions,
)
from calibrex.solvers.borer_rotation_only_solver import (
    BorerRotationOnlyOptions,
    BorerRotationOnlySolver,
    fibonacci_sphere_rotation_perturbations,
)


def _observation() -> DepthToDepthObservation:
    camera = DepthToDepthCameraModel(
        width=48,
        height=36,
        fx=38.0,
        fy=38.0,
        cx=23.5,
        cy=17.5,
    )
    depth = np.full((camera.height, camera.width), np.nan)
    points = []
    for v in range(4, 32):
        for u in range(5, 43):
            distance = (
                8.0
                + 0.05 * u
                + 1.2 * math.sin(0.41 * u)
                + 0.7 * math.cos(0.33 * v)
            )
            ray = np.asarray(
                [(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0]
            )
            ray /= np.linalg.norm(ray)
            points.append(ray * distance)
            depth[v, u] = distance
    return DepthToDepthObservation(
        frame_id="synthetic",
        depth_map=depth,
        lidar_points=np.asarray(points),
        camera=camera,
    )


def _rotation_error_deg(transform: SE3) -> float:
    return math.degrees(
        2.0
        * math.acos(min(1.0, max(-1.0, abs(transform.rotation_quat_xyzw[3]))))
    )


def test_rotation_only_solver_improves_bounded_d2d_pose() -> None:
    angle = math.radians(6.0)
    initial = SE3(
        (0.0, 0.0, 0.0),
        (0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0)),
    )
    options = BorerRotationOnlyOptions(
        bound_deg=10.0,
        initial_step_deg=3.0,
        minimum_step_deg=0.1875,
        max_evaluations=180,
        d2d=DepthToDepthOptions(histogram_bins=20, min_visible_points=100),
    )

    result = BorerRotationOnlySolver().solve([_observation()], initial, options)

    assert result.status == "converged"
    assert result.final_evaluation.mutual_information > (
        result.initial_evaluation.mutual_information
    )
    assert _rotation_error_deg(result.transform_camera_lidar) < (
        _rotation_error_deg(initial)
    )
    assert result.transform_camera_lidar.translation_m == initial.translation_m
    assert result.evaluation_count == len(result.trace)
    assert any(item.accepted for item in result.trace[1:])


def test_rotation_only_solver_rejects_empty_depth_support() -> None:
    observation = _observation()
    empty = DepthToDepthObservation(
        frame_id="empty",
        depth_map=np.full_like(observation.depth_map, np.nan),
        lidar_points=observation.lidar_points,
        camera=observation.camera,
    )

    result = BorerRotationOnlySolver().solve(
        [empty],
        SE3.identity(),
        BorerRotationOnlyOptions(
            d2d=DepthToDepthOptions(min_visible_points=10)
        ),
    )

    assert result.status == "insufficient_observations"
    assert result.transform_camera_lidar == SE3.identity()
    assert result.trace == ()


def test_fibonacci_sphere_protocol_is_deterministic_and_on_radius() -> None:
    first = fibonacci_sphere_rotation_perturbations(count=200, magnitude_deg=10.0)
    second = fibonacci_sphere_rotation_perturbations(count=200, magnitude_deg=10.0)

    assert first == second
    assert len(first) == 200
    assert all(
        math.isclose(math.sqrt(sum(value * value for value in point)), 10.0)
        for point in first
    )
    assert abs(sum(point[2] for point in first)) < 1.0e-12
