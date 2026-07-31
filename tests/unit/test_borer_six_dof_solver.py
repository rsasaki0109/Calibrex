import math

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
    DepthToDepthOptions,
)
from calibrex.solvers.borer_six_dof_solver import (
    BorerSixDofOptions,
    BorerSixDofSolver,
    apply_local_se3_delta,
)


def _observation() -> DepthToDepthObservation:
    camera = DepthToDepthCameraModel(
        width=64,
        height=48,
        fx=45.0,
        fy=45.0,
        cx=31.5,
        cy=23.5,
    )
    depth = np.full((48, 64), np.nan)
    points = []
    for v in range(5, 43):
        for u in range(6, 58):
            distance = (
                8.0
                + 0.03 * u
                + 0.02 * v
                + 1.7 * math.sin(0.31 * u)
                + 1.1 * math.cos(0.27 * v)
            )
            ray = np.asarray(
                [(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0]
            )
            ray /= np.linalg.norm(ray)
            points.append(ray * distance)
            depth[v, u] = distance
    return DepthToDepthObservation(
        frame_id="six-dof",
        depth_map=depth,
        lidar_points=np.asarray(points),
        camera=camera,
    )


def test_apply_local_se3_delta_changes_all_six_dof() -> None:
    result = apply_local_se3_delta(
        SE3.identity(),
        (1.0, 2.0, 3.0),
        (0.1, -0.2, 0.3),
    )

    assert result.translation_m == pytest.approx((0.1, -0.2, 0.3))
    assert result.rotation_quat_xyzw != (0.0, 0.0, 0.0, 1.0)


def test_six_dof_solver_improves_synthetic_perturbation() -> None:
    initial = apply_local_se3_delta(
        SE3.identity(),
        (1.0, 0.0, 0.0),
        (0.10, 0.0, 0.0),
    )
    result = BorerSixDofSolver().solve(
        [_observation()],
        initial,
        BorerSixDofOptions(
            rotation_bound_deg=2.0,
            translation_bound_m=0.2,
            initial_rotation_step_deg=0.5,
            initial_translation_step_m=0.05,
            minimum_rotation_step_deg=0.125,
            minimum_translation_step_m=0.0125,
            max_evaluations=160,
            d2d=DepthToDepthOptions(
                histogram_bins=20,
                min_visible_points=100,
            ),
        ),
    )

    assert result.final_evaluation.mutual_information > (
        result.initial_evaluation.mutual_information
    )
    assert abs(result.transform_camera_lidar.translation_m[0]) < 0.10
    assert result.evaluation_count == len(result.trace)
    assert result.as_dict()["method"] == "bounded_se3_pattern_search/v0.1"
