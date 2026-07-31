import math

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
    DepthToDepthOptions,
    evaluate_depth_to_depth_mi,
    project_depth_pairs,
    resolve_depth_to_depth_options,
)


def _synthetic_observation() -> DepthToDepthObservation:
    camera = DepthToDepthCameraModel(
        width=64,
        height=48,
        fx=48.0,
        fy=48.0,
        cx=31.5,
        cy=23.5,
    )
    depth = np.full((camera.height, camera.width), np.nan, dtype=float)
    points = []
    for v in range(5, 43):
        for u in range(6, 58):
            lidar_range = (
                10.0
                + 0.04 * u
                + 0.02 * v
                + 1.4 * math.sin(0.37 * u)
                + 0.8 * math.cos(0.29 * v)
            )
            ray = np.asarray(
                [(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0],
                dtype=float,
            )
            ray /= np.linalg.norm(ray)
            points.append(ray * lidar_range)
            depth[v, u] = lidar_range
    return DepthToDepthObservation(
        frame_id="synthetic",
        depth_map=depth,
        lidar_points=np.asarray(points),
        camera=camera,
    )


def test_d2d_objective_prefers_aligned_pose() -> None:
    observation = _synthetic_observation()
    options = DepthToDepthOptions(histogram_bins=24, min_visible_points=100)
    angle = math.radians(8.0)
    perturbed = SE3(
        (0.0, 0.0, 0.0),
        (0.0, math.sin(angle / 2.0), 0.0, math.cos(angle / 2.0)),
    )

    aligned = evaluate_depth_to_depth_mi(
        [observation], SE3.identity(), options
    )
    wrong = evaluate_depth_to_depth_mi([observation], perturbed, options)

    assert aligned.evaluated_frame_count == 1
    assert aligned.normalized_mutual_information > 0.99
    assert aligned.mutual_information > wrong.mutual_information
    assert aligned.normalized_mutual_information > wrong.normalized_mutual_information


def test_projection_z_buffer_keeps_nearest_point() -> None:
    camera = DepthToDepthCameraModel(
        width=5,
        height=5,
        fx=2.0,
        fy=2.0,
        cx=2.0,
        cy=2.0,
    )
    observation = DepthToDepthObservation(
        frame_id="occlusion",
        depth_map=np.full((5, 5), 5.0),
        lidar_points=np.asarray([[0.0, 0.0, 10.0], [0.0, 0.0, 5.0]]),
        camera=camera,
    )

    buffered = project_depth_pairs(observation, SE3.identity(), use_z_buffer=True)
    unbuffered = project_depth_pairs(observation, SE3.identity(), use_z_buffer=False)

    assert buffered.projected_count_before_visibility == 2
    assert buffered.lidar_range_m.tolist() == [5.0]
    assert unbuffered.lidar_range_m.tolist() == [10.0, 5.0]


def test_double_sphere_projection_accepts_front_and_rejects_back() -> None:
    camera = DepthToDepthCameraModel(
        width=9,
        height=9,
        fx=4.0,
        fy=4.0,
        cx=4.0,
        cy=4.0,
        projection="double_sphere",
        xi=0.5,
        alpha=0.5,
    )
    observation = DepthToDepthObservation(
        frame_id="fisheye",
        depth_map=np.full((9, 9), 3.0),
        lidar_points=np.asarray([[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]),
        camera=camera,
    )

    projected = project_depth_pairs(observation, SE3.identity())

    assert projected.lidar_range_m.tolist() == [3.0]
    assert projected.pixel_u.tolist() == [4]
    assert projected.pixel_v.tolist() == [4]


def test_mei_projection_applies_radial_tangential_distortion() -> None:
    camera = DepthToDepthCameraModel(
        width=21,
        height=21,
        fx=10.0,
        fy=10.0,
        cx=10.0,
        cy=10.0,
        projection="mei",
        xi=1.0,
        distortion=(0.1, 0.0, 0.01, -0.02),
    )
    observation = DepthToDepthObservation(
        frame_id="mei",
        depth_map=np.full((21, 21), 3.0),
        lidar_points=np.asarray([[1.0, 0.5, 2.0], [0.0, 0.0, -3.0]]),
        camera=camera,
    )

    projected = project_depth_pairs(observation, SE3.identity())

    assert projected.lidar_range_m.tolist() == [pytest.approx(np.sqrt(5.25))]
    assert projected.projected_count_before_visibility == 1
    assert projected.pixel_u.tolist() == [12]
    assert projected.pixel_v.tolist() == [11]


def test_d2d_averages_per_frame_and_records_skips() -> None:
    observation = _synthetic_observation()
    empty = DepthToDepthObservation(
        frame_id="empty",
        depth_map=np.full_like(observation.depth_map, np.nan),
        lidar_points=observation.lidar_points,
        camera=observation.camera,
    )

    result = evaluate_depth_to_depth_mi(
        [observation, observation, empty],
        SE3.identity(),
        DepthToDepthOptions(histogram_bins=24, min_visible_points=100),
    )

    assert result.evaluated_frame_count == 2
    assert result.skipped_frame_ids == ("empty",)
    assert result.mutual_information == result.frames[0].mutual_information
    assert result.as_dict()["primary_source"] == "https://arxiv.org/abs/2311.01905"


def test_resolve_options_freezes_invariant_histogram_ranges() -> None:
    observation = _synthetic_observation()

    resolved = resolve_depth_to_depth_options(
        [observation],
        DepthToDepthOptions(histogram_bins=24),
    )

    assert resolved.camera_depth_range is not None
    assert resolved.lidar_range_m is not None
    assert observation.lidar_range_m.shape == (
        observation.lidar_points.shape[0],
    )
