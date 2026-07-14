from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.solvers.levinson_thrun_online_solver import (
    LevinsonCameraModel,
    LevinsonOnlineFrame,
    LevinsonThrunOnlineOptions,
    LevinsonThrunOnlineSolver,
    calibrated_probability,
    evaluate_levinson_objective,
    extract_near_depth_discontinuities,
    levinson_edge_response,
)


def test_levinson_image_and_lidar_preprocessing_contract() -> None:
    image = np.zeros((9, 11), dtype=float)
    image[4, 5] = 100.0
    response = levinson_edge_response(image, alpha=1.0 / 3.0, gamma=0.8, max_radius=3)

    assert response.shape == image.shape
    assert response[4, 5] > response[4, 9]
    assert response[4, 10] == 0.0

    points = np.asarray(
        [
            [
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 5.0],
            ]
        ]
    )
    selected, weights = extract_near_depth_discontinuities(points)
    assert selected.shape == (1, 3)
    assert np.allclose(selected[0], (0.0, 0.0, 3.0))
    assert np.allclose(weights, (math.sqrt(2.0),))


def test_levinson_monitor_identifies_local_optimum_and_scores_holdout() -> None:
    truth = SE3.identity()
    frames = _frames_for_truths([truth] * 12)
    options = _options(tracking_enabled=False)

    result = LevinsonThrunOnlineSolver().solve(frames, truth, options)

    assert result.status == "monitored"
    assert result.timeline
    assert result.final_transform_camera_lidar == truth
    assert result.timeline[-1].worsening_fraction > 0.95
    assert result.timeline[-1].calibrated_probability > 0.9
    assert result.timeline[-1].classified_calibrated is True
    assert result.holdout_evaluation.projected_point_count > 100
    assert len(result.probes) == 12
    assert result.curvature is not None
    assert result.as_dict()["paper_doi"] == "10.15607/RSS.2013.IX.029"


def test_levinson_tracker_recovers_local_six_dof_offset() -> None:
    truth = _transform(0.02, -0.01, 0.01, 0.05, -0.05, 0.2)
    frames = _frames_for_truths([truth] * 16)
    options = _options(tracking_enabled=True)
    initial_score = evaluate_levinson_objective(frames, SE3.identity())

    result = LevinsonThrunOnlineSolver().solve(frames, SE3.identity(), options)

    assert result.status == "tracked"
    assert any(step.update_applied for step in result.timeline)
    assert math.dist(
        result.final_transform_camera_lidar.translation_m, truth.translation_m
    ) <= 0.011
    assert _rotation_error_deg(result.final_transform_camera_lidar, truth) <= 0.11
    assert result.holdout_evaluation.normalized_objective > initial_score.normalized_objective
    assert sum(probe.detectable for probe in result.probes) >= 8


def test_levinson_monitor_rejects_sudden_miscalibration_without_tracking() -> None:
    nominal = SE3.identity()
    shifted = _transform(0.1, 0.0, 0.0, 0.0, 0.0, 0.25)
    frames = _frames_for_truths([nominal] * 10 + [shifted] * 10)
    result = LevinsonThrunOnlineSolver().solve(
        frames, nominal, _options(tracking_enabled=False)
    )

    before = [
        step.calibrated_probability
        for step in result.timeline
        if int(step.frame_id.split("-")[-1]) < 10
    ]
    after = [
        step.calibrated_probability
        for step in result.timeline
        if int(step.frame_id.split("-")[-1]) >= 12
    ]
    assert before and after
    assert float(np.median(before)) > 0.9
    assert max(after) < 0.1


def test_levinson_tracker_follows_gradual_rotational_drift() -> None:
    truths = [
        _transform(0.0, 0.0, 0.0, 0.02 * index, -0.02 * index, 0.02 * index)
        for index in range(18)
    ]
    frames = _frames_for_truths(truths)

    result = LevinsonThrunOnlineSolver().solve(
        frames, SE3.identity(), _options(tracking_enabled=True)
    )

    truth_by_frame = {
        frame.frame_id: truth for frame, truth in zip(frames, truths, strict=True)
    }
    tracked_errors = [
        _rotation_error_deg(step.transform_camera_lidar, truth_by_frame[step.frame_id])
        for step in result.timeline
    ]
    untracked_errors = [
        _rotation_error_deg(SE3.identity(), truth_by_frame[step.frame_id])
        for step in result.timeline
    ]
    assert sum(step.update_applied for step in result.timeline) >= 3
    assert float(np.mean(tracked_errors)) < 0.6 * float(np.mean(untracked_errors))
    assert tracked_errors[-1] <= 0.25


def test_levinson_probability_matches_reported_distribution_centers() -> None:
    options = LevinsonThrunOnlineOptions()
    assert calibrated_probability(options.correct_fraction_mean, options) > 0.99
    assert calibrated_probability(options.incorrect_fraction_mean, options) < 0.01


def _options(*, tracking_enabled: bool) -> LevinsonThrunOnlineOptions:
    return LevinsonThrunOnlineOptions(
        window_size=3,
        min_train_frames=6,
        min_holdout_frames=2,
        holdout_ratio=0.25,
        split_seed=4,
        tracking_enabled=tracking_enabled,
        translation_grid_step_m=0.01,
        rotation_grid_step_deg=0.05,
        min_projected_points=20,
        known_bad_translation_m=0.05,
        known_bad_rotation_deg=0.25,
    )


def _frames_for_truths(truths: list[SE3]) -> list[LevinsonOnlineFrame]:
    camera = LevinsonCameraModel(
        width=96,
        height=72,
        fx=72.0,
        fy=70.0,
        cx=47.5,
        cy=35.5,
    )
    frames: list[LevinsonOnlineFrame] = []
    for index, truth in enumerate(truths):
        rng = np.random.default_rng(8100 + index)
        u = rng.uniform(10.0, camera.width - 11.0, 48)
        v = rng.uniform(10.0, camera.height - 11.0, 48)
        depth = rng.uniform(3.0, 12.0, 48)
        camera_points = np.column_stack(
            (
                (u - camera.cx) * depth / camera.fx,
                (v - camera.cy) * depth / camera.fy,
                depth,
            )
        )
        inverse = truth.inverse()
        lidar_points = np.asarray(
            [inverse.transform_point(point) for point in camera_points], dtype=float
        )
        response = _spot_response(camera.height, camera.width, u, v)
        frames.append(
            LevinsonOnlineFrame(
                frame_id=f"frame-{index:03d}",
                edge_response=response,
                lidar_points=lidar_points,
                discontinuity_weights=rng.uniform(0.6, 1.4, u.shape),
                camera=camera,
            )
        )
    return frames


def _spot_response(
    height: int, width: int, u: NDArray[np.float64], v: NDArray[np.float64]
) -> NDArray[np.float64]:
    yy, xx = np.mgrid[:height, :width]
    response = np.zeros((height, width), dtype=float)
    for center_u, center_v in zip(u, v, strict=True):
        distance2 = (xx - center_u) ** 2 + (yy - center_v) ** 2
        response = np.maximum(response, np.exp(-0.5 * distance2 / (0.6**2)))
    return response


def _transform(
    x: float,
    y: float,
    z: float,
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> SE3:
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )
    return SE3(
        (x, y, z), quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1))
    )


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(float(np.dot(left.rotation_quat_xyzw, right.rotation_quat_xyzw)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))
