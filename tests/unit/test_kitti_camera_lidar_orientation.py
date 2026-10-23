"""Orientation penalties must not reward mismatch or alter unknown-orientation scores."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.evaluation import camera_lidar_edge as edge

oriented = _load("kitti_camera_lidar_orientation")
probe = _load("kitti_camera_lidar_profile_probe")
CAMERA = edge.EdgeCamera(fx=1, fy=1, cx=2, cy=2)


def _frame(value: float = 1.0) -> edge.PreparedFrame:
    return edge.PreparedFrame(
        time_s=1, gap_s=0, edge=np.full((4, 4), value, dtype=np.float32), scale=1,
        points=np.array([[0, 2, 5]], dtype=np.float32),
        weights=np.ones(1, dtype=np.float32), rings=1,
    )


def _feature(xx: float, yy: float) -> oriented.OrientationFeatures:
    tensor = np.zeros((4, 4, 3), dtype=np.float32)
    tensor[..., 0], tensor[..., 2] = xx, yy
    return oriented.OrientationFeatures(np.ones((4, 4), dtype=np.float32), tensor)


def test_aligned_and_perpendicular_gradients_and_pose_dependent_scan_direction() -> None:
    aligned = oriented.OrientationObjective([_frame()], CAMERA, np.eye(4), [_feature(1, 0)])
    perpendicular = oriented.OrientationObjective([_frame()], CAMERA, np.eye(4), [_feature(0, 1)])
    assert aligned.score(np.zeros(6)) == pytest.approx(1)
    assert perpendicular.score(np.zeros(6)) == pytest.approx(0)
    turn = np.array([0, 0, 90.0, 0, 0, 0])
    assert aligned.score(turn) == pytest.approx(0, abs=1e-12)
    assert perpendicular.score(turn) == pytest.approx(1)


def test_perpendicular_orientation_does_not_raise_negative_background_score() -> None:
    objective = oriented.OrientationObjective([_frame(-0.5)], CAMERA, np.eye(4), [_feature(0, 1)])
    assert objective.score(np.zeros(6)) == pytest.approx(-1.5)


def test_weak_gradient_confidence_reduces_the_orientation_penalty() -> None:
    objective = oriented.OrientationObjective([_frame()], CAMERA, np.eye(4), [_feature(0, 0.25)])
    assert objective.score(np.zeros(6)) == pytest.approx(0.75)


@pytest.mark.parametrize("scale", [1, 2])
def test_unknown_orientation_matches_scalar_scores_with_fixed_margin_support(scale: int) -> None:
    frame = replace(_frame(-0.5), scale=scale)
    frame = replace(frame, points=np.array([[0, 2, 5], [-11, 0, 5]], dtype=np.float32),
                    weights=np.array([1, 2], dtype=np.float32))
    scalar = edge.EdgeObjective([frame], CAMERA, np.eye(4))
    neutral = oriented.OrientationObjective([frame], CAMERA, np.eye(4), [_feature(0, 0)])
    for theta in (np.zeros(6), np.array([1.0, 2, 3, 0.1, 0, 0]),
                  np.array([0, 0, 0, 4.0, 0, 0])):
        assert neutral.score(theta) == pytest.approx(scalar.score(theta), abs=1e-12)
    assert neutral._total_weight == scalar._total_weight == 3


def test_image_features_are_polarity_independent_and_keep_scalar_image_scale() -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[:, 16:] = 255
    options = edge.EdgeAlignmentOptions(image_max_width=16)
    feature, factor = oriented.prepare_orientation_image(image, options)
    opposite, other_factor = oriented.prepare_orientation_image(255 - image, options)
    assert factor == other_factor == 2
    np.testing.assert_allclose(feature.tensor, opposite.tensor, atol=1e-6)
    assert feature.tensor.shape == (16, 16, 3)
    assert not feature.tensor.flags.writeable and not feature.raw_response.flags.writeable
    assert feature.tensor[8, 8, 0] == pytest.approx(1)
    with pytest.raises(ValueError, match="no image edges"):
        oriented.prepare_orientation_image(np.zeros_like(image), options)


def test_unsupported_or_confounded_inputs_are_rejected_before_estimation() -> None:
    with pytest.raises(ValueError, match="organized"):
        oriented.OrientationObjective([replace(_frame(), rings=0)], CAMERA,
                                       np.eye(4), [_feature(1, 0)])
    with pytest.raises(ValueError, match="pinhole"):
        oriented.OrientationObjective([_frame()], replace(CAMERA, distortion_model="radtan"),
                                       np.eye(4), [_feature(1, 0)])
    with pytest.raises(ValueError, match="one objective"):
        probe.profile_snapshot(Path("missing"), Path("missing.json"), Path("unused"),
                               spatial_balance=True, azimuth_orientation=True)
