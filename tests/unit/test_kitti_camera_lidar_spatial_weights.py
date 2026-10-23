"""Tile balancing preserves point support and fixes weights before any pose search."""
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.evaluation import camera_lidar_edge as edge

spatial = _load("kitti_camera_lidar_spatial_weights")
CAMERA = edge.EdgeCamera(fx=1, fy=1, cx=2, cy=2)


def _frame() -> edge.PreparedFrame:
    return edge.PreparedFrame(
        time_s=1, gap_s=0, edge=np.arange(16, dtype=np.float32).reshape(4, 4), scale=1,
        points=np.array([[-1.5, -1.5, 1], [-1.25, -1.5, 1], [0.5, -1.5, 1],
                         [-1.5, 0.5, 1], [0.5, 0.5, 1]], dtype=np.float32),
        weights=np.array([100, 1, 2, 3, 4], dtype=np.float32), rings=1,
    )


def test_equal_tile_mass_preserves_relative_weights_points_and_input() -> None:
    original = _frame()
    balanced, support = spatial.balance_frame(original, CAMERA, np.eye(4))
    assert support.point_counts == [2, 1, 1, 1]
    assert support.original_weight == [101, 2, 3, 4]
    assert support.outside_point_count == 0
    assert support.balanced_weight == pytest.approx([27.5] * 4)
    assert balanced.weights.sum() == pytest.approx(original.weights.sum())
    assert balanced.weights[0] / balanced.weights[1] == pytest.approx(100)
    assert balanced.points is original.points and balanced.edge is original.edge
    np.testing.assert_array_equal(original.weights, [100, 1, 2, 3, 4])
    assert not balanced.weights.flags.writeable
    # Trial scoring uses constant weights; no reassignment or denominator change occurs.
    objective = edge.EdgeObjective([balanced], CAMERA, np.eye(4))
    before = balanced.weights.copy()
    objective.score(np.array([2.0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(balanced.weights, before)
    assert objective._total_weight == pytest.approx(110)


def test_unoccupied_tiles_receive_no_mass_and_frame_total_is_preserved() -> None:
    frame = _frame()
    frame = replace(frame, points=frame.points[:2], weights=frame.weights[:2])
    balanced, support = spatial.balance_frame(frame, CAMERA, np.eye(4))
    assert support.point_counts == [2, 0, 0, 0]
    assert support.balanced_weight == pytest.approx([101, 0, 0, 0])
    assert balanced.weights.sum() == pytest.approx(frame.weights.sum())


@pytest.mark.parametrize("weights", [[], [1], [100, 1, 2, 3, 0], [100, 1, 2, 3, float("nan")]])
def test_bad_support_is_rejected(weights: list[float]) -> None:
    with pytest.raises(ValueError, match="positive finite matching weights"):
        spatial.balance_frame(replace(_frame(), weights=np.array(weights)), CAMERA, np.eye(4))


def test_projection_support_and_image_scale_follow_objective_coordinates() -> None:
    frame = _frame()
    scaled = replace(frame, scale=2)
    # Double pixel coordinates and add the half-pixel shift used by the core objective.
    camera = edge.EdgeCamera(fx=2, fy=2, cx=4.5, cy=4.5)
    _, support = spatial.balance_frame(scaled, camera, np.eye(4))
    assert support.point_counts == [2, 1, 1, 1]
    outside, support = spatial.balance_frame(
        frame, edge.EdgeCamera(fx=1, fy=1, cx=20, cy=20), np.eye(4),
    )
    assert support.outside_point_count == 5 and support.point_counts == [0, 0, 0, 0]
    assert support.outside_original_weight == support.outside_balanced_weight == 110
    np.testing.assert_array_equal(outside.weights, frame.weights)
    with pytest.raises(ValueError, match="finite valid"):
        spatial.balance_frame(replace(frame, points=-frame.points), CAMERA, np.eye(4))


def test_margin_points_keep_their_weights_and_can_enter_the_trial_image() -> None:
    frame = _frame()
    frame = replace(frame, points=np.vstack([frame.points, [-2.1, 0, 1]]).astype(np.float32),
                    weights=np.append(frame.weights, np.float32(5)))
    balanced, support = spatial.balance_frame(frame, CAMERA, np.eye(4))
    assert support.outside_point_count == 1
    assert support.outside_original_weight == support.outside_balanced_weight == 5
    assert len(balanced.points) == len(frame.points) == 6
    assert balanced.weights[-1] == frame.weights[-1] == 5
    assert balanced.weights.sum() == pytest.approx(frame.weights.sum())
    # A translation can bring that originally outside point into the image.
    point_only = replace(balanced, points=balanced.points[-1:], weights=balanced.weights[-1:])
    objective = edge.EdgeObjective([point_only], CAMERA, np.eye(4))
    assert objective.score(np.zeros(6)) == 0
    assert objective.score(np.array([0, 0, 0, 0.2, 0, 0])) > 0
