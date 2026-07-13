import math
from pathlib import Path

from calibrex.core.geometry import SE3
from calibrex.data.kitti import read_velodyne_to_camera_transform
from calibrex.evaluation.lidar_camera_comparison import (
    evaluate_lidar_camera_candidates_on_kitti,
)

_DATASET = Path(
    "examples/public_datasets/kitti_lidar_camera_evidence/2011_09_26/2011_09_26_drive_0005_sync"
)


def test_candidates_share_identical_train_holdout_frame_ids() -> None:
    reference = read_velodyne_to_camera_transform(_DATASET)
    assert reference is not None
    angle = math.radians(1.0)
    perturbation = SE3(
        (0.1, 0.0, 0.0),
        (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)),
    )

    scores = evaluate_lidar_camera_candidates_on_kitti(
        _DATASET,
        {
            "reference": (reference, True),
            "external": (perturbation.compose(reference), False),
        },
        max_frames=2,
        max_points=800,
        holdout_ratio=0.2,
        split_seed=9,
    )

    assert scores["reference"].train_frame_ids == scores["external"].train_frame_ids
    assert scores["reference"].holdout_frame_ids == scores["external"].holdout_frame_ids
    assert len(scores["reference"].train_frame_ids) == 1
    assert len(scores["reference"].holdout_frame_ids) == 1
    assert scores["reference"].scored_frame_count == 2
    assert scores["external"].training_isolation_declared is False
    assert scores["reference"].holdout_edge_alignment is not None
    assert scores["external"].holdout_edge_alignment is not None


def test_empty_dataset_returns_no_candidate_scores(tmp_path: Path) -> None:
    scores = evaluate_lidar_camera_candidates_on_kitti(
        tmp_path,
        {"candidate": (SE3.identity(), True)},
        max_frames=2,
        max_points=100,
        holdout_ratio=0.2,
        split_seed=0,
    )

    assert scores["candidate"].scored_frame_count == 0
    assert scores["candidate"].holdout_projection_ratio is None
