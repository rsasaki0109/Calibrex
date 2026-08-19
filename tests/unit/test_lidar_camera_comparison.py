import math
from pathlib import Path

import pytest

from calibrex.core.config import load_config
from calibrex.core.geometry import SE3
from calibrex.core.result import CalibrationResult, FrameGraphSnapshot, RunInfo, TransformResult
from calibrex.data.inspect import inspect_dataset
from calibrex.data.kitti import read_velodyne_to_camera_transform
from calibrex.evaluation.lidar_camera_comparison import (
    _applied_transform,
    compare_lidar_camera_candidate_scores,
    evaluate_lidar_camera_candidates_on_kitti,
    lidar_camera_comparison_metrics_from_result,
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

    deltas = compare_lidar_camera_candidate_scores(
        scores,
        reference_candidate_id="reference",
    )
    external = deltas["external"]
    assert external.translation_delta_m == pytest.approx(0.1)
    assert external.rotation_delta_deg == pytest.approx(1.0)
    assert external.holdout_edge_alignment_delta == pytest.approx(
        scores["external"].holdout_edge_alignment - scores["reference"].holdout_edge_alignment
    )
    assert external.both_training_isolated is False


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


def test_missing_reference_returns_no_pairwise_deltas() -> None:
    assert compare_lidar_camera_candidate_scores({}, reference_candidate_id="missing") == {}


def test_applied_transform_uses_frame_graph_candidate_when_enabled() -> None:
    reference = read_velodyne_to_camera_transform(_DATASET)
    assert reference is not None
    angle = math.radians(5.0)
    perturbed = reference.compose(
        SE3(
            (0.25, 0.0, 0.0),
            (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)),
        )
    )
    camera = TransformResult(
        parent="base_link",
        child="camera0",
        translation_m=(0.0, 0.0, 0.0),
        rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    result = CalibrationResult(
        run=RunInfo(id="unit", slac_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(root="base_link", frames={"base_link": None}),
    )
    result.transforms = {
        "T_base_link_camera0": camera,
        "T_base_link_lidar0": TransformResult(
            parent="base_link",
            child="lidar0",
            translation_m=reference.translation_m,
            rotation_quat_xyzw=reference.rotation_quat_xyzw,
        ),
    }
    result.candidate_extrinsics = {
        "T_base_link_camera0": camera,
        "T_base_link_lidar0": TransformResult(
            parent="base_link",
            child="lidar0",
            translation_m=perturbed.translation_m,
            rotation_quat_xyzw=perturbed.rotation_quat_xyzw,
        ),
    }
    config = load_config(
        "examples/public_datasets/kitti_lidar_camera_evidence/config.yaml"
    ).model_copy(deep=True)
    config.evaluation.kitti.use_frame_graph_candidate = True

    applied = _applied_transform(result, config=config)
    assert applied is not None
    assert applied.translation_m == pytest.approx(perturbed.translation_m)
    assert applied.rotation_quat_xyzw == pytest.approx(perturbed.rotation_quat_xyzw)


def test_comparison_metrics_report_nonzero_dataset_reference_delta_for_bad_candidate() -> None:
    reference = read_velodyne_to_camera_transform(_DATASET)
    assert reference is not None
    angle = math.radians(5.0)
    perturbed = reference.compose(
        SE3(
            (0.25, 0.0, 0.0),
            (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)),
        )
    )
    camera = TransformResult(
        parent="base_link",
        child="camera0",
        translation_m=(0.0, 0.0, 0.0),
        rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    result = CalibrationResult(
        run=RunInfo(id="unit", slac_version="0.1.0"),
        frame_graph=FrameGraphSnapshot(root="base_link", frames={"base_link": None}),
    )
    result.transforms = {
        "T_base_link_camera0": camera,
        "T_base_link_lidar0": TransformResult(
            parent="base_link",
            child="lidar0",
            translation_m=reference.translation_m,
            rotation_quat_xyzw=reference.rotation_quat_xyzw,
        ),
    }
    result.candidate_extrinsics = {
        "T_base_link_camera0": camera,
        "T_base_link_lidar0": TransformResult(
            parent="base_link",
            child="lidar0",
            translation_m=perturbed.translation_m,
            rotation_quat_xyzw=perturbed.rotation_quat_xyzw,
        ),
    }
    config = load_config(
        "examples/public_datasets/kitti_lidar_camera_evidence/config.yaml"
    ).model_copy(deep=True)
    config.evaluation.kitti.use_frame_graph_candidate = True
    config.pipeline.factors["lidar_camera_baseline_comparison"].enabled = True
    inspection = inspect_dataset(config.dataset)

    metrics = lidar_camera_comparison_metrics_from_result(config, result, inspection)
    translation_delta = metrics[
        "lidar_camera_comparison_calibrex_applied_vs_kitti_dataset_reference_translation_delta_m"
    ].value
    rotation_delta = metrics[
        "lidar_camera_comparison_calibrex_applied_vs_kitti_dataset_reference_rotation_delta_deg"
    ].value

    assert translation_delta == pytest.approx(0.25, abs=0.02)
    assert rotation_delta == pytest.approx(5.0, abs=0.5)
