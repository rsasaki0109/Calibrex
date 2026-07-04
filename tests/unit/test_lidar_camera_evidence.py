"""Unit tests for camera-LiDAR evidence summary rows and grade rollup."""

from __future__ import annotations

from calibrex.core.result import CalibrationResult, FrameGraphSnapshot, MetricResult, RunInfo
from calibrex.evaluation.evidence_summary import (
    _lidar_camera_evidence_items,
    evidence_decision_grade_from_result,
    evidence_summaries_from_result,
)
from calibrex.evaluation.metrics import evaluate_quality


def _camera_lidar_result(**metric_overrides: MetricResult) -> CalibrationResult:
    metrics = {
        "lidar_camera_projection_frame_count": MetricResult(value=2.0, unit="frames", grade="pass"),
        "lidar_camera_projected_points": MetricResult(
            train=12.0,
            holdout=12.0,
            unit="points",
            grade="pass",
        ),
        "lidar_camera_projection_ratio": MetricResult(train=1.0, holdout=1.0, grade="pass"),
        "lidar_camera_projection_horizontal_coverage": MetricResult(
            train=0.2,
            holdout=0.2,
            grade="pass",
        ),
        "lidar_camera_projection_vertical_coverage": MetricResult(
            train=0.2,
            holdout=0.2,
            grade="pass",
        ),
        "lidar_camera_edge_alignment_score": MetricResult(
            train=0.5,
            holdout=0.5,
            grade="pass",
        ),
        "lidar_camera_depth_discontinuity_points": MetricResult(
            train=12.0,
            holdout=12.0,
            unit="points",
            grade="pass",
        ),
        "lidar_camera_depth_edge_alignment_score": MetricResult(
            train=0.5,
            holdout=0.5,
            grade="pass",
        ),
        "lidar_camera_perturbation_detectable_fraction": MetricResult(
            value=1.0,
            grade="pass",
        ),
        "lidar_camera_perturbation_mandatory_case_count": MetricResult(
            value=12.0,
            unit="cases",
            grade="pass",
        ),
        "lidar_camera_perturbation_mandatory_detectable_count": MetricResult(
            value=12.0,
            unit="cases",
            grade="pass",
        ),
        "lidar_camera_perturbation_edge_delta_mean": MetricResult(value=0.1, grade="pass"),
        "lidar_camera_perturbation_depth_edge_delta_mean": MetricResult(value=0.1, grade="pass"),
        "lidar_camera_perturbation_projection_ratio_delta_mean": MetricResult(
            value=0.0,
            grade="pass",
        ),
    }
    metrics.update(metric_overrides)
    return CalibrationResult(
        run=RunInfo(
            id="camera-lidar-evidence-unit",
            calibrex_version="0.1.0",
            domain="autonomous_driving",
            provenance={
                "lidar_camera_evidence": {
                    "protocol_id": "kitti_lidar_camera_projection_edge_holdout/v0.1",
                    "evidence_gates": {
                        "evidence_gate_min_edge_alignment_holdout": 0.20,
                        "evidence_gate_min_depth_edge_alignment_holdout": 0.20,
                        "evidence_gate_min_perturbation_detectable_fraction": 0.50,
                        "evidence_gate_min_mandatory_detectable_count": 8,
                    },
                    "limitations": [
                        "projection evidence observes image-plane alignment only",
                    ],
                }
            },
        ),
        frame_graph=FrameGraphSnapshot(root="base_link", frames={"base_link": None}),
        metrics=metrics,
    )


def test_lidar_camera_evidence_pass_records_thresholds() -> None:
    items = _lidar_camera_evidence_items(_camera_lidar_result())
    by_check = {item.check: item for item in items}

    assert by_check["Holdout Edge Alignment"].status == "pass"
    assert "threshold >= 0.2" in by_check["Holdout Edge Alignment"].evidence
    assert by_check["Known-Bad Controls"].status == "pass"
    assert "threshold >= 0.5" in by_check["Known-Bad Controls"].evidence
    assert by_check["Decision Boundary"].status == "pass"
    assert "Observability Statement" in by_check


def test_lidar_camera_holdout_fail_when_edge_score_below_threshold() -> None:
    items = _lidar_camera_evidence_items(
        _camera_lidar_result(
            lidar_camera_edge_alignment_score=MetricResult(
                train=0.5,
                holdout=0.05,
                grade="fail",
            ),
        )
    )
    by_check = {item.check: item for item in items}

    assert by_check["Holdout Edge Alignment"].status == "warn"
    assert by_check["Decision Boundary"].status == "warn"


def test_lidar_camera_holdout_inconclusive_without_holdout_split() -> None:
    items = _lidar_camera_evidence_items(
        _camera_lidar_result(
            lidar_camera_edge_alignment_score=MetricResult(value=0.5, grade="pass"),
        )
    )
    by_check = {item.check: item for item in items}

    assert by_check["Holdout Edge Alignment"].status == "warn"
    assert "unavailable" in by_check["Holdout Edge Alignment"].evidence


def test_lidar_camera_known_bad_fail_when_mandatory_probes_not_detected() -> None:
    items = _lidar_camera_evidence_items(
        _camera_lidar_result(
            lidar_camera_perturbation_detectable_fraction=MetricResult(
                value=0.0,
                grade="fail",
            ),
            lidar_camera_perturbation_mandatory_detectable_count=MetricResult(
                value=0.0,
                unit="cases",
                grade="warn",
            ),
        )
    )
    by_check = {item.check: item for item in items}

    assert by_check["Known-Bad Controls"].status == "fail"
    assert by_check["Decision Boundary"].status == "warn"


def test_evidence_summaries_include_both_families_without_cross_contamination() -> None:
    result = _camera_lidar_result()
    result.metrics["lidar_pair_source_voxel_recall_in_target"] = MetricResult(
        value=0.25,
        grade="pass",
    )
    result.metrics["lidar_pair_known_bad_detectable_fraction"] = MetricResult(
        value=1.0,
        grade="pass",
    )

    items = evidence_summaries_from_result(result)
    families = {item.family for item in items}

    assert "lidar_camera" in families
    assert "lidar_pair" in families
    assert sum(1 for item in items if item.family == "lidar_camera") == 5


def test_evaluate_quality_includes_camera_lidar_decision_boundary() -> None:
    result = _camera_lidar_result(
        lidar_camera_perturbation_detectable_fraction=MetricResult(
            value=0.0,
            grade="fail",
        ),
        lidar_camera_perturbation_mandatory_detectable_count=MetricResult(
            value=0.0,
            unit="cases",
            grade="warn",
        ),
    )

    evaluate_quality(result)

    assert evidence_decision_grade_from_result(result) == "warn"
    assert any(
        "evidence decision boundary" in item for item in result.quality.warnings
    )
