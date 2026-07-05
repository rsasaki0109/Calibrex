"""Unit tests for LiDAR-IMU evidence summary rows and grade rollup."""

from __future__ import annotations

from slac.core.result import CalibrationResult, FrameGraphSnapshot, MetricResult, RunInfo
from slac.evaluation.evidence_summary import (
    _lidar_imu_evidence_items,
    evidence_decision_grade_from_result,
    evidence_summaries_from_result,
)
from slac.evaluation.metrics import evaluate_quality


def _lidar_imu_result(**metric_overrides: MetricResult) -> CalibrationResult:
    metrics = {
        "lidar_imu_pose_interval_count": MetricResult(value=20.0, unit="intervals", grade="pass"),
        "lidar_imu_imu_sample_count": MetricResult(value=400.0, unit="samples", grade="pass"),
        "lidar_imu_odometry_angular_excitation_p95_dps": MetricResult(
            value=30.0,
            unit="deg/s",
            grade="pass",
        ),
        "lidar_imu_holdout_rotation_rate_rmse_dps": MetricResult(
            holdout=1.5,
            unit="deg/s",
            grade="pass",
        ),
        "lidar_imu_holdout_rotation_rate_mean_abs_dps": MetricResult(
            holdout=1.0,
            unit="deg/s",
            grade="pass",
        ),
        "lidar_imu_known_bad_detectable_fraction": MetricResult(value=1.0, grade="pass"),
        "lidar_imu_known_bad_mandatory_case_count": MetricResult(
            value=12.0,
            unit="cases",
            grade="pass",
        ),
        "lidar_imu_known_bad_mandatory_detectable_count": MetricResult(
            value=12.0,
            unit="cases",
            grade="pass",
        ),
        "lidar_imu_gravity_alignment_deg": MetricResult(value=2.0, unit="deg", grade="pass"),
    }
    metrics.update(metric_overrides)
    return CalibrationResult(
        run=RunInfo(
            id="lidar-imu-evidence-unit",
            slac_version="0.1.0",
            domain="robotics",
            provenance={
                "lidar_imu_evidence": {
                    "protocol_id": "lidar_imu_rotation_gravity_holdout/v0.1",
                    "evidence_gates": {
                        "imu_gate_max_holdout_rotation_rate_rmse_dps": 5.0,
                        "imu_gate_max_gravity_alignment_deg": 10.0,
                        "known_bad_detect_margin": 0.20,
                        "min_excitation_p95_dps": 5.0,
                    },
                }
            },
        ),
        frame_graph=FrameGraphSnapshot(root="velodyne_vlp16", frames={"velodyne_vlp16": None}),
        metrics=metrics,
    )


def test_lidar_imu_evidence_pass_records_thresholds() -> None:
    items = _lidar_imu_evidence_items(_lidar_imu_result())
    by_check = {item.check: item for item in items}

    assert by_check["Holdout Rotation Consistency"].status == "pass"
    assert "threshold <= 5" in by_check["Holdout Rotation Consistency"].evidence
    assert by_check["Known-Bad Controls"].status == "pass"
    assert "threshold >= 0.5" in by_check["Known-Bad Controls"].evidence
    assert by_check["Decision Boundary"].status == "pass"
    assert all(item.family == "lidar_imu" for item in items)


def test_lidar_imu_support_includes_axis_observability_for_z_dominant() -> None:
    result = _lidar_imu_result()
    result.run.provenance["lidar_imu_evidence"] = {
        **result.run.provenance["lidar_imu_evidence"],
        "odometry_angular_excitation_p95_dps_per_axis": {
            "x": 1.0,
            "y": 2.0,
            "z": 35.0,
        },
    }
    items = _lidar_imu_evidence_items(result)
    by_check = {item.check: item for item in items}
    assert "per-axis p95" in by_check["Candidate Support"].evidence
    assert "Axis-observability" in by_check["Candidate Support"].interpretation
    assert "Weakly detectable probe axis: yaw" in by_check["Candidate Support"].interpretation
    assert "Falsifiable probe axes: roll, pitch" in by_check["Candidate Support"].interpretation


def test_lidar_imu_support_inconclusive_low_excitation() -> None:
    items = _lidar_imu_evidence_items(
        _lidar_imu_result(
            lidar_imu_odometry_angular_excitation_p95_dps=MetricResult(
                value=1.0,
                unit="deg/s",
                grade="warn",
            ),
        )
    )
    by_check = {item.check: item for item in items}
    assert by_check["Candidate Support"].status == "warn"
    assert "INCONCLUSIVE" in by_check["Candidate Support"].interpretation
    assert by_check["Decision Boundary"].status == "warn"


def test_lidar_imu_holdout_warn_when_rmse_above_gate() -> None:
    items = _lidar_imu_evidence_items(
        _lidar_imu_result(
            lidar_imu_holdout_rotation_rate_rmse_dps=MetricResult(
                holdout=12.0,
                unit="deg/s",
                grade="warn",
            ),
        )
    )
    by_check = {item.check: item for item in items}
    assert by_check["Holdout Rotation Consistency"].status == "warn"
    assert by_check["Decision Boundary"].status == "warn"


def test_lidar_imu_gravity_support_never_fails_decision() -> None:
    items = _lidar_imu_evidence_items(
        _lidar_imu_result(
            lidar_imu_gravity_alignment_deg=MetricResult(value=45.0, unit="deg", grade="warn"),
        )
    )
    by_check = {item.check: item for item in items}
    assert by_check["Gravity Support"].status == "warn"
    assert by_check["Decision Boundary"].status == "pass"


def test_lidar_imu_no_cross_family_contamination() -> None:
    summaries = evidence_summaries_from_result(_lidar_imu_result())
    assert summaries
    assert all(item.family == "lidar_imu" for item in summaries)


def test_lidar_imu_decision_grade_feeds_evaluate_quality() -> None:
    result = _lidar_imu_result()
    evaluate_quality(result)
    assert evidence_decision_grade_from_result(result) == "pass"
    assert result.quality.grade in {"pass", "warn"}
