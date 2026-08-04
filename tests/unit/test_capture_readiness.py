from __future__ import annotations

import jsonschema

from calibrex.core.capture_readiness import (
    CaptureReadinessArtifact,
    CaptureReadinessGeometry,
    CaptureReadinessMotion,
    CaptureReadinessParameters,
    CaptureReadinessProvenance,
    CaptureReadinessThresholds,
    CaptureReadinessWindow,
    build_capture_readiness_recommendations,
    capture_readiness_json_schema,
    classify_capture_readiness_window,
)


def _geometry(*, dof: int = 6, rank: int = 3) -> CaptureReadinessGeometry:
    return CaptureReadinessGeometry(
        source_scan_count=20,
        source_point_count=2000,
        plane_count=100,
        normal_rank=rank,
        normal_condition_number=4.0,
        normal_diversity_score=0.25,
        normal_eigenvalues=[100.0, 50.0, 25.0],
        estimated_observable_dof=dof,
        weak_directions=[] if dof == 6 else ["yaw_target"],
        correspondence_count=500,
    )


def _motion(*, rotation_deg: float = 60.0) -> CaptureReadinessMotion:
    return CaptureReadinessMotion(
        pose_count=100,
        first_timestamp_ns=1,
        last_timestamp_ns=10_000_000_001,
        time_span_s=10.0,
        path_length_m=3.0,
        endpoint_displacement_m=2.0,
        endpoint_rotation_deg=rotation_deg,
        linear_speed_p95_mps=0.4,
        linear_speed_max_mps=0.8,
        angular_speed_p95_dps=20.0,
        angular_speed_max_dps=30.0,
    )


def _window(index: int, grade: str) -> CaptureReadinessWindow:
    return CaptureReadinessWindow(
        window_index=index,
        requested_start_timestamp_ns=1,
        requested_end_timestamp_ns=10,
        motion=_motion(),
        geometry=_geometry(),
        grade=grade,  # type: ignore[arg-type]
        decision="proceed" if grade == "pass" else "review",
        reasons=["fixture"],
        actions=["fixture action"],
    )


def test_low_rotation_window_requires_recapture() -> None:
    window = classify_capture_readiness_window(
        window_index=0,
        requested_window=(1, 10),
        motion=_motion(rotation_deg=2.0),
        geometry=_geometry(),
        thresholds=CaptureReadinessThresholds(),
    )

    assert window.grade == "fail"
    assert window.decision == "recapture"
    assert any("endpoint rotation" in reason for reason in window.reasons)
    assert any("roll, pitch, and yaw" in action for action in window.actions)


def test_readiness_recommendation_prioritizes_failed_windows() -> None:
    grade, decision, summary, recommendations = build_capture_readiness_recommendations(
        [_window(0, "fail"), _window(1, "pass")]
    )

    assert grade == "fail"
    assert decision == "recapture"
    assert "not ready" in summary
    assert recommendations[0].action == "recapture"
    assert recommendations[0].window_indices == [0]


def test_capture_readiness_artifact_is_schema_valid() -> None:
    artifact = CaptureReadinessArtifact(
        config_path="config.yaml",
        dataset_path="bag",
        source_sensor="livox",
        target_sensor="rslidar",
        odometry_topic="/odom",
        parameters=CaptureReadinessParameters(
            voxel_size_m=0.5,
            correspondence_gate_m=1.5,
            max_scans_per_window=40,
            max_points_per_window=4000,
            source_point_time_field="offset_time",
            odometry_burst_min_interval_s=0.001,
            capture_window_record_prefilter_margin_s=2.0,
            capture_window_sampling_policy="evenly_spaced",
        ),
        thresholds=CaptureReadinessThresholds(),
        windows=[_window(0, "pass")],
        grade="pass",
        decision="proceed",
        summary="fixture",
        recommendations=[
            {
                "action": "proceed",
                "message": "fixture",
                "window_indices": [0],
            }
        ],
        provenance=CaptureReadinessProvenance(
            source_paths=["config.yaml", "bag"],
            source_sha256={"config.yaml": "a" * 64, "bag": "b" * 64},
        ),
    )

    payload = artifact.model_dump(mode="json", exclude_none=True)
    jsonschema.validate(payload, capture_readiness_json_schema())
    assert payload["decision"] == "proceed"
