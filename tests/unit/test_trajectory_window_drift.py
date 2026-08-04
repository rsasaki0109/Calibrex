from __future__ import annotations

import jsonschema

from calibrex.core.trajectory_window_drift import (
    TrajectoryWindowCrossSegmentEvidence,
    TrajectoryWindowDriftArtifact,
    TrajectoryWindowDriftParameters,
    TrajectoryWindowDriftProvenance,
    TrajectoryWindowDriftThresholds,
    TrajectoryWindowDriftWindow,
    TrajectoryWindowOdometryStats,
    interpret_trajectory_window_drift,
    trajectory_window_drift_json_schema,
)


def _cross(status: str, rmse_m: float | None = 0.1) -> TrajectoryWindowCrossSegmentEvidence:
    return TrajectoryWindowCrossSegmentEvidence(
        rmse_m=rmse_m,
        first_half_scan_count=2,
        second_half_scan_count=2,
        first_half_point_count=10,
        second_half_point_count=10,
        sampling_policy="evenly_spaced_time",
        correspondence_count=100,
        gate_status=status,  # type: ignore[arg-type]
        gate_reason="fixture",
    )


def _artifact(statuses: list[str], full_status: str) -> TrajectoryWindowDriftArtifact:
    windows = [
        TrajectoryWindowDriftWindow(
            window_index=index,
            start_timestamp_ns=index * 10,
            end_timestamp_ns=(index + 1) * 10,
            odometry=TrajectoryWindowOdometryStats(
                pose_count=2,
                first_timestamp_ns=index * 10,
                last_timestamp_ns=(index + 1) * 10,
                time_span_s=1.0,
                path_length_m=1.0,
                endpoint_displacement_m=1.0,
                endpoint_rotation_deg=2.0,
            ),
            cross_segment=_cross(status),
        )
        for index, status in enumerate(statuses)
    ]
    return TrajectoryWindowDriftArtifact(
        config_path="config.yaml",
        dataset_path="bag",
        source_sensor="source",
        target_sensor="target",
        odometry_topic="/odom",
        parameters=TrajectoryWindowDriftParameters(
            voxel_size_m=0.5,
            correspondence_gate_m=1.5,
            max_scans_per_half=40,
            max_points_per_half=4000,
            source_point_time_field="offset_time",
            odometry_burst_min_interval_s=0.001,
            capture_window_record_prefilter_margin_s=2.0,
            capture_window_sampling_policy="evenly_spaced",
        ),
        thresholds=TrajectoryWindowDriftThresholds(
            max_cross_segment_rmse_m=0.3,
            min_correspondences=50,
        ),
        windows=windows,
        full_span_cross_segment=_cross(full_status),
        interpretation="local_inconsistency_suspected",
        grade="fail",
        interpretation_reason="fixture",
        provenance=TrajectoryWindowDriftProvenance(
            source_paths=["config.yaml", "bag"],
            source_sha256={"config.yaml": "a" * 64, "bag": "b" * 64},
        ),
    )


def test_window_drift_interpretation_separates_local_failure() -> None:
    interpretation, grade, reason = interpret_trajectory_window_drift(
        ["fail", "pass", "fail"], "fail"
    )

    assert interpretation == "local_inconsistency_suspected"
    assert grade == "fail"
    assert "0, 2" in reason


def test_window_drift_interpretation_identifies_full_span_only_failure() -> None:
    interpretation, grade, reason = interpret_trajectory_window_drift(
        ["pass", "pass"], "fail"
    )

    assert interpretation == "long_span_inconsistency_suspected"
    assert grade == "fail"
    assert "long-span" in reason


def test_window_drift_artifact_is_schema_valid() -> None:
    artifact = _artifact(["fail", "pass", "fail"], "fail")

    jsonschema.validate(
        artifact.model_dump(mode="json", exclude_none=True),
        trajectory_window_drift_json_schema(),
    )
    assert artifact.windows[1].odometry.path_length_m == 1.0
