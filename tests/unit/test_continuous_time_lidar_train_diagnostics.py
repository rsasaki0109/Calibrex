from __future__ import annotations

import pytest

from calibrex.core.continuous_time_lidar_train_diagnostics import (
    ContinuousTimeLidarTrainDiagnostics,
    ContinuousTimeLidarTrainDiagnosticsData,
    ContinuousTimeLidarTrainDiagnosticsProvenance,
    build_train_range_bins,
    summarize_train_residuals,
    train_diagnostics_artifact_from_data,
)


def test_train_residual_summary_uses_deterministic_absolute_percentiles() -> None:
    summary = summarize_train_residuals([-0.2, 0.1, 0.05, -0.4])

    assert summary.count == 4
    assert summary.rmse_m == pytest.approx(0.2304886114323222)
    assert summary.mean_m == -0.1125
    assert summary.mean_abs_m == 0.1875
    assert summary.median_abs_m == pytest.approx(0.15)
    assert summary.p95_abs_m == 0.4
    assert summary.p99_abs_m == 0.4
    assert summary.min_m == -0.4
    assert summary.max_m == 0.1


def test_train_range_bins_have_stable_boundaries_and_empty_summaries() -> None:
    bins = build_train_range_bins(
        [0.0, 4.999, 5.0, 79.999, 80.0, 120.0],
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
    )

    assert len(bins) == 6
    assert [item.correspondence_count for item in bins] == [2, 1, 0, 0, 1, 2]
    assert bins[0].lower_bound_m == 0.0
    assert bins[0].upper_bound_m == 5.0
    assert bins[-1].lower_bound_m == 80.0
    assert bins[-1].upper_bound_m is None
    assert bins[2].residuals.count == 0
    assert bins[2].residuals.rmse_m is None


def test_train_diagnostics_artifact_attaches_provenance_and_excludes_holdout() -> None:
    data = ContinuousTimeLidarTrainDiagnosticsData(
        sampled_source_record_count=100,
        train_target_point_count=80,
        train_candidate_correspondence_count=10,
        train_correspondence_count=8,
        train_outlier_rejected_count=2,
        selected_time_offset_sec=0.012,
        final_residuals=summarize_train_residuals([0.01, -0.02]),
        range_bins=build_train_range_bins([2.0, 12.0], [0.01, -0.02]),
        profile=(),
    )
    artifact = train_diagnostics_artifact_from_data(
        data,
        provenance=ContinuousTimeLidarTrainDiagnosticsProvenance(
            source_paths=["config.yaml", "capture.db3"],
            source_sha256={"config.yaml": "a" * 64, "capture.db3": "b" * 64},
        ),
    )

    assert isinstance(artifact, ContinuousTimeLidarTrainDiagnostics)
    assert artifact.holdout_used_for_selection is False
    assert artifact.train_outlier_rejection_rate == 0.2
    assert not hasattr(artifact, "holdout_rmse_m")
    assert artifact.provenance.source_sha256["capture.db3"] == "b" * 64
