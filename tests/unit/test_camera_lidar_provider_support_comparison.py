from __future__ import annotations

from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.camera_lidar_confidence_calibration import (
    CameraLidarConfidenceCalibrationArtifact,
    CameraLidarConfidenceCalibrationCandidate,
    CameraLidarConfidenceCalibrationProvenance,
    CameraLidarLockedRefinementOptions,
)
from calibrex.core.camera_lidar_correspondence_quality import (
    CameraLidarDistributionSummary,
    CameraLidarPoseObservability,
)
from calibrex.core.camera_lidar_provider_support_comparison import (
    load_camera_lidar_provider_support_comparison,
)
from calibrex.core.probabilistic_correspondence import (
    CorrespondenceProviderIdentity,
)
from calibrex.core.validation import validate_file
from calibrex.evaluation.camera_lidar_provider_support_comparison import (
    compare_camera_lidar_provider_support,
)


def test_provider_support_comparison_keeps_rejected_leader_ineligible(
    tmp_path: Path,
) -> None:
    weak_path = tmp_path / "provider-v03.yaml"
    leader_path = tmp_path / "provider-v05.yaml"
    _calibration(
        calibration_id="provider-v03",
        provider_version="0.3",
        geometric_counts=(4, 2, 0),
    ).save(weak_path)
    _calibration(
        calibration_id="provider-v05",
        provider_version="0.5",
        geometric_counts=(82, 37, 0),
    ).save(leader_path)

    comparison = compare_camera_lidar_provider_support(
        [weak_path, leader_path],
        comparison_id="provider-support-fixture",
    )

    assert comparison.status == "rejected"
    assert comparison.selected_candidate_id is None
    assert comparison.diagnostic_leader_candidate_id == "provider-v05"
    assert not comparison.release_sota_claim_allowed
    leader = next(
        item for item in comparison.candidates if item.candidate_id == "provider-v05"
    )
    weak = next(
        item for item in comparison.candidates if item.candidate_id == "provider-v03"
    )
    assert leader.diagnostic_rank == 1
    assert not leader.runtime_eligible
    assert leader.worst_split_geometric_inlier_count == 0
    assert weak.pareto_dominated_by == ["provider-v05"]


def test_provider_support_comparison_cli_writes_schema_valid_rejection(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    output = tmp_path / "comparison.yaml"
    _calibration(
        calibration_id="first",
        provider_version="1",
        geometric_counts=(12, 5, 0),
    ).save(first_path)
    _calibration(
        calibration_id="second",
        provider_version="2",
        geometric_counts=(24, 9, 0),
    ).save(second_path)

    exit_code = main(
        [
            "camera-lidar",
            "compare-probabilistic-provider-support",
            str(first_path),
            str(second_path),
            "--output",
            str(output),
            "--comparison-id",
            "cli-provider-support-fixture",
            "--json",
        ]
    )

    assert exit_code == 2
    assert validate_file(
        output, "camera-lidar-provider-support-comparison"
    ).valid
    assert validate_file(output, "auto").valid
    loaded = load_camera_lidar_provider_support_comparison(output)
    assert loaded.comparison_id == "cli-provider-support-fixture"
    assert loaded.status == "rejected"
    assert loaded.provenance.command == [
        "calibrex",
        "camera-lidar",
        "compare-probabilistic-provider-support",
        str(first_path),
        str(second_path),
        "--output",
        str(output),
        "--comparison-id",
        "cli-provider-support-fixture",
    ]


def test_provider_support_comparison_selects_only_locked_calibration(
    tmp_path: Path,
) -> None:
    rejected_path = tmp_path / "rejected.yaml"
    locked_path = tmp_path / "locked.yaml"
    _calibration(
        calibration_id="diagnostic-leader-rejected",
        provider_version="rejected",
        geometric_counts=(100, 50, 0),
    ).save(rejected_path)
    _calibration(
        calibration_id="locked-runtime-provider",
        provider_version="locked",
        geometric_counts=(60, 30, 10),
        locked=True,
    ).save(locked_path)

    comparison = compare_camera_lidar_provider_support(
        [rejected_path, locked_path]
    )

    assert comparison.status == "locked"
    assert comparison.selected_candidate_id == "locked-runtime-provider"
    selected = next(
        item
        for item in comparison.candidates
        if item.candidate_id == comparison.selected_candidate_id
    )
    assert selected.runtime_eligible
    assert selected.passing_thresholds == [0.2]


def test_provider_support_comparison_rejects_protocol_mismatch(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first = _calibration(
        calibration_id="first",
        provider_version="1",
        geometric_counts=(12, 5, 0),
    )
    second = _calibration(
        calibration_id="second",
        provider_version="2",
        geometric_counts=(24, 9, 0),
    ).model_copy(update={"problem_sha256": "f" * 64})
    first.save(first_path)
    second.save(second_path)

    with pytest.raises(ValueError, match="one exact development protocol"):
        compare_camera_lidar_provider_support([first_path, second_path])


def test_provider_support_diagnostic_prefers_support_when_geometry_ties(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    _calibration(
        calibration_id="support-rich",
        provider_version="1",
        geometric_counts=(0, 0, 0),
        accepted_counts=(180, 4),
    ).save(first_path)
    _calibration(
        calibration_id="support-middle",
        provider_version="2",
        geometric_counts=(0, 0, 0),
        accepted_counts=(160, 8),
    ).save(second_path)

    comparison = compare_camera_lidar_provider_support(
        [first_path, second_path]
    )

    assert comparison.diagnostic_leader_candidate_id == "support-rich"
    assert all(
        item.diagnostic.minimum_confidence == pytest.approx(0.1)
        for item in comparison.candidates
    )


def _calibration(
    *,
    calibration_id: str,
    provider_version: str,
    geometric_counts: tuple[int, int, int],
    accepted_counts: tuple[int, int] = (200, 200),
    locked: bool = False,
) -> CameraLidarConfidenceCalibrationArtifact:
    total, train, holdout = geometric_counts
    candidates = [
        _candidate(
            threshold=0.1,
            geometric_count=max(total - 1, 0),
            train_count=max(train - 1, 0),
            holdout_count=holdout,
            accepted_count=accepted_counts[0],
            gate_pass=False,
        ),
        _candidate(
            threshold=0.2,
            geometric_count=total,
            train_count=train,
            holdout_count=holdout,
            accepted_count=accepted_counts[1],
            gate_pass=locked,
        ),
    ]
    return CameraLidarConfidenceCalibrationArtifact(
        status="locked" if locked else "rejected",
        calibration_id=calibration_id,
        dataset_id="development-dataset",
        sequence_id="development-sequence",
        split_id="development",
        problem_id="development-problem",
        problem_sha256="a" * 64,
        correspondence_artifact_id=f"correspondence-{calibration_id}",
        correspondence_artifact_sha256="b" * 64,
        provider=CorrespondenceProviderIdentity(
            provider="fixture",
            model="fixture-model",
            version=provider_version,
            source_repository="https://example.test/provider",
            source_commit="c" * 40,
            license_spdx="MIT",
            checkpoint_redistribution="not applicable",
        ),
        evaluation_dataset_ids_excluded=["evaluation-a", "evaluation-b"],
        selection_rule_id=(
            "highest_threshold_passing_support_observability_and_geometry/v0.2"
        ),
        calibration_holdout_ratio=0.25,
        calibration_split_seeds=[0, 1],
        minimum_frame_correspondence_count=4,
        minimum_frame_support_rate=0.95,
        minimum_full_rank_frame_rate=0.95,
        development_reprojection_inlier_threshold_px=8.0,
        minimum_geometric_inlier_rate=0.25,
        minimum_frame_geometric_inlier_count=4,
        minimum_frame_geometric_support_rate=0.95,
        candidates=candidates,
        selected_minimum_confidence=0.2 if locked else None,
        locked_refinement_options=_locked_options() if locked else None,
        provenance=CameraLidarConfidenceCalibrationProvenance(
            generator="pytest",
            generator_version="1",
            source_paths=["correspondence.yaml", "problem.yaml"],
            source_sha256={
                "correspondence.yaml": "d" * 64,
                "problem.yaml": "e" * 64,
            },
        ),
    )


def _candidate(
    *,
    threshold: float,
    geometric_count: int,
    train_count: int,
    holdout_count: int,
    accepted_count: int,
    gate_pass: bool,
) -> CameraLidarConfidenceCalibrationCandidate:
    accepted = accepted_count
    frame_geometric_count = (
        10 if holdout_count > 0 else (6 if geometric_count > 0 else 0)
    )
    return CameraLidarConfidenceCalibrationCandidate(
        minimum_confidence=threshold,
        accepted_correspondence_count=accepted,
        accepted_correspondence_rate=accepted / 200,
        frames_meeting_minimum_count=10,
        frames_meeting_minimum_rate=1.0,
        full_rank_frame_count=10,
        full_rank_frame_rate=1.0,
        minimum_train_correspondence_count_across_seeds=min(120, accepted),
        minimum_holdout_correspondence_count_across_seeds=min(40, accepted),
        aggregate_observability=_observability(),
        geometric_inlier_count=geometric_count,
        geometric_inlier_rate=geometric_count / accepted,
        frames_meeting_geometric_minimum_count=frame_geometric_count,
        frames_meeting_geometric_minimum_rate=frame_geometric_count / 10,
        minimum_train_geometric_inlier_count_across_seeds=train_count,
        minimum_holdout_geometric_inlier_count_across_seeds=holdout_count,
        reprojection_error_px=CameraLidarDistributionSummary(
            count=max(geometric_count, 1),
            minimum=1.0,
            p10=1.0,
            median=1.0,
            p90=1.0,
            p95=1.0,
            maximum=1.0,
            mean=1.0,
        ),
        gate_pass=gate_pass,
        gate_reasons=[] if gate_pass else ["geometric_support_gate_failed"],
    )


def _observability() -> CameraLidarPoseObservability:
    axes = ["rx", "ry", "rz", "tx", "ty", "tz"]
    return CameraLidarPoseObservability(
        parameter_order=axes,
        parameter_scales=[1.0] * 6,
        factor_count=200,
        singular_values=[1.0] * 6,
        rank=6,
        condition_number=1.0,
        axis_information_fraction=dict.fromkeys(axes, 1.0 / 6.0),
        weak_axes=[],
    )


def _locked_options() -> CameraLidarLockedRefinementOptions:
    return CameraLidarLockedRefinementOptions(
        minimum_confidence=0.2,
        holdout_ratio=0.25,
        evaluation_split_seeds=[0, 1],
        minimum_train_correspondences=24,
        minimum_holdout_correspondences=8,
        rotation_bound_deg=5.0,
        translation_bound_m=1.0,
        initial_rotation_step_deg=1.0,
        initial_translation_step_m=0.1,
        minimum_rotation_step_deg=0.1,
        minimum_translation_step_m=0.01,
        max_evaluations=100,
        cauchy_scale=2.0,
        minimum_absolute_train_objective_improvement=1.0e-6,
        minimum_relative_train_objective_improvement=1.0e-4,
        maximum_train_correspondence_loss_fraction=0.1,
        maximum_accepted_bound_fraction=0.95,
    )
