"""Compare development-only Camera--LiDAR provider support calibrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrex import __version__
from calibrex.core.camera_lidar_confidence_calibration import (
    CameraLidarConfidenceCalibrationArtifact,
    CameraLidarConfidenceCalibrationCandidate,
    load_camera_lidar_confidence_calibration,
)
from calibrex.core.camera_lidar_provider_support_comparison import (
    CameraLidarProviderSupportCandidate,
    CameraLidarProviderSupportComparisonArtifact,
    CameraLidarProviderSupportComparisonProvenance,
    CameraLidarProviderSupportProtocol,
)
from calibrex.core.provenance import git_commit, sha256_path

CAMERA_LIDAR_PROVIDER_SUPPORT_COMPARISON_VERSION = (
    "calibrex.camera_lidar_provider_support_comparison/v0.1"
)


@dataclass(frozen=True)
class _SourceCandidate:
    calibration: CameraLidarConfidenceCalibrationArtifact
    path: Path
    digest: str
    diagnostic: CameraLidarConfidenceCalibrationCandidate


def compare_camera_lidar_provider_support(
    calibration_paths: list[str | Path],
    *,
    comparison_id: str | None = None,
    command: list[str] | None = None,
) -> CameraLidarProviderSupportComparisonArtifact:
    """Compare like-for-like development calibrations without evaluation data."""

    if len(calibration_paths) < 2:
        raise ValueError("provider support comparison requires at least two inputs")
    sources = [_load_source(Path(path)) for path in calibration_paths]
    _validate_common_protocol(sources)
    ranked = sorted(sources, key=_rank_key)
    rank_by_id = {
        item.calibration.calibration_id: rank
        for rank, item in enumerate(ranked, start=1)
    }
    candidates = [
        _candidate(
            item,
            diagnostic_rank=rank_by_id[item.calibration.calibration_id],
            dominated_by=_dominating_ids(item, sources),
        )
        for item in sources
    ]
    eligible = sorted(
        (item for item in candidates if item.runtime_eligible),
        key=lambda item: item.diagnostic_rank,
    )
    first = sources[0].calibration
    selected = eligible[0].candidate_id if eligible else None
    source_paths = [str(item.path) for item in sources]
    return CameraLidarProviderSupportComparisonArtifact(
        comparison_id=(
            comparison_id
            or f"{first.dataset_id}-provider-support-comparison-v0.1"
        ),
        status="locked" if selected is not None else "rejected",
        protocol=_protocol(first),
        candidates=candidates,
        selected_candidate_id=selected,
        diagnostic_leader_candidate_id=ranked[0].calibration.calibration_id,
        provenance=CameraLidarProviderSupportComparisonProvenance(
            generator=__name__,
            generator_version=(
                f"{__version__}+{CAMERA_LIDAR_PROVIDER_SUPPORT_COMPARISON_VERSION}"
            ),
            git_commit=git_commit(),
            command=command or [],
            source_paths=source_paths,
            source_sha256={str(item.path): item.digest for item in sources},
            notes=[
                "only schema-valid development confidence calibrations were compared",
                "all inputs use one exact problem, threshold grid, split set, and gate",
                "rejected provider calibrations are never runtime eligible",
                "the diagnostic leader is not a runtime selection unless status is locked",
                "excluded evaluation datasets were not opened",
                "release_sota_claim_allowed remains false",
            ],
        ),
    )


def _load_source(path: Path) -> _SourceCandidate:
    calibration = load_camera_lidar_confidence_calibration(path)
    if calibration.schema_version != "slac.camera_lidar_confidence_calibration/v0.2":
        raise ValueError("provider comparison requires v0.2 geometric calibrations")
    if calibration.status is None:
        raise ValueError("provider comparison calibration status is missing")
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"provider calibration is unreadable: {path}")
    diagnostic = _diagnostic_candidate(calibration)
    return _SourceCandidate(
        calibration=calibration,
        path=path,
        digest=digest,
        diagnostic=diagnostic,
    )


def _diagnostic_candidate(
    calibration: CameraLidarConfidenceCalibrationArtifact,
) -> CameraLidarConfidenceCalibrationCandidate:
    if calibration.status == "locked":
        selected = calibration.selected_minimum_confidence
        candidate = next(
            (
                item
                for item in calibration.candidates
                if item.minimum_confidence == selected
            ),
            None,
        )
        if candidate is None:
            raise ValueError("locked confidence threshold is absent from candidates")
        return candidate
    return max(calibration.candidates, key=_diagnostic_key)


def _diagnostic_key(
    candidate: CameraLidarConfidenceCalibrationCandidate,
) -> tuple[float, ...]:
    return (
        float(candidate.minimum_holdout_geometric_inlier_count_across_seeds or 0),
        float(candidate.minimum_train_geometric_inlier_count_across_seeds or 0),
        float(candidate.frames_meeting_geometric_minimum_rate or 0.0),
        float(candidate.geometric_inlier_count or 0),
        float(candidate.geometric_inlier_rate or 0.0),
        candidate.frames_meeting_minimum_rate,
        candidate.full_rank_frame_rate,
        float(candidate.minimum_holdout_correspondence_count_across_seeds),
        float(candidate.minimum_train_correspondence_count_across_seeds),
        float(candidate.accepted_correspondence_count),
        candidate.accepted_correspondence_rate,
        candidate.minimum_confidence,
    )


def _rank_key(source: _SourceCandidate) -> tuple[Any, ...]:
    return (
        *(-value for value in _diagnostic_key(source.diagnostic)),
        source.calibration.calibration_id,
    )


def _dominance_metrics(source: _SourceCandidate) -> tuple[float, ...]:
    diagnostic = source.diagnostic
    return (
        float(diagnostic.minimum_holdout_geometric_inlier_count_across_seeds or 0),
        float(diagnostic.minimum_train_geometric_inlier_count_across_seeds or 0),
        float(diagnostic.frames_meeting_geometric_minimum_rate or 0.0),
        float(diagnostic.geometric_inlier_rate or 0.0),
    )


def _dominating_ids(
    target: _SourceCandidate,
    sources: list[_SourceCandidate],
) -> list[str]:
    target_metrics = _dominance_metrics(target)
    dominating: list[str] = []
    for other in sources:
        if other is target:
            continue
        other_metrics = _dominance_metrics(other)
        if all(a >= b for a, b in zip(other_metrics, target_metrics, strict=True)) and any(
            a > b for a, b in zip(other_metrics, target_metrics, strict=True)
        ):
            dominating.append(other.calibration.calibration_id)
    return sorted(dominating)


def _candidate(
    source: _SourceCandidate,
    *,
    diagnostic_rank: int,
    dominated_by: list[str],
) -> CameraLidarProviderSupportCandidate:
    calibration = source.calibration
    diagnostic = source.diagnostic
    train = diagnostic.minimum_train_geometric_inlier_count_across_seeds
    holdout = diagnostic.minimum_holdout_geometric_inlier_count_across_seeds
    if train is None or holdout is None:
        raise ValueError("provider comparison geometric split diagnostics are absent")
    return CameraLidarProviderSupportCandidate(
        candidate_id=calibration.calibration_id,
        calibration_id=calibration.calibration_id,
        calibration_path=str(source.path),
        calibration_sha256=source.digest,
        correspondence_artifact_id=calibration.correspondence_artifact_id,
        correspondence_artifact_sha256=calibration.correspondence_artifact_sha256,
        provider=calibration.provider,
        calibration_status=calibration.status or "rejected",
        passing_thresholds=[
            item.minimum_confidence
            for item in calibration.candidates
            if item.gate_pass
        ],
        runtime_eligible=calibration.status == "locked",
        diagnostic=diagnostic,
        worst_split_geometric_inlier_count=min(train, holdout),
        diagnostic_rank=diagnostic_rank,
        pareto_dominated_by=dominated_by,
    )


def _validate_common_protocol(sources: list[_SourceCandidate]) -> None:
    first = _protocol_signature(sources[0].calibration)
    seen_ids: set[str] = set()
    for source in sources:
        calibration = source.calibration
        if calibration.calibration_id in seen_ids:
            raise ValueError("provider calibration IDs must be unique")
        seen_ids.add(calibration.calibration_id)
        if _protocol_signature(calibration) != first:
            raise ValueError(
                "provider calibrations do not share one exact development protocol"
            )


def _protocol_signature(
    calibration: CameraLidarConfidenceCalibrationArtifact,
) -> tuple[Any, ...]:
    return (
        calibration.dataset_id,
        calibration.sequence_id,
        calibration.split_id,
        calibration.problem_id,
        calibration.problem_sha256,
        tuple(calibration.evaluation_dataset_ids_excluded),
        tuple(item.minimum_confidence for item in calibration.candidates),
        calibration.calibration_holdout_ratio,
        tuple(calibration.calibration_split_seeds),
        calibration.minimum_frame_correspondence_count,
        calibration.minimum_frame_support_rate,
        calibration.minimum_full_rank_frame_rate,
        calibration.development_reprojection_inlier_threshold_px,
        calibration.minimum_geometric_inlier_rate,
        calibration.minimum_frame_geometric_inlier_count,
        calibration.minimum_frame_geometric_support_rate,
        calibration.reference_pose_used_for_development_calibration,
        calibration.runtime_holdout_used_for_pose_selection,
        calibration.release_sota_claim_allowed,
    )


def _protocol(
    calibration: CameraLidarConfidenceCalibrationArtifact,
) -> CameraLidarProviderSupportProtocol:
    reprojection = calibration.development_reprojection_inlier_threshold_px
    inlier_rate = calibration.minimum_geometric_inlier_rate
    frame_inliers = calibration.minimum_frame_geometric_inlier_count
    frame_rate = calibration.minimum_frame_geometric_support_rate
    if (
        reprojection is None
        or inlier_rate is None
        or frame_inliers is None
        or frame_rate is None
    ):
        raise ValueError("provider comparison geometric gate settings are incomplete")
    return CameraLidarProviderSupportProtocol(
        dataset_id=calibration.dataset_id,
        sequence_id=calibration.sequence_id,
        problem_id=calibration.problem_id,
        problem_sha256=calibration.problem_sha256,
        evaluation_dataset_ids_excluded=(
            calibration.evaluation_dataset_ids_excluded
        ),
        confidence_thresholds=[
            item.minimum_confidence for item in calibration.candidates
        ],
        calibration_holdout_ratio=calibration.calibration_holdout_ratio,
        calibration_split_seeds=calibration.calibration_split_seeds,
        minimum_frame_correspondence_count=(
            calibration.minimum_frame_correspondence_count
        ),
        minimum_frame_support_rate=calibration.minimum_frame_support_rate,
        minimum_full_rank_frame_rate=calibration.minimum_full_rank_frame_rate,
        development_reprojection_inlier_threshold_px=float(reprojection),
        minimum_geometric_inlier_rate=float(inlier_rate),
        minimum_frame_geometric_inlier_count=int(frame_inliers),
        minimum_frame_geometric_support_rate=float(frame_rate),
    )
