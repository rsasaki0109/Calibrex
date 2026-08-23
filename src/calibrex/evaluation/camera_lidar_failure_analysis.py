"""Derive digest-bound failure categories from Camera--LiDAR artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from calibrex.core.camera_lidar_artifacts import (
    CalibrationCandidateTrace,
    load_calibration_candidate_trace,
    load_camera_lidar_benchmark_protocol,
)
from calibrex.core.camera_lidar_failure_analysis import (
    CameraLidarFailureAnalysisArtifact,
    CameraLidarFailureAnalysisProvenance,
    CameraLidarFailureCase,
    CameraLidarFailureFinding,
    CameraLidarFailureSummary,
)
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticRefinementResultArtifact,
    load_probabilistic_refinement_result,
)
from calibrex.core.provenance import git_commit, sha256_path

CAMERA_LIDAR_FAILURE_ANALYSIS_VERSION = (
    "calibrex.camera_lidar_failure_analysis/v0.1"
)


def analyze_camera_lidar_failures(
    *,
    protocol_path: str | Path,
    initialization_trace_paths: Sequence[str | Path],
    result_paths: Sequence[str | Path],
    command: list[str] | None = None,
) -> CameraLidarFailureAnalysisArtifact:
    """Analyze every retained refinement result against its D2D initializer.

    The function reloads every source artifact and verifies the protocol and
    trace digests recorded by the refinement result.  Categories describe
    observed outcomes; they are not causal diagnoses.
    """

    protocol_file = Path(protocol_path).resolve()
    protocol = load_camera_lidar_benchmark_protocol(protocol_file)
    protocol_digest = _required_digest(protocol_file)
    traces = _load_traces(
        initialization_trace_paths,
        protocol_digest,
        {item.trial_id for item in protocol.perturbations},
        protocol.problem_sha256,
    )
    result_files = tuple(sorted((Path(path).resolve() for path in result_paths), key=str))
    if not result_files:
        raise ValueError("Camera-LiDAR failure analysis requires refinement results")

    cases: list[CameraLidarFailureCase] = []
    source_paths = [protocol_file, *[item[0] for item in traces.values()], *result_files]
    source_sha256 = {str(path): _required_digest(path) for path in source_paths}
    for result_file in result_files:
        result = load_probabilistic_refinement_result(result_file)
        if result.initialization_trace_id is None:
            raise ValueError(
                "failure analysis requires a D2D-initialized result: "
                f"{result_file}"
            )
        trace_entry = traces.get(result.initialization_trace_id)
        if trace_entry is None:
            raise ValueError(
                "refinement result references an unknown initialization trace: "
                f"{result_file}"
            )
        trace_file, trace = trace_entry
        trace_digest = source_sha256[str(trace_file)]
        if result.provenance.initialization_problem_sha256 != protocol.problem_sha256:
            raise ValueError(
                "refinement result problem digest does not match the protocol: "
                f"{result_file}"
            )
        if result.provenance.initialization_trace_sha256 != trace_digest:
            raise ValueError(
                "refinement result and initialization trace digests differ: "
                f"{result_file}"
            )
        seed = _split_seed(result, result_file)
        final_rotation = result.final_rotation_error_deg
        final_translation = result.final_translation_error_m
        candidate_rotation = (
            result.candidate_rotation_error_deg
            if result.candidate_rotation_error_deg is not None
            else final_rotation
        )
        candidate_translation = (
            result.candidate_translation_error_m
            if result.candidate_translation_error_m is not None
            else final_translation
        )
        initial_rmse = result.initial_holdout_evaluation.weighted_reprojection_rmse_px
        final_rmse = result.final_holdout_evaluation.weighted_reprojection_rmse_px
        candidate_holdout = (
            result.candidate_holdout_evaluation
            if result.candidate_holdout_evaluation is not None
            else result.final_holdout_evaluation
        )
        candidate_rmse = candidate_holdout.weighted_reprojection_rmse_px
        candidate_accepted = (
            result.acceptance.accepted
            if result.acceptance is not None
            else result.status == "converged"
        )
        selected_source = (
            result.acceptance.selected_source
            if result.acceptance is not None
            else "refined_candidate"
        )
        acceptance_reasons = (
            result.acceptance.reasons if result.acceptance is not None else []
        )
        candidate_rotation_failure = (
            candidate_rotation is None
            or candidate_rotation >= protocol.hit.rotation_error_max_deg
        )
        candidate_translation_failure = (
            candidate_translation is None
            or candidate_translation >= protocol.hit.translation_error_max_m
        )
        candidate_holdout_degradation = bool(
            initial_rmse is not None
            and candidate_rmse is not None
            and candidate_rmse > initial_rmse + 1.0e-9
        )
        rotation_failure = (
            final_rotation is None
            or final_rotation >= protocol.hit.rotation_error_max_deg
        )
        translation_failure = (
            final_translation is None
            or final_translation >= protocol.hit.translation_error_max_m
        )
        holdout_degradation = bool(
            initial_rmse is not None
            and final_rmse is not None
            and final_rmse > initial_rmse + 1.0e-9
        )
        categories = _failure_categories(
            refinement_status=result.status,
            candidate_accepted=candidate_accepted,
            initializer_rollback=selected_source == "initializer_rollback",
            d2d_initial_rotation_failure=(
                trace.outcome.rotation_error_deg
                >= protocol.hit.rotation_error_max_deg
            ),
            d2d_initial_translation_failure=(
                trace.outcome.translation_error_m
                >= protocol.hit.translation_error_max_m
            ),
            rotation_failure=rotation_failure,
            translation_failure=translation_failure,
            holdout_degradation=holdout_degradation,
            candidate_rotation_failure=candidate_rotation_failure,
            candidate_translation_failure=candidate_translation_failure,
            candidate_holdout_degradation=candidate_holdout_degradation,
            initial_rmse=initial_rmse,
            final_rmse=final_rmse,
        )
        cases.append(
            CameraLidarFailureCase(
                case_id=result.result_id,
                trial_id=trace.trial_id,
                seed=seed,
                result_path=str(result_file),
                result_sha256=source_sha256[str(result_file)],
                initialization_trace_path=str(trace_file),
                initialization_trace_sha256=trace_digest,
                d2d_status=trace.status,
                refinement_status=result.status,
                candidate_accepted=candidate_accepted,
                selected_source=selected_source,
                acceptance_reasons=acceptance_reasons,
                d2d_initial_hit=trace.outcome.hit,
                d2d_initial_rotation_error_deg=trace.outcome.rotation_error_deg,
                d2d_initial_translation_error_m=trace.outcome.translation_error_m,
                initial_rotation_error_deg=result.initial_rotation_error_deg,
                candidate_rotation_error_deg=candidate_rotation,
                final_rotation_error_deg=final_rotation,
                initial_translation_error_m=result.initial_translation_error_m,
                candidate_translation_error_m=candidate_translation,
                final_translation_error_m=final_translation,
                initial_holdout_rmse_px=initial_rmse,
                candidate_holdout_rmse_px=candidate_rmse,
                final_holdout_rmse_px=final_rmse,
                candidate_rotation_failure=candidate_rotation_failure,
                candidate_translation_failure=candidate_translation_failure,
                candidate_holdout_degradation=candidate_holdout_degradation,
                rotation_failure=rotation_failure,
                translation_failure=translation_failure,
                holdout_degradation=holdout_degradation,
                failure_categories=categories,
                diagnosis=(
                    "no_declared_failure"
                    if not categories
                    else "+".join(categories)
                ),
            )
        )

    summary = _summary(cases)
    findings = _findings(cases, summary)
    return CameraLidarFailureAnalysisArtifact(
        artifact_id=f"{protocol.dataset_id}-camera-lidar-failure-analysis",
        dataset_id=protocol.dataset_id,
        protocol_id=protocol.protocol_id,
        protocol_sha256=protocol_digest,
        protocol_path=str(protocol_file),
        rotation_threshold_deg=protocol.hit.rotation_error_max_deg,
        translation_threshold_m=protocol.hit.translation_error_max_m,
        analysis_scope=(
            "all supplied D2D initialization trials and seeded probabilistic "
            "multi-frame refinement results"
        ),
        summary=summary,
        cases=cases,
        findings=findings,
        conclusion=(
            "Observed failure categories are reported against the frozen pose "
            "thresholds; findings are diagnostic associations, not causal proof."
        ),
        provenance=CameraLidarFailureAnalysisProvenance(
            source_paths=[str(path) for path in source_paths],
            source_sha256=source_sha256,
            generator_version=CAMERA_LIDAR_FAILURE_ANALYSIS_VERSION,
            git_commit=git_commit(),
            command=command or [],
            notes=[
                "all source artifacts were reloaded and digest-checked",
                "reference-transform errors are not independent metrology",
            ],
        ),
    )


def _load_traces(
    paths: Sequence[str | Path],
    protocol_digest: str,
    expected_trial_ids: set[str],
    expected_problem_digest: str,
) -> dict[str, tuple[Path, CalibrationCandidateTrace]]:
    if not paths:
        raise ValueError("failure analysis requires D2D initialization traces")
    traces: dict[str, tuple[Path, CalibrationCandidateTrace]] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        trace = load_calibration_candidate_trace(path)
        if trace.protocol_sha256 != protocol_digest:
            raise ValueError(f"D2D trace protocol digest mismatch: {path}")
        if trace.problem_sha256 != expected_problem_digest:
            raise ValueError(f"D2D trace problem digest mismatch: {path}")
        if trace.trial_id not in expected_trial_ids:
            raise ValueError(f"D2D trace trial is absent from the protocol: {path}")
        if trace.trace_id in traces:
            raise ValueError(f"duplicate D2D trace ID: {trace.trace_id}")
        traces[trace.trace_id] = (path, trace)
    return traces


def _failure_categories(
    *,
    refinement_status: str,
    candidate_accepted: bool,
    initializer_rollback: bool,
    d2d_initial_rotation_failure: bool,
    d2d_initial_translation_failure: bool,
    rotation_failure: bool,
    translation_failure: bool,
    holdout_degradation: bool,
    candidate_rotation_failure: bool,
    candidate_translation_failure: bool,
    candidate_holdout_degradation: bool,
    initial_rmse: float | None,
    final_rmse: float | None,
) -> list[str]:
    categories: list[str] = []
    if d2d_initial_rotation_failure:
        categories.append("d2d_initial_rotation_failure")
    if d2d_initial_translation_failure:
        categories.append("d2d_initial_translation_failure")
    if refinement_status != "converged":
        categories.append("refinement_solver_failure")
    if not candidate_accepted:
        categories.append("refinement_candidate_rejected")
    if initializer_rollback:
        categories.append("initializer_rollback")
    if candidate_rotation_failure:
        categories.append("candidate_rotation_failure")
    if candidate_translation_failure:
        categories.append("candidate_translation_failure")
    if candidate_holdout_degradation:
        categories.append("candidate_holdout_degradation")
    if rotation_failure:
        categories.append("refinement_rotation_failure")
    if translation_failure:
        categories.append("refinement_translation_failure")
    if holdout_degradation:
        categories.append("holdout_degradation")
    if initial_rmse is None or final_rmse is None:
        categories.append("missing_holdout_rmse")
    return categories


def _summary(cases: list[CameraLidarFailureCase]) -> CameraLidarFailureSummary:
    count = len(cases)
    rotation_errors = [
        item.final_rotation_error_deg
        for item in cases
        if item.final_rotation_error_deg is not None
    ]
    translation_errors = [
        item.final_translation_error_m
        for item in cases
        if item.final_translation_error_m is not None
    ]
    diagnosis_counts = Counter(item.diagnosis for item in cases)
    candidate_accepted_count = sum(item.candidate_accepted is True for item in cases)
    rollback_count = sum(item.selected_source == "initializer_rollback" for item in cases)
    candidate_rotation_failure_count = sum(
        item.candidate_rotation_failure is True for item in cases
    )
    candidate_translation_failure_count = sum(
        item.candidate_translation_failure is True for item in cases
    )
    candidate_holdout_degradation_count = sum(
        item.candidate_holdout_degradation is True for item in cases
    )
    return CameraLidarFailureSummary(
        case_count=count,
        scored_case_count=count,
        d2d_initial_hit_count=sum(item.d2d_initial_hit for item in cases),
        d2d_initial_hit_rate=_rate(sum(item.d2d_initial_hit for item in cases), count),
        refinement_converged_count=sum(
            item.refinement_status == "converged" for item in cases
        ),
        refinement_converged_rate=_rate(
            sum(item.refinement_status == "converged" for item in cases), count
        ),
        candidate_accepted_count=candidate_accepted_count,
        candidate_accepted_rate=_rate(candidate_accepted_count, count),
        initializer_rollback_count=rollback_count,
        initializer_rollback_rate=_rate(rollback_count, count),
        candidate_rotation_failure_count=candidate_rotation_failure_count,
        candidate_rotation_failure_rate=_rate(
            candidate_rotation_failure_count, count
        ),
        candidate_translation_failure_count=candidate_translation_failure_count,
        candidate_translation_failure_rate=_rate(
            candidate_translation_failure_count, count
        ),
        candidate_holdout_degradation_count=candidate_holdout_degradation_count,
        candidate_holdout_degradation_rate=_rate(
            candidate_holdout_degradation_count, count
        ),
        rotation_failure_count=sum(item.rotation_failure for item in cases),
        rotation_failure_rate=_rate(sum(item.rotation_failure for item in cases), count),
        translation_failure_count=sum(item.translation_failure for item in cases),
        translation_failure_rate=_rate(
            sum(item.translation_failure for item in cases), count
        ),
        holdout_degradation_count=sum(item.holdout_degradation for item in cases),
        holdout_degradation_rate=_rate(
            sum(item.holdout_degradation for item in cases), count
        ),
        mean_final_rotation_error_deg=_mean(rotation_errors),
        p90_final_rotation_error_deg=_percentile(rotation_errors, 0.90),
        mean_final_translation_error_m=_mean(translation_errors),
        p90_final_translation_error_m=_percentile(translation_errors, 0.90),
        diagnosis_counts=dict(sorted(diagnosis_counts.items())),
    )


def _findings(
    cases: list[CameraLidarFailureCase],
    summary: CameraLidarFailureSummary,
) -> list[CameraLidarFailureFinding]:
    findings: list[CameraLidarFailureFinding] = []
    if summary.initializer_rollback_count:
        findings.append(
            CameraLidarFailureFinding(
                id="initializer_rollback_observed",
                severity="warning",
                confidence="high",
                title="Initializer-preserving rollback rejected refinement candidates",
                evidence=(
                    f"{summary.initializer_rollback_count}/{summary.case_count} "
                    "cases retained the initializer"
                ),
                supporting_case_ids=[
                    item.case_id
                    for item in cases
                    if item.selected_source == "initializer_rollback"
                ],
                caveat=(
                    "Rollback prevents unsafe publication but does not repair the "
                    "correspondence objective or solver."
                ),
            )
        )
    if summary.candidate_holdout_degradation_count:
        findings.append(
            CameraLidarFailureFinding(
                id="candidate_holdout_degradation_observed",
                severity="candidate_cause",
                confidence="high",
                title="Retained candidates degraded held-out reprojection",
                evidence=(
                    f"{summary.candidate_holdout_degradation_count}/"
                    f"{summary.case_count} candidates exceeded initializer holdout RMSE"
                ),
                supporting_case_ids=[
                    item.case_id
                    for item in cases
                    if item.candidate_holdout_degradation is True
                ],
                caveat=(
                    "Holdout degradation is evaluation evidence and is never used "
                    "by the acceptance policy."
                ),
            )
        )
    if summary.translation_failure_count > summary.rotation_failure_count:
        findings.append(
            CameraLidarFailureFinding(
                id="translation_failure_dominant",
                severity="candidate_cause",
                confidence="medium",
                title="Translation failures exceed rotation failures",
                evidence=(
                    f"translation failures={summary.translation_failure_count}/"
                    f"{summary.case_count}; rotation failures="
                    f"{summary.rotation_failure_count}/{summary.case_count}"
                ),
                supporting_case_ids=[
                    item.case_id for item in cases if item.translation_failure
                ],
                caveat=(
                    "This association does not identify whether correspondences, "
                    "depth, or optimization caused the error."
                ),
            )
        )
    if summary.holdout_degradation_count:
        findings.append(
            CameraLidarFailureFinding(
                id="holdout_degradation_observed",
                severity="warning",
                confidence="high",
                title="Refinement degraded held-out reprojection",
                evidence=(
                    f"{summary.holdout_degradation_count}/{summary.case_count} "
                    "cases have final holdout RMSE above the D2D initializer"
                ),
                supporting_case_ids=[
                    item.case_id for item in cases if item.holdout_degradation
                ],
                caveat=(
                    "The holdout metric is an evaluation signal and does not "
                    "establish a physical calibration cause."
                ),
            )
        )
    if not findings:
        findings.append(
            CameraLidarFailureFinding(
                id="no_dominant_failure_pattern",
                severity="info",
                confidence="medium",
                title="No dominant failure pattern observed",
                evidence=(
                    "No translation-dominant or holdout-degradation pattern met "
                    "the declared rule."
                ),
                caveat="A null finding does not prove robustness or independent accuracy.",
            )
        )
    return findings


def _split_seed(result: ProbabilisticRefinementResultArtifact, path: Path) -> int:
    value = result.options.get("split_seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"refinement result has no integer split_seed: {path}")
    return value


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required failure-analysis input is not readable: {path}")
    return digest


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _percentile(values: Sequence[float | None], quantile: float) -> float | None:
    finite = sorted(value for value in values if value is not None)
    if not finite:
        return None
    index = max(0, min(len(finite) - 1, int((len(finite) - 1) * quantile + 0.999999)))
    return finite[index]
