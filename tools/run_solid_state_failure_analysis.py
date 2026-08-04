#!/usr/bin/env python3
"""Analyze adaptive-vs-uniform counterexamples in a solid-state benchmark."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calibrex import __version__
from calibrex.core.continuous_time_lidar_artifacts import (
    ContinuousTimeLidarPairArtifact,
)
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.solid_state_cross_dataset_benchmark import (
    SolidStateCrossDatasetBenchmarkDataset,
    SolidStateCrossDatasetBenchmarkManifest,
    SolidStateCrossDatasetBenchmarkReplicate,
    SolidStateCrossDatasetBenchmarkVariant,
)
from calibrex.core.solid_state_failure_analysis import (
    SolidStateFailureAnalysisFinding,
    SolidStateFailureAnalysisManifest,
    SolidStateFailureAnalysisProvenance,
    SolidStateFailureAnalysisReplicate,
    SolidStateFailureAnalysisSummary,
    SolidStateFailureAnalysisVariant,
)
from calibrex.core.solid_state_failure_analysis_report import (
    write_solid_state_failure_analysis_reports,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_manifest", type=Path, help="v0.2 benchmark result YAML")
    parser.add_argument(
        "--dataset-id",
        default="agrob_modular_e",
        help="dataset to analyze (default: agrob_modular_e)",
    )
    parser.add_argument("--output", type=Path, required=True, help="analysis result YAML")
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--json", action="store_true", help="emit the result as JSON")
    return parser


def _resolve(path_text: str, *, base: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    candidates = [(base / path).resolve(), (Path.cwd() / path).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _digest(path: Path) -> str | None:
    return sha256_path(path)


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _variant_path(
    record: SolidStateCrossDatasetBenchmarkVariant,
    *,
    replicate: SolidStateCrossDatasetBenchmarkReplicate,
    benchmark_path: Path,
) -> Path | None:
    if record.result_path is None:
        return None
    ablation_path = _resolve(replicate.ablation_manifest_path, base=benchmark_path.parent)
    return _resolve(record.result_path, base=ablation_path.parent)


def _profile_diagnostics(
    artifact: ContinuousTimeLidarPairArtifact,
) -> tuple[int, int, float | None, float | None, float | None, float | None]:
    iterations = artifact.iterations
    holdouts = [item.holdout_rmse_m for item in iterations if item.holdout_rmse_m is not None]
    offsets = [offset for item in iterations for offset in item.candidate_offsets_sec]
    if not offsets:
        offsets = [artifact.initial_time_offset_sec, artifact.estimated_time_offset_sec]
    best_holdout = min(holdouts) if holdouts else artifact.final_holdout_rmse_m
    holdout_span = max(holdouts) - min(holdouts) if len(holdouts) > 1 else 0.0 if holdouts else None
    offset_span = max(offsets) - min(offsets) if offsets else None
    accepted_count = sum(item.accepted for item in iterations)
    accepted_rate = accepted_count / len(iterations) if iterations else None
    return (
        len(iterations),
        accepted_count,
        accepted_rate,
        best_holdout,
        holdout_span,
        offset_span,
    )


def _variant_diagnostics(
    record: SolidStateCrossDatasetBenchmarkVariant,
    *,
    artifact: ContinuousTimeLidarPairArtifact | None,
    result_path: Path | None,
) -> SolidStateFailureAnalysisVariant:
    path_text = str(result_path) if result_path is not None else (record.result_path or "")
    if artifact is None:
        return SolidStateFailureAnalysisVariant(
            id=record.id,
            voxel_strategy=record.voxel_strategy,
            outlier_policy=record.outlier_policy,
            result_path=path_text,
            result_sha256=record.result_sha256,
            optimization_status=record.optimization_status,
            quality_grade=record.quality_grade,
        )

    (
        profile_iteration_count,
        accepted_iteration_count,
        accepted_iteration_rate,
        best_profile_holdout,
        profile_holdout_span,
        profile_offset_span,
    ) = _profile_diagnostics(artifact)
    total_input = (
        artifact.train_correspondence_count
        + artifact.holdout_correspondence_count
        + artifact.outlier_rejected_count
    )
    outlier_rate = artifact.outlier_rejected_count / total_input if total_input else None
    holdout_gap = None
    holdout_ratio = None
    if artifact.final_train_rmse_m is not None and artifact.final_holdout_rmse_m is not None:
        holdout_gap = artifact.final_holdout_rmse_m - artifact.final_train_rmse_m
        if artifact.final_train_rmse_m > 0.0:
            holdout_ratio = artifact.final_holdout_rmse_m / artifact.final_train_rmse_m
    train_change = None
    if artifact.initial_train_rmse_m is not None and artifact.final_train_rmse_m is not None:
        train_change = artifact.final_train_rmse_m - artifact.initial_train_rmse_m
    return SolidStateFailureAnalysisVariant(
        id=record.id,
        voxel_strategy=record.voxel_strategy,
        outlier_policy=record.outlier_policy,
        result_path=path_text,
        result_sha256=_digest(result_path) if result_path is not None else record.result_sha256,
        artifact_schema_version=artifact.schema_version,
        optimization_status=artifact.status,
        quality_grade=artifact.refined_transform.quality.grade,
        initial_time_offset_sec=artifact.initial_time_offset_sec,
        estimated_time_offset_sec=artifact.estimated_time_offset_sec,
        initial_train_rmse_m=artifact.initial_train_rmse_m,
        final_train_rmse_m=artifact.final_train_rmse_m,
        final_holdout_rmse_m=artifact.final_holdout_rmse_m,
        train_correspondence_count=artifact.train_correspondence_count,
        holdout_correspondence_count=artifact.holdout_correspondence_count,
        observability_rank=artifact.observability.rank,
        condition_number=artifact.observability.condition_number,
        weak_directions=list(artifact.observability.weak_directions),
        residual_count=artifact.observability.residual_count,
        outlier_rejected_count=artifact.outlier_rejected_count,
        outlier_rejection_rate=outlier_rate,
        train_holdout_gap_m=holdout_gap,
        holdout_train_ratio=holdout_ratio,
        train_rmse_change_m=train_change,
        profile_iteration_count=profile_iteration_count,
        accepted_iteration_count=accepted_iteration_count,
        accepted_iteration_rate=accepted_iteration_rate,
        best_profile_holdout_rmse_m=best_profile_holdout,
        profile_holdout_span_m=profile_holdout_span,
        profile_offset_span_sec=profile_offset_span,
        options=artifact.options,
        artifact_source_sha256=dict(artifact.provenance.source_sha256),
    )


def _load_variant(
    record: SolidStateCrossDatasetBenchmarkVariant,
    *,
    replicate: SolidStateCrossDatasetBenchmarkReplicate,
    benchmark_path: Path,
) -> tuple[SolidStateFailureAnalysisVariant, ContinuousTimeLidarPairArtifact | None, Path | None]:
    result_path = _variant_path(record, replicate=replicate, benchmark_path=benchmark_path)
    artifact: ContinuousTimeLidarPairArtifact | None = None
    if result_path is not None and result_path.exists():
        try:
            artifact = ContinuousTimeLidarPairArtifact.model_validate(read_mapping(result_path))
        except (OSError, ValueError):
            artifact = None
    return (
        _variant_diagnostics(record, artifact=artifact, result_path=result_path),
        artifact,
        result_path,
    )


def _pair_metric(
    adaptive: SolidStateFailureAnalysisVariant | None,
    uniform: SolidStateFailureAnalysisVariant | None,
    field: str,
) -> float | None:
    if adaptive is None or uniform is None:
        return None
    adaptive_value = getattr(adaptive, field)
    uniform_value = getattr(uniform, field)
    if adaptive_value is None or uniform_value is None:
        return None
    return float(adaptive_value - uniform_value)


def _diagnose_pair(
    *,
    status: str,
    winner: str,
    adaptive: SolidStateFailureAnalysisVariant | None,
    uniform: SolidStateFailureAnalysisVariant | None,
) -> tuple[str, list[str]]:
    if status != "scored" or adaptive is None or uniform is None:
        return "artifact_or_comparison_incomplete", ["comparison_incomplete"]
    signals: list[str] = []
    holdout_delta = _pair_metric(adaptive, uniform, "final_holdout_rmse_m")
    train_delta = _pair_metric(adaptive, uniform, "final_train_rmse_m")
    if holdout_delta is not None and holdout_delta > 1.0e-9:
        signals.append("adaptive_holdout_worse")
    elif holdout_delta is not None and holdout_delta < -1.0e-9:
        signals.append("adaptive_holdout_better")
    if train_delta is not None and train_delta < -1.0e-9:
        signals.append("adaptive_train_lower")
    elif train_delta is not None and train_delta > 1.0e-9:
        signals.append("adaptive_train_higher")
    if "adaptive_holdout_worse" in signals and "adaptive_train_lower" in signals:
        signals.append("train_holdout_mismatch")
    if adaptive.outlier_rejection_rate is not None and adaptive.outlier_rejection_rate >= 0.05:
        signals.append("adaptive_mad_rejection_at_least_5_percent")
    offset_delta = _pair_metric(adaptive, uniform, "estimated_time_offset_sec")
    if offset_delta is not None and abs(offset_delta) >= 0.02 - 1.0e-12:
        signals.append("offset_difference_at_least_20_ms")
    gap_delta = _pair_metric(adaptive, uniform, "train_holdout_gap_m")
    if gap_delta is not None and gap_delta > 0.02:
        signals.append("adaptive_generalization_gap_larger")
    if adaptive.observability_rank is not None and adaptive.observability_rank < 6:
        signals.append("adaptive_rank_below_six")
    if uniform.observability_rank is not None and uniform.observability_rank < 6:
        signals.append("uniform_rank_below_six")
    if adaptive.optimization_status != "converged":
        signals.append("adaptive_not_converged")
    if uniform.optimization_status != "converged":
        signals.append("uniform_not_converged")
    if winner == "adaptive":
        diagnosis = "adaptive_holdout_advantage"
    elif "train_holdout_mismatch" in signals:
        diagnosis = "uniform_wins_with_adaptive_train_advantage"
    elif winner == "uniform":
        diagnosis = "uniform_wins_without_adaptive_train_advantage"
    elif winner == "tie":
        diagnosis = "tie"
    else:
        diagnosis = "inconclusive"
    return diagnosis, signals


def _replicate_analysis(
    replicate: SolidStateCrossDatasetBenchmarkReplicate,
    *,
    benchmark_path: Path,
) -> tuple[
    SolidStateFailureAnalysisReplicate,
    list[ContinuousTimeLidarPairArtifact],
]:
    records = {item.id: item for item in replicate.variants}
    loaded: dict[str, SolidStateFailureAnalysisVariant | None] = {}
    artifacts: list[ContinuousTimeLidarPairArtifact] = []
    for variant_id in ("adaptive_mad", "uniform_none"):
        record = records.get(variant_id)
        if record is None:
            loaded[variant_id] = None
            continue
        variant, artifact, _result_path = _load_variant(
            record,
            replicate=replicate,
            benchmark_path=benchmark_path,
        )
        loaded[variant_id] = variant
        if artifact is not None:
            artifacts.append(artifact)
    adaptive = loaded["adaptive_mad"]
    uniform = loaded["uniform_none"]
    diagnosis, signals = _diagnose_pair(
        status=replicate.status,
        winner=replicate.winner,
        adaptive=adaptive,
        uniform=uniform,
    )
    return (
        SolidStateFailureAnalysisReplicate(
            id=replicate.id,
            split_id=replicate.split_id,
            seed=replicate.seed,
            status=replicate.status,
            winner=replicate.winner,
            holdout_improvement_percent=replicate.holdout_improvement_percent,
            adaptive=adaptive,
            uniform=uniform,
            adaptive_holdout_minus_uniform_m=_pair_metric(
                adaptive, uniform, "final_holdout_rmse_m"
            ),
            adaptive_train_minus_uniform_m=_pair_metric(adaptive, uniform, "final_train_rmse_m"),
            adaptive_generalization_gap_minus_uniform_m=_pair_metric(
                adaptive, uniform, "train_holdout_gap_m"
            ),
            adaptive_offset_minus_uniform_sec=_pair_metric(
                adaptive, uniform, "estimated_time_offset_sec"
            ),
            adaptive_correspondence_delta=(
                adaptive.train_correspondence_count - uniform.train_correspondence_count
                if adaptive is not None and uniform is not None
                else None
            ),
            adaptive_outlier_rate_minus_uniform=_pair_metric(
                adaptive, uniform, "outlier_rejection_rate"
            ),
            adaptive_condition_number_minus_uniform=_pair_metric(
                adaptive, uniform, "condition_number"
            ),
            diagnosis=diagnosis,
            signals=signals,
        ),
        artifacts,
    )


def _finding(
    *,
    finding_id: str,
    severity: str,
    confidence: str,
    title: str,
    evidence: str,
    replicates: list[str],
    caveat: str,
) -> SolidStateFailureAnalysisFinding:
    return SolidStateFailureAnalysisFinding(
        id=finding_id,
        severity=severity,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        title=title,
        evidence=evidence,
        supporting_replicates=replicates,
        caveat=caveat,
    )


def _findings(
    replicates: list[SolidStateFailureAnalysisReplicate],
    summary: SolidStateFailureAnalysisSummary,
) -> list[SolidStateFailureAnalysisFinding]:
    scored = [item for item in replicates if item.status == "scored"]
    findings: list[SolidStateFailureAnalysisFinding] = []
    mismatch = [
        item for item in scored if item.diagnosis == "uniform_wins_with_adaptive_train_advantage"
    ]
    if mismatch:
        findings.append(
            _finding(
                finding_id="adaptive_train_holdout_mismatch",
                severity="candidate_cause",
                confidence="medium" if len(mismatch) >= 2 else "low",
                title="Adaptive improves train fit but loses temporal holdout",
                evidence=(
                    f"{len(mismatch)}/{len(scored)} scored replicates have lower adaptive "
                    "train RMSE and higher adaptive holdout RMSE than uniform."
                ),
                replicates=[item.id for item in mismatch],
                caveat="This is a generalization signal, not proof of overfitting or leakage.",
            )
        )
    rejection_candidates = [
        item
        for item in mismatch
        if item.adaptive is not None
        and item.adaptive.outlier_rejection_rate is not None
        and item.adaptive.outlier_rejection_rate >= 0.05
    ]
    if rejection_candidates:
        mean_rate = mean(
            item.adaptive.outlier_rejection_rate
            for item in rejection_candidates
            if item.adaptive is not None and item.adaptive.outlier_rejection_rate is not None
        )
        findings.append(
            _finding(
                finding_id="adaptive_mad_rejection_candidate",
                severity="candidate_cause",
                confidence="low",
                title="MAD rejection is active in adaptive counterexamples",
                evidence=(
                    f"Adaptive rejected at least 5% of input correspondences in "
                    f"{len(rejection_candidates)} mismatch replicates; mean estimated "
                    f"rejection rate is {mean_rate:.1%}."
                ),
                replicates=[item.id for item in rejection_candidates],
                caveat=(
                    "The artifact stores aggregate counts, not residual histograms; "
                    "rejection is therefore a candidate mechanism only."
                ),
            )
        )
    split_winners: dict[str, list[SolidStateFailureAnalysisReplicate]] = defaultdict(list)
    for item in scored:
        split_winners[item.split_id].append(item)
    mixed_split = [
        (split_id, items)
        for split_id, items in split_winners.items()
        if len({item.winner for item in items}) > 1
    ]
    if mixed_split:
        replicates_for_finding = [item.id for _, items in mixed_split for item in items]
        findings.append(
            _finding(
                finding_id="sampling_seed_sensitivity",
                severity="candidate_cause",
                confidence="medium" if len(mixed_split) >= 1 else "low",
                title="Outcome changes with sampling seed within a temporal split",
                evidence=(
                    "At least one split contains both adaptive and uniform winners, "
                    "so the result is not invariant to the declared deterministic seed."
                ),
                replicates=replicates_for_finding,
                caveat=(
                    "Four AgRob replicates are insufficient to estimate a stable seed distribution."
                ),
            )
        )
    adaptive_offsets = [
        item.adaptive.estimated_time_offset_sec
        for item in scored
        if item.adaptive is not None and item.adaptive.estimated_time_offset_sec is not None
    ]
    uniform_offsets = [
        item.uniform.estimated_time_offset_sec
        for item in scored
        if item.uniform is not None and item.uniform.estimated_time_offset_sec is not None
    ]
    if adaptive_offsets and uniform_offsets:
        adaptive_span = max(adaptive_offsets) - min(adaptive_offsets)
        uniform_span = max(uniform_offsets) - min(uniform_offsets)
        if adaptive_span >= 0.02 and adaptive_span > uniform_span + 0.01:
            findings.append(
                _finding(
                    finding_id="adaptive_clock_profile_instability",
                    severity="warning",
                    confidence="low",
                    title="Adaptive clock estimates vary more across replicas",
                    evidence=(
                        f"Adaptive estimated-offset span is {adaptive_span:.3f}s versus "
                        f"{uniform_span:.3f}s for uniform."
                    ),
                    replicates=[item.id for item in scored],
                    caveat=(
                        "Offset variation may reflect capture-window motion, not "
                        "voxelization alone."
                    ),
                )
            )
    all_adaptive_rank_six = summary.adaptive_rank_six_count == summary.scored_replicate_count
    all_uniform_rank_six = summary.uniform_rank_six_count == summary.scored_replicate_count
    if all_adaptive_rank_six and all_uniform_rank_six and scored:
        findings.append(
            _finding(
                finding_id="no_observability_rank_failure",
                severity="info",
                confidence="high",
                title="Rank deficiency does not explain the AgRob counterexample",
                evidence=f"Both variants have rank 6 in all {len(scored)} scored replicates.",
                replicates=[item.id for item in scored],
                caveat=(
                    "Condition number and unrecorded residual structure still require "
                    "separate review."
                ),
            )
        )
    all_converged = (
        summary.adaptive_converged_count == summary.scored_replicate_count
        and summary.uniform_converged_count == summary.scored_replicate_count
        and bool(scored)
    )
    if all_converged:
        findings.append(
            _finding(
                finding_id="no_termination_failure",
                severity="info",
                confidence="high",
                title="The counterexample is not caused by solver termination",
                evidence=(
                    f"Both variants report converged status in all {len(scored)} scored replicates."
                ),
                replicates=[item.id for item in scored],
                caveat="Converged means the configured clock-profile stopping condition was met.",
            )
        )
    return findings


def _summary(
    replicates: list[SolidStateFailureAnalysisReplicate],
) -> SolidStateFailureAnalysisSummary:
    scored = [item for item in replicates if item.status == "scored"]
    adaptive_wins = sum(item.winner == "adaptive" for item in scored)
    uniform_wins = sum(item.winner == "uniform" for item in scored)
    ties = sum(item.winner == "tie" for item in scored)
    adaptive_variants = [item.adaptive for item in scored if item.adaptive is not None]
    uniform_variants = [item.uniform for item in scored if item.uniform is not None]
    mismatch_gaps = [
        item.adaptive_generalization_gap_minus_uniform_m
        for item in scored
        if item.adaptive_generalization_gap_minus_uniform_m is not None
    ]
    train_deltas = [
        item.adaptive_train_minus_uniform_m
        for item in scored
        if item.adaptive_train_minus_uniform_m is not None
    ]
    improvements = [
        item.holdout_improvement_percent
        for item in scored
        if item.holdout_improvement_percent is not None
    ]
    rejection_rates = [
        item.adaptive.outlier_rejection_rate
        for item in scored
        if item.adaptive is not None and item.adaptive.outlier_rejection_rate is not None
    ]
    diagnosis_counts = dict(Counter(item.diagnosis for item in replicates))
    return SolidStateFailureAnalysisSummary(
        replicate_count=len(replicates),
        scored_replicate_count=len(scored),
        adaptive_wins=adaptive_wins,
        uniform_wins=uniform_wins,
        ties=ties,
        adaptive_win_rate=adaptive_wins / len(scored) if scored else None,
        mean_holdout_improvement_percent=_mean_or_none(improvements),
        mean_adaptive_train_minus_uniform_m=_mean_or_none(train_deltas),
        mean_adaptive_generalization_gap_minus_uniform_m=_mean_or_none(mismatch_gaps),
        mean_adaptive_outlier_rejection_rate=_mean_or_none(rejection_rates),
        adaptive_rank_six_count=sum(item.observability_rank == 6 for item in adaptive_variants),
        uniform_rank_six_count=sum(item.observability_rank == 6 for item in uniform_variants),
        adaptive_converged_count=sum(
            item.optimization_status == "converged" for item in adaptive_variants
        ),
        uniform_converged_count=sum(
            item.optimization_status == "converged" for item in uniform_variants
        ),
        diagnosis_counts=diagnosis_counts,
    )


def _source_paths(
    benchmark: SolidStateCrossDatasetBenchmarkManifest,
    dataset: SolidStateCrossDatasetBenchmarkDataset,
    replicates: list[SolidStateFailureAnalysisReplicate],
    artifacts: list[ContinuousTimeLidarPairArtifact],
    *,
    benchmark_path: Path,
) -> tuple[list[str], dict[str, str]]:
    paths: set[Path] = {benchmark_path}
    paths.add(_resolve(dataset.config_path, base=benchmark_path.parent))
    paths.add(_resolve(dataset.dataset_manifest_path, base=benchmark_path.parent))
    paths.add(_resolve(dataset.ablation_manifest_path, base=benchmark_path.parent))
    for replicate in replicates:
        paths.add(
            _resolve(replicate.adaptive.result_path, base=benchmark_path.parent)
            if replicate.adaptive
            else benchmark_path
        )
        paths.add(
            _resolve(replicate.uniform.result_path, base=benchmark_path.parent)
            if replicate.uniform
            else benchmark_path
        )
    source_sha256: dict[str, str] = {}
    for path in sorted(paths):
        digest = _digest(path)
        if digest is not None and path.is_file():
            source_sha256[str(path)] = digest
    for artifact in artifacts:
        for source_path, digest in artifact.provenance.source_sha256.items():
            source_sha256.setdefault(source_path, digest)
    return sorted(source_sha256), dict(sorted(source_sha256.items()))


def _conclusion(
    dataset: SolidStateCrossDatasetBenchmarkDataset,
    summary: SolidStateFailureAnalysisSummary,
    findings: list[SolidStateFailureAnalysisFinding],
) -> str:
    candidate_titles = [item.title for item in findings if item.severity == "candidate_cause"]
    if summary.uniform_wins > summary.adaptive_wins:
        lead = (
            f"{dataset.id} is a counterexample for the current adaptive_mad policy: "
            f"uniform wins {summary.uniform_wins}/{summary.scored_replicate_count} scored "
            f"replicates and adaptive wins {summary.adaptive_wins}."
        )
    else:
        lead = (
            f"{dataset.id} does not show a majority uniform counterexample in the "
            f"{summary.scored_replicate_count} scored replicates."
        )
    if candidate_titles:
        lead += " Candidate signals: " + "; ".join(candidate_titles[:2]) + "."
    lead += " Rank and termination evidence should not be treated as causal proof."
    return lead


def run_failure_analysis(
    benchmark_manifest_path: Path,
    *,
    dataset_id: str,
    output_path: Path,
) -> SolidStateFailureAnalysisManifest:
    """Generate a digest-bound diagnostic artifact for one benchmark dataset."""

    benchmark_manifest_path = benchmark_manifest_path.resolve()
    output_path = output_path.resolve()
    benchmark = SolidStateCrossDatasetBenchmarkManifest.model_validate(
        read_mapping(benchmark_manifest_path)
    )
    dataset = next((item for item in benchmark.datasets if item.id == dataset_id), None)
    if dataset is None:
        available = ", ".join(item.id for item in benchmark.datasets)
        raise ValueError(f"dataset {dataset_id!r} not found; available: {available}")
    analysis_replicates: list[SolidStateFailureAnalysisReplicate] = []
    artifacts: list[ContinuousTimeLidarPairArtifact] = []
    for replicate in dataset.replicates:
        analysis_replicate, loaded_artifacts = _replicate_analysis(
            replicate,
            benchmark_path=benchmark_manifest_path,
        )
        analysis_replicates.append(analysis_replicate)
        artifacts.extend(loaded_artifacts)
    summary = _summary(analysis_replicates)
    findings = _findings(analysis_replicates, summary)
    paths, source_sha256 = _source_paths(
        benchmark,
        dataset,
        analysis_replicates,
        artifacts,
        benchmark_path=benchmark_manifest_path,
    )
    source_sha256[str(benchmark_manifest_path)] = _digest(
        benchmark_manifest_path
    ) or benchmark.provenance.source_sha256.get(str(benchmark_manifest_path), "")
    source_sha256 = {key: value for key, value in source_sha256.items() if value}
    paths = sorted(set(paths) | set(source_sha256))
    result = SolidStateFailureAnalysisManifest(
        tool="tools/run_solid_state_failure_analysis.py",
        tool_version=__version__,
        benchmark_manifest_path=str(benchmark_manifest_path),
        benchmark_manifest_sha256=_digest(benchmark_manifest_path) or "0" * 64,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        analysis_scope=(
            "paired adaptive_mad versus uniform_none diagnostics from declared "
            "split/seed replicates; aggregate correspondences are post-filter counts"
        ),
        summary=summary,
        replicates=analysis_replicates,
        findings=findings,
        conclusion=_conclusion(dataset, summary, findings),
        provenance=SolidStateFailureAnalysisProvenance(
            source_paths=paths,
            source_sha256=source_sha256,
            tool_version=__version__,
            git_commit=git_commit(),
            notes=[
                "result artifacts are revalidated before diagnostics are extracted",
                "outlier_rejection_rate denominator is retained correspondences plus "
                "rejected observations",
                "raw residual histograms and per-iteration correspondence counts are "
                "not present in v0.1 artifacts",
            ],
        ),
    )
    write_mapping(output_path, result.model_dump(mode="json", exclude_none=True))
    return result


def main() -> int:
    args = _parser().parse_args()
    result = run_failure_analysis(
        args.benchmark_manifest,
        dataset_id=args.dataset_id,
        output_path=args.output,
    )
    report_paths = write_solid_state_failure_analysis_reports(
        result,
        markdown_path=args.markdown_output,
        html_path=args.html_output,
    )
    if args.json:
        import json

        print(
            json.dumps(result.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True)
        )
    else:
        print(result.conclusion)
        print(f"analysis: {args.output}")
        for report_path in report_paths:
            print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
