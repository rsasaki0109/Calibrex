#!/usr/bin/env python3
"""Aggregate uniform-vs-adaptive solid-state LiDAR evidence across public datasets."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from statistics import mean, median
from typing import Any

from calibrex import __version__
from calibrex.core.continuous_time_lidar_ablation import (
    ContinuousTimeLidarAblationManifest,
)
from calibrex.core.continuous_time_lidar_artifacts import (
    ContinuousTimeLidarPairArtifact,
)
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import sha256_path
from calibrex.core.solid_state_cross_dataset_benchmark import (
    BenchmarkDatasetStatus,
    BenchmarkFailureCategory,
    BenchmarkWinner,
    SolidStateCrossDatasetBenchmarkAggregate,
    SolidStateCrossDatasetBenchmarkConfidenceInterval,
    SolidStateCrossDatasetBenchmarkDataset,
    SolidStateCrossDatasetBenchmarkManifest,
    SolidStateCrossDatasetBenchmarkProvenance,
    SolidStateCrossDatasetBenchmarkReplicate,
    SolidStateCrossDatasetBenchmarkSpec,
    SolidStateCrossDatasetBenchmarkSpecReplicate,
    SolidStateCrossDatasetBenchmarkVariant,
)
from calibrex.core.solid_state_cross_dataset_report import (
    write_solid_state_cross_dataset_benchmark_reports,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="cross-dataset benchmark declaration YAML")
    parser.add_argument("--output", type=Path, required=True, help="generated result YAML")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="also write a reproducibility-oriented Markdown report",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        help="also write a self-contained HTML report",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="return non-zero if any dataset is not scored",
    )
    return parser


def _resolve(path_text: str, *, base: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _digest(path: Path) -> str | None:
    return sha256_path(path)


def _artifact_for_variant(
    variant_record: Any,
    *,
    ablation_path: Path,
) -> tuple[ContinuousTimeLidarPairArtifact | None, Path | None, str | None]:
    if not variant_record.result_path:
        return None, None, None
    result_path = _resolve(variant_record.result_path, base=ablation_path.parent)
    if not result_path.exists():
        return None, result_path, None
    try:
        artifact = ContinuousTimeLidarPairArtifact.model_validate(read_mapping(result_path))
    except (OSError, ValueError):
        return None, result_path, _digest(result_path)
    return artifact, result_path, _digest(result_path)


def _failure_category(
    *,
    artifact: ContinuousTimeLidarPairArtifact | None,
    result_path: Path | None,
    variant_status: str,
    observability_rank: int | None,
    holdout_correspondences: int,
    minimum_holdout_correspondences: int,
) -> BenchmarkFailureCategory:
    if artifact is None:
        if variant_status == "not_run":
            return "not_run"
        if result_path is None or not result_path.exists():
            return "missing_result"
        return "invalid_artifact"
    status_category: dict[str, BenchmarkFailureCategory] = {
        "max_iterations": "max_iterations",
        "insufficient_constraints": "insufficient_constraints",
        "rejected": "rejected",
    }
    if artifact.status in status_category:
        return status_category[artifact.status]
    if observability_rank is None or observability_rank < 6:
        return "rank_deficient"
    if holdout_correspondences < minimum_holdout_correspondences:
        return "insufficient_holdout"
    return "none"


def _variant_summary(
    variant_record: Any,
    *,
    ablation_path: Path,
    minimum_holdout_correspondences: int,
) -> tuple[SolidStateCrossDatasetBenchmarkVariant, list[Path]]:
    artifact, result_path, result_digest = _artifact_for_variant(
        variant_record,
        ablation_path=ablation_path,
    )
    source_paths: list[Path] = []
    if variant_record.config_path:
        config_path = _resolve(variant_record.config_path, base=ablation_path.parent)
        if config_path.exists():
            source_paths.append(config_path)
    if result_path is not None and result_path.exists():
        source_paths.append(result_path)
    if artifact is None:
        holdout_count = variant_record.final_holdout_rmse_m is not None
        rank = variant_record.observability_rank
        failure_category = _failure_category(
            artifact=None,
            result_path=result_path,
            variant_status=variant_record.status,
            observability_rank=rank,
            holdout_correspondences=(
                minimum_holdout_correspondences if holdout_count else 0
            ),
            minimum_holdout_correspondences=minimum_holdout_correspondences,
        )
        summary = SolidStateCrossDatasetBenchmarkVariant(
            id=variant_record.id,
            voxel_strategy=variant_record.voxel_strategy,
            outlier_policy=variant_record.outlier_policy,
            result_path=str(result_path) if result_path is not None else None,
            result_sha256=result_digest,
            optimization_status=variant_record.status,
            quality_grade=variant_record.quality_grade,
            estimated_time_offset_sec=variant_record.estimated_time_offset_sec,
            final_train_rmse_m=variant_record.final_train_rmse_m,
            final_holdout_rmse_m=variant_record.final_holdout_rmse_m,
            observability_rank=rank,
            outlier_rejected_count=variant_record.outlier_rejected_count or 0,
            failure_category=failure_category,
        )
        return summary, source_paths
    failure_category = _failure_category(
        artifact=artifact,
        result_path=result_path,
        variant_status=artifact.status,
        observability_rank=artifact.observability.rank,
        holdout_correspondences=artifact.holdout_correspondence_count,
        minimum_holdout_correspondences=minimum_holdout_correspondences,
    )
    summary = SolidStateCrossDatasetBenchmarkVariant(
        id=variant_record.id,
        voxel_strategy=variant_record.voxel_strategy,
        outlier_policy=variant_record.outlier_policy,
        result_path=str(result_path),
        result_sha256=result_digest,
        optimization_status=artifact.status,
        quality_grade=artifact.refined_transform.quality.grade,
        estimated_time_offset_sec=artifact.estimated_time_offset_sec,
        final_train_rmse_m=artifact.final_train_rmse_m,
        final_holdout_rmse_m=artifact.final_holdout_rmse_m,
        train_correspondence_count=artifact.train_correspondence_count,
        holdout_correspondence_count=artifact.holdout_correspondence_count,
        observability_rank=artifact.observability.rank,
        outlier_rejected_count=artifact.outlier_rejected_count,
        reason=artifact.reason,
        failure_category=failure_category,
    )
    return summary, source_paths


def _placeholder_variant(
    *,
    category: BenchmarkFailureCategory,
) -> SolidStateCrossDatasetBenchmarkVariant:
    return SolidStateCrossDatasetBenchmarkVariant(
        id="not_available",
        voxel_strategy="adaptive",
        outlier_policy="mad",
        optimization_status="not_run",
        failure_category=category,
    )


def _replicate_result(
    dataset_spec: Any,
    replicate_spec: SolidStateCrossDatasetBenchmarkSpecReplicate,
    *,
    spec_base: Path,
    minimum_holdout_correspondences: int,
    require_converged: bool,
) -> tuple[SolidStateCrossDatasetBenchmarkReplicate, list[Path]]:
    ablation_path = _resolve(replicate_spec.ablation_manifest_path, base=spec_base)
    source_paths = [ablation_path] if ablation_path.exists() else []
    variants: list[SolidStateCrossDatasetBenchmarkVariant] = []
    ablation: ContinuousTimeLidarAblationManifest | None = None
    if ablation_path.exists():
        try:
            ablation = ContinuousTimeLidarAblationManifest.model_validate(
                read_mapping(ablation_path)
            )
        except (OSError, ValueError):
            ablation = None
    if ablation is not None:
        for record in ablation.variants:
            summary, variant_sources = _variant_summary(
                record,
                ablation_path=ablation_path,
                minimum_holdout_correspondences=minimum_holdout_correspondences,
            )
            variants.append(summary)
            source_paths.extend(variant_sources)

    if not ablation_path.exists():
        status: BenchmarkDatasetStatus = "unavailable"
        failure_categories: set[BenchmarkFailureCategory] = {"missing_result"}
        comparison_note = "ablation manifest is missing; no comparison was scored"
    elif ablation is None:
        status = "failed"
        failure_categories = {"invalid_artifact"}
        variants = [_placeholder_variant(category="invalid_artifact")]
        comparison_note = "ablation manifest could not be loaded"
    elif not variants:
        status = "failed"
        failure_categories = {"not_run"}
        variants = [_placeholder_variant(category="not_run")]
        comparison_note = "ablation manifest contained no variants"
    else:
        adaptive = next((item for item in variants if item.id == "adaptive_mad"), None)
        uniform = next((item for item in variants if item.id == "uniform_none"), None)
        has_holdout = all(
            item is not None
            and item.final_holdout_rmse_m is not None
            and item.holdout_correspondence_count >= minimum_holdout_correspondences
            and item.observability_rank is not None
            and item.observability_rank >= 6
            and (not require_converged or item.optimization_status == "converged")
            for item in (adaptive, uniform)
        )
        improvement: float | None = None
        winner: BenchmarkWinner = "inconclusive"
        if has_holdout and adaptive is not None and uniform is not None:
            assert adaptive.final_holdout_rmse_m is not None
            assert uniform.final_holdout_rmse_m is not None
            if uniform.final_holdout_rmse_m > 0.0:
                improvement = (
                    (uniform.final_holdout_rmse_m - adaptive.final_holdout_rmse_m)
                    / uniform.final_holdout_rmse_m
                    * 100.0
                )
            if adaptive.final_holdout_rmse_m < uniform.final_holdout_rmse_m - 1.0e-12:
                winner = "adaptive"
            elif uniform.final_holdout_rmse_m < adaptive.final_holdout_rmse_m - 1.0e-12:
                winner = "uniform"
            else:
                winner = "tie"
        failure_categories = {
            item.failure_category for item in variants if item.failure_category != "none"
        }
        if not has_holdout and not failure_categories:
            failure_categories.add("insufficient_holdout")
        status = "scored" if has_holdout else "inconclusive"
        comparison_note = (
            "lower final temporal-holdout point-to-plane RMSE wins; "
            "optimization status and failure category are retained per variant"
            if has_holdout
            else (
                "requires both adaptive_mad and uniform_none with rank >= 6, at least "
                f"{minimum_holdout_correspondences} holdout correspondences"
                + (", and converged status" if require_converged else "")
            )
        )
        return (
            SolidStateCrossDatasetBenchmarkReplicate(
                id=replicate_spec.id,
                split_id=replicate_spec.split_id,
                seed=replicate_spec.seed,
                ablation_manifest_path=str(ablation_path),
                ablation_manifest_sha256=_digest(ablation_path),
                status=status,
                variants=variants,
                holdout_improvement_percent=improvement,
                winner=winner,
                failure_categories=sorted(failure_categories),
                comparison_note=comparison_note,
                notes=list(replicate_spec.notes),
            ),
            source_paths,
        )

    return (
        SolidStateCrossDatasetBenchmarkReplicate(
            id=replicate_spec.id,
            split_id=replicate_spec.split_id,
            seed=replicate_spec.seed,
            ablation_manifest_path=str(ablation_path),
            ablation_manifest_sha256=_digest(ablation_path),
            status=status,
            variants=variants,
            failure_categories=sorted(failure_categories),
            comparison_note=comparison_note,
            notes=list(replicate_spec.notes),
        ),
        source_paths,
    )


def _bootstrap_interval(
    values: list[float],
    *,
    confidence_level: float,
    sample_count: int,
    seed: int,
) -> SolidStateCrossDatasetBenchmarkConfidenceInterval | None:
    if not values:
        return None
    if len(values) == 1:
        lower = upper = values[0]
    else:
        rng = random.Random(seed)
        estimates = [
            mean(values[rng.randrange(len(values))] for _ in values)
            for _ in range(sample_count)
        ]
        estimates.sort()
        lower_index = math.floor((1.0 - confidence_level) * 0.5 * (len(estimates) - 1))
        upper_index = math.ceil(
            (1.0 - (1.0 - confidence_level) * 0.5) * (len(estimates) - 1)
        )
        lower = estimates[max(0, lower_index)]
        upper = estimates[min(len(estimates) - 1, upper_index)]
    return SolidStateCrossDatasetBenchmarkConfidenceInterval(
        confidence_level=confidence_level,
        lower=lower,
        upper=upper,
        statistic="mean_holdout_improvement_percent",
    )


def _dataset_result(
    dataset_spec: Any,
    *,
    spec_base: Path,
    minimum_holdout_correspondences: int,
    minimum_scored_replicates: int,
    require_converged: bool,
    confidence_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[SolidStateCrossDatasetBenchmarkDataset, list[Path]]:
    dataset_manifest_path = _resolve(dataset_spec.dataset_manifest_path, base=spec_base)
    config_path = _resolve(dataset_spec.config_path, base=spec_base)
    declared_replicates = list(dataset_spec.replicates)
    if not declared_replicates:
        declared_replicates = [
            SolidStateCrossDatasetBenchmarkSpecReplicate(
                id="default",
                split_id="default",
                seed=0,
                ablation_manifest_path=dataset_spec.ablation_manifest_path,
            )
        ]
    replicates: list[SolidStateCrossDatasetBenchmarkReplicate] = []
    source_paths = [
        path for path in (dataset_manifest_path, config_path) if path.exists()
    ]
    for replicate_spec in declared_replicates:
        replicate, replicate_sources = _replicate_result(
            dataset_spec,
            replicate_spec,
            spec_base=spec_base,
            minimum_holdout_correspondences=minimum_holdout_correspondences,
            require_converged=require_converged,
        )
        replicates.append(replicate)
        source_paths.extend(replicate_sources)

    scored = [item for item in replicates if item.status == "scored"]
    improvements = [
        item.holdout_improvement_percent
        for item in scored
        if item.holdout_improvement_percent is not None
    ]
    adaptive_wins = sum(item.winner == "adaptive" for item in scored)
    uniform_wins = sum(item.winner == "uniform" for item in scored)
    if len(scored) >= minimum_scored_replicates:
        status: BenchmarkDatasetStatus = "scored"
    elif all(item.status == "unavailable" for item in replicates):
        status = "unavailable"
    elif any(item.status == "failed" for item in replicates):
        status = "failed"
    else:
        status = "inconclusive"
    if adaptive_wins > uniform_wins:
        winner: BenchmarkWinner = "adaptive"
    elif uniform_wins > adaptive_wins:
        winner = "uniform"
    elif scored and adaptive_wins == uniform_wins == len(scored):
        winner = "tie"
    else:
        winner = "inconclusive"
    variants = replicates[0].variants if replicates else [_placeholder_variant(category="not_run")]
    adaptive = next((item for item in variants if item.id == "adaptive_mad"), None)
    uniform = next((item for item in variants if item.id == "uniform_none"), None)
    failure_categories = sorted(
        {category for item in replicates for category in item.failure_categories}
    )
    mean_improvement = mean(improvements) if improvements else None
    ci = _bootstrap_interval(
        improvements,
        confidence_level=confidence_level,
        sample_count=bootstrap_samples,
        seed=bootstrap_seed,
    )
    first_replicate = replicates[0] if replicates else None
    ablation_path = (
        Path(first_replicate.ablation_manifest_path)
        if first_replicate is not None
        else _resolve(dataset_spec.ablation_manifest_path, base=spec_base)
    )
    comparison_note = (
        f"{len(scored)}/{len(replicates)} replicates scored; lower final "
        "temporal-holdout point-to-plane RMSE wins"
    )
    if require_converged:
        comparison_note += "; converged status required"
    dataset_result = SolidStateCrossDatasetBenchmarkDataset(
        id=dataset_spec.id,
        name=dataset_spec.name,
        family=dataset_spec.family,
        dataset_manifest_path=str(dataset_manifest_path),
        dataset_manifest_sha256=_digest(dataset_manifest_path),
        config_path=str(config_path),
        config_sha256=_digest(config_path),
        ablation_manifest_path=str(ablation_path),
        ablation_manifest_sha256=_digest(ablation_path),
        reference_mode=dataset_spec.reference_mode,
        absolute_extrinsic_ground_truth=dataset_spec.absolute_extrinsic_ground_truth,
        identity_control=dataset_spec.identity_control,
        independent_temporal_holdout=dataset_spec.independent_temporal_holdout,
        status=status,
        variants=variants,
        adaptive_variant_id=adaptive.id if adaptive is not None else None,
        uniform_variant_id=uniform.id if uniform is not None else None,
        holdout_improvement_percent=mean_improvement,
        winner=winner,
        comparison_note=comparison_note,
        replicates=replicates,
        replicate_count=len(replicates),
        scored_replicate_count=len(scored),
        adaptive_win_rate=(adaptive_wins / len(scored) if scored else None),
        holdout_improvement_mean_percent=mean_improvement,
        holdout_improvement_median_percent=median(improvements) if improvements else None,
        holdout_improvement_ci=ci,
        failure_categories=failure_categories,
        notes=list(dataset_spec.notes),
    )
    return dataset_result, source_paths


def run_benchmark(spec_path: Path, output_path: Path) -> SolidStateCrossDatasetBenchmarkManifest:
    """Aggregate all declared ablation manifests into one evidence artifact."""

    spec_path = spec_path.resolve()
    output_path = output_path.resolve()
    spec = SolidStateCrossDatasetBenchmarkSpec.model_validate(read_mapping(spec_path))
    spec_digest = _digest(spec_path)
    if spec_digest is None:
        raise RuntimeError(f"could not hash benchmark spec: {spec_path}")

    datasets: list[SolidStateCrossDatasetBenchmarkDataset] = []
    source_paths: list[Path] = [spec_path]
    for dataset_spec in spec.datasets:
        dataset_result, dataset_sources = _dataset_result(
            dataset_spec,
            spec_base=Path.cwd(),
            minimum_holdout_correspondences=spec.protocol.minimum_holdout_correspondences,
            minimum_scored_replicates=spec.protocol.minimum_scored_replicates,
            require_converged=spec.protocol.require_converged,
            confidence_level=spec.protocol.confidence_level,
            bootstrap_samples=spec.protocol.bootstrap_samples,
            bootstrap_seed=spec.protocol.bootstrap_seed + len(datasets),
        )
        datasets.append(dataset_result)
        source_paths.extend(dataset_sources)

    scored = [item for item in datasets if item.status == "scored"]
    improvements = [
        replicate.holdout_improvement_percent
        for dataset in datasets
        for replicate in dataset.replicates
        if replicate.status == "scored"
        and replicate.holdout_improvement_percent is not None
    ]
    scored_replicates = [
        replicate
        for dataset in datasets
        for replicate in dataset.replicates
        if replicate.status == "scored"
    ]
    adaptive_wins = sum(item.winner == "adaptive" for item in scored_replicates)
    uniform_wins = sum(item.winner == "uniform" for item in scored_replicates)
    ties = sum(item.winner == "tie" for item in scored_replicates)
    total_replicates = sum(item.replicate_count for item in datasets)
    failure_categories = sorted(
        {category for item in datasets for category in item.failure_categories}
    )
    aggregate_ci = _bootstrap_interval(
        improvements,
        confidence_level=spec.protocol.confidence_level,
        sample_count=spec.protocol.bootstrap_samples,
        seed=spec.protocol.bootstrap_seed + 100_000,
    )
    conclusion = (
        f"adaptive wins {adaptive_wins}/{len(scored_replicates)} scored replicates "
        f"across {len(scored)}/{len(datasets)} scored datasets; "
        "this is ground-truth-free temporal-holdout evidence unless a dataset "
        "declares an absolute extrinsic reference, and is not a universal SOTA claim"
    )
    source_sha256: dict[str, str] = {}
    normalized_source_paths: list[str] = []
    for path in source_paths:
        digest = _digest(path)
        if digest is None:
            continue
        path_text = str(path)
        normalized_source_paths.append(path_text)
        source_sha256[path_text] = digest

    manifest = SolidStateCrossDatasetBenchmarkManifest(
        tool="tools/run_solid_state_cross_dataset_benchmark.py",
        tool_version=__version__,
        spec_path=str(spec_path),
        spec_sha256=spec_digest,
        protocol=spec.protocol,
        datasets=datasets,
        aggregate=SolidStateCrossDatasetBenchmarkAggregate(
            dataset_count=len(datasets),
            scored_dataset_count=len(scored),
            inconclusive_dataset_count=sum(item.status == "inconclusive" for item in datasets),
            adaptive_wins=adaptive_wins,
            uniform_wins=uniform_wins,
            ties=ties,
            mean_holdout_improvement_percent=mean(improvements) if improvements else None,
            median_holdout_improvement_percent=median(improvements) if improvements else None,
            total_replicate_count=total_replicates,
            scored_replicate_count=len(scored_replicates),
            adaptive_win_rate=(
                adaptive_wins / len(scored_replicates) if scored_replicates else None
            ),
            holdout_improvement_ci=aggregate_ci,
            failure_categories=failure_categories,
            conclusion=conclusion,
        ),
        provenance=SolidStateCrossDatasetBenchmarkProvenance(
            source_paths=normalized_source_paths,
            source_sha256=source_sha256,
            tool_version=__version__,
            notes=[
                "raw dataset digests are inherited from each continuous-time result artifact",
                "dataset-local replay caps remain explicit in each referenced config and result",
            ],
        ),
    )
    write_mapping(output_path, manifest.model_dump(mode="json", exclude_none=True))
    return manifest


def main() -> int:
    args = _parser().parse_args()
    manifest = run_benchmark(args.spec, args.output)
    report_paths = write_solid_state_cross_dataset_benchmark_reports(
        manifest,
        markdown_path=args.markdown_output,
        html_path=args.html_output,
    )
    if args.json:
        import json

        print(
            json.dumps(
                manifest.model_dump(mode="json", exclude_none=True),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for dataset in manifest.datasets:
            adaptive = next(
                (item for item in dataset.variants if item.id == dataset.adaptive_variant_id),
                None,
            )
            uniform = next(
                (item for item in dataset.variants if item.id == dataset.uniform_variant_id),
                None,
            )
            adaptive_rmse = adaptive.final_holdout_rmse_m if adaptive else None
            uniform_rmse = uniform.final_holdout_rmse_m if uniform else None
            print(
                f"{dataset.id}: status={dataset.status} winner={dataset.winner} "
                f"adaptive_holdout={adaptive_rmse} uniform_holdout={uniform_rmse} "
                f"improvement_percent={dataset.holdout_improvement_percent} "
                f"replicates={dataset.scored_replicate_count}/{dataset.replicate_count} "
                f"ci={dataset.holdout_improvement_ci}"
            )
        print(f"manifest: {args.output}")
        for report_path in report_paths:
            print(f"report: {report_path}")
    return 1 if args.enforce and manifest.aggregate.inconclusive_dataset_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
