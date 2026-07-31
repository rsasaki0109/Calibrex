"""Paired uncertainty ablations for probabilistic Camera--LiDAR refinement."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from calibrex.core.benchmark import (
    BenchmarkArtifact,
    BenchmarkDefinition,
    BenchmarkMethodDefinition,
    BenchmarkMetricDefinition,
    BenchmarkProtocol,
    BenchmarkProvenance,
    BenchmarkSplit,
    BenchmarkTrial,
    BenchmarkTrialProvenance,
    aggregate_benchmark_definition,
)
from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
from calibrex.core.probabilistic_correspondence import (
    load_probabilistic_correspondence,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.evaluation.holdout import split_indices
from calibrex.evaluation.probabilistic_camera_lidar_run import (
    run_probabilistic_camera_lidar_refinement,
)
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefinementOptions,
)

PROBABILISTIC_ABLATION_VERSION = (
    "calibrex.probabilistic_refinement_ablation/v0.1"
)


def run_probabilistic_refinement_ablation(
    correspondence_path: str | Path,
    initialization_problem_path: str | Path,
    *,
    initialization_trace_path: str | Path | None = None,
    result_directory: str | Path,
    command: str,
    split_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    base_options: ProbabilisticCameraLidarRefinementOptions | None = None,
    bootstrap_samples: int = 2000,
) -> tuple[BenchmarkDefinition, BenchmarkArtifact]:
    """Run paired full/uncertainty-ablated refinements over fixed splits."""

    if not split_seeds or len(set(split_seeds)) != len(split_seeds):
        raise ValueError("split_seeds must be non-empty and unique")
    correspondence_file = Path(correspondence_path)
    problem_file = Path(initialization_problem_path)
    correspondence = load_probabilistic_correspondence(correspondence_file)
    problem = load_camera_lidar_problem(problem_file)
    correspondence_digest = _required_digest(correspondence_file)
    problem_digest = _required_digest(problem_file)
    trace_file = (
        Path(initialization_trace_path)
        if initialization_trace_path is not None
        else None
    )
    trace_digest = _required_digest(trace_file) if trace_file is not None else None
    combined_input_digest = _text_digest(
        f"{correspondence_digest}\n{problem_digest}\n{trace_digest or ''}"
    )
    problem_ids = {item.frame_id for item in problem.observations}
    ordered_ids = sorted(item.frame_id for item in correspondence.frames)
    if not set(ordered_ids).issubset(problem_ids):
        raise ValueError("correspondence frames must exist in initialization problem")
    settings = base_options or ProbabilisticCameraLidarRefinementOptions()
    variants = {
        "full_uncertainty": settings,
        "without_covariance": replace(settings, use_covariance=False),
        "without_outlier_probability": replace(
            settings, use_outlier_probability=False
        ),
        "without_reliability": replace(settings, use_reliability=False),
    }
    output_root = Path(result_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    splits = [
        _split(seed, ordered_ids, settings.holdout_ratio) for seed in split_seeds
    ]
    trials: list[BenchmarkTrial] = []
    for seed in split_seeds:
        for method_id, variant in variants.items():
            options = replace(variant, split_seed=seed)
            config_digest = _options_digest(options)
            output = output_root / f"{method_id}.seed-{seed:04d}.yaml"
            started = time.perf_counter()
            result = run_probabilistic_camera_lidar_refinement(
                correspondence_file,
                problem_file,
                initialization_trace_path=trace_file,
                result_id=f"{method_id}-seed-{seed:04d}",
                options=options,
                command=command.split(),
            )
            runtime = time.perf_counter() - started
            result.save(output)
            output_digest = _required_digest(output)
            holdout_rmse = (
                result.final_holdout_evaluation.weighted_reprojection_rmse_px
            )
            succeeded = (
                result.status == "converged"
                and holdout_rmse is not None
                and result.final_rotation_error_deg is not None
                and result.final_translation_error_m is not None
            )
            metrics: dict[str, float] = {}
            if succeeded:
                assert holdout_rmse is not None
                assert result.final_rotation_error_deg is not None
                assert result.final_translation_error_m is not None
                metrics["holdout_reprojection_rmse_px"] = holdout_rmse
                metrics["rotation_error_deg"] = (
                    result.final_rotation_error_deg
                )
                metrics["translation_error_m"] = (
                    result.final_translation_error_m
                )
            trials.append(
                BenchmarkTrial(
                    method_id=method_id,
                    split_id=f"seed-{seed:04d}",
                    status="success" if succeeded else "failed",
                    metrics=metrics,
                    runtime_seconds=runtime,
                    failure_reason=(
                        None
                        if succeeded
                        else (
                            result.reason
                            if result.status != "converged"
                            else "converged result lacks complete evaluation metrics"
                        )
                    ),
                    provenance=BenchmarkTrialProvenance(
                        command=command,
                        config_sha256=config_digest,
                        input_sha256=combined_input_digest,
                        output_sha256=output_digest,
                        notes=[
                            f"solver_status={result.status}",
                            "paired frame split shared by every ablation",
                        ],
                    ),
                )
            )
    provenance = BenchmarkProvenance(
        generator=__name__,
        generator_version=PROBABILISTIC_ABLATION_VERSION,
        git_commit=git_commit(),
        command=command,
        source_artifacts={
            str(correspondence_file): correspondence_digest,
            str(problem_file): problem_digest,
            **(
                {str(trace_file): trace_digest}
                if trace_file is not None and trace_digest is not None
                else {}
            ),
        },
        data_verified=(
            correspondence.dataset_license_spdx is not None
            and bool(correspondence.provenance.input_sha256)
        ),
    )
    definition = BenchmarkDefinition(
        benchmark_id=f"{correspondence.artifact_id}-uncertainty-ablation",
        title="Probabilistic Camera-LiDAR uncertainty ablation",
        protocol=BenchmarkProtocol(
            protocol_id="probabilistic-refinement-paired-frame-holdout/v0.1",
            dataset_id=correspondence.dataset_id,
            dataset_source_sha256=combined_input_digest,
            data_license=(
                correspondence.dataset_license_spdx or "not declared"
            ),
            split_policy="seeded disjoint frame holdout",
            splits=splits,
            initial_estimate_policy=(
                (
                    "same digest-pinned D2D candidate trace output for every "
                    "method/split"
                )
                if trace_file is not None
                else (
                    "same digest-pinned problem initial pose for every "
                    "method/split; not claimed as a solved D2D initialization"
                )
            ),
            tuning_policy=(
                "all thresholds and split seeds fixed before evaluation"
            ),
            failure_policy=(
                "only converged results are successful; failures retain no metrics"
            ),
        ),
        metrics=[
            BenchmarkMetricDefinition(
                name="holdout_reprojection_rmse_px",
                label="Holdout reprojection RMSE",
                unit="px",
                direction="lower",
                primary=True,
                interpretation=(
                    "confidence-weighted pixel RMSE on disjoint frames"
                ),
            ),
            BenchmarkMetricDefinition(
                name="rotation_error_deg",
                label="Reference rotation error",
                unit="deg",
                direction="lower",
                primary=True,
                interpretation=(
                    "quaternion geodesic error to the digest-pinned reference"
                ),
            ),
            BenchmarkMetricDefinition(
                name="translation_error_m",
                label="Reference translation error",
                unit="m",
                direction="lower",
                primary=True,
                interpretation=(
                    "Euclidean translation error to the digest-pinned reference"
                ),
            ),
        ],
        methods=[
            BenchmarkMethodDefinition(
                method_id=method_id,
                label=method_id.replace("_", " "),
                implementation="calibrex_native",
                tool_name="Calibrex probabilistic multi-frame refiner",
                tool_version="0.1",
                source_commit=git_commit(),
                license_spdx="Apache-2.0",
            )
            for method_id in variants
        ],
        trials=trials,
        reference_method_id="full_uncertainty",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=0,
        limitations=[
            (
                "This ablation isolates estimator inputs; it does not replace "
                "the frozen U1-U5 accuracy/runtime protocols."
            )
        ],
        provenance=provenance,
    )
    return definition, aggregate_benchmark_definition(definition)


def _split(
    seed: int,
    ordered_ids: list[str],
    holdout_ratio: float,
) -> BenchmarkSplit:
    fit_indices, holdout_indices = split_indices(
        len(ordered_ids), holdout_ratio, seed
    )
    fit_ids = [ordered_ids[index] for index in fit_indices]
    holdout_ids = [ordered_ids[index] for index in holdout_indices]
    return BenchmarkSplit(
        split_id=f"seed-{seed:04d}",
        seed=seed,
        fit_count=len(fit_ids),
        holdout_count=len(holdout_ids),
        fit_ids_sha256=_text_digest("\n".join(fit_ids)),
        holdout_ids_sha256=_text_digest("\n".join(holdout_ids)),
    )


def _options_digest(
    options: ProbabilisticCameraLidarRefinementOptions,
) -> str:
    return _text_digest(
        json.dumps(asdict(options), sort_keys=True, separators=(",", ":"))
    )


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required benchmark file is not readable: {path}")
    return digest


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
