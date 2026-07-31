"""Paired temporal-model ablations for continuous-time Camera--LiDAR."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

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
from calibrex.core.continuous_time_camera_lidar_artifacts import (
    load_continuous_time_camera_lidar_problem,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.evaluation.continuous_time_camera_lidar_run import (
    run_continuous_time_camera_lidar_problem,
)
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.continuous_time_camera_lidar_solver import (
    ContinuousTimeCameraLidarOptions,
)

CONTINUOUS_TIME_ABLATION_VERSION = (
    "calibrex.continuous_time_camera_lidar_ablation/v0.1"
)


def run_continuous_time_camera_lidar_ablation(
    problem_path: str | Path,
    *,
    result_directory: str | Path,
    command: str,
    split_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    bootstrap_samples: int = 2000,
) -> tuple[BenchmarkDefinition, BenchmarkArtifact]:
    """Run full/fixed-clock/per-point/covariance temporal ablations."""

    if not split_seeds or len(set(split_seeds)) != len(split_seeds):
        raise ValueError("split_seeds must be non-empty and unique")
    path = Path(problem_path)
    problem = load_continuous_time_camera_lidar_problem(path)
    problem_digest = _required_digest(path)
    base = ContinuousTimeCameraLidarOptions(
        **cast(dict[str, Any], problem.options)
    )
    variants = {
        "full_continuous_time": base,
        "fixed_clock_offset": replace(base, estimate_time_offset=False),
        "without_per_point_time": replace(base, use_per_point_time=False),
        "without_rolling_shutter": replace(base, use_rolling_shutter=False),
        "without_covariance": replace(base, use_covariance=False),
    }
    ordered_ids = sorted(item.capture_id for item in problem.captures)
    splits = [
        _split(seed, ordered_ids, base.holdout_ratio) for seed in split_seeds
    ]
    output_root = Path(result_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    trials: list[BenchmarkTrial] = []
    for seed in split_seeds:
        for method_id, variant in variants.items():
            options = replace(variant, split_seed=seed)
            output = output_root / f"{method_id}.seed-{seed:04d}.yaml"
            started = time.perf_counter()
            result = run_continuous_time_camera_lidar_problem(
                path,
                result_id=f"{method_id}-seed-{seed:04d}",
                command=command.split(),
                options=options,
            )
            runtime = time.perf_counter() - started
            result.save(output)
            succeeded = result.status == "converged"
            metrics: dict[str, float] = {}
            holdout_rmse = (
                result.final_holdout_evaluation.weighted_reprojection_rmse_px
            )
            if succeeded and holdout_rmse is not None:
                metrics["holdout_reprojection_rmse_px"] = holdout_rmse
                if result.final_rotation_error_deg is not None:
                    metrics["rotation_error_deg"] = (
                        result.final_rotation_error_deg
                    )
                if result.final_translation_error_m is not None:
                    metrics["translation_error_m"] = (
                        result.final_translation_error_m
                    )
                if result.final_time_offset_error_sec is not None:
                    metrics["time_offset_abs_error_sec"] = (
                        result.final_time_offset_error_sec
                    )
            trials.append(
                BenchmarkTrial(
                    method_id=method_id,
                    split_id=f"seed-{seed:04d}",
                    status="success" if succeeded else "failed",
                    metrics=metrics,
                    runtime_seconds=runtime,
                    failure_reason=None if succeeded else result.reason,
                    provenance=BenchmarkTrialProvenance(
                        command=command,
                        config_sha256=_options_digest(options),
                        input_sha256=problem_digest,
                        output_sha256=_required_digest(output),
                        notes=[
                            f"solver_status={result.status}",
                            "paired capture split shared by every ablation",
                        ],
                    ),
                )
            )
    provenance = BenchmarkProvenance(
        generator=__name__,
        generator_version=CONTINUOUS_TIME_ABLATION_VERSION,
        git_commit=git_commit(),
        command=command,
        source_artifacts={str(path): problem_digest},
        data_verified=problem.dataset_license_spdx is not None,
    )
    definition = BenchmarkDefinition(
        benchmark_id=f"{problem.problem_id}-temporal-ablation",
        title="Continuous-time Camera-LiDAR temporal-model ablation",
        protocol=BenchmarkProtocol(
            protocol_id="continuous-time-paired-capture-holdout/v0.1",
            dataset_id=problem.dataset_id,
            dataset_source_sha256=problem_digest,
            data_license=problem.dataset_license_spdx or "not declared",
            split_policy="seeded disjoint capture holdout",
            splits=splits,
            initial_estimate_policy=(
                "same problem extrinsic and clock state for all methods"
            ),
            tuning_policy="all variants and seeds fixed before evaluation",
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
                    "probability-weighted reprojection on disjoint captures"
                ),
            ),
            *(
                [
                    BenchmarkMetricDefinition(
                        name="rotation_error_deg",
                        label="Reference rotation error",
                        unit="deg",
                        direction="lower",
                        primary=True,
                        interpretation=(
                            "quaternion geodesic error to the frozen reference"
                        ),
                    ),
                    BenchmarkMetricDefinition(
                        name="translation_error_m",
                        label="Reference translation error",
                        unit="m",
                        direction="lower",
                        primary=True,
                        interpretation=(
                            "translation distance to the frozen reference"
                        ),
                    ),
                    BenchmarkMetricDefinition(
                        name="time_offset_abs_error_sec",
                        label="Reference clock-offset absolute error",
                        unit="s",
                        direction="lower",
                        primary=True,
                        interpretation=(
                            "absolute scalar clock error to the frozen reference"
                        ),
                    ),
                ]
                if problem.reference_transform_camera_lidar is not None
                else []
            ),
        ],
        methods=[
            BenchmarkMethodDefinition(
                method_id=method_id,
                label=method_id.replace("_", " "),
                implementation="calibrex_native",
                tool_name="Calibrex continuous-time Camera-LiDAR solver",
                tool_version="0.1",
                source_commit=git_commit(),
                license_spdx="Apache-2.0",
            )
            for method_id in variants
        ],
        trials=trials,
        reference_method_id="full_continuous_time",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=0,
        limitations=[
            (
                "This isolates temporal factors but does not establish the "
                "asynchronous real-data improvement gate by itself."
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


def _options_digest(options: ContinuousTimeCameraLidarOptions) -> str:
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
