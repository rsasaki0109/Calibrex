"""Reproducible Borer rotation-only D2D benchmark execution."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

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
from calibrex.core.camera_lidar_artifacts import (
    CalibrationCandidateEvaluation,
    CalibrationCandidateOutcome,
    CalibrationCandidateTrace,
    CameraLidarArtifactProvenance,
    CameraLidarBenchmarkProtocol,
    CameraLidarCalibrationProblem,
    CameraLidarHitDefinition,
    CameraLidarIsolationDeclaration,
    CameraLidarPerturbation,
    load_calibration_candidate_trace,
    load_camera_lidar_benchmark_protocol,
    load_camera_lidar_problem,
)
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import (
    EstimateEvidenceLevel,
    EstimateProducer,
    TransformEstimateProvenance,
    TransformResult,
)
from calibrex.data.depth import (
    DepthFileReference,
    DepthProviderArtifact,
    load_depth_provider,
    verify_depth_provider_files,
)
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
    DepthToDepthOptions,
)
from calibrex.solvers.borer_rotation_only_solver import (
    BorerRotationOnlyOptions,
    BorerRotationOnlyResult,
    BorerRotationOnlySolver,
    apply_local_euler_delta,
    fibonacci_sphere_rotation_perturbations,
)

FloatArray: TypeAlias = NDArray[np.float64]
BORER_ROTATION_BENCHMARK_VERSION = "calibrex.borer_rotation_benchmark/v0.1"


@dataclass(frozen=True)
class LoadedCameraLidarProblem:
    """Verified arrays and transforms loaded from a problem artifact."""

    problem: CameraLidarCalibrationProblem
    depth_provider: DepthProviderArtifact
    observations: tuple[DepthToDepthObservation, ...]
    reference_transform_camera_lidar: SE3
    problem_sha256: str
    depth_provider_sha256: str


@dataclass(frozen=True)
class BorerRotationTrialExecution:
    """One in-memory solver execution retained in frozen protocol order."""

    perturbation: CameraLidarPerturbation
    initial_transform_camera_lidar: SE3
    result: BorerRotationOnlyResult
    runtime_seconds: float


def build_borer_rotation_protocol(
    problem_path: str | Path,
    *,
    perturbation_count: int = 200,
    rotation_magnitude_deg: float = 10.0,
    hit_rotation_error_deg: float = 0.5,
    hit_translation_error_m: float = 0.20,
    histogram_bins: int = 32,
    min_visible_points: int = 64,
    bound_deg: float = 20.0,
    initial_step_deg: float = 4.0,
    minimum_step_deg: float = 0.05,
    max_evaluations: int = 400,
    command: tuple[str, ...] = (),
) -> CameraLidarBenchmarkProtocol:
    """Build the explicit 200-point Fibonacci-sphere paper protocol."""

    path = Path(problem_path)
    problem = load_camera_lidar_problem(path)
    problem_digest = _required_digest(path)
    perturbations = [
        CameraLidarPerturbation(
            trial_id=f"rotation-{rotation_magnitude_deg:g}deg-{index:03d}",
            rotation_deg_xyz=list(rotation),
            translation_m_xyz=[0.0, 0.0, 0.0],
        )
        for index, rotation in enumerate(
            fibonacci_sphere_rotation_perturbations(
                count=perturbation_count,
                magnitude_deg=rotation_magnitude_deg,
            )
        )
    ]
    return CameraLidarBenchmarkProtocol(
        protocol_id=(
            f"borer-rotation-{problem.dataset_id}-"
            f"{rotation_magnitude_deg:g}deg-{perturbation_count}"
        ),
        primary_source="https://arxiv.org/abs/2311.01905",
        dataset_id=problem.dataset_id,
        problem_sha256=problem_digest,
        degrees_of_freedom="rotation_only",
        frame_sampling="uniform_over_sequence",
        frame_ids=[item.frame_id for item in problem.observations],
        perturbation_method="fibonacci_sphere",
        perturbation_count=perturbation_count,
        rotation_magnitude_deg=rotation_magnitude_deg,
        translation_magnitude_m=0.0,
        perturbations=perturbations,
        hit=CameraLidarHitDefinition(
            rotation_error_max_deg=hit_rotation_error_deg,
            translation_error_max_m=hit_translation_error_m,
        ),
        objective="per_frame_mean_mutual_information",
        histogram_bins=histogram_bins,
        min_visible_points=min_visible_points,
        visibility="z_buffer_nearest_range",
        optimizer="bounded_so3_pattern_search/v0.1",
        optimizer_options={
            "bound_deg": bound_deg,
            "initial_step_deg": initial_step_deg,
            "minimum_step_deg": minimum_step_deg,
            "max_evaluations": max_evaluations,
            "improvement_tolerance": 1.0e-9,
        },
        isolation=CameraLidarIsolationDeclaration(
            provider_selection_split="external provider training declaration",
            threshold_selection_split="paper-prespecified thresholds",
            evaluation_split=problem.sequence_id,
            test_data_used_for_selection=False,
            evidence=(
                "frame IDs, perturbations, optimizer settings, and hit thresholds "
                "are frozen before trial execution"
            ),
        ),
        metric_definitions={
            "rotation_error_deg": (
                "quaternion geodesic error to dataset reference, degrees"
            ),
            "translation_error_m": (
                "Euclidean translation error to dataset reference, metres"
            ),
            "hit": (
                "1 iff rotation and translation errors are strictly below "
                "the prespecified paper thresholds"
            ),
        },
        provenance=CameraLidarArtifactProvenance(
            generator="calibrex.evaluation.borer_rotation_benchmark",
            generator_version=BORER_ROTATION_BENCHMARK_VERSION,
            git_commit=git_commit(),
            command=list(command),
            source_sha256=problem_digest,
        ),
    )


def build_borer_six_dof_protocol(
    problem_path: str | Path,
    *,
    perturbation_count: int = 200,
    rotation_magnitude_deg: float = 0.5,
    translation_magnitude_m: float = 0.50,
    hit_rotation_error_deg: float = 0.5,
    hit_translation_error_m: float = 0.20,
    histogram_bins: int = 32,
    min_visible_points: int = 64,
    rotation_bound_deg: float = 2.0,
    translation_bound_m: float = 1.0,
    initial_rotation_step_deg: float = 0.25,
    initial_translation_step_m: float = 0.10,
    minimum_rotation_step_deg: float = 0.01,
    minimum_translation_step_m: float = 0.005,
    max_evaluations: int = 800,
    command: tuple[str, ...] = (),
) -> CameraLidarBenchmarkProtocol:
    """Freeze the Borer six-DoF paired Fibonacci-sphere protocol."""

    path = Path(problem_path)
    problem = load_camera_lidar_problem(path)
    problem_digest = _required_digest(path)
    if rotation_bound_deg > problem.rotation_bound_deg:
        raise ValueError(
            "six-DoF rotation bound exceeds the problem rotation bound"
        )
    if translation_bound_m > problem.translation_bound_m:
        raise ValueError(
            "six-DoF translation bound exceeds the problem translation bound"
        )
    rotations = fibonacci_sphere_rotation_perturbations(
        count=perturbation_count,
        magnitude_deg=rotation_magnitude_deg,
    )
    translations = _fibonacci_sphere_vectors(
        count=perturbation_count,
        magnitude=translation_magnitude_m,
    )
    perturbations = [
        CameraLidarPerturbation(
            trial_id=(
                f"se3-{rotation_magnitude_deg:g}deg-"
                f"{translation_magnitude_m:g}m-{index:03d}"
            ),
            rotation_deg_xyz=list(rotation),
            translation_m_xyz=list(translation),
        )
        for index, (rotation, translation) in enumerate(
            zip(rotations, translations, strict=True)
        )
    ]
    return CameraLidarBenchmarkProtocol(
        protocol_id=(
            f"borer-six-dof-{problem.dataset_id}-"
            f"{rotation_magnitude_deg:g}deg-"
            f"{translation_magnitude_m:g}m-{perturbation_count}"
        ),
        primary_source="https://arxiv.org/abs/2311.01905",
        dataset_id=problem.dataset_id,
        problem_sha256=problem_digest,
        degrees_of_freedom="six_dof",
        frame_sampling="uniform_over_sequence",
        frame_ids=[item.frame_id for item in problem.observations],
        perturbation_method="fibonacci_sphere",
        perturbation_count=perturbation_count,
        rotation_magnitude_deg=rotation_magnitude_deg,
        translation_magnitude_m=translation_magnitude_m,
        perturbations=perturbations,
        hit=CameraLidarHitDefinition(
            rotation_error_max_deg=hit_rotation_error_deg,
            translation_error_max_m=hit_translation_error_m,
        ),
        objective="per_frame_mean_mutual_information",
        histogram_bins=histogram_bins,
        min_visible_points=min_visible_points,
        visibility="z_buffer_nearest_range",
        optimizer="bounded_se3_pattern_search/v0.1",
        optimizer_options={
            "rotation_bound_deg": rotation_bound_deg,
            "translation_bound_m": translation_bound_m,
            "initial_rotation_step_deg": initial_rotation_step_deg,
            "initial_translation_step_m": initial_translation_step_m,
            "minimum_rotation_step_deg": minimum_rotation_step_deg,
            "minimum_translation_step_m": minimum_translation_step_m,
            "max_evaluations": max_evaluations,
            "improvement_tolerance": 1.0e-9,
        },
        isolation=CameraLidarIsolationDeclaration(
            provider_selection_split="external provider training declaration",
            threshold_selection_split="paper-prespecified thresholds",
            evaluation_split=problem.sequence_id,
            test_data_used_for_selection=False,
            evidence=(
                "frame IDs, paired Fibonacci perturbations, optimizer settings, "
                "and hit thresholds are frozen before trial execution"
            ),
        ),
        metric_definitions={
            "rotation_error_deg": (
                "quaternion geodesic error to dataset reference, degrees"
            ),
            "translation_error_m": (
                "Euclidean translation error to dataset reference, metres"
            ),
            "hit": (
                "1 iff rotation and translation errors are strictly below "
                "the prespecified paper thresholds"
            ),
        },
        provenance=CameraLidarArtifactProvenance(
            generator="calibrex.evaluation.borer_rotation_benchmark",
            generator_version=BORER_ROTATION_BENCHMARK_VERSION,
            git_commit=git_commit(),
            command=list(command),
            source_sha256=problem_digest,
        ),
    )


def _fibonacci_sphere_vectors(
    *,
    count: int,
    magnitude: float,
) -> tuple[tuple[float, float, float], ...]:
    unit_scaled = fibonacci_sphere_rotation_perturbations(
        count=count,
        magnitude_deg=magnitude,
    )
    return tuple(unit_scaled)


def load_borer_problem(
    problem_path: str | Path,
) -> LoadedCameraLidarProblem:
    """Verify all digests and load depth/LiDAR arrays for native D2D."""

    path = Path(problem_path)
    problem = load_camera_lidar_problem(path)
    problem_digest = _required_digest(path)
    depth_provider_path = _resolve(path.parent, problem.depth_provider_path)
    depth_digest = _required_digest(depth_provider_path)
    if depth_digest != problem.depth_provider_sha256:
        raise ValueError(
            "depth provider digest mismatch: "
            f"expected {problem.depth_provider_sha256}, observed {depth_digest}"
        )
    provider = load_depth_provider(depth_provider_path)
    provider_issues = verify_depth_provider_files(
        provider,
        base_path=depth_provider_path.parent,
    )
    if provider_issues:
        raise ValueError("depth provider verification failed: " + "; ".join(provider_issues))
    depth_by_frame = {item.frame_id: item for item in provider.observations}
    observations = []
    for binding in problem.observations:
        try:
            depth_observation = depth_by_frame[binding.depth_observation_frame_id]
        except KeyError as exc:
            raise ValueError(
                "problem references unknown depth observation: "
                f"{binding.depth_observation_frame_id}"
            ) from exc
        lidar_path = _resolve(path.parent, binding.lidar.path)
        _verify_reference(lidar_path, binding.lidar)
        depth_path = _resolve(depth_provider_path.parent, depth_observation.depth.path)
        depth: FloatArray = np.array(
            _load_depth(depth_path, depth_observation.depth.encoding),
            dtype=float,
            copy=True,
        )
        if depth_observation.scale_to_m is not None:
            depth = depth * depth_observation.scale_to_m
        for invalid in depth_observation.invalid_values:
            depth[depth == invalid] = np.nan
        intrinsics = depth_observation.intrinsics
        camera = DepthToDepthCameraModel(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            projection=intrinsics.projection,
            xi=intrinsics.xi or 0.0,
            alpha=intrinsics.alpha or 0.5,
            distortion=tuple(intrinsics.distortion),  # type: ignore[arg-type]
        )
        observations.append(
            DepthToDepthObservation(
                frame_id=binding.frame_id,
                depth_map=depth,
                lidar_points=_load_lidar(lidar_path, binding.lidar.encoding),
                camera=camera,
            )
        )
    return LoadedCameraLidarProblem(
        problem=problem,
        depth_provider=provider,
        observations=tuple(observations),
        reference_transform_camera_lidar=(
            problem.reference_transform_camera_lidar.as_se3()
        ),
        problem_sha256=problem_digest,
        depth_provider_sha256=depth_digest,
    )


def run_borer_rotation_benchmark(
    problem_path: str | Path,
    protocol_path: str | Path,
    *,
    trace_directory: str | Path,
    command: str,
    bootstrap_samples: int = 2000,
    workers: int = 1,
    resume: bool = False,
) -> tuple[BenchmarkDefinition, BenchmarkArtifact]:
    """Execute every frozen perturbation and retain misses/failures."""

    if workers < 1:
        raise ValueError("workers must be at least one")
    problem_file = Path(problem_path)
    protocol_file = Path(protocol_path)
    loaded = load_borer_problem(problem_file)
    protocol = load_camera_lidar_benchmark_protocol(protocol_file)
    protocol_digest = _required_digest(protocol_file)
    if protocol.problem_sha256 != loaded.problem_sha256:
        raise ValueError("protocol problem_sha256 does not match the problem artifact")
    if protocol.frame_ids != [item.frame_id for item in loaded.problem.observations]:
        raise ValueError("protocol frame IDs do not match the problem artifact")
    trace_root = Path(trace_directory)
    trace_root.mkdir(parents=True, exist_ok=True)
    splits = []
    baseline_trials = []
    native_trials = []
    trace_digests: dict[str, str] = {}
    fit_digest = _text_digest("\n".join(protocol.frame_ids))
    reference_digest = _text_digest(
        str(
            loaded.problem.reference_transform_camera_lidar.model_dump(
                mode="json"
            )
        )
    )
    options = _solver_options(protocol)
    cached_traces: dict[str, CalibrationCandidateTrace] = {}
    pending_perturbations = []
    if resume:
        for perturbation in protocol.perturbations:
            trace_path = trace_root / f"{perturbation.trial_id}.trace.yaml"
            if trace_path.is_file():
                cached_traces[perturbation.trial_id] = _validated_resumed_trace(
                    trace_path,
                    perturbation=perturbation,
                    problem_sha256=loaded.problem_sha256,
                    protocol_sha256=protocol_digest,
                )
            else:
                pending_perturbations.append(perturbation)
    else:
        pending_perturbations = list(protocol.perturbations)
    executions = iter(
        _execute_rotation_trials(
            pending_perturbations,
            loaded=loaded,
            options=options,
            workers=workers,
        )
    )
    for perturbation in protocol.perturbations:
        initial = apply_local_euler_delta(
            loaded.reference_transform_camera_lidar,
            perturbation.rotation_deg_xyz,
        )
        split = BenchmarkSplit(
            split_id=perturbation.trial_id,
            seed=0,
            fit_count=len(protocol.frame_ids),
            holdout_count=1,
            fit_ids_sha256=fit_digest,
            holdout_ids_sha256=reference_digest,
        )
        splits.append(split)
        initial_rotation_error, initial_translation_error = _transform_errors(
            initial,
            loaded.reference_transform_camera_lidar,
        )
        initial_hit = _hit(
            initial_rotation_error,
            initial_translation_error,
            protocol.hit,
        )
        baseline_trials.append(
            BenchmarkTrial(
                method_id="unoptimized_initial",
                split_id=perturbation.trial_id,
                status="success",
                metrics={
                    "rotation_error_deg": initial_rotation_error,
                    "translation_error_m": initial_translation_error,
                    "hit": float(initial_hit),
                },
                runtime_seconds=0.0,
                provenance=BenchmarkTrialProvenance(
                    command=command,
                    config_sha256=protocol_digest,
                    input_sha256=loaded.problem_sha256,
                    notes=["prespecified Fibonacci-sphere initial perturbation"],
                ),
            )
        )
        trace_path = trace_root / f"{perturbation.trial_id}.trace.yaml"
        trace = cached_traces.get(perturbation.trial_id)
        if trace is None:
            execution = next(executions)
            if execution.perturbation.trial_id != perturbation.trial_id:
                raise RuntimeError("rotation trial execution order changed")
            trace = _candidate_trace(
                result=execution.result,
                perturbation=perturbation,
                loaded=loaded,
                protocol=protocol,
                protocol_digest=protocol_digest,
                runtime_seconds=execution.runtime_seconds,
                command=command,
            )
            trace.save(trace_path)
        runtime = trace.runtime_seconds
        trace_digest = _required_digest(trace_path)
        trace_digests[perturbation.trial_id] = trace_digest
        if trace.status == "insufficient_observations":
            native_trials.append(
                BenchmarkTrial(
                    method_id="native_borer_d2d_rotation",
                    split_id=perturbation.trial_id,
                    status="failed",
                    runtime_seconds=runtime,
                    failure_reason=trace.stopping_reason,
                    provenance=BenchmarkTrialProvenance(
                        command=command,
                        config_sha256=protocol_digest,
                        input_sha256=loaded.problem_sha256,
                        output_sha256=trace_digest,
                    ),
                )
            )
        else:
            native_trials.append(
                BenchmarkTrial(
                    method_id="native_borer_d2d_rotation",
                    split_id=perturbation.trial_id,
                    status="success",
                    metrics={
                        "rotation_error_deg": trace.outcome.rotation_error_deg,
                        "translation_error_m": trace.outcome.translation_error_m,
                        "hit": float(trace.outcome.hit),
                    },
                    runtime_seconds=runtime,
                    provenance=BenchmarkTrialProvenance(
                        command=command,
                        config_sha256=protocol_digest,
                        input_sha256=loaded.problem_sha256,
                        output_sha256=trace_digest,
                    ),
                )
            )
    benchmark_protocol = BenchmarkProtocol(
        protocol_id=protocol.protocol_id,
        dataset_id=protocol.dataset_id,
        dataset_source_sha256=loaded.problem_sha256,
        data_license="dataset-specific; see problem/depth-provider provenance",
        split_policy=(
            "one frozen Fibonacci-sphere perturbation per split; all selected "
            "frames optimize pose; dataset reference evaluates error"
        ),
        splits=splits,
        initial_estimate_policy=(
            f"{protocol.perturbation_method} at "
            f"{protocol.rotation_magnitude_deg:g} degrees"
        ),
        tuning_policy=protocol.isolation.evidence,
        failure_policy="retain every miss and computational failure in denominator",
    )
    definition = BenchmarkDefinition(
        benchmark_id=protocol.protocol_id,
        title=(
            f"Borer rotation-only D2D recovery on {loaded.problem.dataset_id}"
        ),
        protocol=benchmark_protocol,
        metrics=_benchmark_metrics(),
        methods=_benchmark_methods(),
        trials=[*baseline_trials, *native_trials],
        reference_method_id="unoptimized_initial",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=0,
        limitations=[
            "The external monocular-depth provider is evaluated as a frozen input.",
            "Dataset-reference extrinsics are not independent metrology.",
            "This artifact supports only the declared rotation-only protocol.",
        ],
        provenance=BenchmarkProvenance(
            generator="calibrex.evaluation.borer_rotation_benchmark",
            generator_version=BORER_ROTATION_BENCHMARK_VERSION,
            git_commit=git_commit(),
            command=command,
            source_artifacts={
                "problem": loaded.problem_sha256,
                "depth_provider": loaded.depth_provider_sha256,
                "protocol": protocol_digest,
                **{
                    f"trace:{trial_id}": digest
                    for trial_id, digest in sorted(trace_digests.items())
                },
            },
            data_verified=True,
        ),
    )
    return definition, aggregate_benchmark_definition(definition)


def _execute_rotation_trial(
    perturbation: CameraLidarPerturbation,
    *,
    loaded: LoadedCameraLidarProblem,
    options: BorerRotationOnlyOptions,
) -> BorerRotationTrialExecution:
    initial = apply_local_euler_delta(
        loaded.reference_transform_camera_lidar,
        perturbation.rotation_deg_xyz,
    )
    started = time.perf_counter()
    result = BorerRotationOnlySolver().solve(
        loaded.observations,
        initial,
        options,
    )
    return BorerRotationTrialExecution(
        perturbation=perturbation,
        initial_transform_camera_lidar=initial,
        result=result,
        runtime_seconds=time.perf_counter() - started,
    )


def _validated_resumed_trace(
    path: Path,
    *,
    perturbation: CameraLidarPerturbation,
    problem_sha256: str,
    protocol_sha256: str,
) -> CalibrationCandidateTrace:
    trace = load_calibration_candidate_trace(path)
    if trace.trial_id != perturbation.trial_id:
        raise ValueError(f"resumed trace trial_id mismatch: {path}")
    if trace.problem_sha256 != problem_sha256:
        raise ValueError(f"resumed trace problem_sha256 mismatch: {path}")
    if trace.protocol_sha256 != protocol_sha256:
        raise ValueError(f"resumed trace protocol_sha256 mismatch: {path}")
    if trace.solver_version != BORER_ROTATION_BENCHMARK_VERSION:
        raise ValueError(f"resumed trace solver_version mismatch: {path}")
    return trace


def _execute_rotation_trials(
    perturbations: list[CameraLidarPerturbation],
    *,
    loaded: LoadedCameraLidarProblem,
    options: BorerRotationOnlyOptions,
    workers: int,
) -> Iterator[BorerRotationTrialExecution]:
    execute = partial(
        _execute_rotation_trial,
        loaded=loaded,
        options=options,
    )
    if workers == 1:
        yield from map(execute, perturbations)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(execute, perturbations)


def _candidate_trace(
    *,
    result: BorerRotationOnlyResult,
    perturbation: CameraLidarPerturbation,
    loaded: LoadedCameraLidarProblem,
    protocol: CameraLidarBenchmarkProtocol,
    protocol_digest: str,
    runtime_seconds: float,
    command: str,
) -> CalibrationCandidateTrace:
    rotation_error, translation_error = _transform_errors(
        result.transform_camera_lidar,
        loaded.reference_transform_camera_lidar,
    )
    hit = _hit(rotation_error, translation_error, protocol.hit)
    return CalibrationCandidateTrace(
        trace_id=f"{protocol.protocol_id}:{perturbation.trial_id}",
        trial_id=perturbation.trial_id,
        problem_sha256=loaded.problem_sha256,
        protocol_sha256=protocol_digest,
        solver=result.method,
        solver_version=BORER_ROTATION_BENCHMARK_VERSION,
        status=result.status,
        initial_transform_camera_lidar=_transform_result(
            result.initial_transform_camera_lidar,
            producer="dataset_provider",
            evidence="dataset_provided",
            note="dataset reference plus frozen perturbation",
        ),
        output_transform_camera_lidar=_transform_result(
            result.transform_camera_lidar,
            producer="slac_native",
            evidence="algorithmically_refined",
            note="native D2D rotation-only output; translation fixed",
        ),
        evaluations=[
            CalibrationCandidateEvaluation(
                evaluation=item.evaluation,
                parameters={
                    "roll_delta_deg": item.delta_rotation_deg_xyz[0],
                    "pitch_delta_deg": item.delta_rotation_deg_xyz[1],
                    "yaw_delta_deg": item.delta_rotation_deg_xyz[2],
                },
                objective=item.objective,
                mutual_information=item.mutual_information,
                normalized_mutual_information=item.normalized_mutual_information,
                evaluated_frame_count=item.evaluated_frame_count,
                accepted=item.accepted,
            )
            for item in result.trace
        ],
        stopping_reason=result.stopping_reason,
        runtime_seconds=runtime_seconds,
        outcome=CalibrationCandidateOutcome(
            rotation_error_deg=rotation_error,
            translation_error_m=translation_error,
            hit=hit,
            hit_reason=(
                f"rotation {rotation_error:g} < "
                f"{protocol.hit.rotation_error_max_deg:g} deg and translation "
                f"{translation_error:g} < "
                f"{protocol.hit.translation_error_max_m:g} m"
            ),
        ),
        warnings=(
            []
            if result.status != "insufficient_observations"
            else [result.stopping_reason]
        ),
        provenance=CameraLidarArtifactProvenance(
            generator="calibrex.evaluation.borer_rotation_benchmark",
            generator_version=BORER_ROTATION_BENCHMARK_VERSION,
            git_commit=git_commit(),
            command=command.split(),
            config_sha256=protocol_digest,
            source_sha256=loaded.problem_sha256,
        ),
    )


def _solver_options(
    protocol: CameraLidarBenchmarkProtocol,
) -> BorerRotationOnlyOptions:
    values = protocol.optimizer_options
    return BorerRotationOnlyOptions(
        bound_deg=float(values["bound_deg"]),
        initial_step_deg=float(values["initial_step_deg"]),
        minimum_step_deg=float(values["minimum_step_deg"]),
        max_evaluations=int(values["max_evaluations"]),
        improvement_tolerance=float(values["improvement_tolerance"]),
        d2d=DepthToDepthOptions(
            histogram_bins=protocol.histogram_bins,
            min_visible_points=protocol.min_visible_points,
            use_z_buffer=True,
        ),
    )


def _benchmark_metrics() -> list[BenchmarkMetricDefinition]:
    return [
        BenchmarkMetricDefinition(
            name="rotation_error_deg",
            label="Rotation error",
            unit="deg",
            direction="lower",
            primary=True,
            interpretation="geodesic rotation error to dataset reference",
        ),
        BenchmarkMetricDefinition(
            name="translation_error_m",
            label="Translation error",
            unit="m",
            direction="lower",
            interpretation="fixed-translation error to dataset reference",
        ),
        BenchmarkMetricDefinition(
            name="hit",
            label="Hit rate",
            unit="fraction",
            direction="higher",
            primary=True,
            interpretation="paper-prespecified strict success indicator",
        ),
    ]


def _benchmark_methods() -> list[BenchmarkMethodDefinition]:
    return [
        BenchmarkMethodDefinition(
            method_id="unoptimized_initial",
            label="Initial perturbation",
            implementation="calibrex_native",
            tool_name="fibonacci_sphere",
            tool_version="v0.1",
            paper_doi="10.48550/arXiv.2311.01905",
            license_spdx="Apache-2.0",
        ),
        BenchmarkMethodDefinition(
            method_id="native_borer_d2d_rotation",
            label="Calibrex native D2D rotation",
            implementation="calibrex_native",
            tool_name="calibrex",
            tool_version=BORER_ROTATION_BENCHMARK_VERSION,
            paper_doi="10.48550/arXiv.2311.01905",
            license_spdx="Apache-2.0",
        ),
    ]


def _transform_result(
    transform: SE3,
    *,
    producer: EstimateProducer,
    evidence: EstimateEvidenceLevel,
    note: str,
) -> TransformResult:
    return TransformResult(
        parent="camera0",
        child="lidar0",
        translation_m=list(transform.translation_m),
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        provenance=TransformEstimateProvenance(
            producer=producer,
            execution_mode="offline_batch",
            role_in_comparison="output",
            evidence_level=evidence,
            source="borer_rotation_benchmark",
            tool_name="calibrex",
            notes=[note],
        ),
    )


def _transform_errors(candidate: SE3, reference: SE3) -> tuple[float, float]:
    dot = abs(
        sum(
            left * right
            for left, right in zip(
                candidate.rotation_quat_xyzw,
                reference.rotation_quat_xyzw,
                strict=True,
            )
        )
    )
    rotation = math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))
    translation = math.sqrt(
        sum(
            (left - right) ** 2
            for left, right in zip(
                candidate.translation_m,
                reference.translation_m,
                strict=True,
            )
        )
    )
    return rotation, translation


def _hit(
    rotation_error_deg: float,
    translation_error_m: float,
    definition: CameraLidarHitDefinition,
) -> bool:
    return (
        rotation_error_deg < definition.rotation_error_max_deg
        and translation_error_m < definition.translation_error_max_m
    )


def _load_depth(path: Path, encoding: str) -> FloatArray:
    if encoding not in {"npy_float32", "npy_float64"}:
        raise ValueError(f"unsupported native D2D depth encoding: {encoding}")
    data = np.load(path, allow_pickle=False)
    depth = np.asarray(data, dtype=float)
    if depth.ndim != 2:
        raise ValueError(f"depth array must be HxW: {path}")
    return depth


def _load_lidar(path: Path, encoding: str) -> FloatArray:
    if encoding == "kitti_velodyne_f32x4":
        raw: NDArray[np.float32] = np.fromfile(path, dtype="<f4")
        if raw.size % 4:
            raise ValueError(f"KITTI LiDAR file is not float32 Nx4: {path}")
        return np.asarray(raw.reshape(-1, 4)[:, :3], dtype=float)
    if encoding in {"npy_xyz_float32", "npy_xyz_float64"}:
        raw = np.load(path, allow_pickle=False)
        points = np.asarray(raw, dtype=float)
        if points.ndim != 2 or points.shape[1] not in {3, 4}:
            raise ValueError(f"LiDAR NPY must have shape Nx3 or Nx4: {path}")
        return points[:, :3]
    raise ValueError(f"unsupported native D2D LiDAR encoding: {encoding}")


def _verify_reference(path: Path, reference: DepthFileReference) -> None:
    observed = _required_digest(path)
    if observed != reference.sha256:
        raise ValueError(
            f"LiDAR digest mismatch for {reference.path}: "
            f"expected {reference.sha256}, observed {observed}"
        )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"cannot digest required artifact: {path}")
    return digest


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
