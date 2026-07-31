"""Reproducible Borer six-DoF D2D benchmark execution."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
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
from calibrex.core.camera_lidar_artifacts import (
    CalibrationCandidateEvaluation,
    CalibrationCandidateOutcome,
    CalibrationCandidateTrace,
    CameraLidarArtifactProvenance,
    CameraLidarBenchmarkProtocol,
    CameraLidarHitDefinition,
    CameraLidarPerturbation,
    load_calibration_candidate_trace,
    load_camera_lidar_benchmark_protocol,
)
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import (
    EstimateEvidenceLevel,
    EstimateProducer,
    TransformEstimateProvenance,
    TransformResult,
)
from calibrex.evaluation.borer_rotation_benchmark import (
    LoadedCameraLidarProblem,
    load_borer_problem,
)
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthOptions
from calibrex.solvers.borer_six_dof_solver import (
    BorerSixDofOptions,
    BorerSixDofResult,
    BorerSixDofSolver,
    apply_local_se3_delta,
)

BORER_SIX_DOF_BENCHMARK_VERSION = "calibrex.borer_six_dof_benchmark/v0.1"


@dataclass(frozen=True)
class BorerSixDofTrialExecution:
    """One in-memory six-DoF solver execution."""

    perturbation: CameraLidarPerturbation
    initial_transform_camera_lidar: SE3
    result: BorerSixDofResult
    runtime_seconds: float


def run_borer_six_dof_benchmark(
    problem_path: str | Path,
    protocol_path: str | Path,
    *,
    trace_directory: str | Path,
    command: str,
    bootstrap_samples: int = 2000,
    workers: int = 1,
    resume: bool = False,
) -> tuple[BenchmarkDefinition, BenchmarkArtifact]:
    """Execute every frozen six-DoF perturbation and retain all outcomes."""

    if workers < 1:
        raise ValueError("workers must be at least one")
    problem_file = Path(problem_path)
    protocol_file = Path(protocol_path)
    loaded = load_borer_problem(problem_file)
    protocol = load_camera_lidar_benchmark_protocol(protocol_file)
    protocol_digest = _required_digest(protocol_file)
    if protocol.degrees_of_freedom != "six_dof":
        raise ValueError("six-DoF runner requires a six_dof protocol")
    if protocol.problem_sha256 != loaded.problem_sha256:
        raise ValueError("protocol problem_sha256 does not match the problem artifact")
    if protocol.frame_ids != [item.frame_id for item in loaded.problem.observations]:
        raise ValueError("protocol frame IDs do not match the problem artifact")

    trace_root = Path(trace_directory)
    trace_root.mkdir(parents=True, exist_ok=True)
    cached: dict[str, CalibrationCandidateTrace] = {}
    pending = []
    if resume:
        for perturbation in protocol.perturbations:
            trace_path = trace_root / f"{perturbation.trial_id}.trace.yaml"
            if trace_path.is_file():
                cached[perturbation.trial_id] = _validated_trace(
                    trace_path,
                    perturbation=perturbation,
                    problem_sha256=loaded.problem_sha256,
                    protocol_sha256=protocol_digest,
                )
            else:
                pending.append(perturbation)
    else:
        pending = list(protocol.perturbations)
    executions = iter(
        _execute_trials(
            pending,
            loaded=loaded,
            options=_solver_options(protocol),
            workers=workers,
        )
    )

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
    for perturbation in protocol.perturbations:
        initial = apply_local_se3_delta(
            loaded.reference_transform_camera_lidar,
            perturbation.rotation_deg_xyz,
            perturbation.translation_m_xyz,
        )
        splits.append(
            BenchmarkSplit(
                split_id=perturbation.trial_id,
                seed=0,
                fit_count=len(protocol.frame_ids),
                holdout_count=1,
                fit_ids_sha256=fit_digest,
                holdout_ids_sha256=reference_digest,
            )
        )
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
                    notes=["prespecified paired Fibonacci-sphere perturbation"],
                ),
            )
        )
        trace_path = trace_root / f"{perturbation.trial_id}.trace.yaml"
        trace = cached.get(perturbation.trial_id)
        if trace is None:
            execution = next(executions)
            if execution.perturbation.trial_id != perturbation.trial_id:
                raise RuntimeError("six-DoF trial execution order changed")
            trace = _candidate_trace(
                execution.result,
                perturbation=perturbation,
                loaded=loaded,
                protocol=protocol,
                protocol_digest=protocol_digest,
                runtime_seconds=execution.runtime_seconds,
                command=command,
            )
            trace.save(trace_path)
        trace_digest = _required_digest(trace_path)
        trace_digests[perturbation.trial_id] = trace_digest
        native_trials.append(
            BenchmarkTrial(
                method_id="native_borer_d2d_six_dof",
                split_id=perturbation.trial_id,
                status=(
                    "failed"
                    if trace.status == "insufficient_observations"
                    else "success"
                ),
                metrics=(
                    {}
                    if trace.status == "insufficient_observations"
                    else {
                        "rotation_error_deg": trace.outcome.rotation_error_deg,
                        "translation_error_m": trace.outcome.translation_error_m,
                        "hit": float(trace.outcome.hit),
                    }
                ),
                runtime_seconds=trace.runtime_seconds,
                failure_reason=(
                    trace.stopping_reason
                    if trace.status == "insufficient_observations"
                    else None
                ),
                provenance=BenchmarkTrialProvenance(
                    command=command,
                    config_sha256=protocol_digest,
                    input_sha256=loaded.problem_sha256,
                    output_sha256=trace_digest,
                ),
            )
        )

    definition = BenchmarkDefinition(
        benchmark_id=protocol.protocol_id,
        title=f"Borer six-DoF D2D recovery on {loaded.problem.dataset_id}",
        protocol=BenchmarkProtocol(
            protocol_id=protocol.protocol_id,
            dataset_id=protocol.dataset_id,
            dataset_source_sha256=loaded.problem_sha256,
            data_license="dataset-specific; see problem/depth-provider provenance",
            split_policy=(
                "one paired Fibonacci rotation/translation perturbation per "
                "split; all selected frames optimize pose"
            ),
            splits=splits,
            initial_estimate_policy=(
                f"paired fibonacci_sphere at "
                f"{protocol.rotation_magnitude_deg:g} degrees and "
                f"{protocol.translation_magnitude_m:g} metres"
            ),
            tuning_policy=protocol.isolation.evidence,
            failure_policy="retain every miss and computational failure in denominator",
        ),
        metrics=_benchmark_metrics(),
        methods=_benchmark_methods(),
        trials=[*baseline_trials, *native_trials],
        reference_method_id="unoptimized_initial",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=0,
        limitations=[
            "The external monocular-depth provider is a frozen input.",
            "Dataset-reference extrinsics are not independent metrology.",
            "The paired rotation/translation direction policy is Calibrex-frozen.",
        ],
        provenance=BenchmarkProvenance(
            generator="calibrex.evaluation.borer_six_dof_benchmark",
            generator_version=BORER_SIX_DOF_BENCHMARK_VERSION,
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


def _execute_trial(
    perturbation: CameraLidarPerturbation,
    *,
    loaded: LoadedCameraLidarProblem,
    options: BorerSixDofOptions,
) -> BorerSixDofTrialExecution:
    initial = apply_local_se3_delta(
        loaded.reference_transform_camera_lidar,
        perturbation.rotation_deg_xyz,
        perturbation.translation_m_xyz,
    )
    started = time.perf_counter()
    result = BorerSixDofSolver().solve(loaded.observations, initial, options)
    return BorerSixDofTrialExecution(
        perturbation=perturbation,
        initial_transform_camera_lidar=initial,
        result=result,
        runtime_seconds=time.perf_counter() - started,
    )


def _execute_trials(
    perturbations: list[CameraLidarPerturbation],
    *,
    loaded: LoadedCameraLidarProblem,
    options: BorerSixDofOptions,
    workers: int,
) -> Iterator[BorerSixDofTrialExecution]:
    execute = partial(_execute_trial, loaded=loaded, options=options)
    if workers == 1:
        yield from map(execute, perturbations)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(execute, perturbations)


def _candidate_trace(
    result: BorerSixDofResult,
    *,
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
        solver_version=BORER_SIX_DOF_BENCHMARK_VERSION,
        status=result.status,
        initial_transform_camera_lidar=_transform_result(
            result.initial_transform_camera_lidar,
            producer="dataset_provider",
            evidence="dataset_provided",
            note="dataset reference plus frozen paired perturbation",
        ),
        output_transform_camera_lidar=_transform_result(
            result.transform_camera_lidar,
            producer="slac_native",
            evidence="algorithmically_refined",
            note="native D2D six-DoF output",
        ),
        evaluations=[
            CalibrationCandidateEvaluation(
                evaluation=item.evaluation,
                parameters={
                    "roll_delta_deg": item.delta_rotation_deg_xyz[0],
                    "pitch_delta_deg": item.delta_rotation_deg_xyz[1],
                    "yaw_delta_deg": item.delta_rotation_deg_xyz[2],
                    "tx_delta_m": item.delta_translation_m_xyz[0],
                    "ty_delta_m": item.delta_translation_m_xyz[1],
                    "tz_delta_m": item.delta_translation_m_xyz[2],
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
                f"rotation {rotation_error:.6g} < "
                f"{protocol.hit.rotation_error_max_deg:g} deg and translation "
                f"{translation_error:.6g} < "
                f"{protocol.hit.translation_error_max_m:g} m"
            ),
        ),
        provenance=CameraLidarArtifactProvenance(
            generator="calibrex.evaluation.borer_six_dof_benchmark",
            generator_version=BORER_SIX_DOF_BENCHMARK_VERSION,
            git_commit=git_commit(),
            command=command.split(),
            config_sha256=protocol_digest,
            source_sha256=loaded.problem_sha256,
        ),
    )


def _solver_options(protocol: CameraLidarBenchmarkProtocol) -> BorerSixDofOptions:
    values = protocol.optimizer_options
    return BorerSixDofOptions(
        rotation_bound_deg=float(values["rotation_bound_deg"]),
        translation_bound_m=float(values["translation_bound_m"]),
        initial_rotation_step_deg=float(values["initial_rotation_step_deg"]),
        initial_translation_step_m=float(values["initial_translation_step_m"]),
        minimum_rotation_step_deg=float(values["minimum_rotation_step_deg"]),
        minimum_translation_step_m=float(values["minimum_translation_step_m"]),
        max_evaluations=int(values["max_evaluations"]),
        improvement_tolerance=float(values["improvement_tolerance"]),
        d2d=DepthToDepthOptions(
            histogram_bins=protocol.histogram_bins,
            min_visible_points=protocol.min_visible_points,
            use_z_buffer=True,
        ),
    )


def _validated_trace(
    path: Path,
    *,
    perturbation: CameraLidarPerturbation,
    problem_sha256: str,
    protocol_sha256: str,
) -> CalibrationCandidateTrace:
    trace = load_calibration_candidate_trace(path)
    if (
        trace.trial_id != perturbation.trial_id
        or trace.problem_sha256 != problem_sha256
        or trace.protocol_sha256 != protocol_sha256
        or trace.solver_version != BORER_SIX_DOF_BENCHMARK_VERSION
    ):
        raise ValueError(f"resumed six-DoF trace identity mismatch: {path}")
    return trace


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
            source="borer_six_dof_benchmark",
            tool_name="calibrex",
            notes=[note],
        ),
    )


def _transform_errors(estimate: SE3, reference: SE3) -> tuple[float, float]:
    estimate_inverse = estimate.inverse()
    relative = reference.compose(estimate_inverse)
    quaternion = relative.rotation_quat_xyzw
    angle = 2.0 * math.degrees(
        math.acos(min(1.0, max(-1.0, abs(quaternion[3]))))
    )
    translation = math.sqrt(
        sum(
            (estimate.translation_m[index] - reference.translation_m[index]) ** 2
            for index in range(3)
        )
    )
    return angle, translation


def _hit(
    rotation_error_deg: float,
    translation_error_m: float,
    definition: CameraLidarHitDefinition,
) -> bool:
    return (
        rotation_error_deg < definition.rotation_error_max_deg
        and translation_error_m < definition.translation_error_max_m
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
            primary=True,
            interpretation="Euclidean translation error to dataset reference",
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
            label="Initial paired perturbation",
            implementation="calibrex_native",
            tool_name="fibonacci_sphere",
            tool_version="v0.1",
            paper_doi="10.48550/arXiv.2311.01905",
            license_spdx="Apache-2.0",
        ),
        BenchmarkMethodDefinition(
            method_id="native_borer_d2d_six_dof",
            label="Calibrex native D2D six-DoF",
            implementation="calibrex_native",
            tool_name="calibrex",
            tool_version=BORER_SIX_DOF_BENCHMARK_VERSION,
            paper_doi="10.48550/arXiv.2311.01905",
            license_spdx="Apache-2.0",
        ),
    ]


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required artifact is unreadable: {path}")
    return digest


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
