"""End-to-end falsification evidence for targetless Camera--LiDAR refinement.

The analytical D2D trace and the probabilistic multi-frame refiner are useful
independently, but a calibration claim needs one reproducible evaluation
boundary.  This module joins them with fixed frame holdouts, twelve signed
6-DoF known-bad controls, and the standard Calibrex evidence contract.

External depth/correspondence providers remain input artifacts.  No provider
code or runtime is imported by this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from calibrex import __version__
from calibrex.core.assessment import AssessmentArtifact, assess_report_evidence
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
    CalibrationCandidateTrace,
    CameraLidarBenchmarkProtocol,
    load_calibration_candidate_trace,
    load_camera_lidar_benchmark_protocol,
    load_camera_lidar_problem,
)
from calibrex.core.camera_lidar_failure_analysis import (
    CameraLidarFailureAnalysisArtifact,
)
from calibrex.core.evidence_bundle import (
    EvidenceBundleManifest,
    EvidenceBundleVerification,
    verify_evidence_bundle,
    write_evidence_bundle,
    write_evidence_bundle_verification,
)
from calibrex.core.evidence_contract import (
    ProtocolArtifact,
    policy_artifact_from_assessment,
)
from calibrex.core.geometry import SE3
from calibrex.core.io import write_mapping
from calibrex.core.probabilistic_correspondence import (
    ProbabilisticCorrespondenceFrame,
    ProbabilisticRefinementEvaluationArtifact,
    load_probabilistic_correspondence,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.report_artifacts import (
    EvidenceCaseItem,
    EvidenceInputFileItem,
    EvidenceMaterializationInfo,
    EvidenceProtocolItem,
    EvidenceSummaryItem,
    ReportEvidenceArtifact,
    ReportRunInfo,
)
from calibrex.core.result import Grade
from calibrex.evaluation.borer_six_dof_benchmark import (
    BORER_SIX_DOF_BENCHMARK_VERSION,
)
from calibrex.evaluation.camera_lidar_failure_analysis import (
    analyze_camera_lidar_failures,
)
from calibrex.evaluation.probabilistic_camera_lidar_run import (
    run_probabilistic_camera_lidar_refinement,
)
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta
from calibrex.solvers.probabilistic_camera_lidar_refiner import (
    ProbabilisticCameraLidarRefinementOptions,
    evaluate_probabilistic_camera_lidar_pose,
)

PROBABILISTIC_FALSIFICATION_VERSION = (
    "calibrex.probabilistic_camera_lidar_falsification/v0.2"
)
PROBABILISTIC_FALSIFICATION_PROTOCOL_ID = (
    "camera_lidar_targetless_6dof_refinement/v0.2"
)


@dataclass(frozen=True)
class ProbabilisticCameraLidarFalsificationArtifacts:
    """Materialized artifacts produced by one integrated evaluation run."""

    result_paths: tuple[Path, ...]
    initialization_trace_paths: tuple[Path, ...]
    definition_path: Path
    benchmark_path: Path
    evidence_path: Path
    failure_analysis_path: Path
    assessment_path: Path
    protocol_path: Path
    policy_path: Path
    bundle_path: Path
    verification_path: Path
    definition: BenchmarkDefinition
    benchmark: BenchmarkArtifact
    evidence: ReportEvidenceArtifact
    failure_analysis: CameraLidarFailureAnalysisArtifact
    assessment: AssessmentArtifact
    bundle: EvidenceBundleManifest
    verification: EvidenceBundleVerification


def run_probabilistic_camera_lidar_falsification(
    correspondence_path: str | Path,
    initialization_problem_path: str | Path,
    *,
    initialization_trace_path: str | Path | None = None,
    initialization_trace_directory: str | Path | None = None,
    benchmark_protocol_path: str | Path,
    output_directory: str | Path,
    command: str,
    split_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    base_options: ProbabilisticCameraLidarRefinementOptions | None = None,
    known_bad_rotation_deg: float = 5.0,
    known_bad_translation_m: float = 0.5,
    known_bad_min_holdout_rmse_delta_px: float = 0.5,
    max_final_holdout_rmse_px: float | None = None,
    bootstrap_samples: int = 2000,
) -> ProbabilisticCameraLidarFalsificationArtifacts:
    """Run integrated D2D-seeded refinement and materialize evidence sidecars.

    The D2D trace(s) and immutable six-DoF source protocol are required by
    this v0.2 integrated path. Supplying a trace directory requires one
    digest-valid trace for every protocol perturbation; each trace becomes an
    independent probabilistic initialization. Each seed uses a deterministic
    disjoint frame holdout. Known-bad controls perturb the final candidate
    without re-optimizing, so their holdout degradation is measured
    independently of fitting.
    """

    if not split_seeds or len(set(split_seeds)) != len(split_seeds):
        raise ValueError("split_seeds must be non-empty and unique")
    if known_bad_rotation_deg <= 0.0 or known_bad_translation_m <= 0.0:
        raise ValueError("known-bad rotation and translation magnitudes must be positive")
    if known_bad_min_holdout_rmse_delta_px <= 0.0:
        raise ValueError("known-bad holdout RMSE delta must be positive")
    if max_final_holdout_rmse_px is not None and max_final_holdout_rmse_px <= 0.0:
        raise ValueError("max_final_holdout_rmse_px must be positive when supplied")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    correspondence_file = Path(correspondence_path).resolve()
    problem_file = Path(initialization_problem_path).resolve()
    if (initialization_trace_path is None) == (
        initialization_trace_directory is None
    ):
        raise ValueError(
            "provide exactly one of initialization_trace_path or "
            "initialization_trace_directory"
        )
    trace_file = (
        Path(initialization_trace_path).resolve()
        if initialization_trace_path is not None
        else None
    )
    trace_directory = (
        Path(initialization_trace_directory).resolve()
        if initialization_trace_directory is not None
        else None
    )
    protocol_file = Path(benchmark_protocol_path).resolve()
    output_root = Path(output_directory).resolve()
    results_root = output_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    correspondence = load_probabilistic_correspondence(correspondence_file)
    problem = load_camera_lidar_problem(problem_file)
    correspondence_digest = _required_digest(correspondence_file)
    problem_digest = _required_digest(problem_file)
    protocol_digest = _required_digest(protocol_file)
    if correspondence.dataset_id != problem.dataset_id:
        raise ValueError("correspondence and initialization problem dataset IDs differ")
    if not correspondence.frames:
        raise ValueError("probabilistic correspondence artifact contains no frames")
    protocol = load_camera_lidar_benchmark_protocol(protocol_file)
    trace_paths, trace_paths_by_trial, traces_by_trial = _load_initialization_traces(
        protocol=protocol,
        trace_path=trace_file,
        trace_directory=trace_directory,
        problem_sha256=problem_digest,
        protocol_sha256=protocol_digest,
    )
    if protocol.degrees_of_freedom != "six_dof":
        raise ValueError("probabilistic falsification requires a six_dof protocol")
    if protocol.translation_magnitude_m <= 0.0 or any(
        not any(abs(value) > 0.0 for value in item.translation_m_xyz)
        for item in protocol.perturbations
    ):
        raise ValueError("six_dof protocol must contain non-zero translation perturbations")
    if protocol.dataset_id != correspondence.dataset_id:
        raise ValueError(
            "benchmark protocol dataset ID does not match correspondence"
        )
    if protocol.problem_sha256 != problem_digest:
        raise ValueError(
            "benchmark protocol problem digest does not match the problem"
        )
    if protocol.frame_ids != [item.frame_id for item in problem.observations]:
        raise ValueError("benchmark protocol frame IDs do not match the problem")

    settings = base_options or ProbabilisticCameraLidarRefinementOptions()
    frames_by_id = {frame.frame_id: frame for frame in correspondence.frames}
    trace_digests = {
        trial_id: _required_digest(path)
        for trial_id, path in trace_paths_by_trial.items()
    }
    combined_input_digest = _text_digest(
        "\n".join(
            (
                correspondence_digest,
                problem_digest,
                protocol_digest,
                *trace_digests.values(),
            )
        )
    )
    run_id = (
        f"{correspondence.artifact_id}-camera-lidar-falsification-v02"
    )
    command_tokens = command.split()

    result_paths: list[Path] = []
    trials: list[BenchmarkTrial] = []
    splits: list[BenchmarkSplit] = []
    known_bad_cases: list[EvidenceCaseItem] = []
    candidate_holdout_passes: list[bool] = []
    known_bad_passes: list[bool] = []
    if trace_directory is None:
        selected_perturbations = [
            item
            for item in protocol.perturbations
            if item.trial_id in traces_by_trial
        ]
    else:
        selected_perturbations = list(protocol.perturbations)
    if not selected_perturbations:
        raise ValueError("no protocol perturbation is represented by the D2D trace")
    full_d2d_coverage = len(selected_perturbations) == len(protocol.perturbations)

    for trial_index, perturbation in enumerate(selected_perturbations):
        trace = traces_by_trial[perturbation.trial_id]
        trial_trace_path = trace_paths_by_trial[perturbation.trial_id]
        for seed in split_seeds:
            options = replace(settings, split_seed=seed)
            result = run_probabilistic_camera_lidar_refinement(
                correspondence_file,
                problem_file,
                initialization_trace_path=trial_trace_path,
                result_id=f"{run_id}-{trace.trial_id}-seed-{seed:04d}",
                options=options,
                command=command_tokens,
            )
            result_path = (
                results_root
                / f"result.trial-{trial_index:03d}.seed-{seed:04d}.yaml"
            )
            result.save(result_path)
            result_digest = _required_digest(result_path)
            result_paths.append(result_path)

            fit_ids = list(result.train_frame_ids)
            holdout_ids = list(result.holdout_frame_ids)
            split_id = f"{trace.trial_id}::seed-{seed:04d}"
            splits.append(
                BenchmarkSplit(
                    split_id=split_id,
                    seed=seed,
                    fit_count=len(fit_ids),
                    holdout_count=len(holdout_ids),
                    fit_ids_sha256=_text_digest("\n".join(fit_ids)),
                    holdout_ids_sha256=_text_digest("\n".join(holdout_ids)),
                )
            )
            final_holdout = result.final_holdout_evaluation
            initial_holdout = result.initial_holdout_evaluation
            final_rmse = final_holdout.weighted_reprojection_rmse_px
            initial_rmse = initial_holdout.weighted_reprojection_rmse_px
            candidate_pass = _candidate_holdout_pass(
                result.status,
                result.acceptance is not None and result.acceptance.accepted,
                final_rmse,
                initial_rmse,
                final_holdout.valid_correspondence_count,
                options.minimum_holdout_correspondences,
                max_final_holdout_rmse_px,
            )
            candidate_holdout_passes.append(candidate_pass)
            baseline_metrics = _benchmark_metrics_for_result(
                initial_rmse,
                initial_holdout.objective,
                result.initial_rotation_error_deg,
                result.initial_translation_error_m,
                _improvement(initial_rmse, initial_rmse),
            )
            candidate_metrics = _benchmark_metrics_for_result(
                final_rmse,
                final_holdout.objective,
                result.final_rotation_error_deg,
                result.final_translation_error_m,
                _improvement(initial_rmse, final_rmse),
            )
            trials.append(
                BenchmarkTrial(
                    method_id="d2d_initialized_probabilistic",
                    split_id=split_id,
                    status="success" if candidate_pass else "failed",
                    metrics=candidate_metrics,
                    failure_reason=None if candidate_pass else result.reason,
                    runtime_seconds=None,
                    provenance=BenchmarkTrialProvenance(
                        command=command,
                        config_sha256=_options_digest(options),
                        input_sha256=combined_input_digest,
                        output_sha256=result_digest,
                        notes=[
                            "D2D candidate trace output initialized the shared SE(3) refinement",
                            f"D2D initialization trial: {trace.trial_id}",
                            "fit and holdout frame IDs are retained in the result artifact",
                        ],
                    ),
                )
            )
            trials.append(
                BenchmarkTrial(
                    method_id="d2d_candidate_initial",
                    split_id=split_id,
                    status="success" if initial_rmse is not None else "failed",
                    metrics=baseline_metrics,
                    failure_reason=(
                        None
                        if initial_rmse is not None
                        else "initial holdout has no valid correspondences"
                    ),
                    runtime_seconds=0.0,
                    provenance=BenchmarkTrialProvenance(
                        command=command,
                        config_sha256=_options_digest(options),
                        input_sha256=combined_input_digest,
                        notes=[
                            "paired D2D initializer baseline; no holdout fitting",
                            f"D2D initialization trial: {trace.trial_id}",
                        ],
                    ),
                )
            )

            holdout_frames = tuple(frames_by_id[item] for item in holdout_ids)
            controls_for_seed = _known_bad_controls(
                seed=seed,
                trial_id=trace.trial_id,
                candidate=result.transform_camera_lidar.as_se3(),
                holdout_frames=holdout_frames,
                options=options,
                candidate_evaluation=final_holdout,
                rotation_deg=known_bad_rotation_deg,
                translation_m=known_bad_translation_m,
                minimum_rmse_delta_px=known_bad_min_holdout_rmse_delta_px,
            )
            known_bad_cases.extend(controls_for_seed)
            known_bad_passes.extend(
                case.status == "pass" for case in controls_for_seed
            )

    failure_analysis = analyze_camera_lidar_failures(
        protocol_path=protocol_file,
        initialization_trace_paths=trace_paths,
        result_paths=result_paths,
        command=command_tokens,
    )
    failure_analysis_path = output_root / "failure-analysis.json"
    failure_analysis.save(failure_analysis_path)

    limitations = [
        "Correspondence extraction/provider execution is outside the ROS-independent core.",
        (
            "Problem reference transforms are retained for diagnostics and "
            "are not independent metrology unless declared as such."
        ),
        (
            "Known-bad controls are evaluated without re-optimization on "
            "the same untouched holdout frames."
        ),
    ]
    if not full_d2d_coverage:
        limitations.insert(
            2,
            (
                "A single declared D2D initialization trial was supplied; "
                "use --trace-dir for full protocol coverage."
            ),
        )

    benchmark_protocol = BenchmarkProtocol(
        protocol_id=PROBABILISTIC_FALSIFICATION_PROTOCOL_ID,
        dataset_id=correspondence.dataset_id,
        dataset_source_sha256=combined_input_digest,
        data_license=correspondence.dataset_license_spdx or "not declared",
        split_policy=(
            "D2D perturbation trial x seeded disjoint capture-frame holdout; "
            "no holdout fitting"
        ),
        splits=splits,
        initial_estimate_policy="digest-pinned D2D candidate trace output",
        tuning_policy=(
            "all optimizer settings, seeds, and known-bad magnitudes fixed "
            "before evaluation"
        ),
        failure_policy="retain every solver failure and every undetected known-bad control",
    )
    definition = BenchmarkDefinition(
        benchmark_id=run_id,
        title=(
            "D2D-initialized probabilistic targetless Camera-LiDAR six-DoF "
            f"refinement on {correspondence.dataset_id}"
        ),
        protocol=benchmark_protocol,
        metrics=[
            BenchmarkMetricDefinition(
                name="holdout_reprojection_rmse_px",
                label="Holdout reprojection RMSE",
                unit="px",
                direction="lower",
                primary=True,
                interpretation="confidence-weighted reprojection error on untouched frames",
            ),
            BenchmarkMetricDefinition(
                name="holdout_objective",
                label="Holdout robust objective",
                unit="objective",
                direction="lower",
                primary=False,
                interpretation="robust covariance-aware objective on untouched frames",
            ),
            BenchmarkMetricDefinition(
                name="rotation_error_deg",
                label="Reference rotation error",
                unit="deg",
                direction="lower",
                primary=True,
                interpretation=(
                    "error to the declared problem reference; not independent "
                    "metrology by itself"
                ),
            ),
            BenchmarkMetricDefinition(
                name="translation_error_m",
                label="Reference translation error",
                unit="m",
                direction="lower",
                primary=True,
                interpretation=(
                    "error to the declared problem reference; not independent "
                    "metrology by itself"
                ),
            ),
            BenchmarkMetricDefinition(
                name="holdout_improvement_px",
                label="Holdout RMSE improvement",
                unit="px",
                direction="higher",
                primary=False,
                interpretation="initial D2D holdout RMSE minus refined holdout RMSE",
            ),
        ],
        methods=[
            BenchmarkMethodDefinition(
                method_id="d2d_candidate_initial",
                label="D2D candidate initial",
                implementation="calibrex_native",
                tool_name="Calibrex D2D candidate trace",
                tool_version=BORER_SIX_DOF_BENCHMARK_VERSION,
                source_commit=git_commit(),
                license_spdx="Apache-2.0",
            ),
            BenchmarkMethodDefinition(
                method_id="d2d_initialized_probabilistic",
                label="D2D-initialized probabilistic refinement",
                implementation="calibrex_native",
                tool_name="Calibrex probabilistic multi-frame refiner",
                tool_version="0.3",
                source_commit=git_commit(),
                license_spdx="Apache-2.0",
            ),
        ],
        trials=trials,
        reference_method_id="d2d_candidate_initial",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=0,
        limitations=limitations,
        provenance=BenchmarkProvenance(
            generator=__name__,
            generator_version=PROBABILISTIC_FALSIFICATION_VERSION,
            git_commit=git_commit(),
            command=command,
            source_artifacts={
                "correspondence": correspondence_digest,
                "initialization_problem": problem_digest,
                "benchmark_protocol": protocol_digest,
                **{
                    f"initialization_trace:{trial_id}": digest
                    for trial_id, digest in sorted(trace_digests.items())
                },
            },
            data_verified=(
                correspondence.dataset_license_spdx is not None
                and bool(correspondence.provenance.input_sha256)
            ),
        ),
    )
    benchmark = aggregate_benchmark_definition(definition)

    holdout_status = _aggregate_status(candidate_holdout_passes)
    known_bad_status = _aggregate_status(known_bad_passes)
    decision_status: Grade = (
        "pass"
        if holdout_status == "pass" and known_bad_status == "pass"
        else "fail"
        if "fail" in {holdout_status, known_bad_status}
        else "warn"
    )
    run_status: Literal["success", "warning", "failed"] = (
        "success" if decision_status == "pass" else "warning"
    )
    run = ReportRunInfo(
        id=run_id,
        status=run_status,
        domain="camera_lidar",
        slac_version=__version__,
        git_commit=git_commit(),
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_type="camera_lidar",
        dataset_path=str(correspondence_file),
    )
    selected_trial_ids = [item.trial_id for item in selected_perturbations]
    protocol_item = EvidenceProtocolItem(
        family="lidar_camera",
        protocol_id=PROBABILISTIC_FALSIFICATION_PROTOCOL_ID,
        status=decision_status,
        split_policy=(
            "D2D perturbation trial x seeded disjoint capture-frame holdout; "
            "D2D output fixed before refinement"
        ),
        independent_holdout=True,
        candidate_transform="T_camera_lidar",
        transform_convention="T_parent_child",
        known_bad_perturbation=(
            "signed roll/pitch/yaw and x/y/z controls applied to the final candidate"
        ),
        known_bad_case_count=len(known_bad_cases),
        metric_ids=[
            "camera_lidar_holdout_reprojection_rmse_px",
            "camera_lidar_holdout_objective",
            "camera_lidar_known_bad_holdout_rmse_delta_px",
        ],
        parameters={
            "split_seeds": list(split_seeds),
            "d2d_protocol_id": protocol.protocol_id,
            "d2d_protocol_sha256": protocol_digest,
            "d2d_initialization_trial_ids": selected_trial_ids,
            "d2d_trace_count": len(selected_trial_ids),
            "d2d_protocol_perturbation_count": len(protocol.perturbations),
            "d2d_full_protocol_coverage": full_d2d_coverage,
            "holdout_ratio": settings.holdout_ratio,
            "known_bad_rotation_deg": known_bad_rotation_deg,
            "known_bad_translation_m": known_bad_translation_m,
            "known_bad_min_holdout_rmse_delta_px": known_bad_min_holdout_rmse_delta_px,
            "mandatory_rotation_deg": known_bad_rotation_deg,
            "mandatory_translation_m": known_bad_translation_m,
        },
        limitations=list(definition.limitations),
    )
    evidence = ReportEvidenceArtifact(
        run=run,
        materialization=EvidenceMaterializationInfo(
            metrics_origin="recomputed",
            data_verified=definition.provenance.data_verified,
            computed_at=run.created_at,
            report_generated_at=run.created_at,
        ),
        protocols=[protocol_item],
        input_files=[
            _input_file(correspondence_file, role="probabilistic correspondence"),
            _input_file(problem_file, role="camera-LiDAR initialization problem"),
            _input_file(protocol_file, role="fixed six-DoF D2D benchmark protocol"),
            *[
                _input_file(
                    path,
                    role=f"D2D initialization trace ({trial_id})",
                )
                for trial_id, path in trace_paths_by_trial.items()
            ],
        ],
        summaries=[
            EvidenceSummaryItem(
                family="lidar_camera",
                check="Holdout Edge Alignment",
                status=holdout_status,
                evidence=(
                    "final probabilistic reprojection was evaluated on disjoint frames "
                    "without fitting the holdout"
                ),
                interpretation=(
                    "candidate holdout error did not degrade from the D2D initializer"
                    if holdout_status == "pass"
                    else "candidate holdout gate was not satisfied"
                ),
                metric_ids=["camera_lidar_holdout_reprojection_rmse_px"],
            ),
            EvidenceSummaryItem(
                family="lidar_camera",
                check="Known-Bad Controls",
                status=known_bad_status,
                evidence=(
                    "twelve signed 6-DoF perturbations per split were scored on "
                    "untouched frames for every supplied D2D initialization trial"
                ),
                interpretation=(
                    "known-bad transforms were separated from the selected candidate"
                    if known_bad_status == "pass"
                    else "at least one known-bad transform was not separated"
                ),
                metric_ids=["camera_lidar_known_bad_holdout_rmse_delta_px"],
            ),
            EvidenceSummaryItem(
                family="lidar_camera",
                check="Decision Boundary",
                status=decision_status,
                evidence="holdout non-degradation and known-bad rejection are jointly required",
                interpretation=(
                    "integrated targetless refinement passed its declared evidence gates"
                    if decision_status == "pass"
                    else "integrated targetless refinement is not supported by this run"
                ),
                metric_ids=[
                    "camera_lidar_holdout_reprojection_rmse_px",
                    "camera_lidar_known_bad_holdout_rmse_delta_px",
                ],
            ),
        ],
        cases=known_bad_cases,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    definition_path = output_root / "benchmark.definition.yaml"
    benchmark_path = output_root / "benchmark.yaml"
    evidence_path = output_root / "evidence.json"
    assessment_path = output_root / "assessment.json"
    protocol_path = output_root / "protocol.json"
    policy_path = output_root / "policy.json"
    bundle_path = output_root / "bundle.json"
    verification_path = output_root / "verification.json"
    definition.save(definition_path)
    benchmark.save(benchmark_path)
    write_mapping(evidence_path, evidence.model_dump(mode="json", exclude_none=True))
    assessment = assess_report_evidence(
        evidence,
        source_path="evidence.json",
        source_sha256=_required_digest(evidence_path),
    )
    write_mapping(assessment_path, assessment.model_dump(mode="json", exclude_none=True))
    write_mapping(
        protocol_path,
        ProtocolArtifact(run=run, protocols=[protocol_item]).model_dump(
            mode="json", exclude_none=True
        ),
    )
    policy = policy_artifact_from_assessment(assessment, run=run)
    write_mapping(policy_path, policy.model_dump(mode="json", exclude_none=True))
    bundle = write_evidence_bundle(
        bundle_path,
        run_id=run_id,
        primary_evidence_path=evidence_path,
        artifacts=[
            *[(path, "probabilistic-refinement-result") for path in result_paths],
            (definition_path, "benchmark-definition"),
            (benchmark_path, "benchmark"),
            (evidence_path, "report-evidence"),
            (failure_analysis_path, "camera-lidar-failure-analysis"),
            (assessment_path, "assessment"),
            (protocol_path, "protocol"),
            (policy_path, "policy"),
        ],
    )
    verification = verify_evidence_bundle(bundle_path, require_raw_recomputed=True)
    write_evidence_bundle_verification(
        verification_path,
        verification,
        source_bundle_path=bundle_path,
    )
    return ProbabilisticCameraLidarFalsificationArtifacts(
        result_paths=tuple(result_paths),
        initialization_trace_paths=trace_paths,
        definition_path=definition_path,
        benchmark_path=benchmark_path,
        evidence_path=evidence_path,
        failure_analysis_path=failure_analysis_path,
        assessment_path=assessment_path,
        protocol_path=protocol_path,
        policy_path=policy_path,
        bundle_path=bundle_path,
        verification_path=verification_path,
        definition=definition,
        benchmark=benchmark,
        evidence=evidence,
        failure_analysis=failure_analysis,
        assessment=assessment,
        bundle=bundle,
        verification=verification,
    )


def _load_initialization_traces(
    *,
    protocol: CameraLidarBenchmarkProtocol,
    trace_path: Path | None,
    trace_directory: Path | None,
    problem_sha256: str,
    protocol_sha256: str,
) -> tuple[
    tuple[Path, ...],
    dict[str, Path],
    dict[str, CalibrationCandidateTrace],
]:
    """Load one trace or the complete protocol trace set with identity checks."""

    expected = {item.trial_id for item in protocol.perturbations}
    if trace_path is not None:
        trace = load_calibration_candidate_trace(trace_path)
        _validate_initialization_trace(
            trace,
            path=trace_path,
            expected_trial_ids=expected,
            problem_sha256=problem_sha256,
            protocol_sha256=protocol_sha256,
        )
        return (trace_path,), {trace.trial_id: trace_path}, {trace.trial_id: trace}

    if trace_directory is None or not trace_directory.is_dir():
        raise ValueError(
            "initialization_trace_directory must be an existing directory"
        )
    paths: list[Path] = []
    paths_by_trial: dict[str, Path] = {}
    traces_by_trial: dict[str, CalibrationCandidateTrace] = {}
    expected_solver: str | None = None
    for perturbation in protocol.perturbations:
        path = trace_directory / f"{perturbation.trial_id}.trace.yaml"
        if not path.is_file():
            raise ValueError(
                "missing D2D trace for protocol trial "
                f"{perturbation.trial_id}: {path}"
            )
        trace = load_calibration_candidate_trace(path)
        _validate_initialization_trace(
            trace,
            path=path,
            expected_trial_ids=expected,
            problem_sha256=problem_sha256,
            protocol_sha256=protocol_sha256,
        )
        if expected_solver is None:
            expected_solver = trace.solver
        elif trace.solver != expected_solver:
            raise ValueError(
                "mixed D2D solver identities in initialization trace directory: "
                f"expected {expected_solver!r}, got {trace.solver!r} at {path}"
            )
        if trace.trial_id in traces_by_trial:
            raise ValueError(f"duplicate D2D initialization trial: {trace.trial_id}")
        paths.append(path)
        paths_by_trial[trace.trial_id] = path
        traces_by_trial[trace.trial_id] = trace
    return tuple(paths), paths_by_trial, traces_by_trial


def _validate_initialization_trace(
    trace: CalibrationCandidateTrace,
    *,
    path: Path,
    expected_trial_ids: set[str],
    problem_sha256: str,
    protocol_sha256: str,
) -> None:
    """Validate a D2D trace's protocol membership and source digests."""

    if trace.trial_id not in expected_trial_ids:
        raise ValueError(f"initialization trace trial_id is absent from protocol: {path}")
    if trace.protocol_sha256 != protocol_sha256:
        raise ValueError(
            f"initialization trace protocol digest does not match the protocol: {path}"
        )
    if trace.problem_sha256 != problem_sha256:
        raise ValueError(
            f"initialization trace problem digest does not match the problem: {path}"
        )
    if trace.solver_version != BORER_SIX_DOF_BENCHMARK_VERSION:
        raise ValueError(
            "initialization trace is not a native six-DoF D2D trace: "
            f"{path} (solver_version={trace.solver_version!r})"
        )


def _known_bad_controls(
    *,
    seed: int,
    trial_id: str,
    candidate: SE3,
    holdout_frames: tuple[ProbabilisticCorrespondenceFrame, ...],
    options: ProbabilisticCameraLidarRefinementOptions,
    candidate_evaluation: ProbabilisticRefinementEvaluationArtifact,
    rotation_deg: float,
    translation_m: float,
    minimum_rmse_delta_px: float,
) -> list[EvidenceCaseItem]:
    controls = (
        ("roll+", [rotation_deg, 0.0, 0.0], [0.0, 0.0, 0.0], "deg", rotation_deg),
        ("roll-", [-rotation_deg, 0.0, 0.0], [0.0, 0.0, 0.0], "deg", -rotation_deg),
        ("pitch+", [0.0, rotation_deg, 0.0], [0.0, 0.0, 0.0], "deg", rotation_deg),
        ("pitch-", [0.0, -rotation_deg, 0.0], [0.0, 0.0, 0.0], "deg", -rotation_deg),
        ("yaw+", [0.0, 0.0, rotation_deg], [0.0, 0.0, 0.0], "deg", rotation_deg),
        ("yaw-", [0.0, 0.0, -rotation_deg], [0.0, 0.0, 0.0], "deg", -rotation_deg),
        ("x+", [0.0, 0.0, 0.0], [translation_m, 0.0, 0.0], "m", translation_m),
        ("x-", [0.0, 0.0, 0.0], [-translation_m, 0.0, 0.0], "m", -translation_m),
        ("y+", [0.0, 0.0, 0.0], [0.0, translation_m, 0.0], "m", translation_m),
        ("y-", [0.0, 0.0, 0.0], [0.0, -translation_m, 0.0], "m", -translation_m),
        ("z+", [0.0, 0.0, 0.0], [0.0, 0.0, translation_m], "m", translation_m),
        ("z-", [0.0, 0.0, 0.0], [0.0, 0.0, -translation_m], "m", -translation_m),
    )
    candidate_rmse = candidate_evaluation.weighted_reprojection_rmse_px
    candidate_objective = candidate_evaluation.objective
    cases: list[EvidenceCaseItem] = []
    for name, rotation, translation, unit, amount in controls:
        control_transform = apply_local_se3_delta(candidate, rotation, translation)
        control_evaluation = evaluate_probabilistic_camera_lidar_pose(
            holdout_frames,
            control_transform,
            options,
        )
        control_rmse = control_evaluation.weighted_reprojection_rmse_px
        control_objective = control_evaluation.objective
        rmse_delta = _delta(control_rmse, candidate_rmse)
        objective_delta = _delta(control_objective, candidate_objective)
        detected = bool(
            control_rmse is not None
            and candidate_rmse is not None
            and rmse_delta is not None
            and rmse_delta >= minimum_rmse_delta_px
            and control_evaluation.valid_correspondence_count
            >= options.minimum_holdout_correspondences
        )
        cases.append(
            EvidenceCaseItem(
                family="lidar_camera",
                case_id=f"{trial_id}::seed-{seed:04d}-{name}",
                check="Known-Bad Controls",
                status="pass" if detected else "fail",
                dof=name.rstrip("+-"),
                amount=amount,
                unit=unit,
                convention="T_parent_child; perturb final candidate locally",
                metric_values={
                    "camera_lidar_candidate_holdout_rmse_px": candidate_rmse,
                    "camera_lidar_known_bad_holdout_rmse_px": control_rmse,
                    "camera_lidar_candidate_holdout_objective": candidate_objective,
                    "camera_lidar_known_bad_holdout_objective": control_objective,
                    "camera_lidar_known_bad_holdout_valid_count": float(
                        control_evaluation.valid_correspondence_count
                    ),
                },
                delta_values={
                    "camera_lidar_known_bad_holdout_rmse_delta_px": rmse_delta,
                    "camera_lidar_known_bad_holdout_objective_delta": objective_delta,
                },
            )
        )
    return cases


def _candidate_holdout_pass(
    status: str,
    candidate_accepted: bool,
    final_rmse: float | None,
    initial_rmse: float | None,
    valid_count: int,
    minimum_valid_count: int,
    maximum_rmse: float | None,
) -> bool:
    if (
        status != "converged"
        or not candidate_accepted
        or final_rmse is None
        or initial_rmse is None
    ):
        return False
    if valid_count < minimum_valid_count or final_rmse > initial_rmse + 1.0e-9:
        return False
    return maximum_rmse is None or final_rmse <= maximum_rmse


def _aggregate_status(values: list[bool]) -> Literal["pass", "fail", "warn"]:
    if not values:
        return "warn"
    return "pass" if all(values) else "fail"


def _benchmark_metrics_for_result(
    rmse: float | None,
    objective: float,
    rotation_error: float | None,
    translation_error: float | None,
    improvement: float | None,
) -> dict[str, float]:
    metrics = {"holdout_objective": objective}
    if rmse is not None:
        metrics["holdout_reprojection_rmse_px"] = rmse
    if rotation_error is not None:
        metrics["rotation_error_deg"] = rotation_error
    if translation_error is not None:
        metrics["translation_error_m"] = translation_error
    if improvement is not None:
        metrics["holdout_improvement_px"] = improvement
    return metrics


def _improvement(initial: float | None, final: float | None) -> float | None:
    if initial is None or final is None:
        return None
    return initial - final


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _input_file(path: Path, *, role: str) -> EvidenceInputFileItem:
    digest = _required_digest(path)
    return EvidenceInputFileItem(
        path=str(path),
        role=role,
        sha256=digest,
        size_bytes=path.stat().st_size,
    )


def _options_digest(options: ProbabilisticCameraLidarRefinementOptions) -> str:
    return _text_digest(json.dumps(asdict(options), sort_keys=True, separators=(",", ":")))


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required Camera-LiDAR artifact is not readable: {path}")
    return digest


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
