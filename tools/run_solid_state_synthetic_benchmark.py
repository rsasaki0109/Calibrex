#!/usr/bin/env python3
"""Run a deterministic ground-truth benchmark for the solid-state LiDAR solver."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from calibrex import __version__
from calibrex.core.geometry import SE3
from calibrex.core.io import write_mapping, write_text
from calibrex.core.provenance import sha256_path
from calibrex.core.result import (
    EstimateEvidenceLevel,
    EstimateRole,
    TransformEstimateProvenance,
    TransformResult,
)
from calibrex.core.solid_state_synthetic_benchmark import (
    SOLID_STATE_SYNTHETIC_BENCHMARK_SCHEMA_VERSION,
    SolidStateSyntheticBenchmarkAggregate,
    SolidStateSyntheticBenchmarkArtifact,
    SolidStateSyntheticBenchmarkCase,
    SolidStateSyntheticBenchmarkProtocol,
    SolidStateSyntheticBenchmarkProvenance,
    SolidStateSyntheticBenchmarkThresholds,
    SyntheticBenchmarkCaseType,
    SyntheticBenchmarkExpectedOutcome,
)
from calibrex.data.livox import LivoxPointRecord
from calibrex.data.odometry_track import OdometryPoseSample, OdometryTrack
from calibrex.solvers.continuous_time_lidar_pair_solver import (
    ContinuousTimeLidarPairOptions,
    ContinuousTimeLidarPairProblem,
    ContinuousTimeLidarPairResult,
    ContinuousTimeLidarPairSolver,
)
from calibrex.solvers.fixed_trajectory_se3_solver import FixedTrajectorySe3SolverOptions

TOOL_PATH = "tools/run_solid_state_synthetic_benchmark.py"
RANDOM_SEED = 17


@dataclass(frozen=True)
class _SyntheticScene:
    problem: ContinuousTimeLidarPairProblem
    true_transform: SE3
    true_time_offset_sec: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _yaw_quaternion(degrees: float) -> tuple[float, float, float, float]:
    half = math.radians(degrees) * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _world_plane_records() -> list[LivoxPointRecord]:
    records: list[LivoxPointRecord] = []
    for x in range(0, 9):
        for y in range(-4, 5):
            records.append(
                LivoxPointRecord(
                    point=(0.5 * x, 0.5 * y, 0.0, 0.0),
                    normal_xyz=(0.0, 0.0, 1.0),
                )
            )
    for y in range(-4, 5):
        for z in range(0, 7):
            records.append(
                LivoxPointRecord(
                    point=(4.0, 0.5 * y, 0.5 * z, 0.0),
                    normal_xyz=(1.0, 0.0, 0.0),
                )
            )
    for x in range(0, 9):
        for z in range(0, 7):
            records.append(
                LivoxPointRecord(
                    point=(0.5 * x, -2.0, 0.5 * z, 0.0),
                    normal_xyz=(0.0, 1.0, 0.0),
                )
            )
    return records


def _scene() -> _SyntheticScene:
    source_records = _world_plane_records()
    true_transform = SE3((0.12, -0.05, 0.04), _yaw_quaternion(8.0))
    odometry_samples = [
        OdometryPoseSample(
            timestamp_ns=1_000_000_000 + index * 100_000_000,
            pose=SE3(
                (0.12 * index, 0.0, 0.0),
                _yaw_quaternion(4.0 * index),
            ),
        )
        for index in range(12)
    ]
    true_time_offset_sec = 0.03
    target_points: list[tuple[float, float, float]] = []
    target_timestamps_ns: list[int] = []
    selected_world_points = [record.point[:3] for record in source_records[::7]]
    for sample in odometry_samples[1:-1]:
        for point_world in selected_world_points[:12]:
            point_source = sample.pose.inverse().transform_point(point_world)
            target_points.append(true_transform.inverse().transform_point(point_source))
            target_timestamps_ns.append(
                sample.timestamp_ns - round(true_time_offset_sec * 1e9)
            )
    split = int(len(target_points) * 0.75)
    return _SyntheticScene(
        problem=ContinuousTimeLidarPairProblem(
            source_records=source_records,
            target_points=target_points,
            target_capture_timestamps_ns=target_timestamps_ns,
            odometry_track=OdometryTrack(odometry_samples),
            t_base_source=SE3.identity(),
            initial_t_source_target=SE3(
                (0.18, -0.02, 0.08),
                _yaw_quaternion(3.0),
            ),
            variable="T_base_lidar_target",
            sensor="lidar_target",
            voxel_size_m=0.5,
            correspondence_gate_m=0.8,
            train_indices=tuple(range(split)),
            holdout_indices=tuple(range(split, len(target_points))),
        ),
        true_transform=true_transform,
        true_time_offset_sec=true_time_offset_sec,
    )


def _solver_options(*, max_abs_time_offset_sec: float) -> ContinuousTimeLidarPairOptions:
    return ContinuousTimeLidarPairOptions(
        max_abs_time_offset_sec=max_abs_time_offset_sec,
        initial_time_step_sec=0.02,
        minimum_time_step_sec=0.001,
        max_iterations=16,
        min_correspondences=6,
        max_odometry_extrapolation_s=0.05,
        max_source_records=None,
        max_target_points_per_split=None,
        sampling_seed=RANDOM_SEED,
        holdout_start_fraction=0.75,
        outlier_policy="none",
        voxel_strategy="uniform",
        correspondence_refinement_iterations=1,
        solver=FixedTrajectorySe3SolverOptions(
            max_iterations=24,
            robust_loss="none",
            max_rotation_step_rad=math.radians(2.0),
        ),
    )


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(
        sum(
            left_value * right_value
            for left_value, right_value in zip(
                left.rotation_quat_xyzw,
                right.rotation_quat_xyzw,
                strict=True,
            )
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def _translation_error_m(left: SE3, right: SE3) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(
                left.translation_m,
                right.translation_m,
                strict=True,
            )
        )
    )


def _transform_result(
    transform: SE3,
    *,
    role: EstimateRole,
    evidence_level: EstimateEvidenceLevel,
) -> TransformResult:
    return TransformResult(
        parent="source",
        child="target",
        translation_m=list(transform.translation_m),
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        estimate_id=f"synthetic-{role}",
        provenance=TransformEstimateProvenance(
            producer="factory" if evidence_level == "synthetic_truth" else "slac_native",
            execution_mode="manual" if evidence_level == "synthetic_truth" else "offline_batch",
            role_in_comparison=role,
            evidence_level=evidence_level,
            source=TOOL_PATH,
            tool_name=TOOL_PATH if evidence_level != "synthetic_truth" else None,
            tool_version=__version__ if evidence_level != "synthetic_truth" else None,
            notes=[
                "deterministic synthetic three-plane motion fixture",
                "not a public-dataset accuracy claim",
            ],
        ),
    )


def _failure_reasons(
    result: ContinuousTimeLidarPairResult,
    *,
    true_transform: SE3,
    true_time_offset_sec: float,
    thresholds: SolidStateSyntheticBenchmarkThresholds,
) -> list[str]:
    reasons: list[str] = []
    if thresholds.require_converged and result.status != "converged":
        reasons.append(f"status={result.status}")
    if result.observability is None or result.observability.rank != 6:
        rank = None if result.observability is None else result.observability.rank
        reasons.append(f"observability_rank={rank}")
    if _rotation_error_deg(result.refined_t_source_target, true_transform) > (
        thresholds.max_rotation_error_deg
    ):
        reasons.append("final_rotation_error_exceeds_threshold")
    if _translation_error_m(result.refined_t_source_target, true_transform) > (
        thresholds.max_translation_error_m
    ):
        reasons.append("final_translation_error_exceeds_threshold")
    if abs(result.estimated_time_offset_sec - true_time_offset_sec) > (
        thresholds.max_time_offset_error_sec
    ):
        reasons.append("final_time_offset_error_exceeds_threshold")
    if result.final_holdout_rmse_m is None:
        reasons.append("missing_final_holdout_rmse")
    if result.train_correspondence_count < 6 or result.holdout_correspondence_count < 6:
        reasons.append("insufficient_correspondences")
    return reasons


def _case(
    scene: _SyntheticScene,
    *,
    case_id: str,
    case_type: SyntheticBenchmarkCaseType,
    expected_outcome: SyntheticBenchmarkExpectedOutcome,
    options: ContinuousTimeLidarPairOptions,
    thresholds: SolidStateSyntheticBenchmarkThresholds,
    notes: list[str],
) -> SolidStateSyntheticBenchmarkCase:
    result = ContinuousTimeLidarPairSolver().solve(scene.problem, options)
    initial = scene.problem.initial_t_source_target
    true_transform = scene.true_transform
    final_rotation_error = _rotation_error_deg(result.refined_t_source_target, true_transform)
    final_translation_error = _translation_error_m(result.refined_t_source_target, true_transform)
    failure_reasons = _failure_reasons(
        result,
        true_transform=true_transform,
        true_time_offset_sec=scene.true_time_offset_sec,
        thresholds=thresholds,
    )
    gate_passed = not failure_reasons
    expected_detected = (
        gate_passed if expected_outcome == "pass" else not gate_passed
    )
    return SolidStateSyntheticBenchmarkCase(
        id=case_id,
        case_type=case_type,
        expected_outcome=expected_outcome,
        status=result.status,
        initial_transform=_transform_result(
            initial,
            role="initial",
            evidence_level="synthetic_truth",
        ),
        true_transform=_transform_result(
            true_transform,
            role="selected_reference",
            evidence_level="synthetic_truth",
        ),
        estimated_transform=_transform_result(
            result.refined_t_source_target,
            role="output",
            evidence_level="algorithmically_refined",
        ),
        true_time_offset_sec=scene.true_time_offset_sec,
        estimated_time_offset_sec=result.estimated_time_offset_sec,
        initial_rotation_error_deg=_rotation_error_deg(initial, true_transform),
        final_rotation_error_deg=final_rotation_error,
        initial_translation_error_m=_translation_error_m(initial, true_transform),
        final_translation_error_m=final_translation_error,
        initial_time_offset_error_sec=abs(scene.true_time_offset_sec),
        final_time_offset_error_sec=abs(
            result.estimated_time_offset_sec - scene.true_time_offset_sec
        ),
        initial_train_rmse_m=result.initial_train_rmse_m,
        final_train_rmse_m=result.final_train_rmse_m,
        final_holdout_rmse_m=result.final_holdout_rmse_m,
        train_correspondence_count=result.train_correspondence_count,
        holdout_correspondence_count=result.holdout_correspondence_count,
        observability_rank=(
            result.observability.rank if result.observability is not None else None
        ),
        iterations=len(result.iterations),
        gate_passed=gate_passed,
        expected_outcome_detected=expected_detected,
        failure_reasons=failure_reasons,
        notes=notes,
    )


def build_solid_state_synthetic_benchmark(
    *,
    command: list[str] | None = None,
) -> SolidStateSyntheticBenchmarkArtifact:
    """Build the deterministic reference and fixed-clock control cases."""

    source_path = Path(__file__).resolve()
    source_digest = sha256_path(source_path)
    if source_digest is None:
        raise RuntimeError(f"could not hash benchmark generator: {source_path}")
    thresholds = SolidStateSyntheticBenchmarkThresholds(
        max_rotation_error_deg=0.5,
        max_translation_error_m=0.02,
        max_time_offset_error_sec=0.005,
        require_converged=True,
    )
    scene = _scene()
    cases = [
        _case(
            scene,
            case_id="joint_extrinsic_clock_reference",
            case_type="reference",
            expected_outcome="pass",
            options=_solver_options(max_abs_time_offset_sec=0.1),
            thresholds=thresholds,
            notes=[
                "known transform and +30 ms clock offset are recovered jointly",
                "truth is generated by this tool, not measured hardware truth",
            ],
        ),
        _case(
            scene,
            case_id="fixed_clock_known_bad_control",
            case_type="known_bad",
            expected_outcome="fail",
            options=_solver_options(max_abs_time_offset_sec=0.0),
            thresholds=thresholds,
            notes=[
                "clock offset is deliberately fixed to zero while truth is +30 ms",
                "the control must be rejected by the absolute time-error gate",
            ],
        ),
    ]
    reference_cases = [item for item in cases if item.case_type == "reference"]
    known_bad_cases = [item for item in cases if item.case_type == "known_bad"]
    reference_pass_count = sum(item.gate_passed for item in reference_cases)
    known_bad_detected_count = sum(
        not item.gate_passed for item in known_bad_cases
    )
    all_detected = (
        reference_pass_count == len(reference_cases)
        and known_bad_detected_count == len(known_bad_cases)
    )
    return SolidStateSyntheticBenchmarkArtifact(
        benchmark_id="solid-state-synthetic-ground-truth-v0.1",
        protocol=SolidStateSyntheticBenchmarkProtocol(
            name="synthetic_joint_extrinsic_clock_recovery",
            solver="continuous_time_lidar_pair",
            scene="three_plane_motion_fixture",
            point_time_convention=(
                "target capture timestamp = source odometry timestamp - true offset; "
                "positive solver offset evaluates target at t_capture + offset"
            ),
            train_fraction=0.75,
            known_bad_control="fixed clock at zero against a +30 ms truth",
            notes=[
                "all points and poses are generated deterministically in memory",
                "the final holdout is disjoint and chronological",
            ],
        ),
        thresholds=thresholds,
        cases=cases,
        aggregate=SolidStateSyntheticBenchmarkAggregate(
            reference_case_count=len(reference_cases),
            reference_pass_count=reference_pass_count,
            known_bad_case_count=len(known_bad_cases),
            known_bad_detected_count=known_bad_detected_count,
            all_expected_outcomes_detected=all_detected,
            conclusion="pass" if all_detected else "fail",
            notes=[
                "passing this benchmark establishes solver recovery on this fixture only",
                "it does not establish public-dataset or metrology accuracy",
            ],
        ),
        provenance=SolidStateSyntheticBenchmarkProvenance(
            generator=TOOL_PATH,
            generator_version=__version__,
            source_sha256=source_digest,
            command=command or [],
            random_seed=RANDOM_SEED,
            notes=[
                "the generator source digest binds the synthetic scene and thresholds",
                "no ROS, GPL, or external dataset is used",
            ],
        ),
    )


def render_solid_state_synthetic_benchmark_markdown(
    artifact: SolidStateSyntheticBenchmarkArtifact,
) -> str:
    """Render a compact human-readable report for the truth-backed artifact."""

    lines = [
        "# Solid-state LiDAR synthetic ground-truth benchmark",
        "",
        f"- Schema: `{artifact.schema_version}`",
        f"- Generator: `{artifact.provenance.generator}`",
        f"- Generator SHA-256: `{artifact.provenance.source_sha256}`",
        "",
        artifact.aggregate.conclusion.upper(),
        "",
        "This is a deterministic solver gate, not a public-dataset accuracy claim.",
        "",
        "| Case | Expected | Status | Rotation error (deg) | Translation error (m) | "
        "Clock error (ms) | Gate | Detected |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for case in artifact.cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    case.id,
                    case.expected_outcome,
                    case.status,
                    f"{case.final_rotation_error_deg:.6f}",
                    f"{case.final_translation_error_m:.6f}",
                    f"{case.final_time_offset_error_sec * 1000.0:.3f}",
                    "PASS" if case.gate_passed else "FAIL",
                    "yes" if case.expected_outcome_detected else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The reference case must pass all absolute parameter-error gates.",
            "- The fixed-clock control must fail the time-error gate.",
            "- Passing here does not replace independent extrinsic/clock ground truth "
            "on a real sensor pair.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write the schema-valid artifact."""

    args = _parser().parse_args(argv)
    command = [TOOL_PATH, *(argv or sys.argv[1:])]
    artifact = build_solid_state_synthetic_benchmark(command=command)
    write_mapping(args.output, artifact.model_dump(mode="json", exclude_none=True))
    if args.markdown_output is not None:
        write_text(
            args.markdown_output,
            render_solid_state_synthetic_benchmark_markdown(artifact),
        )
    payload = {
        "schema_version": SOLID_STATE_SYNTHETIC_BENCHMARK_SCHEMA_VERSION,
        "output": str(args.output),
        "conclusion": artifact.aggregate.conclusion,
        "reference_pass_count": artifact.aggregate.reference_pass_count,
        "known_bad_detected_count": artifact.aggregate.known_bad_detected_count,
    }
    if args.json:
        import json

        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"synthetic benchmark: {artifact.aggregate.conclusion} "
            f"reference={artifact.aggregate.reference_pass_count}/"
            f"{artifact.aggregate.reference_case_count} "
            f"known_bad={artifact.aggregate.known_bad_detected_count}/"
            f"{artifact.aggregate.known_bad_case_count}"
        )
    if args.enforce and not artifact.aggregate.all_expected_outcomes_detected:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
