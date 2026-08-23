#!/usr/bin/env python3
"""Analyze frozen I2PNet pose-initializer correction responses."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from calibrex.core.benchmark import BenchmarkDefinition, BenchmarkTrial
from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
from calibrex.core.camera_lidar_correspondence_export import (
    load_camera_lidar_correspondence_export,
)
from calibrex.core.camera_lidar_pose_initializer_benchmark import (
    CameraLidarPoseInitializerBenchmarkProtocol,
    load_camera_lidar_pose_initializer_protocol,
)
from calibrex.core.camera_lidar_pose_initializer_failure_analysis import (
    CameraLidarCorrectionVector,
    CameraLidarPoseInitializerFailureAnalysis,
    CameraLidarPoseInitializerFailureCase,
    CameraLidarPoseInitializerFailureFinding,
    CameraLidarPoseInitializerFailureProvenance,
    CameraLidarPoseInitializerFailureSummary,
    CameraLidarPoseInitializerFrameResponse,
)
from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import TransformResult

I2PNET_POSE_FAILURE_ANALYSIS_VERSION = "calibrex.i2pnet_pose_failure_analysis/v0.1"

FloatArray = NDArray[np.float64]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a frozen I2PNet pose-initializer benchmark"
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("problem", type=Path)
    parser.add_argument("definition", type=Path)
    parser.add_argument("trial_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate benchmark lineage and write correction-response analysis."""

    raw_args = sys.argv[1:] if argv is None else argv
    args = _parser().parse_args(raw_args)
    paths = {
        "protocol": args.protocol.resolve(),
        "problem": args.problem.resolve(),
        "definition": args.definition.resolve(),
        "trial_directory": args.trial_directory.resolve(),
        "analyzer": Path(__file__).resolve(),
        "schema_source": (
            Path(__file__).resolve().parent.parent
            / "src"
            / "calibrex"
            / "core"
            / "camera_lidar_pose_initializer_failure_analysis.py"
        ),
    }
    protocol = load_camera_lidar_pose_initializer_protocol(paths["protocol"])
    problem = load_camera_lidar_problem(paths["problem"])
    definition = BenchmarkDefinition.model_validate(read_mapping(paths["definition"]))
    digests = {name: _required_digest(path) for name, path in paths.items() if path.is_file()}
    _verify_top_level(protocol, problem=problem, definition=definition, digests=digests)

    source_paths = {name: str(path) for name, path in paths.items() if path.is_file()}
    source_digests = dict(digests)
    cases: list[CameraLidarPoseInitializerFailureCase] = []
    for perturbation in protocol.perturbations:
        trial_id = perturbation.trial_id
        trial_directory = paths["trial_directory"] / trial_id
        initial_path = trial_directory / "initial.yaml"
        manifest_path = trial_directory / "correspondence-manifest.yaml"
        pose_path = trial_directory / "pose.yaml"
        trial_paths = {
            f"initial:{trial_id}": initial_path,
            f"manifest:{trial_id}": manifest_path,
            f"pose:{trial_id}": pose_path,
        }
        trial_digests = {name: _required_digest(path) for name, path in trial_paths.items()}
        for name, digest in trial_digests.items():
            if definition.provenance.source_artifacts.get(name) != digest:
                raise ValueError(f"benchmark definition digest mismatch for {name}")
            source_paths[name] = str(trial_paths[name])
            source_digests[name] = digest
        case, export_sources = _analyze_trial(
            trial_id,
            protocol=protocol,
            problem=problem,
            definition=definition,
            initial_path=initial_path,
            manifest_path=manifest_path,
            pose_path=pose_path,
            trial_digests=trial_digests,
        )
        cases.append(case)
        for name, path in export_sources.items():
            source_paths[name] = str(path)
            source_digests[name] = _required_digest(path)

    summary = _summary(cases)
    artifact = CameraLidarPoseInitializerFailureAnalysis(
        analysis_id=f"{protocol.protocol_id}-failure-analysis-v0.1",
        protocol_id=protocol.protocol_id,
        protocol_sha256=digests["protocol"],
        problem_id=problem.problem_id,
        problem_sha256=digests["problem"],
        benchmark_definition_id=definition.benchmark_id,
        benchmark_definition_sha256=digests["definition"],
        dataset_id=protocol.dataset_id,
        partition=protocol.partition,
        provider_id=protocol.provider.provider_id,
        hit=protocol.hit,
        summary=summary,
        cases=cases,
        findings=_findings(cases, summary),
        conclusion=(
            "The frozen response matrix measures correction anisotropy for this provider, "
            "checkpoint, dataset, and perturbation set. Findings are diagnostic associations, "
            "not causal proof or evaluation-partition evidence."
        ),
        provenance=CameraLidarPoseInitializerFailureProvenance(
            generator="tools.analyze_i2pnet_pose_benchmark",
            generator_version=I2PNET_POSE_FAILURE_ANALYSIS_VERSION,
            git_commit=git_commit(),
            command=[sys.executable, str(Path(__file__).resolve()), *raw_args],
            source_paths=source_paths,
            source_sha256=source_digests,
            notes=[
                "all benchmark and frame-export digests were recomputed",
                "required correction is T_reference @ inverse(T_initial)",
                "predicted correction is T_output @ inverse(T_initial)",
                "rotation vectors use the shortest canonical quaternion arc",
                "reference errors are dataset-derived and not independent metrology",
            ],
        ),
    )
    artifact.save(args.output.resolve())
    print(
        f"analysis={args.output.resolve()} trials={summary.trial_count} "
        f"frames={summary.frame_response_count} hits={summary.hit_count}",
        flush=True,
    )
    return 0


def _verify_top_level(
    protocol: CameraLidarPoseInitializerBenchmarkProtocol,
    *,
    problem: Any,
    definition: BenchmarkDefinition,
    digests: dict[str, str],
) -> None:
    if protocol.problem_id != problem.problem_id or protocol.problem_sha256 != digests["problem"]:
        raise ValueError("protocol and problem identity or digest differ")
    if definition.benchmark_id != protocol.protocol_id:
        raise ValueError("benchmark definition ID differs from the protocol")
    declared = definition.provenance.source_artifacts
    if declared.get("protocol") != digests["protocol"]:
        raise ValueError("benchmark definition protocol digest mismatch")
    if declared.get("problem") != digests["problem"]:
        raise ValueError("benchmark definition problem digest mismatch")


def _analyze_trial(
    trial_id: str,
    *,
    protocol: CameraLidarPoseInitializerBenchmarkProtocol,
    problem: Any,
    definition: BenchmarkDefinition,
    initial_path: Path,
    manifest_path: Path,
    pose_path: Path,
    trial_digests: dict[str, str],
) -> tuple[CameraLidarPoseInitializerFailureCase, dict[str, Path]]:
    initial = TransformResult.model_validate(read_mapping(initial_path))
    output = TransformResult.model_validate(read_mapping(pose_path))
    manifest = load_camera_lidar_correspondence_export(manifest_path)
    if [item.frame_id for item in manifest.frames] != protocol.frame_ids:
        raise ValueError(f"manifest frame IDs differ from protocol for {trial_id}")
    if manifest.provenance.input_sha256.get("initial_transform") != trial_digests[
        f"initial:{trial_id}"
    ]:
        raise ValueError(f"manifest initial-transform digest mismatch for {trial_id}")
    reference = problem.reference_transform_camera_lidar.as_se3()
    initial_se3 = initial.as_se3()
    output_se3 = output.as_se3()
    required = reference.compose(initial_se3.inverse())
    predicted = output_se3.compose(initial_se3.inverse())
    initial_errors = _transform_errors(initial_se3, reference)
    output_errors = _transform_errors(output_se3, reference)
    candidate_trial = _trial(definition, "i2pnet_kitti_large", trial_id)
    baseline_trial = _trial(definition, "unoptimized_initial", trial_id)
    _verify_metrics(candidate_trial, output_errors, protocol=protocol)
    _verify_metrics(baseline_trial, initial_errors, protocol=protocol)
    required_rotation = _rotation_vector_deg(required)
    predicted_rotation = _rotation_vector_deg(predicted)
    required_translation = np.asarray(required.translation_m, dtype=np.float64)
    predicted_translation = np.asarray(predicted.translation_m, dtype=np.float64)
    rotation_response = _response(required_rotation, predicted_rotation)
    translation_response = _response(required_translation, predicted_translation)

    frames: list[CameraLidarPoseInitializerFrameResponse] = []
    export_sources: dict[str, Path] = {}
    for frame in manifest.frames:
        export_path = _resolve_reference(manifest_path.parent, frame.export.path)
        export_digest = _required_digest(export_path)
        if export_digest != frame.export.sha256:
            raise ValueError(f"frame export digest mismatch for {trial_id}/{frame.frame_id}")
        with np.load(export_path, allow_pickle=False) as payload:
            if "i2pnet_pose_correction_wxyz_t" not in payload:
                raise ValueError(f"frame export has no pose correction: {export_path}")
            correction = _correction_se3(payload["i2pnet_pose_correction_wxyz_t"])
        frame_output = correction.compose(initial_se3)
        frame_errors = _transform_errors(frame_output, reference)
        frames.append(
            CameraLidarPoseInitializerFrameResponse(
                frame_id=frame.frame_id,
                export_path=str(export_path),
                export_sha256=export_digest,
                predicted_correction=_correction_vector(correction),
                output_rotation_error_deg=frame_errors[0],
                output_translation_error_m=frame_errors[1],
            )
        )
        export_sources[f"export:{trial_id}:{frame.frame_id}"] = export_path
    return (
        CameraLidarPoseInitializerFailureCase(
            trial_id=trial_id,
            initial_path=str(initial_path),
            initial_sha256=trial_digests[f"initial:{trial_id}"],
            manifest_path=str(manifest_path),
            manifest_sha256=trial_digests[f"manifest:{trial_id}"],
            pose_path=str(pose_path),
            pose_sha256=trial_digests[f"pose:{trial_id}"],
            initial_rotation_error_deg=initial_errors[0],
            initial_translation_error_m=initial_errors[1],
            required_correction=_correction_vector(required),
            aggregate_predicted_correction=_correction_vector(predicted),
            rotation_parallel_gain=rotation_response[0],
            rotation_orthogonal_ratio=rotation_response[1],
            rotation_alignment_cosine=rotation_response[2],
            translation_parallel_gain=translation_response[0],
            translation_orthogonal_ratio=translation_response[1],
            translation_alignment_cosine=translation_response[2],
            output_rotation_error_deg=output_errors[0],
            output_translation_error_m=output_errors[1],
            hit=_hit(output_errors, protocol),
            frames=frames,
        ),
        export_sources,
    )


def _trial(definition: BenchmarkDefinition, method_id: str, split_id: str) -> BenchmarkTrial:
    matches = [
        trial
        for trial in definition.trials
        if trial.method_id == method_id and trial.split_id == split_id
    ]
    if len(matches) != 1 or matches[0].status != "success":
        raise ValueError(f"expected one successful {method_id}/{split_id} benchmark trial")
    return matches[0]


def _verify_metrics(
    trial: BenchmarkTrial,
    errors: tuple[float, float],
    *,
    protocol: CameraLidarPoseInitializerBenchmarkProtocol,
) -> None:
    expected = {
        "rotation_error_deg": errors[0],
        "translation_error_m": errors[1],
        "hit": float(_hit(errors, protocol)),
    }
    if set(trial.metrics) != set(expected) or any(
        not math.isclose(trial.metrics[name], value, rel_tol=0.0, abs_tol=1.0e-10)
        for name, value in expected.items()
    ):
        raise ValueError(
            f"benchmark metrics do not reproduce for {trial.method_id}/{trial.split_id}"
        )


def _summary(
    cases: list[CameraLidarPoseInitializerFailureCase],
) -> CameraLidarPoseInitializerFailureSummary:
    rotation_gain: dict[str, float] = {}
    rotation_orthogonal: dict[str, float] = {}
    translation_gain: dict[str, float] = {}
    translation_orthogonal: dict[str, float] = {}
    for case in cases:
        rotation_axis = _single_axis(case.required_correction.rotation_vector_deg)
        if rotation_axis is not None and case.rotation_parallel_gain is not None:
            rotation_gain[rotation_axis] = case.rotation_parallel_gain
            rotation_orthogonal[rotation_axis] = case.rotation_orthogonal_ratio or 0.0
        translation_axis = _single_axis(case.required_correction.translation_m)
        if translation_axis is not None and case.translation_parallel_gain is not None:
            translation_gain[translation_axis] = case.translation_parallel_gain
            translation_orthogonal[translation_axis] = case.translation_orthogonal_ratio or 0.0
    return CameraLidarPoseInitializerFailureSummary(
        trial_count=len(cases),
        frame_response_count=sum(len(case.frames) for case in cases),
        hit_count=sum(case.hit for case in cases),
        hit_rate=sum(case.hit for case in cases) / len(cases),
        mean_initial_rotation_error_deg=float(
            np.mean([case.initial_rotation_error_deg for case in cases])
        ),
        mean_output_rotation_error_deg=float(
            np.mean([case.output_rotation_error_deg for case in cases])
        ),
        mean_initial_translation_error_m=float(
            np.mean([case.initial_translation_error_m for case in cases])
        ),
        mean_output_translation_error_m=float(
            np.mean([case.output_translation_error_m for case in cases])
        ),
        single_axis_rotation_parallel_gain=rotation_gain,
        single_axis_rotation_orthogonal_ratio=rotation_orthogonal,
        single_axis_translation_parallel_gain=translation_gain,
        single_axis_translation_orthogonal_ratio=translation_orthogonal,
    )


def _findings(
    cases: list[CameraLidarPoseInitializerFailureCase],
    summary: CameraLidarPoseInitializerFailureSummary,
) -> list[CameraLidarPoseInitializerFailureFinding]:
    rotation = summary.single_axis_rotation_parallel_gain
    translation = summary.single_axis_translation_parallel_gain
    findings = [
        CameraLidarPoseInitializerFailureFinding(
            finding_id="strict-gate-missed",
            severity="warning",
            confidence="high",
            title="No frozen perturbation meets the joint pose gate",
            evidence=(
                f"{summary.hit_count}/{summary.trial_count} trials satisfy strict rotation "
                "and translation thresholds."
            ),
            supporting_trial_ids=[case.trial_id for case in cases if not case.hit],
            caveat="This development matrix is small and does not estimate deployment prevalence.",
        )
    ]
    if set(rotation) == {"x", "y", "z"}:
        findings.append(
            CameraLidarPoseInitializerFailureFinding(
                finding_id="rotation-response-anisotropy",
                severity="candidate_cause",
                confidence="medium",
                title="Rotation correction is concentrated on the Y axis",
                evidence=(
                    "single-axis signed parallel gains are "
                    f"x={rotation['x']:.6f}, y={rotation['y']:.6f}, z={rotation['z']:.6f}"
                ),
                supporting_trial_ids=[case.trial_id for case in cases],
                caveat=(
                    "The association is specific to this checkpoint, sequence, frames, and "
                    "perturbation magnitudes; it does not identify a causal network component."
                ),
            )
        )
    if set(translation) == {"x", "y", "z"}:
        findings.append(
            CameraLidarPoseInitializerFailureFinding(
                finding_id="translation-response-anisotropy",
                severity="candidate_cause",
                confidence="medium",
                title="Translation correction suppresses the Y component",
                evidence=(
                    "single-axis signed parallel gains are "
                    f"x={translation['x']:.6f}, y={translation['y']:.6f}, "
                    f"z={translation['z']:.6f}"
                ),
                supporting_trial_ids=[case.trial_id for case in cases],
                caveat=(
                    "Axis response may combine training-distribution priors, image geometry, "
                    "and model architecture; this artifact does not separate those causes."
                ),
            )
        )
    return findings


def _response(
    required: FloatArray,
    predicted: FloatArray,
) -> tuple[float | None, float | None, float | None]:
    norm_squared = float(required @ required)
    predicted_norm = float(np.linalg.norm(predicted))
    if norm_squared <= 1.0e-20:
        return None, None, None
    required_norm = math.sqrt(norm_squared)
    gain = float(predicted @ required / norm_squared)
    orthogonal = predicted - gain * required
    orthogonal_ratio = float(np.linalg.norm(orthogonal) / required_norm)
    cosine = None if predicted_norm <= 1.0e-20 else float(
        np.clip((predicted @ required) / (predicted_norm * required_norm), -1.0, 1.0)
    )
    return gain, orthogonal_ratio, cosine


def _single_axis(values: list[float], tolerance: float = 1.0e-8) -> str | None:
    active = [index for index, value in enumerate(values) if abs(value) > tolerance]
    return "xyz"[active[0]] if len(active) == 1 else None


def _correction_vector(transform: SE3) -> CameraLidarCorrectionVector:
    return CameraLidarCorrectionVector(
        rotation_vector_deg=_rotation_vector_deg(transform).tolist(),
        translation_m=list(transform.translation_m),
    )


def _rotation_vector_deg(transform: SE3) -> FloatArray:
    quaternion = np.asarray(transform.rotation_quat_xyzw, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    sine = float(np.linalg.norm(quaternion[:3]))
    if sine <= 1.0e-15:
        return np.zeros(3, dtype=np.float64)
    angle_deg = math.degrees(2.0 * math.atan2(sine, float(quaternion[3])))
    return angle_deg * quaternion[:3] / sine


def _correction_se3(values: NDArray[Any]) -> SE3:
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("I2PNet correction must contain seven finite wxyz+t values")
    w, x, y, z, tx, ty, tz = (float(value) for value in pose)
    return SE3((tx, ty, tz), (x, y, z, w))


def _transform_errors(estimate: SE3, reference: SE3) -> tuple[float, float]:
    relative = reference.compose(estimate.inverse())
    rotation = 2.0 * math.degrees(
        math.acos(min(1.0, max(-1.0, abs(relative.rotation_quat_xyzw[3]))))
    )
    return rotation, math.dist(estimate.translation_m, reference.translation_m)


def _hit(
    errors: tuple[float, float],
    protocol: CameraLidarPoseInitializerBenchmarkProtocol,
) -> bool:
    return (
        errors[0] < protocol.hit.rotation_error_max_deg
        and errors[1] < protocol.hit.translation_error_max_m
    )


def _resolve_reference(base: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base / path).resolve()


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        raise ValueError(f"required source is missing or unreadable: {path}")
    return digest


if __name__ == "__main__":
    raise SystemExit(main())
