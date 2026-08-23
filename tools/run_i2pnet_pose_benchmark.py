#!/usr/bin/env python3
"""Run a frozen I2PNet pose-initializer perturbation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

if __package__:
    from tools import run_i2pnet_correspondence_provider as i2pnet
else:  # Direct ``python tools/<script>.py`` execution.
    import run_i2pnet_correspondence_provider as i2pnet

from calibrex import __version__
from calibrex.core.benchmark import (
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
    CameraLidarCalibrationProblem,
    CameraLidarHitDefinition,
    CameraLidarPerturbation,
    load_camera_lidar_problem,
)
from calibrex.core.camera_lidar_correspondence_export import (
    load_camera_lidar_correspondence_export,
)
from calibrex.core.camera_lidar_pose_initializer_benchmark import (
    CameraLidarPoseInitializerBenchmarkProtocol,
    load_camera_lidar_pose_initializer_protocol,
)
from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import (
    TransformEstimateProvenance,
    TransformQuality,
    TransformResult,
)

I2PNET_POSE_BENCHMARK_VERSION = "calibrex.i2pnet_pose_benchmark/v0.1"

FloatArray = NDArray[np.float64]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a frozen external I2PNet pose-initializer benchmark"
    )
    parser.add_argument("protocol", type=Path)
    parser.add_argument("problem", type=Path)
    parser.add_argument("sequence_path", type=Path)
    parser.add_argument("--depth-provider", type=Path, required=True)
    parser.add_argument("--provider-python", type=Path, required=True)
    parser.add_argument("--provider-adapter", type=Path, required=True)
    parser.add_argument("--i2pnet-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path)
    parser.add_argument("--provider-patch", type=Path)
    parser.add_argument("--lidar-directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--definition-output", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-evaluation",
        action="store_true",
        help="explicitly unlock an evaluation-partition protocol",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute all protocol perturbations and aggregate paired evidence."""

    raw_args = sys.argv[1:] if argv is None else argv
    args = _parser().parse_args(raw_args)
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    paths = _resolved_paths(args)
    protocol = load_camera_lidar_pose_initializer_protocol(paths["protocol"])
    problem = load_camera_lidar_problem(paths["problem"])
    if protocol.partition == "evaluation" and not args.allow_evaluation:
        raise ValueError(
            "evaluation protocol is locked; pass --allow-evaluation only after "
            "all provider and threshold choices are frozen"
        )
    protocol_digest = _required_digest(paths["protocol"])
    problem_digest = _required_digest(paths["problem"])
    _verify_protocol_inputs(
        protocol,
        problem=problem,
        problem_digest=problem_digest,
        adapter=paths["provider_adapter"],
        checkpoint=paths["checkpoint"],
    )
    paths["output_directory"].mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), *raw_args]
    command_text = subprocess.list2cmdline(command)
    trials: list[BenchmarkTrial] = []
    splits: list[BenchmarkSplit] = []
    source_artifacts = {
        "protocol": protocol_digest,
        "problem": problem_digest,
        "depth_provider": _required_digest(paths["depth_provider"]),
        "provider_adapter": _required_digest(paths["provider_adapter"]),
        "checkpoint": _required_digest(paths["checkpoint"]),
        "benchmark_runner": _required_digest(Path(__file__).resolve()),
    }
    if paths.get("checkpoint_archive") is not None:
        source_artifacts["checkpoint_archive"] = _required_digest(paths["checkpoint_archive"])
    if paths.get("provider_patch") is not None:
        source_artifacts["provider_patch"] = _required_digest(paths["provider_patch"])

    frame_digest = _text_digest("\n".join(protocol.frame_ids))
    reference_digest = _text_digest(
        json.dumps(
            problem.reference_transform_camera_lidar.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    for position, perturbation in enumerate(protocol.perturbations, start=1):
        split_id = perturbation.trial_id
        trial_directory = paths["output_directory"] / split_id
        trial_directory.mkdir(parents=True, exist_ok=True)
        initial_path = trial_directory / "initial.yaml"
        manifest_path = trial_directory / "correspondence-manifest.yaml"
        pose_path = trial_directory / "pose.yaml"
        export_directory = trial_directory / "exports"
        initial = _initial_result(
            problem,
            perturbation=perturbation,
            protocol=protocol,
            problem_path=paths["problem"],
            problem_digest=problem_digest,
            protocol_digest=protocol_digest,
        )
        write_mapping(
            initial_path,
            initial.model_dump(mode="json", exclude_none=True),
        )
        initial_digest = _required_digest(initial_path)
        source_artifacts[f"initial:{split_id}"] = initial_digest
        trial_input_digest = _combined_digest(
            {
                "protocol": protocol_digest,
                "problem": problem_digest,
                "depth_provider": source_artifacts["depth_provider"],
                "checkpoint": source_artifacts["checkpoint"],
                "initial": initial_digest,
            }
        )
        initial_rotation, initial_translation = _transform_errors(
            initial.as_se3(),
            problem.reference_transform_camera_lidar.as_se3(),
        )
        trials.append(
            _benchmark_trial(
                method_id="unoptimized_initial",
                split_id=split_id,
                rotation_error_deg=initial_rotation,
                translation_error_m=initial_translation,
                hit=_hit(initial_rotation, initial_translation, protocol.hit),
                runtime_seconds=0.0,
                command=command_text,
                protocol_digest=protocol_digest,
                input_digest=trial_input_digest,
                output_digest=initial_digest,
                notes=["explicit left-camera-frame SE(3) perturbation"],
            )
        )
        splits.append(
            BenchmarkSplit(
                split_id=split_id,
                seed=0,
                fit_count=len(protocol.frame_ids),
                holdout_count=1,
                fit_ids_sha256=frame_digest,
                holdout_ids_sha256=reference_digest,
            )
        )
        provider_command = _provider_command(
            args,
            paths=paths,
            protocol=protocol,
            initial_path=initial_path,
            export_directory=export_directory,
            manifest_path=manifest_path,
            pose_path=pose_path,
        )
        started = time.perf_counter()
        failure_reason: str | None = None
        if not (args.resume and manifest_path.is_file() and pose_path.is_file()):
            completed = subprocess.run(provider_command, check=False)
            if completed.returncode != 0:
                failure_reason = f"provider exited with code {completed.returncode}"
        runtime_seconds = time.perf_counter() - started
        if failure_reason is None:
            try:
                output = _verified_provider_output(
                    pose_path,
                    manifest_path=manifest_path,
                    initial=initial,
                    initial_digest=initial_digest,
                    protocol=protocol,
                )
            except (OSError, ValueError) as exc:
                failure_reason = str(exc)
        if failure_reason is not None:
            trials.append(
                BenchmarkTrial(
                    method_id="i2pnet_kitti_large",
                    split_id=split_id,
                    status="failed",
                    runtime_seconds=runtime_seconds,
                    failure_reason=failure_reason,
                    provenance=BenchmarkTrialProvenance(
                        command=subprocess.list2cmdline(provider_command),
                        config_sha256=protocol_digest,
                        input_sha256=trial_input_digest,
                        execution_host=platform.node(),
                    ),
                )
            )
            print(
                f"[{position}/{len(protocol.perturbations)}] {split_id} failed: {failure_reason}",
                flush=True,
            )
            continue
        pose_digest = _required_digest(pose_path)
        manifest_digest = _required_digest(manifest_path)
        source_artifacts[f"manifest:{split_id}"] = manifest_digest
        source_artifacts[f"pose:{split_id}"] = pose_digest
        rotation_error, translation_error = _transform_errors(
            output.as_se3(),
            problem.reference_transform_camera_lidar.as_se3(),
        )
        trials.append(
            _benchmark_trial(
                method_id="i2pnet_kitti_large",
                split_id=split_id,
                rotation_error_deg=rotation_error,
                translation_error_m=translation_error,
                hit=_hit(rotation_error, translation_error, protocol.hit),
                runtime_seconds=runtime_seconds,
                command=subprocess.list2cmdline(provider_command),
                protocol_digest=protocol_digest,
                input_digest=trial_input_digest,
                output_digest=pose_digest,
                notes=[
                    f"manifest_sha256={manifest_digest}",
                    f"frame_count={len(protocol.frame_ids)}",
                    "provider process never receives the benchmark problem/reference",
                ],
            )
        )
        print(
            f"[{position}/{len(protocol.perturbations)}] {split_id} "
            f"rotation={rotation_error:.6f}deg translation={translation_error:.6f}m",
            flush=True,
        )

    definition = BenchmarkDefinition(
        benchmark_id=protocol.protocol_id,
        title=(f"I2PNet pose-initializer {protocol.partition} benchmark on {protocol.dataset_id}"),
        protocol=BenchmarkProtocol(
            protocol_id=protocol.protocol_id,
            dataset_id=protocol.dataset_id,
            dataset_version=protocol.sequence_id,
            dataset_source_sha256=problem_digest,
            data_license=protocol.dataset_license_spdx,
            split_policy=(
                "one paired trial per frozen left-camera-frame SE(3) perturbation; "
                f"each trial aggregates {len(protocol.frame_ids)} fixed frames"
            ),
            splits=splits,
            initial_estimate_policy=protocol.perturbation_convention,
            tuning_policy=protocol.isolation.evidence,
            failure_policy=protocol.failure_policy,
        ),
        metrics=_benchmark_metrics(),
        methods=_benchmark_methods(protocol),
        trials=trials,
        reference_method_id="unoptimized_initial",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=0,
        limitations=[
            f"partition={protocol.partition}; no claim transfers to another partition",
            "I2PNet checkpoint training overlap with KITTI raw is unknown",
            "frame aggregation spread is not calibrated predictive uncertainty",
            "diagnostic cost-volume correspondences are rejected independently of pose output",
        ],
        provenance=BenchmarkProvenance(
            generator="tools.run_i2pnet_pose_benchmark",
            generator_version=I2PNET_POSE_BENCHMARK_VERSION,
            git_commit=git_commit(),
            command=command_text,
            source_artifacts=source_artifacts,
            data_verified=True,
        ),
    )
    definition.save(paths["definition_output"])
    benchmark = aggregate_benchmark_definition(definition)
    benchmark.save(paths["benchmark_output"])
    candidate = benchmark.method_summaries["i2pnet_kitti_large"]
    strict_hits = sum(
        int(trial.metrics.get("hit", 0.0))
        for trial in definition.trials
        if trial.method_id == "i2pnet_kitti_large" and trial.status == "success"
    )
    print(
        f"benchmark={paths['benchmark_output']} completed={candidate.success_count}/"
        f"{candidate.trial_count} strict_hits={strict_hits}/{candidate.trial_count} "
        f"failure_rate={candidate.failure_rate:.6f}",
        flush=True,
    )
    return 0 if candidate.failure_count == 0 else 2


def _resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    names = (
        "protocol",
        "problem",
        "sequence_path",
        "depth_provider",
        "provider_python",
        "provider_adapter",
        "i2pnet_repository",
        "checkpoint",
        "output_directory",
        "definition_output",
        "benchmark_output",
    )
    values = {name: getattr(args, name).resolve() for name in names}
    for name in ("checkpoint_archive", "provider_patch", "lidar_directory"):
        value = getattr(args, name)
        if value is not None:
            values[name] = value.resolve()
    return values


def _verify_protocol_inputs(
    protocol: CameraLidarPoseInitializerBenchmarkProtocol,
    *,
    problem: CameraLidarCalibrationProblem,
    problem_digest: str,
    adapter: Path,
    checkpoint: Path,
) -> None:
    if protocol.problem_sha256 != problem_digest:
        raise ValueError("protocol problem_sha256 does not match the problem artifact")
    if (
        protocol.problem_id != problem.problem_id
        or protocol.dataset_id != problem.dataset_id
        or protocol.sequence_id != problem.sequence_id
    ):
        raise ValueError("protocol dataset identity does not match the problem artifact")
    available = {item.frame_id for item in problem.observations}
    missing = sorted(set(protocol.frame_ids) - available)
    if missing:
        raise ValueError("protocol frame IDs are absent from the problem: " + ", ".join(missing))
    if protocol.provider.provider_id != "i2pnet_kitti_large":
        raise ValueError("runner requires provider_id=i2pnet_kitti_large")
    if protocol.provider.source_repository != i2pnet.I2PNET_REPOSITORY:
        raise ValueError("protocol I2PNet source repository differs from the adapter")
    if protocol.provider.source_commit != i2pnet.I2PNET_COMMIT:
        raise ValueError("protocol I2PNet source commit differs from the adapter")
    if protocol.provider.checkpoint_sha256 != _required_digest(checkpoint):
        raise ValueError("protocol checkpoint digest differs from the runtime checkpoint")
    if protocol.provider.adapter_sha256 != _required_digest(adapter):
        raise ValueError("protocol adapter digest differs from the runtime adapter")
    if protocol.provider.adapter_version != i2pnet.I2PNET_ADAPTER_VERSION:
        raise ValueError("protocol adapter version differs from the runtime adapter")


def _initial_result(
    problem: CameraLidarCalibrationProblem,
    *,
    perturbation: CameraLidarPerturbation,
    protocol: CameraLidarPoseInitializerBenchmarkProtocol,
    problem_path: Path,
    problem_digest: str,
    protocol_digest: str,
) -> TransformResult:
    transform = _left_camera_frame_perturbation(
        problem.reference_transform_camera_lidar.as_se3(),
        perturbation.rotation_deg_xyz,
        perturbation.translation_m_xyz,
    )
    return TransformResult(
        parent=problem.reference_transform_camera_lidar.parent,
        child=problem.reference_transform_camera_lidar.child,
        translation_m=list(transform.translation_m),
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        quality=TransformQuality(grade="warn"),
        estimate_id=f"{protocol.protocol_id}-{perturbation.trial_id}-initial",
        provenance=TransformEstimateProvenance(
            producer="slac_native",
            execution_mode="offline_batch",
            role_in_comparison="initial",
            evidence_level="algorithmically_refined",
            source="frozen reference-derived pose-initializer perturbation protocol",
            source_path=str(problem_path),
            tool_name="calibrex",
            tool_version=__version__,
            adapter_version=I2PNET_POSE_BENCHMARK_VERSION,
            command=(
                "left_camera_frame_se3 rotation_deg_xyz="
                f"{perturbation.rotation_deg_xyz} translation_m_xyz="
                f"{perturbation.translation_m_xyz}"
            ),
            notes=[
                f"problem_sha256={problem_digest}",
                f"protocol_sha256={protocol_digest}",
                f"partition={protocol.partition}",
                "reference-derived perturbation is explicit; provider receives only this initial",
            ],
        ),
    )


def _provider_command(
    args: argparse.Namespace,
    *,
    paths: dict[str, Path],
    protocol: CameraLidarPoseInitializerBenchmarkProtocol,
    initial_path: Path,
    export_directory: Path,
    manifest_path: Path,
    pose_path: Path,
) -> list[str]:
    command = [
        str(paths["provider_python"]),
        str(paths["provider_adapter"]),
        str(paths["sequence_path"]),
        "--depth-provider",
        str(paths["depth_provider"]),
        "--i2pnet-repository",
        str(paths["i2pnet_repository"]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--initial-transform",
        str(initial_path),
        "--output-directory",
        str(export_directory),
        "--manifest-output",
        str(manifest_path),
        "--pose-output",
        str(pose_path),
        "--dataset-family",
        protocol.dataset_family,
        "--split-id",
        protocol.partition,
        "--camera-stream",
        protocol.camera_stream,
        "--frame-ids",
        ",".join(protocol.frame_ids),
        "--neighbor-backend",
        protocol.provider.neighbor_backend,
        "--device",
        args.device,
    ]
    for key, flag in (
        ("checkpoint_archive", "--checkpoint-archive"),
        ("provider_patch", "--provider-patch"),
        ("lidar_directory", "--lidar-directory"),
    ):
        if paths.get(key) is not None:
            command.extend((flag, str(paths[key])))
    return command


def _verified_provider_output(
    pose_path: Path,
    *,
    manifest_path: Path,
    initial: TransformResult,
    initial_digest: str,
    protocol: CameraLidarPoseInitializerBenchmarkProtocol,
) -> TransformResult:
    manifest = load_camera_lidar_correspondence_export(manifest_path)
    manifest_identity = (
        manifest.dataset_id,
        manifest.dataset_family,
        manifest.sequence_id,
        manifest.split_id,
        manifest.dataset_license_spdx,
    )
    protocol_identity = (
        protocol.dataset_id,
        protocol.dataset_family,
        protocol.sequence_id,
        protocol.partition,
        protocol.dataset_license_spdx,
    )
    if manifest_identity != protocol_identity:
        raise ValueError("provider manifest dataset identity differs from the frozen protocol")
    if [item.frame_id for item in manifest.frames] != protocol.frame_ids:
        raise ValueError("provider manifest frame IDs differ from the frozen protocol")
    provider = manifest.provider
    provider_identity = (
        provider.model,
        provider.source_repository,
        provider.source_commit,
        provider.license_spdx,
    )
    expected_provider_identity = (
        protocol.provider.model,
        protocol.provider.source_repository,
        protocol.provider.source_commit,
        protocol.provider.license_spdx,
    )
    if provider_identity != expected_provider_identity:
        raise ValueError("provider manifest identity differs from the frozen protocol")
    if (
        provider.checkpoint is None
        or provider.checkpoint.sha256 != protocol.provider.checkpoint_sha256
    ):
        raise ValueError("provider manifest checkpoint differs from the frozen protocol")
    inputs = manifest.provenance.input_sha256
    expected_inputs = {
        "adapter_source": protocol.provider.adapter_sha256,
        "initial_transform": initial_digest,
        "i2pnet_checkpoint": protocol.provider.checkpoint_sha256,
    }
    if manifest.provenance.generator_version != protocol.provider.adapter_version or any(
        inputs.get(name) != digest for name, digest in expected_inputs.items()
    ):
        raise ValueError("provider manifest provenance differs from the frozen protocol")
    if _command_option(manifest.provenance.command, "--neighbor-backend") != (
        protocol.provider.neighbor_backend
    ):
        raise ValueError("provider manifest neighbor backend differs from the frozen protocol")
    if _command_option(manifest.provenance.command, "--camera-stream") != (
        protocol.camera_stream
    ):
        raise ValueError("provider manifest camera stream differs from the frozen protocol")
    output = TransformResult.model_validate(read_mapping(pose_path))
    if (output.parent, output.child) != (initial.parent, initial.child):
        raise ValueError("provider pose frames differ from the initial transform")
    if output.provenance.role_in_comparison != "output":
        raise ValueError("provider pose provenance role must be output")
    notes = set(output.provenance.notes)
    expected = {
        f"initial transform sha256={initial_digest}",
        (f"source sha256 correspondence_manifest={_required_digest(manifest_path)}"),
        f"source sha256 adapter_source={protocol.provider.adapter_sha256}",
        f"source sha256 i2pnet_checkpoint={protocol.provider.checkpoint_sha256}",
    }
    if not expected.issubset(notes):
        raise ValueError("provider pose does not bind its initial and manifest digests")
    return output


def _command_option(command: list[str], option: str) -> str | None:
    try:
        position = command.index(option)
        return command[position + 1]
    except (ValueError, IndexError):
        return None


def _left_camera_frame_perturbation(
    reference: SE3,
    rotation_deg_xyz: list[float],
    translation_m_xyz: list[float],
) -> SE3:
    rotation = _euler_xyz_matrix(rotation_deg_xyz)
    quaternion = quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1))
    delta = SE3(
        tuple(float(value) for value in translation_m_xyz),
        quaternion,
    )
    return delta.compose(reference)


def _euler_xyz_matrix(rotation_deg_xyz: list[float]) -> FloatArray:
    values = np.asarray(rotation_deg_xyz, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("rotation_deg_xyz must contain three finite values")
    roll, pitch, yaw = np.radians(values)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _transform_errors(estimate: SE3, reference: SE3) -> tuple[float, float]:
    relative = reference.compose(estimate.inverse())
    rotation = 2.0 * math.degrees(
        math.acos(min(1.0, max(-1.0, abs(relative.rotation_quat_xyzw[3]))))
    )
    translation = math.dist(estimate.translation_m, reference.translation_m)
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


def _benchmark_trial(
    *,
    method_id: str,
    split_id: str,
    rotation_error_deg: float,
    translation_error_m: float,
    hit: bool,
    runtime_seconds: float,
    command: str,
    protocol_digest: str,
    input_digest: str,
    output_digest: str,
    notes: list[str],
) -> BenchmarkTrial:
    return BenchmarkTrial(
        method_id=method_id,
        split_id=split_id,
        status="success",
        metrics={
            "rotation_error_deg": rotation_error_deg,
            "translation_error_m": translation_error_m,
            "hit": float(hit),
        },
        runtime_seconds=runtime_seconds,
        provenance=BenchmarkTrialProvenance(
            command=command,
            config_sha256=protocol_digest,
            input_sha256=input_digest,
            output_sha256=output_digest,
            execution_host=platform.node(),
            notes=notes,
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
            interpretation="quaternion-geodesic error to the digest-pinned reference",
        ),
        BenchmarkMetricDefinition(
            name="translation_error_m",
            label="Translation error",
            unit="m",
            direction="lower",
            primary=True,
            interpretation="Euclidean translation error to the digest-pinned reference",
        ),
        BenchmarkMetricDefinition(
            name="hit",
            label="Strict hit rate",
            unit="fraction",
            direction="higher",
            primary=True,
            interpretation="strict protocol rotation and translation success indicator",
        ),
    ]


def _benchmark_methods(
    protocol: CameraLidarPoseInitializerBenchmarkProtocol,
) -> list[BenchmarkMethodDefinition]:
    return [
        BenchmarkMethodDefinition(
            method_id="unoptimized_initial",
            label="Frozen initial perturbation",
            implementation="calibrex_native",
            tool_name="calibrex",
            tool_version=I2PNET_POSE_BENCHMARK_VERSION,
            license_spdx="Apache-2.0",
        ),
        BenchmarkMethodDefinition(
            method_id="i2pnet_kitti_large",
            label="I2PNet KITTI-large pose initializer",
            implementation="subprocess",
            tool_name="I2PNet",
            tool_version=protocol.provider.model,
            source_commit=protocol.provider.source_commit,
            license_spdx=protocol.provider.license_spdx,
            adapter_version=protocol.provider.adapter_version,
        ),
    ]


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        raise ValueError(f"required file is missing or unreadable: {path}")
    return digest


def _combined_digest(values: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
