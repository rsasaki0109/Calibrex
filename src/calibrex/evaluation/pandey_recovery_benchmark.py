"""Shared recovery benchmark for the native Camera-LiDAR I2I solver."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np

from calibrex import __version__
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
    BenchmarkTrialStatus,
    aggregate_benchmark_definition,
)
from calibrex.core.exceptions import DatasetError
from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.core.provenance import git_commit
from calibrex.data.kitti import (
    read_camera_projection_model,
    read_png_luminance,
    read_velodyne_bin,
    read_velodyne_to_camera_transform,
)
from calibrex.data.kitti_benchmark import (
    KITTIBenchmarkInputManifest,
    load_kitti_benchmark_input,
    verify_kitti_benchmark_input,
)
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.pandey_mutual_information_solver import (
    MutualInformationCameraModel,
    MutualInformationObservation,
    PandeyMutualInformationOptions,
    PandeyMutualInformationResult,
    PandeyMutualInformationSolver,
)


@dataclass(frozen=True)
class KITTII2IBenchmarkData:
    """Verified KITTI observations and the dataset reference transform."""

    manifest: KITTIBenchmarkInputManifest
    observations: tuple[MutualInformationObservation, ...]
    reference_transform_camera_lidar: SE3


@dataclass(frozen=True)
class _RecoveryCase:
    case_id: str
    initial_transform_camera_lidar: SE3


def load_kitti_i2i_benchmark_data(
    input_manifest_path: str | Path,
    *,
    sequence_path: str | Path | None = None,
    max_points_per_frame: int = 2_000,
) -> KITTII2IBenchmarkData:
    """Load the digest-verified fixed KITTI frame set as I2I observations."""

    if max_points_per_frame < 64:
        raise ValueError("max_points_per_frame must be at least 64")
    manifest = load_kitti_benchmark_input(input_manifest_path)
    sequence = Path(sequence_path or manifest.source_sequence_path)
    verify_kitti_benchmark_input(manifest, sequence_path=sequence)
    projection = read_camera_projection_model(sequence, camera="image_02")
    t_unrectified_camera_lidar = read_velodyne_to_camera_transform(sequence)
    if projection is None or t_unrectified_camera_lidar is None:
        raise DatasetError("KITTI camera projection or Velodyne-camera transform is missing")

    p = projection.projection_matrix
    fx, fy = p[0], p[5]
    if fx <= 0.0 or fy <= 0.0:
        raise DatasetError("KITTI camera projection has non-positive focal lengths")
    t_rectified_camera_unrectified_camera = SE3(
        (p[3] / fx, p[7] / fy, 0.0),
        quaternion_xyzw_from_rotation_matrix(projection.rectification_matrix),
    )
    reference = t_rectified_camera_unrectified_camera.compose(t_unrectified_camera_lidar)

    observations: list[MutualInformationObservation] = []
    for frame_id in manifest.frame_ids:
        image_path = sequence / "image_02" / "data" / f"{frame_id}.png"
        lidar_path = sequence / "velodyne_points" / "data" / f"{frame_id}.bin"
        decoded = read_png_luminance(image_path)
        if decoded is None:
            raise DatasetError(f"unsupported KITTI PNG input: {image_path}")
        points = read_velodyne_bin(lidar_path)
        if not points:
            raise DatasetError(f"empty KITTI Velodyne input: {lidar_path}")
        sampled = _sample_points(points, max_points_per_frame)
        camera = MutualInformationCameraModel(
            width=decoded.width,
            height=decoded.height,
            fx=fx,
            fy=fy,
            cx=p[2],
            cy=p[6],
        )
        observations.append(
            MutualInformationObservation(
                frame_id=frame_id,
                luminance=np.asarray(decoded.rows, dtype=float),
                lidar_points=np.asarray([item[:3] for item in sampled], dtype=float),
                reflectivity=np.asarray([item[3] for item in sampled], dtype=float),
                camera=camera,
            )
        )
    return KITTII2IBenchmarkData(manifest, tuple(observations), reference)


def run_kitti_i2i_recovery_benchmark(
    data: KITTII2IBenchmarkData,
    *,
    command: str,
    max_iterations: int = 20,
    holdout_ratio: float = 0.2,
    split_seed: int = 0,
) -> tuple[BenchmarkDefinition, BenchmarkArtifact]:
    """Compare baseline and coarse I2I recovery on shared perturbation cases."""

    base_options = PandeyMutualInformationOptions(
        holdout_ratio=holdout_ratio,
        split_seed=split_seed,
        histogram_bins=32,
        min_projected_points=64,
        max_iterations=max_iterations,
        histogram_smoothing_backend="scalar",
    )
    coarse_options = replace(
        base_options,
        histogram_smoothing_backend="vectorized",
        coarse_search_translation_steps_m=(0.10, 0.05, 0.02),
        coarse_search_rotation_steps_deg=(10.0, 5.0, 2.0),
        coarse_search_sweeps_per_level=2,
    )
    cases = _recovery_cases(data.reference_transform_camera_lidar)
    sorted_observations = sorted(data.observations, key=lambda item: item.frame_id)
    fit_indices, holdout_indices = split_indices(
        len(sorted_observations),
        holdout_ratio,
        seed=split_seed,
    )
    fit_ids = [sorted_observations[index].frame_id for index in fit_indices]
    holdout_ids = [sorted_observations[index].frame_id for index in holdout_indices]
    splits = [
        BenchmarkSplit(
            split_id=case.case_id,
            seed=split_seed,
            fit_count=len(fit_ids),
            holdout_count=len(holdout_ids),
            fit_ids_sha256=_string_list_sha256(fit_ids),
            holdout_ids_sha256=_string_list_sha256(holdout_ids),
        )
        for case in cases
    ]
    methods = [
        BenchmarkMethodDefinition(
            method_id="pandey_i2i_scalar_bb_v01",
            label="Calibrex I2I scalar-reference BB",
            implementation="calibrex_native",
            tool_name="calibrex",
            tool_version=__version__,
            source_commit=git_commit(),
            license_spdx="Apache-2.0",
        ),
        BenchmarkMethodDefinition(
            method_id="pandey_i2i_vectorized_safe_coarse_bb_v03",
            label="Calibrex I2I vectorized safe coarse + BB",
            implementation="calibrex_native",
            tool_name="calibrex",
            tool_version=__version__,
            source_commit=git_commit(),
            license_spdx="Apache-2.0",
        ),
    ]
    trials: list[BenchmarkTrial] = []
    for method, options in zip(methods, (base_options, coarse_options), strict=True):
        for case in cases:
            trials.append(
                _run_trial(
                    method.method_id,
                    case,
                    data.observations,
                    data.reference_transform_camera_lidar,
                    options,
                    command=command,
                    input_sha256=data.manifest.input_sha256,
                )
            )

    metrics = _metric_definitions()
    provenance = BenchmarkProvenance(
        generator="calibrex.evaluation.pandey_recovery_benchmark",
        generator_version=__version__,
        git_commit=git_commit(),
        command=command,
        source_artifacts={
            "kitti_benchmark_input_sha256": data.manifest.input_sha256,
            "calibrex_python_source_sha256": _python_source_tree_sha256(),
        },
        data_verified=True,
    )
    definition = BenchmarkDefinition(
        benchmark_id="kitti_raw_0005_pandey_i2i_recovery_2026_07",
        title="KITTI raw 0005 native I2I recovery benchmark",
        protocol=BenchmarkProtocol(
            protocol_id="kitti_raw_0005_i2i_fixed20_recovery/v0.1",
            dataset_id=data.manifest.dataset_id,
            dataset_source_sha256=data.manifest.input_sha256,
            data_license="KITTI raw data terms; raw files are not redistributed",
            split_policy="one shared seeded frame holdout reused for every perturbation",
            splits=splits,
            initial_estimate_policy=(
                "six prespecified signed 0.10 m or 10 deg single-axis perturbations"
            ),
            tuning_policy="fixed options before holdout scoring; no per-case tuning",
            failure_policy="retain every solver failure in the denominator",
        ),
        metrics=metrics,
        methods=methods,
        trials=trials,
        reference_method_id="pandey_i2i_scalar_bb_v01",
        bootstrap_samples=2_000,
        bootstrap_seed=71,
        limitations=[
            "KITTI distributed calibration is a dataset reference, not independent metrology.",
            "I2I uses LiDAR reflectivity and image luminance and can be scene-dependent.",
            "Runtime is host-dependent and is reported only for within-run comparison.",
        ],
        provenance=provenance,
    )
    return definition, aggregate_benchmark_definition(definition)


def _run_trial(
    method_id: str,
    case: _RecoveryCase,
    observations: tuple[MutualInformationObservation, ...],
    reference: SE3,
    options: PandeyMutualInformationOptions,
    *,
    command: str,
    input_sha256: str,
) -> BenchmarkTrial:
    start = perf_counter()
    result = PandeyMutualInformationSolver().solve(
        observations,
        case.initial_transform_camera_lidar,
        options,
    )
    runtime = perf_counter() - start
    transform = result.transform_camera_lidar
    status: BenchmarkTrialStatus = "success" if transform is not None else "failed"
    metrics = _trial_metrics(result, reference) if transform is not None else {}
    result_payload = result.as_dict()
    coarse_search = result.coarse_search
    final_selection = coarse_search.final_selection if coarse_search is not None else None
    coarse_notes = (
        [
            f"coarse_initial_accepted={coarse_search.accepted}",
            "coarse_endpoint_accepted="
            f"{final_selection.candidate_accepted if final_selection is not None else False}",
        ]
        if coarse_search is not None
        else []
    )
    return BenchmarkTrial(
        method_id=method_id,
        split_id=case.case_id,
        status=status,
        metrics=metrics,
        runtime_seconds=runtime,
        failure_reason=None if transform is not None else result.reason,
        provenance=BenchmarkTrialProvenance(
            command=f"{command} --method {method_id} --case {case.case_id}",
            config_sha256=_mapping_sha256(asdict(options)),
            input_sha256=input_sha256,
            output_sha256=_mapping_sha256(result_payload),
            execution_host=platform.platform(),
            notes=[
                f"solver_status={result.status}",
                f"solver_reason={result.reason}",
                f"histogram_smoothing_backend={options.histogram_smoothing_backend}",
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                *coarse_notes,
            ],
        ),
    )


def _trial_metrics(
    result: PandeyMutualInformationResult,
    reference: SE3,
) -> dict[str, float]:
    assert result.transform_camera_lidar is not None
    transform = result.transform_camera_lidar
    translation_error = math.dist(transform.translation_m, reference.translation_m)
    rotation_error = _rotation_error_deg(transform, reference)
    return {
        "translation_error_m": translation_error,
        "rotation_error_deg": rotation_error,
        "holdout_normalized_mutual_information": (
            result.holdout_evaluation.normalized_mutual_information
        ),
        "recovered": float(translation_error <= 0.05 and rotation_error <= 1.0),
    }


def _metric_definitions() -> list[BenchmarkMetricDefinition]:
    return [
        BenchmarkMetricDefinition(
            name="translation_error_m",
            label="Translation error",
            unit="m",
            direction="lower",
            primary=True,
            interpretation="SE(3) translation distance to the KITTI dataset reference.",
        ),
        BenchmarkMetricDefinition(
            name="rotation_error_deg",
            label="Rotation error",
            unit="deg",
            direction="lower",
            primary=True,
            interpretation="Quaternion angular distance to the KITTI dataset reference.",
        ),
        BenchmarkMetricDefinition(
            name="holdout_normalized_mutual_information",
            label="Holdout normalized MI",
            unit="ratio",
            direction="higher",
            interpretation="I2I normalized mutual information on untouched frames.",
        ),
        BenchmarkMetricDefinition(
            name="recovered",
            label="Recovered",
            unit="bool",
            direction="higher",
            interpretation="One when error is at most 0.05 m and 1.0 deg.",
        ),
    ]


def _recovery_cases(reference: SE3) -> tuple[_RecoveryCase, ...]:
    cases: list[_RecoveryCase] = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        translation = [0.0, 0.0, 0.0]
        translation[axis_index] = 0.10
        cases.append(
            _RecoveryCase(
                f"{axis_name}_plus_0p10m",
                SE3(
                    (translation[0], translation[1], translation[2]),
                    (0.0, 0.0, 0.0, 1.0),
                ).compose(reference),
            )
        )
    for axis_index, axis_name in enumerate(("roll", "pitch", "yaw")):
        axis = [0.0, 0.0, 0.0]
        axis[axis_index] = 1.0
        cases.append(
            _RecoveryCase(
                f"{axis_name}_plus_10deg",
                SE3(
                    (0.0, 0.0, 0.0),
                    quaternion_xyzw_from_rotation_matrix(
                        _axis_angle_matrix(
                            (axis[0], axis[1], axis[2]),
                            math.radians(10.0),
                        ).reshape(-1)
                    ),
                ).compose(reference),
            )
        )
    return tuple(cases)


def _sample_points(
    points: list[tuple[float, float, float, float]],
    limit: int,
) -> list[tuple[float, float, float, float]]:
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=int)
    return [points[int(index)] for index in indices]


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(float(np.dot(left.rotation_quat_xyzw, right.rotation_quat_xyzw)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def _axis_angle_matrix(
    axis: tuple[float, float, float],
    angle: float,
) -> np.ndarray:
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    x, y, z = vector
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return cast(
        np.ndarray,
        np.asarray(
            np.eye(3)
            + math.sin(angle) * skew
            + (1.0 - math.cos(angle)) * (skew @ skew)
        ),
    )


def _string_list_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _mapping_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _python_source_tree_sha256() -> str:
    """Digest the executed Python source, including uncommitted changes."""

    source_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()
