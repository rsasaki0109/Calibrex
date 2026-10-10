"""Reproducible KITTI raw six-DoF demonstration with a fixed public-data split."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

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
    MetricDirection,
    aggregate_benchmark_definition,
)
from calibrex.core.camera_lidar_artifacts import CameraLidarArtifactProvenance
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import CalibrationResult, TransformEstimateProvenance, TransformQuality
from calibrex.data.depth import DepthProviderArtifact, load_depth_provider
from calibrex.data.kitti_camera_lidar_problem import build_kitti_raw_camera_lidar_problem
from calibrex.evaluation.borer_rotation_benchmark import load_borer_problem
from calibrex.pipelines.camera_lidar_six_dof import run_camera_lidar_six_dof_estimate
from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions, apply_local_se3_delta
from calibrex.solvers.depth_to_depth_runtime import ProjectionBackend, validate_cuda_cache_bytes
from calibrex.visualization.public_six_dof_demo import write_public_six_dof_report

PUBLIC_SIX_DOF_DEMO_VERSION = "calibrex.public_six_dof_demo/v0.2"
_SPLIT_ID = "uniform-25-fit20-holdout5"


@dataclass(frozen=True)
class PublicSixDofDemoResult:
    """Saved evidence, including incomplete executions and backend failures."""

    benchmark: BenchmarkArtifact
    results: dict[str, CalibrationResult]
    numerical_parity: bool
    report_path: Path
    trial_results: dict[tuple[str, str], CalibrationResult]

    @property
    def successful(self) -> bool:
        """Require two converged runs, evaluable holdout, and numerical parity."""

        return self.numerical_parity and all(
            trial.status == "success" for trial in self.benchmark.trials
        )


def run_public_camera_lidar_six_dof_demo(
    sequence_path: str | Path,
    depth_provider_path: str | Path,
    *,
    output_directory: str | Path,
    candidate_backend: ProjectionBackend = "cuda",
    repetitions: int = 1,
    cuda_cache_bytes: int | None = None,
    options: BorerSixDofOptions | None = None,
    rotation_offset_deg: tuple[float, float, float] = (0.5, 0.0, 0.0),
    translation_offset_m: tuple[float, float, float] = (0.5, 0.0, 0.0),
    command: Sequence[str],
) -> PublicSixDofDemoResult:
    """Compare NumPy with one explicitly selected backend on 20/5 KITTI frames.

    Dataset calibration is used only to construct a declared perturbed initial
    estimate and to measure reference differences. Both optimizations receive
    the same initial estimate, never the reference as their search target.
    Images must use the source-aligned rectified pinhole depth grid. No raw
    inputs are changed, no provider is trained, and no backend falls back.
    """

    validate_cuda_cache_bytes(candidate_backend, cuda_cache_bytes)
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if not command or not all(item.strip() for item in command):
        raise ValueError("demo provenance requires a non-empty command argv")
    if candidate_backend not in ("numpy", "numba_cpu", "cuda"):
        raise ValueError("unsupported candidate backend")
    settings = options or BorerSixDofOptions()
    settings_payload = {
        field.name: getattr(settings, field.name)
        for field in fields(settings)
        if field.name not in ("d2d", "projector", "projection_backend")
    }
    settings_payload["d2d"] = asdict(settings.d2d)
    _digest(settings_payload)
    if settings.objective not in ("mutual_information", "normalized_mutual_information"):
        raise ValueError("unsupported six-DoF objective")
    for name, offset, bound in (
        ("rotation offset", rotation_offset_deg, settings.rotation_bound_deg),
        ("translation offset", translation_offset_m, settings.translation_bound_m),
    ):
        if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
            raise ValueError(f"{name} must contain three finite values")
        if any(abs(value) > bound for value in offset):
            raise ValueError(f"{name} exceeds the search bound")
    output = Path(output_directory).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty; existing evidence is preserved")
    source = build_kitti_raw_camera_lidar_problem(
        sequence_path,
        depth_provider_path,
        rotation_bound_deg=settings.rotation_bound_deg,
        translation_bound_m=settings.translation_bound_m,
        command=tuple(command),
    )
    ordered = sorted(source.observations, key=lambda item: int(item.frame_id))
    if len(ordered) < 25:
        raise ValueError("public six-DoF demo requires at least 25 frozen depth frames")
    # Uniform positions in the supplied provider, not an assumed complete drive.
    selected = [ordered[(index * (len(ordered) - 1) + 12) // 24] for index in range(25)]
    frame_ids = [item.frame_id for item in selected]
    holdout_ids = frame_ids[4::5]
    fit_ids = [item for item in frame_ids if item not in holdout_ids]
    provider_path = Path(depth_provider_path).resolve()
    provider = load_depth_provider(provider_path)
    _validate_image_grids(provider, provider_path.parent, frame_ids)
    configuration = {
        "version": PUBLIC_SIX_DOF_DEMO_VERSION,
        "solver_options": settings_payload,
        "rotation_offset_deg_xyz": rotation_offset_deg,
        "translation_offset_m_xyz": translation_offset_m,
        "selected_frame_ids": frame_ids,
        "train_frame_ids": fit_ids,
        "holdout_frame_ids": holdout_ids,
        "candidate_backend": candidate_backend,
        "cuda_cache_bytes": cuda_cache_bytes,
        "repetitions": repetitions,
        "execution_order": "CPU first on odd-numbered repetitions; candidate first on even ones",
    }
    config_digest = _digest(configuration)
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "source.problem.json"
    source.save(source_path)
    initial = source.initial_transform_camera_lidar.model_copy(deep=True)
    perturbed = apply_local_se3_delta(
        source.reference_transform_camera_lidar.as_se3(), rotation_offset_deg, translation_offset_m
    )
    initial.translation_m = list(perturbed.translation_m)
    initial.rotation_quat_xyzw = list(perturbed.rotation_quat_xyzw)
    initial.covariance = None
    initial.quality = TransformQuality()
    initial.estimate_id = "controlled-public-demo-initial"
    initial.provenance = TransformEstimateProvenance(
        producer="slac_native",
        execution_mode="offline_batch",
        role_in_comparison="candidate",
        evidence_level="unknown",
        source="controlled perturbation of dataset reference",
        tool_name="Calibrex",
        tool_version=__version__,
        adapter_version=PUBLIC_SIX_DOF_DEMO_VERSION,
        source_commit=git_commit(),
        command=shlex.join(command),
        notes=[
            "Controlled demo initialization: dataset reference plus fixed local Euler offsets. "
            f"Rotation deg xyz={rotation_offset_deg}; camera-frame translation m xyz="
            f"{translation_offset_m}. This is not an independently measured prior."
        ],
    )
    problem = source.model_copy(
        update={
            "problem_id": source.problem_id + "-public-six-dof-demo",
            "observations": selected,
            "initial_transform_camera_lidar": initial,
            "provenance": CameraLidarArtifactProvenance(
                generator="calibrex.pipelines.public_camera_lidar_demo",
                generator_version=PUBLIC_SIX_DOF_DEMO_VERSION,
                git_commit=git_commit(),
                command=list(command),
                config_sha256=config_digest,
                source_sha256=_required_digest(source_path),
            ),
        },
        deep=True,
    )
    problem_path = output / "demo.problem.json"
    problem.save(problem_path)
    loaded = load_borer_problem(problem_path)
    candidate_id = candidate_backend if candidate_backend != "numpy" else "numpy_repeat"
    methods: list[tuple[str, ProjectionBackend]] = [
        ("numpy", "numpy"),
        (candidate_id, candidate_backend),
    ]
    split_ids = [
        _SPLIT_ID if repetitions == 1 else f"{_SPLIT_ID}-repeat-{index + 1:02d}"
        for index in range(repetitions)
    ]
    trial_results: dict[tuple[str, str], CalibrationResult] = {}
    failures: dict[tuple[str, str], str] = {}
    execution_keys: list[tuple[str, str]] = []
    result_files: dict[tuple[str, str], str] = {}
    result: CalibrationResult | None
    for index, split_id in enumerate(split_ids):
        ordered_methods = methods if index % 2 == 0 else list(reversed(methods))
        for method_id, backend in ordered_methods:
            key = (method_id, split_id)
            execution_keys.append(key)
            filename = (
                f"{method_id}.result.json"
                if index == 0
                else f"{method_id}.repeat-{index + 1:02d}.result.json"
            )
            result_files[key] = filename
            try:
                # Each invocation starts from the shared problem's declared
                # initial pose. No output is reused as a later initialization.
                result = run_camera_lidar_six_dof_estimate(
                    problem_path,
                    options=settings,
                    projection_backend=backend,
                    cuda_cache_bytes=cuda_cache_bytes if backend == "cuda" else None,
                    holdout_frame_ids=holdout_ids,
                    command=command,
                )
                result.run.provenance["public_demo_repetition"] = {
                    "number": index + 1,
                    "count": repetitions,
                    "execution_order": [item[0] for item in ordered_methods],
                    "shared_problem_sha256": loaded.problem_sha256,
                }
                result.save(output / filename)
                trial_results[key] = result
                if result.run.provenance["solver_result"]["status"] != "converged":
                    failures[key] = result.run.provenance["solver_result"]["stopping_reason"]
                elif result.metrics["final_d2d_mutual_information"].holdout is None:
                    failures[key] = "holdout has insufficient visible depth pairs"
            except (OSError, ValueError, RuntimeError, MemoryError) as exc:
                failures[key] = f"{type(exc).__name__}: {exc}"
    baseline = next(
        (
            _numerical_payload(result)
            for (method_id, _split_id), result in trial_results.items()
            if method_id == "numpy"
        ),
        None,
    )
    for key, result in trial_results.items():
        if baseline is not None and _numerical_payload(result) != baseline:
            failures[key] = (
                "CPU/candidate numerical mismatch in trace, poses, or metrics, "
                "including repetitions"
            )
    pair_parity = {
        split_id: baseline is not None
        and all(
            (method_id, split_id) in trial_results
            and _numerical_payload(trial_results[method_id, split_id]) == baseline
            for method_id, _backend in methods
        )
        for split_id in split_ids
    }
    parity = all(pair_parity.values())
    # Keep the original API's first-pair lookup and file names; all repetitions
    # are available by (method_id, split_id) in trial_results.
    results = {
        method_id: trial_results[method_id, split_ids[0]]
        for method_id, _backend in methods
        if (method_id, split_ids[0]) in trial_results
    }
    metric_definitions = [
        BenchmarkMetricDefinition(
            name=name,
            label=label,
            unit=unit,
            direction=cast(MetricDirection, direction),
            interpretation=interpretation,
        )
        for name, label, unit, direction, interpretation in (
            ("solve_seconds", "Solve time", "s", "lower", "One invocation; includes lazy JIT."),
            ("pipeline_seconds", "Pipeline time", "s", "lower", "Load, prepare, solve, holdout."),
            (
                "initial_rotation_difference_deg",
                "Initial rotation difference",
                "deg",
                "lower",
                "Controlled initial difference to dataset reference.",
            ),
            (
                "initial_translation_difference_m",
                "Initial translation difference",
                "m",
                "lower",
                "Controlled initial difference to dataset reference.",
            ),
            (
                "rotation_difference_deg",
                "Rotation difference",
                "deg",
                "lower",
                "Difference to dataset-provided calibration, not metrology error.",
            ),
            (
                "translation_difference_m",
                "Translation difference",
                "m",
                "lower",
                "Difference to dataset-provided calibration, not metrology error.",
            ),
            (
                "train_mi",
                "Training mutual information",
                "nat",
                "higher",
                "Fit-only frozen histogram ranges.",
            ),
            (
                "holdout_mi",
                "Holdout mutual information",
                "nat",
                "higher",
                "Prespecified five frames excluded from fit and histogram range selection.",
            ),
            (
                "numerical_parity",
                "Exact numerical parity",
                "ratio",
                "higher",
                "Full trace, output poses, metrics, and holdout evaluations match exactly.",
            ),
        )
    ]
    trials = []
    for key in execution_keys:
        method_id, split_id = key
        result = trial_results.get(key)
        metrics: dict[str, float] = {}
        if result is not None:
            timing = result.run.provenance["timing_seconds"]
            metrics.update(solve_seconds=timing["solve"], pipeline_seconds=sum(timing.values()))
            metrics["numerical_parity"] = float(pair_parity[split_id])
            initial_rotation, initial_translation = _reference_difference(
                initial.as_se3(),
                loaded.reference_transform_camera_lidar,
            )
            metrics.update(
                initial_rotation_difference_deg=initial_rotation,
                initial_translation_difference_m=initial_translation,
            )
            if result.transforms:
                rotation, translation = _reference_difference(
                    next(iter(result.transforms.values())).as_se3(),
                    loaded.reference_transform_camera_lidar,
                )
                metrics.update(
                    rotation_difference_deg=rotation, translation_difference_m=translation
                )
            final_mi = result.metrics["final_d2d_mutual_information"]
            if final_mi.train is not None:
                metrics["train_mi"] = final_mi.train
            if final_mi.holdout is not None:
                metrics["holdout_mi"] = final_mi.holdout
        trials.append(
            BenchmarkTrial(
                method_id=method_id,
                split_id=split_id,
                status="failed" if key in failures else "success",
                failure_reason=failures.get(key),
                metrics=metrics,
                runtime_seconds=metrics.get("pipeline_seconds"),
                provenance=BenchmarkTrialProvenance(
                    command=shlex.join(command),
                    config_sha256=config_digest,
                    input_sha256=loaded.problem_sha256,
                    output_sha256=sha256_path(output / result_files[key]),
                    notes=[
                        "Same controlled initial pose, problem bytes, and 20/5 split "
                        "for every run.",
                        f"Result file: {result_files[key]}",
                        "CPU first on odd-numbered repetitions; candidate first on even ones. "
                        "Each invocation verifies/loads inputs and prepares a fresh projector. "
                        "The first invocation is included; kernel caches persist "
                        "in this process.",
                    ],
                ),
            )
        )
    training = provider.provider.training
    limitations = [
        f"One sequence, one perturbation, {repetitions} invocation(s) per backend. "
        "Repetitions measure timing on the same frames, not independent recovery trials. "
        "Bootstrap intervals do not establish statistical confidence for this demo.",
        "All invocations, including the first, contribute to timing. There is no separate "
        "warm-up or cold-process guarantee; lazy compilation and process-wide caches "
        "affect latency.",
        "Optimizer convergence and CPU/GPU parity do not establish calibration accuracy. "
        "Dataset reference is not independent metrology; covariance/observability remain unknown.",
        "Only the supplied frozen provider frames are eligible; uniform selection is by "
        "position within that provider. Learned depth and frame synchronization can bias the fit.",
        "Provider training/evaluation overlap: "
        + (training.evaluation_overlap + ". " + training.evidence if training else "undeclared."),
        *provider.warnings,
    ]
    source_artifacts = {
        "source.problem.json": _required_digest(source_path),
        "demo.problem.json": loaded.problem_sha256,
        "depth_provider": loaded.depth_provider_sha256,
    }
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "visualization/public_six_dof_demo.py",
    ):
        source_artifacts[path.name] = _required_digest(path)
    preview_ids = [holdout_ids[index] for index in (0, 2, 4)]
    for observation in provider.observations:
        if observation.frame_id in preview_ids:
            source_artifacts[f"assets/frame-{observation.frame_id}.png"] = observation.image.sha256
    definition = BenchmarkDefinition(
        benchmark_id=problem.problem_id,
        title="KITTI raw six-DoF public-data demo",
        protocol=BenchmarkProtocol(
            protocol_id=PUBLIC_SIX_DOF_DEMO_VERSION,
            dataset_id=problem.dataset_id,
            dataset_source_sha256=source.provenance.source_sha256,
            data_license="CC-BY-NC-SA-3.0",
            split_policy="25 numeric-ordered, uniformly selected provider positions; every fifth "
            "frame is holdout (20 fit / 5 holdout), fixed before optimization.",
            splits=[
                BenchmarkSplit(
                    split_id=split_id,
                    seed=0,
                    fit_count=20,
                    holdout_count=5,
                    fit_ids_sha256=_digest(fit_ids),
                    holdout_ids_sha256=_digest(holdout_ids),
                )
                for split_id in split_ids
            ],
            initial_estimate_policy=initial.provenance.notes[0],
            tuning_policy="Same solver settings for both backends; no holdout tuning.",
            failure_policy="Preserve failed/partial trials and available standard results; "
            "never replace an unavailable candidate with CPU output.",
        ),
        metrics=metric_definitions,
        methods=[
            BenchmarkMethodDefinition(
                method_id=method_id,
                label="NumPy CPU reference"
                if method_id == "numpy"
                else ("NumPy CPU replay" if backend == "numpy" else f"{backend} candidate"),
                implementation="calibrex_native",
                tool_name="Calibrex",
                tool_version=__version__,
                source_commit=git_commit(),
            )
            for method_id, backend in methods
        ],
        trials=trials,
        reference_method_id="numpy",
        bootstrap_samples=1,
        limitations=limitations,
        provenance=BenchmarkProvenance(
            generator="calibrex.pipelines.public_camera_lidar_demo",
            generator_version=PUBLIC_SIX_DOF_DEMO_VERSION,
            git_commit=git_commit(),
            command=shlex.join(command),
            source_artifacts=source_artifacts,
            data_verified=True,
        ),
    )
    definition.save(output / "comparison.definition.json")
    benchmark = aggregate_benchmark_definition(definition)
    benchmark.save(output / "comparison.benchmark.json")
    report = write_public_six_dof_report(
        output,
        loaded=loaded,
        provider_base=provider_path.parent,
        benchmark=benchmark,
        results=trial_results,
        result_files=result_files,
        numerical_parity=parity,
        preview_ids=preview_ids,
    )
    return PublicSixDofDemoResult(benchmark, results, parity, report, trial_results)


def _validate_image_grids(
    provider: DepthProviderArtifact,
    base: Path,
    frame_ids: Sequence[str],
) -> None:
    for observation in provider.observations:
        if observation.frame_id not in frame_ids:
            continue
        transform, camera = observation.image_transform, observation.intrinsics
        if (
            camera.projection != "pinhole"
            or any(camera.distortion)
            or transform.scale_x != 1.0
            or transform.scale_y != 1.0
            or transform.crop_left_px != 0.0
            or transform.crop_top_px != 0.0
            or (transform.source_width, transform.output_width) != (camera.width, camera.width)
            or (transform.source_height, transform.output_height) != (camera.height, camera.height)
        ):
            raise ValueError("demo overlays require source-aligned rectified pinhole depth grids")
        path = base / observation.image.path
        with path.open("rb") as stream:
            header = stream.read(24)
        if len(header) != 24 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
            raise ValueError("demo source images must be PNG files with an IHDR header")
        if struct.unpack(">II", header[16:24]) != (camera.width, camera.height):
            raise ValueError("demo source PNG dimensions do not match the depth grid")


def _numerical_payload(result: CalibrationResult) -> dict[str, Any]:
    solver = dict(result.run.provenance["solver_result"])
    solver.pop("projection_backend", None)
    return {
        "solver": solver,
        "metrics": {name: value.model_dump(mode="json") for name, value in result.metrics.items()},
        "poses": {name: value.as_se3().as_dict() for name, value in result.transforms.items()},
        "initial_holdout": result.run.provenance["initial_holdout_evaluation"],
        "final_holdout": result.run.provenance["final_holdout_evaluation"],
        "status": result.run.status,
    }


def _reference_difference(candidate: SE3, reference: SE3) -> tuple[float, float]:
    dot = abs(
        sum(
            left * right
            for left, right in zip(
                candidate.rotation_quat_xyzw, reference.rotation_quat_xyzw, strict=True
            )
        )
    )
    return (
        math.degrees(2 * math.acos(min(1.0, dot))),
        math.dist(candidate.translation_m, reference.translation_m),
    )


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"cannot hash required artifact: {path}")
    return digest


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
