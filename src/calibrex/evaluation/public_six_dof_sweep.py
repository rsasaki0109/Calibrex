"""Prespecified public-data perturbations with isolated CPU worker processes."""

from __future__ import annotations

import math
import multiprocessing
import shlex
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

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
    aggregate_benchmark_definition,
)
from calibrex.core.provenance import git_commit
from calibrex.data.kitti_camera_lidar_problem import build_kitti_raw_camera_lidar_problem
from calibrex.pipelines.public_camera_lidar_demo import (
    PublicSixDofDemoResult,
    _digest,
    _required_digest,
    run_public_camera_lidar_six_dof_demo,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions
from calibrex.solvers.depth_to_depth_runtime import (
    ProjectionBackend,
    validate_cuda_cache_bytes,
    validate_frame_workers,
)
from calibrex.visualization.public_six_dof_sweep import write_public_six_dof_sweep_report

PUBLIC_SIX_DOF_SWEEP_VERSION = "calibrex.public_six_dof_sweep/v0.2"


@dataclass(frozen=True)
class PublicSixDofSweepCase:
    """One fixed local-Euler/camera-translation perturbation of the reference."""

    case_id: str
    rotation_deg_xyz: tuple[float, float, float]
    translation_m_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class PublicSixDofSweepResult:
    """All six case comparisons, including partial or failed attempts."""

    benchmark: BenchmarkArtifact
    demos: dict[str, PublicSixDofDemoResult]
    report_path: Path
    elapsed_seconds: float

    @property
    def successful(self) -> bool:
        """Require every attempt to converge, evaluate holdout, and match CPU."""

        return all(demo.successful for demo in self.demos.values())


def public_six_dof_axis_cases(
    rotation_deg: float = 0.5,
    translation_m: float = 0.5,
) -> tuple[PublicSixDofSweepCase, ...]:
    """Fix x+/x-/y+/y-/z+/z- before looking at any solver output."""

    if not all(math.isfinite(value) and value > 0 for value in (rotation_deg, translation_m)):
        raise ValueError("sweep rotation and translation magnitudes must be finite and positive")
    cases = []
    for axis, label in enumerate("xyz"):
        for sign, suffix in ((1, "positive"), (-1, "negative")):
            rotation = tuple(sign * rotation_deg if index == axis else 0.0 for index in range(3))
            translation = tuple(
                sign * translation_m if index == axis else 0.0 for index in range(3)
            )
            cases.append(PublicSixDofSweepCase(
                f"{label}-{suffix}", (rotation[0], rotation[1], rotation[2]),
                (translation[0], translation[1], translation[2]),
            ))
    return tuple(cases)


def run_public_six_dof_sweep(
    sequence_path: str | Path,
    depth_provider_path: str | Path,
    *,
    output_directory: str | Path,
    command: Sequence[str],
    candidate_backend: ProjectionBackend = "cuda",
    candidate_frame_workers: int = 1,
    workers: int = 1,
    repetitions: int = 1,
    rotation_deg: float = 0.5,
    translation_m: float = 0.5,
    cuda_cache_bytes: int | None = None,
    options: BorerSixDofOptions | None = None,
) -> PublicSixDofSweepResult:
    """Run six paired cases; CPU-only comparisons may use worker processes.

    A schema-valid plan with explicitly unexecuted trials is saved before
    optimization. Workers own their inputs/projectors and result directories.
    CUDA stays serial to avoid concurrent GPU contexts and timing contention.
    """

    started = time.perf_counter()
    validate_cuda_cache_bytes(candidate_backend, cuda_cache_bytes)
    validate_frame_workers(candidate_backend, candidate_frame_workers)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if candidate_backend == "cuda" and workers != 1:
        raise ValueError("CUDA comparisons require workers=1; CPU-only sweeps support processes")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if candidate_backend not in ("numpy", "numba_cpu", "cuda"):
        raise ValueError("unsupported candidate backend")
    if not command or not all(item.strip() for item in command):
        raise ValueError("sweep provenance requires a non-empty command argv")
    settings = options or BorerSixDofOptions()
    cases = public_six_dof_axis_cases(rotation_deg, translation_m)
    if rotation_deg > settings.rotation_bound_deg or translation_m > settings.translation_bound_m:
        raise ValueError("sweep magnitude exceeds the solver search bound")
    output = Path(output_directory).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty; existing evidence is preserved")
    source = build_kitti_raw_camera_lidar_problem(
        sequence_path, depth_provider_path,
        rotation_bound_deg=settings.rotation_bound_deg,
        translation_bound_m=settings.translation_bound_m,
        command=tuple(command),
    )
    ordered = sorted(source.observations, key=lambda item: int(item.frame_id))
    if len(ordered) < 25:
        raise ValueError("public six-DoF sweep requires at least 25 frozen depth frames")
    ids = [ordered[(index * (len(ordered) - 1) + 12) // 24].frame_id for index in range(25)]
    holdout_ids = ids[4::5]
    fit_ids = [item for item in ids if item not in holdout_ids]
    solver_settings = {
        field.name: getattr(settings, field.name)
        for field in fields(settings)
        if field.name not in ("d2d", "projector", "projection_backend")
    }
    solver_settings["d2d"] = asdict(settings.d2d)
    config_digest = _digest({
        "version": PUBLIC_SIX_DOF_SWEEP_VERSION,
        "cases": [asdict(case) for case in cases],
        "solver_settings": solver_settings,
        "candidate_backend": candidate_backend,
        "candidate_frame_workers": candidate_frame_workers,
        "workers": workers,
        "repetitions": repetitions,
        "cuda_cache_bytes": cuda_cache_bytes,
    })
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "sweep.source.problem.json"
    source.save(source_path)
    source_digest = _required_digest(source_path)
    candidate_id = "numpy_repeat" if candidate_backend == "numpy" else candidate_backend
    splits = [
        BenchmarkSplit(
            split_id=f"{case.case_id}/repeat-{repetition + 1:02d}", seed=0,
            fit_count=20, holdout_count=5,
            fit_ids_sha256=_digest(fit_ids), holdout_ids_sha256=_digest(holdout_ids),
        ) for case in cases for repetition in range(repetitions)
    ]
    protocol = BenchmarkProtocol(
        protocol_id=PUBLIC_SIX_DOF_SWEEP_VERSION, dataset_id=source.dataset_id,
        dataset_source_sha256=source.provenance.source_sha256, data_license="CC-BY-NC-SA-3.0",
        split_policy="Six fixed signed-axis initializations on the same uniform 20-fit/5-holdout "
        "provider frames; repetitions are timing observations, not independent recovery trials.",
        splits=splits,
        initial_estimate_policy=f"Dataset reference plus local Euler rotation +/-{rotation_deg:g} "
        f"deg and camera-frame translation +/-{translation_m:g} m, on x, y, z in that order. "
        "Controlled prior; reference is never the optimization target.",
        tuning_policy="All cases/settings fixed before execution; no holdout tuning or filtering.",
        failure_policy="Keep failed/partial attempts; never substitute a backend "
        "or discard a case.",
    )
    methods = [
        BenchmarkMethodDefinition(
            method_id=method_id, label=label, implementation="calibrex_native",
            tool_name="Calibrex", tool_version=__version__, source_commit=git_commit(),
        ) for method_id, label in (
            ("numpy", "NumPy CPU reference"), (candidate_id, f"{candidate_backend} candidate")
        )
    ]
    provenance = BenchmarkProvenance(
        generator="calibrex.evaluation.public_six_dof_sweep",
        generator_version=PUBLIC_SIX_DOF_SWEEP_VERSION, git_commit=git_commit(),
        command=shlex.join(command), data_verified=True,
        source_artifacts={
            source_path.name: source_digest,
            "depth_provider": _required_digest(Path(depth_provider_path)),
            Path(__file__).name: _required_digest(Path(__file__)),
            "public_six_dof_sweep_report.py": _required_digest(
                Path(__file__).parents[1] / "visualization/public_six_dof_sweep.py"
            ),
        },
    )
    plan = BenchmarkDefinition(
        benchmark_id=source.problem_id + "-signed-axis-sweep", title="Public six-DoF sweep plan",
        protocol=protocol, methods=methods, reference_method_id="numpy", bootstrap_samples=1,
        metrics=[BenchmarkMetricDefinition(
            name="numerical_parity", label="Exact parity", unit="ratio", direction="higher",
            interpretation="Checked only within each case, including its repetitions.",
        )],
        trials=[BenchmarkTrial(
            method_id=method.method_id, split_id=split.split_id, status="failed",
            failure_reason="not executed; frozen plan entry",
            provenance=BenchmarkTrialProvenance(
                command=shlex.join(command), config_sha256=config_digest,
                input_sha256=source_digest,
                notes=["Planning record only; input hash binds the verified unperturbed source.",
                       f"CPU worker processes: {workers}; CUDA requires one worker."],
            ),
        ) for split in splits for method in methods],
        limitations=["This is a prespecified plan, not completed execution evidence."],
        provenance=provenance,
    )
    plan_path = output / "sweep.plan.definition.json"
    plan.save(plan_path)
    tasks = [
        (str(Path(sequence_path).resolve()), str(Path(depth_provider_path).resolve()),
         str(output / case.case_id), tuple(command), candidate_backend, repetitions,
         cuda_cache_bytes, settings, case, candidate_frame_workers) for case in cases
    ]
    if workers == 1:
        demos = dict(_run_sweep_case(task) for task in tasks)
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(cases)), mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            demos = dict(executor.map(_run_sweep_case, tasks))
    elapsed = time.perf_counter() - started
    trials = []
    source_artifacts = {**provenance.source_artifacts, plan_path.name: _required_digest(plan_path)}
    template = demos[cases[0].case_id].benchmark
    for case in cases:
        demo = demos[case.case_id]
        if demo.benchmark.protocol.dataset_source_sha256 != protocol.dataset_source_sha256:
            raise ValueError("sweep dataset binding changed between cases")
        for index, split in enumerate(demo.benchmark.protocol.splits):
            planned = splits[cases.index(case) * repetitions + index]
            if (split.fit_ids_sha256, split.holdout_ids_sha256) != (
                planned.fit_ids_sha256, planned.holdout_ids_sha256
            ):
                raise ValueError("sweep fit/holdout split changed between cases")
            for trial in demo.benchmark.trials:
                if trial.split_id == split.split_id:
                    trial = trial.model_copy(deep=True)
                    trial.split_id = planned.split_id
                    trial.provenance.notes += [
                        f"Case directory: {case.case_id}",
                        f"Frozen sweep plan SHA256: {source_artifacts[plan_path.name]}",
                        f"CPU worker processes: {workers}; "
                        "invocations inside each case are serial.",
                    ]
                    trials.append(trial)
        for name in ("comparison.definition.json", "comparison.benchmark.json"):
            source_artifacts[f"{case.case_id}/{name}"] = _required_digest(
                output / case.case_id / name
            )
    definition = BenchmarkDefinition(
        benchmark_id=plan.benchmark_id, title="KITTI raw signed-axis CPU/candidate six-DoF sweep",
        protocol=protocol, metrics=template.metrics, methods=template.methods,
        trials=trials, reference_method_id="numpy", bootstrap_samples=1,
        limitations=[
            "Six reference-derived axis perturbations on one development drive; not independent "
            "metrology, a complete recovery basin, or multiple independent datasets.",
            "Different initializations can follow different traces; parity is checked within "
            "each case only. Cross-case timing summaries mix different search workloads.",
            f"CPU comparisons use {workers} worker process(es); each owns its arrays/projectors. "
            "Parallel timings include contention and are not isolated serial latencies. "
            "First invocations are included; no separate warm-up or cold-process guarantee.",
            "Convergence is optimizer termination, not an accuracy acceptance gate. "
            "Bootstrap fields are not statistical confidence for these fixed cases.",
            *("Case-level limitation: " + note for note in template.limitations),
        ],
        provenance=provenance.model_copy(update={"source_artifacts": source_artifacts}, deep=True),
    )
    definition.save(output / "sweep.definition.json")
    benchmark = aggregate_benchmark_definition(definition)
    benchmark.save(output / "sweep.benchmark.json")
    report = write_public_six_dof_sweep_report(
        output, benchmark=benchmark, demos=demos,
        cases=[(case.case_id, case.rotation_deg_xyz, case.translation_m_xyz) for case in cases],
        workers=workers, elapsed_seconds=elapsed,
    )
    return PublicSixDofSweepResult(benchmark, demos, report, elapsed)


def _run_sweep_case(
    task: tuple[
        str, str, str, tuple[str, ...], ProjectionBackend, int, int | None,
        BorerSixDofOptions, PublicSixDofSweepCase, int,
    ],
) -> tuple[str, PublicSixDofDemoResult]:
    sequence, provider, output, command, backend, repetitions, cache, settings, case, frames = task
    demo = run_public_camera_lidar_six_dof_demo(
        sequence, provider, output_directory=output, command=command,
        candidate_backend=backend, repetitions=repetitions, cuda_cache_bytes=cache,
        candidate_frame_workers=frames,
        options=settings, rotation_offset_deg=case.rotation_deg_xyz,
        translation_offset_m=case.translation_m_xyz,
    )
    return case.case_id, demo
