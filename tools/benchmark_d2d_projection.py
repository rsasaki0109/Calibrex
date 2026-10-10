"""Compare warmed D2D backends, including transfers and native histogram MI.

Use --problem for digest-verified real inputs, or a seeded synthetic workload.
Reports use the existing benchmark schemas; timing repetitions are paired and
are explicitly not train/holdout accuracy trials. CUDA calls return host arrays,
so perf_counter includes kernel execution and GPU/CPU synchronization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import numpy as np

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
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.evaluation.borer_rotation_benchmark import load_borer_problem
from calibrex.evaluation.borer_six_dof_benchmark import ProjectionBackend, _projection_runtime
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthPairProjector,
    DepthToDepthCameraModel,
    DepthToDepthObservation,
    evaluate_depth_to_depth_mi,
    resolve_depth_to_depth_options,
)
from calibrex.solvers.borer_six_dof_solver import (
    BorerSixDofOptions,
    BorerSixDofSolver,
    apply_local_se3_delta,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path)
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--projection", choices=("pinhole", "double_sphere", "mei"), default="pinhole"
    )
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("numpy", "numba_cpu", "cuda"),
        default=["numpy", "numba_cpu", "cuda"],
    )
    parser.add_argument("--solve-evaluations", type=int, default=0)
    parser.add_argument("--solver-repeats", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _timed(call: Callable[[], object]) -> tuple[float, object]:
    started = time.perf_counter()
    result = call()
    return time.perf_counter() - started, result


def _input_digest(observations: tuple[DepthToDepthObservation, ...]) -> str:
    digest = hashlib.sha256()
    for observation in observations:
        digest.update(observation.frame_id.encode())
        digest.update(json.dumps(asdict(observation.camera), sort_keys=True).encode())
        digest.update(observation.lidar_points.tobytes())
        digest.update(observation.depth_map.tobytes())
    return digest.hexdigest()


def _gpu_memory(projector: DepthPairProjector) -> dict[str, int]:
    """Sample retained CUDA allocations outside the timed region.

    Pool reservation persists after temporary arrays are released. Device-wide
    usage also includes the CUDA context and any other running applications;
    neither value is a sampled process peak.
    """
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    return {
        "cached_input_bytes": int(getattr(projector, "cached_input_bytes", 0)),
        "pool_used_bytes": pool.used_bytes(),
        "pool_reserved_bytes": pool.total_bytes(),
        "device_used_bytes": total - free,
        "device_total_bytes": total,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.repeats < 1 or args.solver_repeats < 1 or args.points < 4:
        raise ValueError("repeats/solver-repeats must be positive and points must be at least four")
    if args.width < 2 or args.height < 2:
        raise ValueError("image dimensions must exceed one pixel")
    if args.solve_evaluations != 0 and args.solve_evaluations < 13:
        raise ValueError("solve-evaluations must be zero or at least thirteen")
    backends = list(dict.fromkeys(args.backends))
    if "numpy" not in backends or len(backends) < 2:
        raise ValueError("select numpy and at least one other backend")
    rng = np.random.default_rng(args.seed)
    if args.problem:
        loaded = load_borer_problem(args.problem)
        observations = loaded.observations
        reference = loaded.reference_transform_camera_lidar
        input_digest = loaded.problem_sha256
        dataset = loaded.problem.dataset_id
    else:
        camera = DepthToDepthCameraModel(
            args.width,
            args.height,
            0.55 * args.width,
            0.55 * args.width,
            (args.width - 1) / 2.0,
            (args.height - 1) / 2.0,
            projection=args.projection,
            xi=0.35 if args.projection == "double_sphere" else 1.4,
            alpha=0.55,
            distortion=(0.01, 0.02, 0.001, -0.002),
        )
        observations = (
            DepthToDepthObservation(
                "synthetic",
                rng.uniform(1.0, 60.0, size=(args.height, args.width)),
                rng.uniform((-30.0, -15.0, -4.0), (30.0, 15.0, 60.0), size=(args.points, 3)),
                camera,
            ),
        )
        reference = SE3.identity()
        input_digest = _input_digest(observations)
        dataset = f"synthetic-{args.projection}-{args.points}-{args.width}x{args.height}"
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    options = resolve_depth_to_depth_options(observations)
    projectors: dict[str, DepthPairProjector] = {}
    identities = {}
    startup = {}
    gpu_memory = {}
    for backend in backends:
        started = time.perf_counter()
        projector, identity = _projection_runtime(cast(ProjectionBackend, backend), observations)
        setup = time.perf_counter() - started
        first_call, _ = _timed(
            lambda p=projector: evaluate_depth_to_depth_mi(
                observations,
                reference,
                options,
                projector=p,
            )
        )
        projectors[backend] = projector
        identities[backend] = identity
        startup[backend] = {"setup_seconds": setup, "first_evaluation_seconds": first_call}
        if backend == "cuda":
            gpu_memory["after_first_evaluation"] = _gpu_memory(projector)
        print(f"prepared {backend}: {identity}", flush=True)
    rows: dict[str, list[dict[str, float]]] = {backend: [] for backend in backends}
    for repeat in range(args.repeats):
        pose = apply_local_se3_delta(
            reference, tuple(rng.uniform(-1.0, 1.0, 3)), (0.12, -0.04, 0.08)
        )
        expected_pairs = tuple(
            projectors["numpy"](observation, pose) for observation in observations
        )
        expected_mi = evaluate_depth_to_depth_mi(observations, pose, options)
        # Rotate ordering to limit bias from thermal/clock changes over the run.
        order = backends[repeat % len(backends) :] + backends[: repeat % len(backends)]
        for backend in order:
            project = projectors[backend]
            projection_seconds, actual_pairs = _timed(
                lambda p=project, t=pose: tuple(p(observation, t) for observation in observations)
            )
            evaluation_seconds, actual_mi = _timed(
                lambda p=project, t=pose: evaluate_depth_to_depth_mi(
                    observations,
                    t,
                    options,
                    projector=p,
                )
            )
            for expected, actual in zip(expected_pairs, actual_pairs, strict=True):
                if (
                    actual.projected_count_before_visibility
                    != expected.projected_count_before_visibility
                ):
                    raise ValueError(f"projected point count mismatch: {backend}")
                for field in ("camera_depth", "lidar_range_m", "pixel_u", "pixel_v"):
                    np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
            if actual_mi != expected_mi:
                raise ValueError(f"MI mismatch: {backend}")
            rows[backend].append(
                {"projection_seconds": projection_seconds, "evaluation_seconds": evaluation_seconds}
            )
    if "cuda" in projectors:
        gpu_memory["after_projection_timings"] = _gpu_memory(projectors["cuda"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {"input": input_digest, "config": config_digest}
    root = Path(__file__).resolve().parents[1]
    for source in (
        Path(__file__).resolve(),
        root / "src/calibrex/solvers/cuda_depth_to_depth_adapter.py",
        root / "src/calibrex/solvers/borer_depth_to_depth_solver.py",
        root / "src/calibrex/solvers/borer_six_dof_solver.py",
    ):
        sources[str(source.relative_to(root))] = sha256_path(source) or ""
    command = " ".join([sys.executable, *sys.argv])
    provenance = BenchmarkProvenance(
        generator="tools.benchmark_d2d_projection",
        generator_version="v0.2",
        git_commit=git_commit(),
        command=command,
        source_artifacts=sources,
        data_verified=True,
    )
    grids = sorted(
        {(item.camera.width, item.camera.height, item.camera.projection) for item in observations}
    )
    notes = [
        "All timed projections and MI records exactly match the NumPy reference.",
        "Timing repetitions are not held-out calibration accuracy trials.",
        "Transfers and synchronization are included; compilation/preparation are separate.",
        f"Startup times: {json.dumps(startup, sort_keys=True)}",
        f"Environment: Python {platform.python_version()}; NumPy {np.__version__}; "
        f"{platform.platform()}; CPU {platform.processor()}",
        f"Workload: {len(observations)} frames; "
        f"{sum(len(item.lidar_points) for item in observations)} total points; "
        f"camera grids: {grids}",
        (
            "Digest-verified real inputs measure backend equivalence and latency; "
            "this timing experiment is not a calibration accuracy benchmark."
            if args.problem
            else "Synthetic random depth/points measure throughput, not recovery accuracy."
        ),
    ]
    if gpu_memory:
        notes.append(f"CUDA allocation snapshots (bytes): {json.dumps(gpu_memory, sort_keys=True)}")
        notes.append(
            "CUDA pool reservation includes reusable scratch allocations; device usage "
            "includes context/other applications. Snapshots are not process peak VRAM."
        )
    _save_report(
        args.output_dir, "projection", rows, identities, observations, dataset, provenance, notes
    )
    if args.solve_evaluations:
        solver_rows: dict[str, list[dict[str, float]]] = {backend: [] for backend in backends}
        settings = BorerSixDofOptions(max_evaluations=args.solve_evaluations, d2d=options)
        for repeat in range(args.solver_repeats):
            initial = apply_local_se3_delta(reference, (1.0, -0.5, 0.2), (0.10, 0.0, 0.0))
            results = {}
            order = backends[repeat % len(backends) :] + backends[: repeat % len(backends)]
            for backend in order:
                seconds, result = _timed(
                    lambda b=backend, pose=initial: BorerSixDofSolver().solve(
                        observations,
                        pose,
                        replace(
                            settings, projector=projectors[b], projection_backend=identities[b]
                        ),
                    )
                )
                results[backend] = result
                solver_rows[backend].append({"solver_seconds": seconds})
                print(
                    f"solver {repeat} {backend}: {seconds:.3f}s, "
                    f"{result.evaluation_count} evaluations",
                    flush=True,
                )
            expected = results["numpy"]
            for backend, result in results.items():
                if replace(result, projection_backend=expected.projection_backend) != expected:
                    raise ValueError(f"solver result/trace mismatch: {backend}")
        solver_notes = list(notes)
        solver_notes.append("All full solver results and traces exactly match NumPy.")
        if "cuda" in projectors:
            solver_notes.append(
                "CUDA allocation snapshot after solver timings (bytes): "
                + json.dumps(_gpu_memory(projectors["cuda"]), sort_keys=True)
            )
        _save_report(
            args.output_dir,
            "solver",
            solver_rows,
            identities,
            observations,
            dataset,
            provenance,
            solver_notes,
        )


def _save_report(
    output: Path,
    phase: str,
    rows: dict[str, list[dict[str, float]]],
    identities: dict[str, str],
    observations: tuple[DepthToDepthObservation, ...],
    dataset: str,
    provenance: BenchmarkProvenance,
    notes: list[str],
) -> None:
    digest = provenance.source_artifacts["input"]
    names = list(next(iter(rows.values()))[0])
    definition = BenchmarkDefinition(
        benchmark_id=f"d2d-{phase}-{dataset}",
        title=f"D2D {phase} backend latency: {dataset}",
        protocol=BenchmarkProtocol(
            protocol_id="d2d-backend-latency/v0.1",
            dataset_id=dataset,
            dataset_source_sha256=digest,
            data_license="synthetic Apache-2.0 or input dataset-specific",
            split_policy=(
                "paired timing repetitions; reference projection is the equivalence anchor, "
                "not accuracy holdout"
            ),
            splits=[
                BenchmarkSplit(
                    split_id=str(index),
                    seed=index,
                    fit_count=len(observations),
                    holdout_count=1,
                    fit_ids_sha256=digest,
                    holdout_ids_sha256=digest,
                )
                for index in range(len(rows["numpy"]))
            ],
            initial_estimate_policy="prespecified pose perturbations shared across backends",
            tuning_policy="FP64 and original histogram settings; no solver or metric retuning",
            failure_policy="abort and report any projection, MI or solver-trace mismatch",
        ),
        metrics=[
            BenchmarkMetricDefinition(
                name=name,
                label=name,
                unit="s",
                display_unit="ms",
                display_scale=1000.0,
                direction="lower",
                primary=True,
                interpretation="synchronized warm host-to-host latency",
            )
            for name in names
        ],
        methods=[
            BenchmarkMethodDefinition(
                method_id=backend,
                label=backend,
                implementation="adapter",
                tool_name=backend,
                tool_version=identity,
                license_spdx="Apache-2.0 AND MIT AND BSD-2-Clause",
            )
            for backend, identity in identities.items()
        ],
        trials=[
            BenchmarkTrial(
                method_id=backend,
                split_id=str(index),
                status="success",
                metrics=values,
                runtime_seconds=values[names[-1]],
                provenance=BenchmarkTrialProvenance(
                    command=provenance.command,
                    config_sha256=provenance.source_artifacts["config"],
                    input_sha256=digest,
                    execution_host=platform.node(),
                    notes=notes[:1],
                ),
            )
            for backend, samples in rows.items()
            for index, values in enumerate(samples)
        ],
        reference_method_id="numpy",
        bootstrap_samples=1000,
        bootstrap_seed=42,
        limitations=notes,
        provenance=provenance,
    )
    definition.save(output / f"{phase}.definition.json")
    benchmark = aggregate_benchmark_definition(definition)
    benchmark.save(output / f"{phase}.benchmark.json")
    print(
        json.dumps(
            {
                backend: {name: summary.metrics[name].distribution.median for name in names}
                for backend, summary in benchmark.method_summaries.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
