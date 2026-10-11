"""Measure CUDA kernel stream spans alongside ordinary D2D evaluation latency.

This is an instrumented diagnostic, not a solver speedup or accuracy benchmark.
CUDA events can include stream scheduling and host enqueue gaps; their spans are
not pure kernel execution times and must not be subtracted from wall time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.evaluation.borer_rotation_benchmark import load_borer_problem
from calibrex.solvers.borer_depth_to_depth_solver import (
    evaluate_depth_to_depth_mi,
    resolve_depth_to_depth_options,
)
from calibrex.solvers.depth_to_depth_runtime import DEFAULT_CUDA_CACHE_BYTES

_STAGES = {
    "_initialize": "initialize_stream_seconds",
    "_project": "project_stream_seconds",
    "_select": "select_stream_seconds",
    "_histogram": "histogram_stream_seconds",
}


@contextmanager
def _instrument(projector: Any, cuda: Any) -> Iterator[dict[str, list[tuple[Any, Any]]]]:
    events: dict[str, list[tuple[Any, Any]]] = {name: [] for name in _STAGES.values()}
    originals = {attribute: getattr(projector, attribute) for attribute in _STAGES}

    def wrap(kernel: Any, name: str) -> Any:
        def measured(*args: Any, **kwargs: Any) -> Any:
            start, end = cuda.Event(), cuda.Event()
            start.record()
            result = kernel(*args, **kwargs)
            end.record()
            events[name].append((start, end))
            return result

        return measured

    try:
        for attribute, name in _STAGES.items():
            setattr(projector, attribute, wrap(originals[attribute], name))
        yield events
    finally:
        for attribute, kernel in originals.items():
            setattr(projector, attribute, kernel)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-frame-id", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--poses-per-repeat", type=int, default=20)
    parser.add_argument("--warmup-poses", type=int, default=3)
    parser.add_argument("--cuda-cache-mib", type=int, default=DEFAULT_CUDA_CACHE_BYTES // 2**20)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.repeats < 1 or args.poses_per_repeat < 1 or args.warmup_poses < 0:
        raise ValueError("repeats/poses-per-repeat must be positive; warmup-poses non-negative")
    if args.cuda_cache_mib < 0:
        raise ValueError("cuda-cache-mib must be non-negative")
    # Keep --help and argument validation usable without the optional dependency.
    import cupy as cp

    from calibrex.solvers.cuda_depth_to_depth_adapter import CudaDepthPairProjector

    started = time.perf_counter()
    loaded = load_borer_problem(args.problem)
    loading_seconds = time.perf_counter() - started
    excluded = set(args.exclude_frame_id)
    unknown = excluded - {frame.frame_id for frame in loaded.observations}
    if unknown:
        raise ValueError(f"unknown excluded frame IDs: {sorted(unknown)}")
    observations = tuple(frame for frame in loaded.observations if frame.frame_id not in excluded)
    if not observations:
        raise ValueError("at least one frame must remain")
    options = resolve_depth_to_depth_options(observations)
    pose = loaded.problem.initial_transform_camera_lidar.as_se3()
    expected = evaluate_depth_to_depth_mi(observations, pose, options)
    started = time.perf_counter()
    projector = CudaDepthPairProjector(observations, cache_bytes=args.cuda_cache_mib * 2**20)
    # Finish preparation before timing ordinary or instrumented evaluations.
    cp.cuda.get_current_stream().synchronize()
    preparation_seconds = time.perf_counter() - started

    def evaluate() -> None:
        actual = evaluate_depth_to_depth_mi(observations, pose, options, projector=projector)
        if actual != expected:
            raise ValueError("complete evaluation differs from NumPy")

    for _ in range(args.warmup_poses):
        evaluate()
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config["pose"] = pose.as_dict()
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    frame_digest = hashlib.sha256(
        json.dumps([f.frame_id for f in observations]).encode()
    ).hexdigest()
    command = " ".join([sys.executable, *sys.argv])
    trials = []
    for repeat in range(args.repeats):
        metrics = {}
        order = ["ordinary", "instrumented"] if repeat % 2 == 0 else ["instrumented", "ordinary"]
        calls = dict.fromkeys(_STAGES.values(), 0)
        for mode in order:
            wall = 0.0
            spans = dict.fromkeys(_STAGES.values(), 0.0)
            for _ in range(args.poses_per_repeat):
                if mode == "ordinary":
                    started = time.perf_counter()
                    evaluate()
                    wall += time.perf_counter() - started
                else:
                    with _instrument(projector, cp.cuda) as events:
                        started = time.perf_counter()
                        evaluate()
                        wall += time.perf_counter() - started
                    # Host histogram downloads have already completed this stream.
                    # Query timing outside wall measurements; retain one pose's events.
                    for name, pairs in events.items():
                        calls[name] += len(pairs)
                        spans[name] += sum(cp.cuda.get_elapsed_time(a, b) for a, b in pairs) / 1000
            metrics[f"{mode}_wall_seconds"] = wall / args.poses_per_repeat
            if mode == "instrumented":
                metrics.update(
                    {name: value / args.poses_per_repeat for name, value in spans.items()}
                )
        metrics["numerical_parity"] = 1.0
        trials.append(
            BenchmarkTrial(
                method_id="cuda_profile",
                split_id=str(repeat),
                status="success",
                metrics=metrics,
                runtime_seconds=metrics["instrumented_wall_seconds"],
                provenance=BenchmarkTrialProvenance(
                    command=command,
                    config_sha256=config_digest,
                    input_sha256=loaded.problem_sha256,
                    notes=[
                        f"Execution order: {order}",
                        f"Kernel calls: {json.dumps(calls, sort_keys=True)}",
                    ],
                ),
            )
        )
        print(json.dumps({"repeat": repeat, "mean_seconds_per_pose": metrics}), flush=True)
    root = Path(__file__).resolve().parents[1]
    sources = {
        "problem": loaded.problem_sha256,
        "depth_provider": loaded.depth_provider_sha256,
    }
    for source in (
        Path(__file__).resolve(),
        root / "src/calibrex/solvers/cuda_depth_to_depth_adapter.py",
        root / "src/calibrex/solvers/borer_depth_to_depth_solver.py",
    ):
        sources[source.relative_to(root).as_posix()] = sha256_path(source) or ""
    names = list(trials[0].metrics)
    definition = BenchmarkDefinition(
        benchmark_id="d2d-cuda-stage-profile",
        title="CUDA D2D stream spans and observer overhead",
        protocol=BenchmarkProtocol(
            protocol_id="d2d-cuda-stage-profile/v0.1",
            dataset_id=loaded.problem.dataset_id,
            dataset_source_sha256=loaded.problem_sha256,
            data_license="input dataset-specific",
            split_policy=(
                "Repeated timing at the declared prior; no accuracy split. "
                "The holdout anchor is NumPy evaluation of the same pose."
            ),
            splits=[
                BenchmarkSplit(
                    split_id=str(i),
                    seed=i,
                    fit_count=len(observations),
                    holdout_count=1,
                    fit_ids_sha256=frame_digest,
                    holdout_ids_sha256=config_digest,
                )
                for i in range(args.repeats)
            ],
            initial_estimate_policy="Problem's declared initial pose; unchanged across all samples",
            tuning_policy="Native histogram settings; ordinary/instrumented order alternates",
            failure_policy="Abort on CUDA or evaluation mismatch; no partial success report",
        ),
        methods=[
            BenchmarkMethodDefinition(
                method_id="cuda_profile",
                label=projector.identity(),
                implementation="calibrex_native",
                tool_name="Calibrex",
                tool_version=__version__,
                source_commit=git_commit(),
            )
        ],
        metrics=[
            BenchmarkMetricDefinition(
                name=name,
                label=name,
                unit="ratio" if name == "numerical_parity" else "s",
                direction="higher" if name == "numerical_parity" else "lower",
                interpretation=(
                    "Exact full evaluation equality with NumPy"
                    if name == "numerical_parity"
                    else "Per pose; stream spans include scheduling/enqueue gaps and event overhead"
                    if "stream" in name
                    else "Host evaluation wall time including histogram transfer and MI"
                ),
            )
            for name in names
        ],
        trials=trials,
        bootstrap_samples=1,
        limitations=[
            "Instrumented stream spans are not pure kernel execution times. "
            "Do not subtract them from wall time or infer a speedup ceiling.",
            "Event recording changes scheduling/latency; ordinary and instrumented walls "
            "expose observer overhead.",
            "Single pose, shared host, warmed runtime. "
            "No accuracy, portability or confidence claim.",
            f"Loading/verification {loading_seconds}s and CUDA preparation "
            f"{preparation_seconds}s are excluded from samples.",
            f"Frames: {len(observations)}; "
            f"points: {sum(len(f.lidar_points) for f in observations)}; "
            f"samples per repeat: {args.poses_per_repeat}",
            f"Environment: Python {platform.python_version()}; "
            f"{platform.platform()}; {platform.processor()}",
            f"Transfer statistics: {json.dumps(projector.transfer_statistics(), sort_keys=True)}",
        ],
        provenance=BenchmarkProvenance(
            generator="tools.profile_d2d_cuda",
            generator_version="v0.1",
            git_commit=git_commit(),
            command=command,
            data_verified=True,
            source_artifacts=sources,
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    definition.save(args.output_dir / "profile.definition.json")
    aggregate_benchmark_definition(definition).save(args.output_dir / "profile.benchmark.json")


if __name__ == "__main__":
    main()
