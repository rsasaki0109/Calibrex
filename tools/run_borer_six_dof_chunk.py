"""Run a contiguous, digest-preserving chunk of a six-DoF benchmark.

The native benchmark runner uses a thread pool for convenience.  This tool
keeps each worker process single-threaded and partitions the frozen protocol
outside the solver, which avoids low-level numerical-library failures while
retaining the full protocol digest in every trace.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import cast

from calibrex.core.camera_lidar_artifacts import (
    load_camera_lidar_benchmark_protocol,
)
from calibrex.evaluation.borer_rotation_benchmark import load_borer_problem
from calibrex.evaluation.borer_six_dof_benchmark import (
    ProjectionBackend,
    _candidate_trace,
    _execute_trial,
    _required_digest,
    _solver_options,
    _trace_solver_identity,
    _validated_trace,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one contiguous chunk of a frozen six-DoF protocol and "
            "write traces compatible with the full-protocol runner."
        )
    )
    parser.add_argument("problem", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument(
        "--projection-backend",
        choices=("numpy", "numba_cpu"),
        default="numpy",
    )
    return parser


def run_chunk(
    problem_path: Path,
    protocol_path: Path,
    *,
    trace_directory: Path,
    start: int,
    stop: int,
    projection_backend: ProjectionBackend = "numpy",
) -> None:
    """Execute ``[start, stop)`` from the full frozen protocol."""

    loaded = load_borer_problem(problem_path)
    protocol = load_camera_lidar_benchmark_protocol(protocol_path)
    protocol_digest = _required_digest(protocol_path)
    if protocol.degrees_of_freedom != "six_dof":
        raise ValueError("chunk runner requires a six_dof protocol")
    if protocol.problem_sha256 != loaded.problem_sha256:
        raise ValueError("protocol problem_sha256 does not match the problem")
    if protocol.frame_ids != [item.frame_id for item in loaded.problem.observations]:
        raise ValueError("protocol frame IDs do not match the problem")
    if not 0 <= start < stop <= len(protocol.perturbations):
        raise ValueError(
            f"chunk range must satisfy 0 <= start < stop <= {len(protocol.perturbations)}"
        )

    trace_directory.mkdir(parents=True, exist_ok=True)
    options = _solver_options(
        protocol,
        projection_backend=projection_backend,
    )
    expected_solver = _trace_solver_identity(options.projection_backend)
    command = " ".join([sys.executable, *sys.argv])
    for index in range(start, stop):
        perturbation = protocol.perturbations[index]
        trace_path = trace_directory / f"{perturbation.trial_id}.trace.yaml"
        if trace_path.is_file():
            _validated_trace(
                trace_path,
                perturbation=perturbation,
                problem_sha256=loaded.problem_sha256,
                protocol_sha256=protocol_digest,
                expected_solver=expected_solver,
            )
            print(f"cached {index} {perturbation.trial_id}", flush=True)
            continue

        started = time.perf_counter()
        execution = _execute_trial(
            perturbation,
            loaded=loaded,
            options=options,
        )
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
        print(
            f"saved {index} {perturbation.trial_id} "
            f"runtime={execution.runtime_seconds:.3f}s "
            f"elapsed={time.perf_counter() - started:.3f}s "
            f"status={trace.status}",
            flush=True,
        )


def main() -> None:
    """Parse arguments and execute one chunk."""

    args = _parser().parse_args()
    run_chunk(
        args.problem,
        args.protocol,
        trace_directory=args.trace_dir,
        start=args.start,
        stop=args.stop,
        projection_backend=cast(ProjectionBackend, args.projection_backend),
    )


if __name__ == "__main__":
    main()
