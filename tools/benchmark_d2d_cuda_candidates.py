"""Compare candidate batching against the pinned, already accelerated CUDA solver.

Both the reference solver and CUDA adapter are loaded from their exact Git
blobs. Every invocation prepares a fresh backend; only an explicitly selected
verified input snapshot is reused. Complete numeric payloads must match the
supplied frozen NumPy oracle, including all trace entries and holdout metrics.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import platform
import re
import statistics
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

from calibrex import __version__
from calibrex.core.benchmark import (
    BenchmarkDefinition,
    BenchmarkMethodDefinition,
    BenchmarkMetricDefinition,
    BenchmarkProvenance,
    BenchmarkTrial,
    BenchmarkTrialProvenance,
    aggregate_benchmark_definition,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import load_result
from calibrex.pipelines import camera_lidar_six_dof as pipeline
from calibrex.pipelines.camera_lidar_input_snapshot import CameraLidarSixDofInputSnapshot
from calibrex.pipelines.public_camera_lidar_demo import _digest, _numerical_payload

REFERENCE_COMMIT = "2017b3a3bb2742430170bff8ef842d7be9e5d9a7"
CASES = ("x-positive", "x-negative", "y-positive", "y-negative", "z-positive", "z-negative")
_SOURCES = (
    "src/calibrex/solvers/borer_depth_to_depth_solver.py",
    "src/calibrex/solvers/cuda_depth_to_depth_adapter.py",
    "src/calibrex/solvers/borer_six_dof_solver.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sweep-root", type=Path)
    selection.add_argument("--problem", type=Path)
    parser.add_argument("--oracle-result", type=Path)
    parser.add_argument("--protocol-template", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-commit", default=REFERENCE_COMMIT)
    parser.add_argument("--input-mode", choices=("verified_snapshot", "fresh_files"),
                        default="verified_snapshot")
    return parser


def _load_reference(commit: str, output: Path) -> tuple[ModuleType, ModuleType]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("baseline commit must be a full lowercase Git SHA-1")
    modules = []
    for index, source in enumerate(_SOURCES):
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{source}"], cwd=Path(__file__).resolve().parents[1],
        )
        path = output/f"baseline.{Path(source).name}.txt"
        path.write_bytes(content)
        name = f"_calibrex_cuda_reference_{index}"
        loader = importlib.machinery.SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        modules.append(module)
    evaluator, adapter, solver = modules
    # The pinned solver's objective imports must also use the pinned evaluator;
    # otherwise a future native arithmetic change could silently alter the reference.
    solver.evaluate_depth_to_depth_mi = evaluator.evaluate_depth_to_depth_mi
    solver.resolve_depth_to_depth_options = evaluator.resolve_depth_to_depth_options
    adapter._rotation_matrix = evaluator._rotation_matrix
    adapter._empty_projection = evaluator._empty_projection
    return adapter, solver


@contextmanager
def _reference_execution(adapter: ModuleType, solver: ModuleType) -> Iterator[None]:
    factory, solver_class = pipeline.create_depth_pair_projector, pipeline.BorerSixDofSolver

    def reference_factory(backend: str, observations: Any, **kwargs: Any) -> Any:
        assert backend == "cuda"
        budget = kwargs.get("cuda_cache_bytes")
        projector = adapter.CudaDepthPairProjector(
            observations, cache_bytes=256*1024**2 if budget is None else budget,
        )
        return projector, projector.identity()

    try:
        pipeline.create_depth_pair_projector = reference_factory
        pipeline.BorerSixDofSolver = solver.BorerSixDofSolver
        yield
    finally:
        pipeline.create_depth_pair_projector, pipeline.BorerSixDofSolver = factory, solver_class


def main() -> None:
    args = _parser().parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.baseline_commit):
        raise ValueError("baseline commit must be a full lowercase Git SHA-1")
    if args.problem and (args.oracle_result is None or args.protocol_template is None):
        raise ValueError("--problem requires --oracle-result and --protocol-template")
    if args.output_dir.exists():
        raise ValueError("output directory already exists; preserve previous measurements")
    cases = (
        [(name, args.sweep_root/name/"demo.problem.json",
          args.sweep_root/name/"numpy.result.json") for name in CASES]
        if args.sweep_root else [("single", args.problem, args.oracle_result)]
    )
    template_path = args.protocol_template or args.sweep_root/CASES[0]/"comparison.definition.json"
    template = BenchmarkDefinition.model_validate_json(template_path.read_bytes())
    args.output_dir.mkdir(parents=True)
    adapter, solver = _load_reference(args.baseline_commit, args.output_dir)
    source_hashes = {
        source: sha256_path(Path(source)) for source in (
            *_SOURCES, "src/calibrex/solvers/cuda_depth_to_depth_batch.py",
            "src/calibrex/pipelines/camera_lidar_six_dof.py", str(Path(__file__)),
        )
    }
    command = " ".join(sys.argv)
    protocol = template.protocol.model_copy(deep=True)
    protocol.protocol_id = "public_kitti_cuda_candidates/v0.1"
    protocol.splits = []
    protocol.failure_policy = "Retain all numerical mismatches, nonconvergence, and invocations."
    rows, trials, snapshots = [], [], {}
    for case_index, (case, problem, oracle_path) in enumerate(cases):
        oracle = load_result(oracle_path)
        expected = _numerical_payload(oracle)
        holdout = oracle.run.provenance["configuration"]["holdout_frame_ids"]
        snapshot = (CameraLidarSixDofInputSnapshot(problem)
                    if args.input_mode == "verified_snapshot" else None)
        snapshots[case] = snapshot.provenance() if snapshot else None
        source_hashes[f"{case}.problem"] = sha256_path(problem)
        source_hashes[f"{case}.oracle"] = sha256_path(oracle_path)
        for repetition in range(1, args.repeats+1):
            split_id = f"{case}-repeat-{repetition:02d}"
            protocol.splits.append(template.protocol.splits[0].model_copy(
                update={"split_id": split_id}, deep=True,
            ))
            methods = ["baseline_cuda", "candidate_cuda"]
            if (case_index+repetition) % 2 == 0:
                methods.reverse()
            for method in methods:
                kwargs = {
                    "projection_backend": "cuda", "input_snapshot": snapshot,
                    "holdout_frame_ids": holdout, "command": sys.argv,
                }
                if method == "baseline_cuda":
                    with _reference_execution(adapter, solver):
                        result = pipeline.run_camera_lidar_six_dof_estimate(problem, **kwargs)
                else:
                    result = pipeline.run_camera_lidar_six_dof_estimate(problem, **kwargs)
                parity = _numerical_payload(result) == expected
                solved = result.run.provenance["solver_result"]
                success = parity and solved["status"] == "converged"
                timing = result.run.provenance["timing_seconds"]
                pipeline_seconds = sum(timing.values())
                result.run.provenance["candidate_batch_benchmark"] = {
                    "baseline_commit": args.baseline_commit, "method": method,
                    "repetition": repetition, "execution_order": methods,
                    "source_sha256": source_hashes.copy(), "complete_numerical_parity": parity,
                    "input_mode": args.input_mode,
                }
                output = args.output_dir/f"{case}.{method}.repeat-{repetition:02d}.result.json"
                result.save(output)
                metrics = {
                    "solve_seconds": timing["solve"], "pipeline_seconds": pipeline_seconds,
                    "numerical_parity": float(parity),
                    "evaluation_count": solved["evaluation_count"],
                }
                trials.append(BenchmarkTrial(
                    method_id=method, split_id=split_id, status="success" if success else "failed",
                    runtime_seconds=pipeline_seconds, metrics=metrics,
                    failure_reason=None if success else "Numerical mismatch or nonconvergence",
                    provenance=BenchmarkTrialProvenance(
                        command=command, config_sha256=result.run.config_sha256,
                        input_sha256=sha256_path(problem), output_sha256=sha256_path(output),
                        execution_host=platform.platform(), notes=[
                            f"Result: {output.name}",
                            f"Numerical payload SHA256: {_digest(expected)}",
                            f"Input mode: {args.input_mode}; fresh backend/solver each invocation.",
                            "No separate warm-up; compiled disk/process caches persist.",
                            "Snapshot preparation, if selected, is outside per-run phases and "
                            "reported separately; charge once per case for repeated usage.",
                            f"Execution order: {methods}; repetition: {repetition}",
                        ],
                    ),
                ))
                row = {
                    "case": case, "method": method, "repetition": repetition, "parity": parity,
                    **metrics, "transfers": result.run.provenance["projection_transfer_statistics"],
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    for source in _SOURCES:
        filename = f"baseline.{Path(source).name}.txt"
        source_hashes[filename] = sha256_path(args.output_dir/filename)
    definition = BenchmarkDefinition(
        benchmark_id="public_kitti_cuda_candidates",
        title="Existing CUDA vs bounded candidate CUDA",
        protocol=protocol, reference_method_id="baseline_cuda", bootstrap_samples=1,
        methods=[BenchmarkMethodDefinition(
            method_id=name, label=label, implementation="calibrex_native", tool_name="Calibrex",
            tool_version=__version__, source_commit=commit,
        ) for name, label, commit in (
            ("baseline_cuda", "Pinned existing CUDA v0.3", args.baseline_commit),
            ("candidate_cuda", "Conservative candidate batches CUDA v0.4", git_commit()),
        )],
        metrics=[BenchmarkMetricDefinition(
            name=name, label=label, unit=unit, direction=direction,
            primary=name == "solve_seconds", interpretation=interpretation,
        ) for name, label, unit, direction, interpretation in (
            ("solve_seconds", "Solve", "s", "lower", "Includes bounded search preparation."),
            ("pipeline_seconds", "Pipeline", "s", "lower", "Sum of all recorded pipeline phases."),
            ("numerical_parity", "Full numerical parity", "boolean", "higher",
             "Full numeric payload equality with the same frozen NumPy oracle."),
            ("evaluation_count", "Evaluations", "count", "lower", "Unchanged search budget/trace."),
        )],
        trials=trials, limitations=[*template.limitations,
            "Shared host, alternating method order, three repeats by default; no speed guarantee.",
            "This compares new implementation against an already accelerated GPU baseline.",
            "Input mode and separately charged snapshot preparation are explicit. Fresh-file "
            "timings must not be inferred from verified-snapshot invocations.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools.benchmark_d2d_cuda_candidates", generator_version="v0.1",
            command=command, git_commit=git_commit(),
            data_verified=all(trial.status == "success" for trial in trials),
            source_artifacts=source_hashes,
        ),
    )
    definition.save(args.output_dir/"comparison.definition.json")
    aggregate_benchmark_definition(definition).save(args.output_dir/"comparison.benchmark.json")
    medians = {case: {method: {
        metric: statistics.median(row[metric] for row in rows
                                  if row["case"] == case and row["method"] == method)
        for metric in ("solve_seconds", "pipeline_seconds")
    } for method in ("baseline_cuda", "candidate_cuda")} for case, _, _ in cases}
    summary = {"input_mode": args.input_mode, "snapshots": snapshots,
               "case_medians": medians, "rows": rows, "source_sha256": source_hashes}
    (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(medians, indent=2), flush=True)
    if not definition.provenance.data_verified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
