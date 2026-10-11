"""Measure fresh-file verification overlap against a pinned sequential pipeline."""

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
import time
from pathlib import Path
from types import ModuleType

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
from calibrex.pipelines.camera_lidar_six_dof import run_camera_lidar_six_dof_estimate
from calibrex.pipelines.public_camera_lidar_demo import _digest, _numerical_payload

REFERENCE_COMMIT = "9180b4c610f8a32e2b3ba328458cf530594e136f"
CASES = ("x-positive", "x-negative", "y-positive", "y-negative", "z-positive", "z-negative")
SOURCES = (
    "src/calibrex/core/provenance.py",
    "src/calibrex/data/depth.py",
    "src/calibrex/evaluation/borer_rotation_benchmark.py",
    "src/calibrex/pipelines/camera_lidar_six_dof.py",
)


def _reference(commit: str, output: Path) -> ModuleType:
    modules = []
    for index, source in enumerate(SOURCES):
        raw = subprocess.check_output(["git", "show", f"{commit}:{source}"])
        path = output / f"baseline.{Path(source).name}.txt"
        path.write_bytes(raw)
        name = f"_calibrex_input_reference_{index}"
        loader = importlib.machinery.SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        modules.append(module)
    provenance, depth, problem_loader, pipeline = modules
    depth.sha256_path = provenance.sha256_path
    problem_loader.sha256_path = provenance.sha256_path
    problem_loader.verify_depth_provider_files = depth.verify_depth_provider_files
    problem_loader.DepthProviderArtifact = depth.DepthProviderArtifact
    pipeline.load_borer_problem = problem_loader.load_borer_problem
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-commit", default=REFERENCE_COMMIT)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if not re.fullmatch(r"[0-9a-f]{7,40}", args.baseline_commit):
        raise ValueError("baseline commit must be a lowercase Git commit hash")
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{args.baseline_commit}^{{commit}}"],
        text=True,
    ).strip()
    if args.output_dir.exists():
        raise ValueError("output directory exists; preserve previous measurements")
    args.output_dir.mkdir(parents=True)
    reference = _reference(commit, args.output_dir)
    cases = args.case or CASES
    template = BenchmarkDefinition.model_validate_json(
        (args.sweep_root / CASES[0] / "comparison.definition.json").read_bytes(),
    )
    protocol = template.protocol.model_copy(deep=True)
    protocol.protocol_id = "public_kitti_input_overlap/v0.1"
    protocol.splits = []
    rows, trials = [], []
    source_paths = (
        *SOURCES,
        "src/calibrex/solvers/borer_depth_to_depth_solver.py",
        "src/calibrex/solvers/borer_six_dof_solver.py",
        "src/calibrex/solvers/cuda_depth_to_depth_adapter.py",
        "src/calibrex/solvers/cuda_depth_to_depth_batch.py",
        str(Path(__file__)),
    )
    sources = {source: sha256_path(Path(source)) for source in source_paths}
    sources.update(
        {path.name: sha256_path(path) for path in args.output_dir.glob("baseline.*.txt")}
    )
    methods = {
        "sequential": reference.run_camera_lidar_six_dof_estimate,
        "buffered_sequential": run_camera_lidar_six_dof_estimate,
        "checkpoint_overlap": run_camera_lidar_six_dof_estimate,
    }
    for case_index, case in enumerate(cases):
        root = args.sweep_root / case
        problem = root / "demo.problem.json"
        oracle = load_result(root / "numpy.result.json")
        expected = _numerical_payload(oracle)
        sources[f"problem:{case}"] = sha256_path(problem)
        sources[f"oracle:{case}"] = sha256_path(root / "numpy.result.json")
        for repetition in range(1, args.repeats + 1):
            split = f"{case}-repeat-{repetition:02d}"
            protocol.splits.append(
                template.protocol.splits[0].model_copy(
                    update={"split_id": split},
                    deep=True,
                )
            )
            order = list(methods)
            position = (case_index + repetition - 1) % len(order)
            order = order[position:] + order[:position]
            for method in order:
                started = time.perf_counter()
                kwargs = (
                    {"overlap_input_verification": False} if method == "buffered_sequential" else {}
                )
                result = methods[method](
                    problem,
                    projection_backend="cuda",
                    command=sys.argv,
                    holdout_frame_ids=oracle.run.provenance["configuration"]["holdout_frame_ids"],
                    **kwargs,
                )
                wall = time.perf_counter() - started
                parity = _numerical_payload(result) == expected
                success = parity and result.run.provenance["solver_result"]["status"] == "converged"
                phases = result.run.provenance["timing_seconds"]
                output = args.output_dir / f"{case}.{method}.repeat-{repetition:02d}.result.json"
                result.run.provenance["input_overlap_benchmark"] = {
                    "method": method,
                    "baseline_commit": commit,
                    "source_sha256": sources.copy(),
                    "repetition": repetition,
                    "execution_order": order,
                    "complete_numerical_parity": parity,
                    "wall_seconds": wall,
                    "numerical_payload_sha256": _digest(expected),
                }
                result.save(output)
                metrics = {
                    "wall_seconds": wall,
                    "pipeline_seconds": sum(phases.values()),
                    "solve_seconds": phases["solve"],
                    "numerical_parity": float(parity),
                }
                trials.append(
                    BenchmarkTrial(
                        method_id=method,
                        split_id=split,
                        status="success" if success else "failed",
                        runtime_seconds=wall,
                        metrics=metrics,
                        failure_reason=None if success else "Numerical mismatch or nonconvergence",
                        provenance=BenchmarkTrialProvenance(
                            command=" ".join(sys.argv),
                            config_sha256=result.run.config_sha256,
                            input_sha256=sha256_path(problem),
                            output_sha256=sha256_path(output),
                            execution_host=platform.platform(),
                            notes=[
                            f"Output: {output.name}; numerical payload "
                            f"SHA256: {_digest(expected)}",
                                f"Order: {order}; no separate warmup; caches persist; shared host.",
                            "Every invocation re-verifies all external files, "
                            "including checkpoint.",
                                "Wall includes verification completion and result construction; "
                                "serialization and aggregation are excluded.",
                            ],
                        ),
                    )
                )
                row = {
                    "case": case,
                    "method": method,
                    "repetition": repetition,
                    **metrics,
                    "phases": phases,
                    "verification": result.run.provenance["input_verification"],
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    definition = BenchmarkDefinition(
        benchmark_id="public_kitti_input_overlap",
        title="Fresh verification overlap",
        protocol=protocol,
        reference_method_id="sequential",
        bootstrap_samples=1,
        methods=[
            BenchmarkMethodDefinition(
                method_id=method,
                label=method,
                implementation="calibrex_native",
                tool_name="Calibrex",
                tool_version=__version__,
                source_commit=source_commit,
            )
            for method, source_commit in (
                ("sequential", commit),
                ("buffered_sequential", git_commit()),
                ("checkpoint_overlap", git_commit()),
            )
        ],
        metrics=[
            BenchmarkMetricDefinition(
                name=name,
                label=name,
                unit=unit,
                direction=direction,
                primary=name == "wall_seconds",
                interpretation=interpretation,
            )
            for name, unit, direction, interpretation in (
                (
                    "wall_seconds",
                    "s",
                    "lower",
                    "Full function call including all verification and output construction",
                ),
                (
                    "pipeline_seconds",
                    "s",
                    "lower",
                    "Nonoverlapping phases; final checkpoint wait charged once",
                ),
                ("solve_seconds", "s", "lower", "Unchanged CUDA v0.4 solver in both methods"),
                (
                    "numerical_parity",
                    "ratio",
                    "higher",
                    "Exact full historical NumPy payload equality",
                ),
            )
        ],
        trials=trials,
        limitations=[
            f"One public sequence, {len(cases)} frozen case(s), "
            f"{args.repeats} repetitions per method.",
            "Same CUDA v0.4 solver/backend; no input snapshot or persistent verification cache.",
            "No separate warmup; existing disk/process/OS caches persist; host is shared.",
            "All attempts are retained. Repeated timings are not independent accuracy trials "
            "or statistical confidence. No portable speed guarantee.",
            *template.limitations[2:],
        ],
        provenance=BenchmarkProvenance(
            generator="tools.benchmark_camera_lidar_input_overlap",
            generator_version="v0.1",
            git_commit=git_commit(),
            command=" ".join(sys.argv),
            source_artifacts=sources,
            data_verified=all(trial.status == "success" for trial in trials),
        ),
    )
    definition.save(args.output_dir / "comparison.definition.json")
    aggregate_benchmark_definition(definition).save(args.output_dir / "comparison.benchmark.json")
    medians = {
        case: {
            method: {
                metric: statistics.median(
                    row[metric] for row in rows if row["case"] == case and row["method"] == method
                )
                for metric in ("wall_seconds", "pipeline_seconds", "solve_seconds")
            }
            for method in methods
        }
        for case in cases
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps({"medians": medians, "rows": rows}, indent=2)
    )
    print(json.dumps(medians, indent=2), flush=True)
    if not definition.provenance.data_verified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
