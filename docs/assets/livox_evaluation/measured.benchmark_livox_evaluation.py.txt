"""Measure fresh public Livox calibration subprocesses and full result parity.

The pinned baseline replaces only the Livox reader/evaluator modules. All
registration solvers and artifact writers use the same checkout in both methods.
No input, output, report, control or external backend is skipped to save time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

BASELINE = "c6fd98b7b5af73be41fa01b5e4938773f4696fb1"
MODULES = ("calibrex.data.livox", "calibrex.evaluation.lidar")
INPUT_DIGESTS = {
    "base_horizon_100432.pcd": "19b3864cbee76b785d207fcb3fe48f6f506d142d8233f842d0a87fe4330fe407",
    "target_horizon_100538.pcd": "61a74eb8b8ff4eeba4ca0ff824a4046ae220e947e8156f27d93da452538f5b6f",
}
SOURCES = (
    "src/calibrex/data/livox.py",
    "src/calibrex/evaluation/lidar.py",
    "src/calibrex/pipelines/calibrate.py",
    "src/calibrex/solvers/native_registration_comparison_solver.py",
    "src/calibrex/solvers/robust_point_to_point_icp_solver.py",
    "src/calibrex/solvers/chen_medioni_point_to_plane_icp_solver.py",
    "src/calibrex/solvers/registration_adapters.py",
    "tools/benchmark_livox_evaluation.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_result(payload, output_dir: Path):
    # Only volatile run identity, timestamps and destination paths change.
    # Preserve all numerical values, grades, reason strings and provenance.
    payload = json.loads(json.dumps(payload))
    payload["run"]["id"] = "${RUN_ID}"
    payload["run"]["created_at"] = "${CREATED_AT}"
    payload["run"]["provenance"]["generated_at"] = "${GENERATED_AT}"

    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            return (
                value.replace(str(output_dir), "${RUN_OUTPUT}")
                .replace(output_dir.as_posix(), "${RUN_OUTPUT}")
                .replace("\\", "/")
            )
        return value

    return visit(payload)


def _worker(args) -> None:
    if args.worker == "baseline":
        from importlib.abc import MetaPathFinder
        from importlib.machinery import SourceFileLoader

        class ArchivedFinder(MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname not in MODULES:
                    return None
                source = args.output_dir.parent / (
                    f"baseline.{fullname.rsplit('.', 1)[1]}.py.txt"
                )
                return importlib.util.spec_from_loader(
                    fullname, SourceFileLoader(fullname, str(source)),
                )

        # Use normal import order, including data/__init__.py's eager exports.
        sys.meta_path.insert(0, ArchivedFinder())
    from calibrex.core.config import load_config
    from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration

    result_filename = load_config(args.config).outputs.result
    started = time.perf_counter()
    result = run_calibration(args.config, CalibrationRunOptions(output_dir=args.output_dir))
    pipeline_seconds = time.perf_counter() - started
    if result is None:
        raise ValueError("calibration returned no result")
    summary = {
        "pipeline_seconds": pipeline_seconds,
        "loaded_evaluation_modules": {name: sys.modules[name].__file__ for name in MODULES},
        "result": _normalized_result(result.model_dump(mode="json"), args.output_dir),
        "result_sha256": _sha(args.output_dir / result_filename),
        "artifacts": {
            str(path.relative_to(args.output_dir)): _sha(path)
            for path in sorted(args.output_dir.rglob("*"))
            if path.is_file()
        },
    }
    (args.output_dir / "measurement.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/public_datasets/livox_horizon_horizon_pcd_sample/icp_comparison_config.yaml"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-commit", default=BASELINE)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--worker", choices=("baseline", "batched"), help=argparse.SUPPRESS)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    args.output_dir = args.output_dir.resolve()
    if args.worker:
        _worker(args)
        return
    if args.repetitions < 1 or args.omp_threads < 1:
        raise ValueError("repetitions and OMP threads must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be new or empty")
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
    from calibrex.core.config import load_config
    from calibrex.core.provenance import git_commit

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = subprocess.check_output(
        ["git", "rev-parse", f"{args.baseline_commit}^{{commit}}"],
        text=True,
    ).strip()
    source_commit = git_commit()
    bindings = {path: _sha(Path(path)) for path in SOURCES}
    for name in MODULES:
        path = "src/" + name.replace(".", "/") + ".py"
        target = args.output_dir / f"baseline.{name.rsplit('.', 1)[1]}.py.txt"
        target.write_bytes(subprocess.check_output(["git", "show", f"{baseline}:{path}"]))
        bindings[str(target)] = _sha(target)
    config_digest = _sha(args.config)
    dataset = Path(load_config(args.config).dataset.path)
    inputs = {str(path): _sha(path) for path in sorted(dataset.rglob("*.pcd"))}
    if len(inputs) != 2:
        raise ValueError("benchmark requires the complete public two-file Livox sample")
    if {Path(path).name: digest for path, digest in inputs.items()} != INPUT_DIGESTS:
        raise ValueError("public Livox input bytes do not match the pinned capture")
    input_digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS=str(args.omp_threads))
    trials = []
    anchor = None
    for repeat in range(1, args.repetitions + 1):
        order = ["baseline", "batched"] if repeat % 2 else ["batched", "baseline"]
        for method in order:
            directory = args.output_dir / f"{method}.repeat-{repeat:02d}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                method,
                "--config",
                str(args.config),
                "--output-dir",
                str(directory),
            ]
            if inputs != {path: _sha(Path(path)) for path in inputs}:
                raise ValueError("inputs changed before calibration")
            print(f"START {repeat} {method}", flush=True)
            started = time.perf_counter()
            with (args.output_dir / f"{method}.repeat-{repeat:02d}.log").open(
                "w",
                encoding="utf-8",
            ) as log:
                subprocess.run(
                    command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
            elapsed = time.perf_counter() - started
            row = json.loads((directory / "measurement.json").read_bytes())
            expected_modules = {
                name: str(
                    args.output_dir / f"baseline.{name.rsplit('.', 1)[1]}.py.txt"
                    if method == "baseline"
                    else Path("src", *name.split(".")).with_suffix(".py").resolve()
                )
                for name in MODULES
            }
            if row["loaded_evaluation_modules"] != expected_modules:
                raise ValueError("baseline/candidate module binding mismatch")
            canonical = json.dumps(
                row["result"], sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            if anchor is None:
                anchor = canonical
            matches = anchor == canonical
            trials.append(
                BenchmarkTrial(
                    method_id=method,
                    split_id=f"repeat-{repeat:02d}",
                    status="success" if matches else "failed",
                    runtime_seconds=elapsed,
                    metrics={
                        "subprocess_wall_seconds": elapsed,
                        "pipeline_seconds": row["pipeline_seconds"],
                        "complete_result_matches": float(matches),
                    },
                    provenance=BenchmarkTrialProvenance(
                        command=subprocess.list2cmdline(command),
                        config_sha256=config_digest,
                        input_sha256=input_digest,
                        output_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                        execution_host=platform.platform(),
                        notes=[
                            json.dumps(
                                {
                                    "repeat": repeat,
                                    "order": order,
                                    "result_sha256": row["result_sha256"],
                                    "artifacts": row["artifacts"],
                                    "loaded_evaluation_modules": row["loaded_evaluation_modules"],
                                },
                                sort_keys=True,
                            ),
                            canonical,
                        ],
                    ),
                )
            )
            print(f"DONE {repeat} {method} {elapsed:.6f}s matches={matches}", flush=True)
    if (
        git_commit() != source_commit
        or bindings != {path: _sha(Path(path)) for path in bindings}
        or config_digest != _sha(args.config)
        or inputs != {path: _sha(Path(path)) for path in inputs}
    ):
        raise ValueError("source, configuration or input bytes changed during benchmark")
    definition = BenchmarkDefinition(
        benchmark_id="livox-complete-pipeline-evaluation",
        title="Public Livox full calibration subprocess with scoped batched evaluation",
        reference_method_id="baseline",
        bootstrap_samples=1,
        protocol=BenchmarkProtocol(
            protocol_id="livox-fresh-subprocess-v1",
            dataset_id="livox_horizon_horizon_pcd_sample",
            dataset_source_sha256=input_digest,
            data_license="Official public Livox sample terms",
            split_policy="Repeated same complete pair; not independent accuracy trials",
            splits=[
                BenchmarkSplit(
                    split_id=f"repeat-{i:02d}",
                    seed=7,
                    fit_count=23921,
                    holdout_count=23666,
                    fit_ids_sha256=input_digest,
                    holdout_ids_sha256=input_digest,
                )
                for i in range(1, args.repetitions + 1)
            ],
            initial_estimate_policy="Same official comparison config and all 24 probe controls",
            tuning_policy=(
                f"Reverse order every other repeat; child OMP_NUM_THREADS={args.omp_threads}"
            ),
            failure_policy="Preserve mismatches and all timings; no outlier exclusion",
        ),
        methods=[
            BenchmarkMethodDefinition(
                method_id=method,
                label=method,
                implementation="subprocess",
                tool_name="Calibrex",
                tool_version="0.5.1",
                adapter_version="livox-evaluation/v1",
                source_commit=baseline if method == "baseline" else source_commit,
            )
            for method in ("baseline", "batched")
        ],
        metrics=[
            BenchmarkMetricDefinition(
                name=name,
                label=name,
                unit="bool" if name.endswith("matches") else "s",
                direction="higher" if name.endswith("matches") else "lower",
                primary=name == "subprocess_wall_seconds",
                interpretation=(
                    "Full result equality after explicit volatile-field normalization"
                    if name.endswith("matches") else "Fresh process wall or its pipeline subset"
                ),
            )
            for name in trials[0].metrics
        ],
        trials=trials,
        limitations=[
            "Fresh interpreters, no warmup; OS caches persist and unrelated user work continues.",
            "Wall includes startup, all reads, inspection, 24 controls, native ICP, Open3D GICP, "
            "NDT availability check and result/report/artifact saves. Parent audits are excluded.",
            "Pipeline timer starts at run_calibration entry; it is a subset of subprocess wall.",
            "Full result parity normalizes only run ID/timestamps and destination paths.",
            "Only two Livox modules are replaced for baseline; all other code is shared.",
            "The single pair is limited geometry evidence; quality failures remain failures.",
            "Timings are descriptive ranges, not confidence or portable speedup estimates.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools.benchmark_livox_evaluation",
            generator_version="v0.1",
            git_commit=source_commit,
            command=subprocess.list2cmdline([sys.executable, __file__, *arguments]),
            data_verified=True,
            source_artifacts={**bindings, **inputs, str(args.config): config_digest},
        ),
    )
    definition.save(args.output_dir / "pipeline.definition.json")
    aggregate_benchmark_definition(definition).save(args.output_dir / "pipeline.benchmark.json")
    if any(trial.status != "success" for trial in trials):
        raise ValueError("complete result parity failed; all measurements retained")


if __name__ == "__main__":
    main()
