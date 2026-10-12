"""Fresh complete continuous-time public Livox runs with exact result parity.

Download and convert the GLIM fixture using the public-dataset tutorial first.
The adaptive scope substitutes adaptive_voxel; runtime replaces the seven
geometry/query/interpolation modules together from the baseline revision.
The wall timer includes interpreter startup, imports, input replay, all clock
candidates, extrinsic refinements, holdout diagnostics, input hashing and saves.
Parent provenance audits are outside that timer. No warmup or trial is discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calibrex.solvers.continuous_time_lidar_pair_solver import (
        ContinuousTimeLidarPairOptions,
        ContinuousTimeLidarPairProblem,
    )

BASELINE = "240ed2cf885f9bbcffc0be68a6afac1d51b29280"
MODULE = "calibrex.data.adaptive_voxel"
RUNTIME_MODULES = (
    "calibrex.data.livox", "calibrex.data.odometry_track",
    "calibrex.graph.lidar_point_to_plane", "calibrex.evaluation.lidar",
    "calibrex.solvers.continuous_time_lidar_pair_solver",
    "calibrex.data.adaptive_voxel", "calibrex.pipelines.online",
)
CONFIG = Path(
    "examples/public_datasets/glim_versatile/online_continuous_time_benchmark_config.yaml"
)
PUBLIC_ROOT = Path("data/public/glim_versatile")
REPLAY = PUBLIC_ROOT / "livox_rosbag2_gt_restamped"
CONVERSION_ARGUMENTS = (
    "tools/rosbag1_to_rosbag2_online_pair.py",
    "--src", (PUBLIC_ROOT / "livox.bag").as_posix(),
    "--dst", REPLAY.as_posix(),
    "--topic", "/livox/points",
    "--odom-source", "external-trajectory",
    "--trajectory-file", (PUBLIC_ROOT / "gt/livox.txt").as_posix(),
    "--trajectory-time-offset-s", "1655977530.260883",
    "--restamp-topic", "/livox/points",
    "--duplicate-topic", "/livox/points:/livox/points_copy",
)
SOURCES = (
    "src/calibrex/data/adaptive_voxel.py",
    "src/calibrex/data/odometry_track.py",
    "src/calibrex/graph/lidar_point_to_plane.py",
    "src/calibrex/solvers/continuous_time_lidar_pair_solver.py",
    "src/calibrex/solvers/fixed_trajectory_se3_solver.py",
    "src/calibrex/evaluation/lidar.py",
    "src/calibrex/data/livox.py",
    "src/calibrex/pipelines/online.py",
    "src/calibrex/core/continuous_time_lidar_artifacts.py",
    "src/calibrex/core/continuous_time_lidar_train_diagnostics.py",
    "src/calibrex/core/provenance.py",
    "tools/rosbag1_to_rosbag2_online_pair.py",
    "tools/lidar_scan_deskew.py",
    "tools/benchmark_adaptive_plane_search.py",
)


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical(payload: dict) -> str:
    """Normalize only platform path separators; keep all values and provenance."""
    def visit(value):
        if isinstance(value, dict):
            return {key.replace("\\", "/"): visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value.replace("\\", "/") if isinstance(value, str) else value

    return json.dumps(visit(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _split_binding(
    problem: ContinuousTimeLidarPairProblem,
    options: ContinuousTimeLidarPairOptions | None,
) -> dict[str, int | str]:
    from calibrex.solvers.continuous_time_lidar_pair_solver import (
        ContinuousTimeLidarPairOptions,
        _resolve_indices,
        _select_evenly_spaced_indices,
    )

    settings = options or ContinuousTimeLidarPairOptions()
    result: dict[str, int | str] = {}
    for name, indices in zip(("train", "holdout"), _resolve_indices(problem), strict=True):
        selected = _select_evenly_spaced_indices(
            indices, settings.max_target_points_per_split, seed=settings.sampling_seed,
        )
        identities = [
            [index, problem.target_capture_timestamps_ns[index], *problem.target_points[index]]
            for index in selected
        ]
        result[name + "_count"] = len(selected)
        result[name + "_ids_sha256"] = hashlib.sha256(
            json.dumps(identities, separators=(",", ":"), allow_nan=False).encode(),
        ).hexdigest()
    return result


def _worker(args: argparse.Namespace) -> None:
    imported = time.perf_counter()
    modules = RUNTIME_MODULES if args.baseline_scope == "runtime" else (MODULE,)
    if args.worker == "baseline":
        from importlib.abc import MetaPathFinder
        from importlib.machinery import SourceFileLoader
        from importlib.util import spec_from_loader

        class ArchivedFinder(MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname in modules:
                    source = args.output_dir.parent / (
                        f"baseline.{fullname.rsplit('.', 1)[1]}.py.txt"
                    )
                    return spec_from_loader(fullname, SourceFileLoader(fullname, str(source)))
                return None

        sys.meta_path.insert(0, ArchivedFinder())
    from calibrex.core.io import read_mapping, write_mapping
    from calibrex.data import livox, odometry_track
    from calibrex.data.adaptive_voxel import AdaptiveVoxelPlaneMap
    from calibrex.pipelines.online import evaluate_continuous_time_lidar_pair
    from calibrex.solvers.continuous_time_lidar_pair_solver import ContinuousTimeLidarPairSolver

    import_seconds = time.perf_counter() - imported
    query_seconds = solver_seconds = 0.0
    query_count = 0
    sampled_split = None
    normal_calculations = pose_calculations = 0
    fallback = livox._fallback_plane_normal
    interpolate_pose = odometry_track.interpolate_se3
    prepared_interval = getattr(odometry_track, "_PreparedInterval", None)

    def measured_normal(*arguments, **keywords):
        nonlocal normal_calculations
        normal_calculations += 1
        return fallback(*arguments, **keywords)

    def measured_pose(*arguments, **keywords):
        nonlocal pose_calculations
        pose_calculations += 1
        return interpolate_pose(*arguments, **keywords)

    livox._fallback_plane_normal = measured_normal
    odometry_track.interpolate_se3 = measured_pose
    if prepared_interval is not None:
        prepared_interpolate = prepared_interval.interpolate

        def measured_prepared_pose(*arguments, **keywords):
            nonlocal pose_calculations
            pose_calculations += 1
            return prepared_interpolate(*arguments, **keywords)

        prepared_interval.interpolate = measured_prepared_pose
    nearest = AdaptiveVoxelPlaneMap.nearest_plane
    solve = ContinuousTimeLidarPairSolver.solve

    def measured_nearest(*arguments, **keywords):
        nonlocal query_seconds, query_count
        query_count += 1
        started = time.perf_counter()
        try:
            return nearest(*arguments, **keywords)
        finally:
            query_seconds += time.perf_counter() - started

    def measured_solve(*arguments, **keywords):
        nonlocal solver_seconds, sampled_split
        sampled_split = _split_binding(
            arguments[1], arguments[2] if len(arguments) > 2 else keywords.get("options"),
        )
        started = time.perf_counter()
        try:
            return solve(*arguments, **keywords)
        finally:
            solver_seconds += time.perf_counter() - started

    AdaptiveVoxelPlaneMap.nearest_plane = measured_nearest
    ContinuousTimeLidarPairSolver.solve = measured_solve
    started = time.perf_counter()
    artifact = evaluate_continuous_time_lidar_pair(args.config)
    pipeline_seconds = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = artifact.model_dump(mode="json", exclude_none=True)
    output = args.output_dir / "result.yaml"
    write_mapping(output, result)
    canonical = _canonical(result)
    if _canonical(read_mapping(output)) != canonical:
        raise ValueError("saved artifact differs from API result")
    measurement = {
        "pipeline_seconds": pipeline_seconds,
        "solver_seconds": solver_seconds,
        "adaptive_query_seconds": query_seconds,
        "adaptive_query_calls": query_count,
        "worker_import_seconds": import_seconds,
        "loaded_module": sys.modules[MODULE].__file__,
        "loaded_evaluation_modules": {name: sys.modules[name].__file__ for name in modules},
        "uniform_normal_calculations": normal_calculations,
        "interpolated_pose_calculations": pose_calculations,
        "sampled_split": sampled_split,
        "result": json.loads(canonical),
        "result_sha256": _digest(output),
    }
    (args.output_dir / "measurement.json").write_text(
        json.dumps(measurement, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def _verify_raw_inputs() -> None:
    for name, size, md5 in (
        ("livox.bag", 613350674, "fb17a6d4e0bc1765e276f0dcca8a5063"),
        ("gt.tar.gz", 140570, "ea25f7703ec5d076ba91dba63666d92c"),
    ):
        path = PUBLIC_ROOT / name
        if path.stat().st_size != size or _digest(path, "md5") != md5:
            raise ValueError(f"official public input checksum mismatch: {path}")


def _input_digests(dataset: Path) -> dict[str, str]:
    paths = [
        PUBLIC_ROOT / "livox.bag", PUBLIC_ROOT / "gt.tar.gz", PUBLIC_ROOT / "gt/livox.txt",
    ]
    paths += [path for path in sorted(dataset.rglob("*")) if path.is_file()]
    return {path.as_posix(): _digest(path) for path in paths}


def _prepare_inputs() -> None:
    """Verify official downloads and run the tutorial's unchanged conversion."""
    import tarfile

    _verify_raw_inputs()
    if REPLAY.exists():
        raise ValueError("conversion destination already exists; preserve it for audit")
    with tarfile.open(PUBLIC_ROOT / "gt.tar.gz") as archive:
        member = next(member for member in archive.getmembers()
                      if member.isfile() and member.name in ("gt/livox.txt", "livox.txt"))
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("public trajectory archive lacks Livox trajectory bytes")
        target = PUBLIC_ROOT / "gt/livox.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(stream.read())
    converter_digest = _digest(Path(CONVERSION_ARGUMENTS[0]))
    with (PUBLIC_ROOT / "conversion.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, *CONVERSION_ARGUMENTS], check=True,
                       stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if _digest(Path(CONVERSION_ARGUMENTS[0])) != converter_digest:
        raise ValueError("converter bytes changed during preparation")
    lineage = {
        "source": "https://zenodo.org/records/6864654",
        "command": list(CONVERSION_ARGUMENTS),
        "converter_sha256": converter_digest,
        "python_version": platform.python_version(),
        "input_and_replay_sha256": _input_digests(REPLAY),
        "conversion_log_sha256": _digest(PUBLIC_ROOT / "conversion.log"),
    }
    (PUBLIC_ROOT / "conversion-provenance.json").write_text(
        json.dumps(lineage, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    print("Official input checksums and converted replay lineage recorded.", flush=True)


def _verified_inputs(dataset: Path) -> dict[str, str]:
    if dataset != REPLAY:
        raise ValueError("benchmark requires the tutorial's complete public converted replay")
    _verify_raw_inputs()
    lineage_path = PUBLIC_ROOT / "conversion-provenance.json"
    lineage = json.loads(lineage_path.read_bytes())
    if lineage["command"] != list(CONVERSION_ARGUMENTS):
        raise ValueError("conversion command differs from the published tutorial protocol")
    if lineage["converter_sha256"] != _digest(Path("tools/rosbag1_to_rosbag2_online_pair.py")):
        raise ValueError("conversion tool no longer matches recorded lineage")
    digests = _input_digests(dataset)
    if digests != lineage["input_and_replay_sha256"]:
        raise ValueError("raw/trajectory/replay bytes no longer match conversion lineage")
    return {**digests, lineage_path.as_posix(): _digest(lineage_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-commit", default=BASELINE)
    parser.add_argument("--baseline-scope", choices=("adaptive", "runtime"), default="adaptive")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument(
        "--prepare", action="store_true",
        help="Verify downloaded official bag/GT, convert once and record replay lineage.",
    )
    parser.add_argument("--worker", choices=("baseline", "sparse"), help=argparse.SUPPRESS)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    args.output_dir = args.output_dir.resolve()
    if args.prepare:
        _prepare_inputs()
        return
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

    config = load_config(args.config)
    if config.dataset.type != "rosbag2":
        raise ValueError("public adaptive benchmark requires the converted rosbag2 fixture")
    inputs = _verified_inputs(Path(config.dataset.path))
    source_commit = git_commit()
    full_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    bindings = {path: _digest(Path(path)) for path in SOURCES}
    baseline = subprocess.check_output(
        ["git", "rev-parse", f"{args.baseline_commit}^{{commit}}"], text=True,
    ).strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modules = RUNTIME_MODULES if args.baseline_scope == "runtime" else (MODULE,)
    archives = {}
    for name in modules:
        archive = args.output_dir / f"baseline.{name.rsplit('.', 1)[1]}.py.txt"
        archive.write_bytes(subprocess.check_output(
            ["git", "show", f"{baseline}:src/{name.replace('.', '/')}.py"],
        ))
        bindings[archive.as_posix()] = _digest(archive)
        archives[name] = archive
    config_digest = _digest(args.config)
    input_digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    environment = dict(os.environ, PYTHONIOENCODING="utf-8",
                       OMP_NUM_THREADS=str(args.omp_threads))

    def guard():
        if (
            full_revision != subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
            ).strip()
            or bindings != {path: _digest(Path(path)) for path in bindings}
            or config_digest != _digest(args.config)
            or inputs != {path: _digest(Path(path)) for path in inputs}
        ):
            raise ValueError("source, HEAD, configuration or inputs changed during benchmark")

    trials = []
    anchor = None
    split_anchor = None
    for repeat in range(1, args.repetitions + 1):
        order = ["baseline", "sparse"] if repeat % 2 else ["sparse", "baseline"]
        for method in order:
            guard()
            directory = args.output_dir / f"{method}.repeat-{repeat:02d}"
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker", method,
                "--baseline-scope", args.baseline_scope,
                "--config", str(args.config), "--output-dir", str(directory),
            ]
            print(f"START {repeat} {method}", flush=True)
            started = time.perf_counter()
            with (args.output_dir / f"{method}.repeat-{repeat:02d}.log").open(
                "w", encoding="utf-8",
            ) as log:
                subprocess.run(command, check=True, env=environment,
                               stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - started
            row = json.loads((directory / "measurement.json").read_bytes())
            expected = {
                name: str(archives[name] if method == "baseline" else
                          Path("src", *name.split(".")).with_suffix(".py").resolve())
                for name in modules
            }
            if row["loaded_evaluation_modules"] != expected:
                raise ValueError("baseline/candidate module binding mismatch")
            if split_anchor is None:
                split_anchor = row["sampled_split"]
            if row["sampled_split"] != split_anchor:
                raise ValueError("train/holdout sampled point identities changed")
            canonical = _canonical(row["result"])
            if anchor is None:
                anchor = canonical
            matches = canonical == anchor
            metrics = {name: float(row[name]) for name in (
                "pipeline_seconds", "solver_seconds", "adaptive_query_seconds",
                "adaptive_query_calls", "worker_import_seconds",
                "uniform_normal_calculations", "interpolated_pose_calculations",
            )}
            metrics.update(subprocess_wall_seconds=elapsed, complete_result_matches=float(matches))
            trials.append(BenchmarkTrial(
                method_id=method, split_id=f"repeat-{repeat:02d}",
                status="success" if matches else "failed", runtime_seconds=elapsed,
                metrics=metrics,
                provenance=BenchmarkTrialProvenance(
                    command=subprocess.list2cmdline(command), config_sha256=config_digest,
                    input_sha256=input_digest,
                    output_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                    execution_host=platform.platform(),
                    notes=[json.dumps({"repeat": repeat, "order": order,
                                       "result_sha256": row["result_sha256"],
                                       "sampled_split": row["sampled_split"],
                                       "loaded_evaluation_modules": (
                                           row["loaded_evaluation_modules"]
                                       ),
                                       "loaded_module": row["loaded_module"]}, sort_keys=True),
                           canonical],
                ),
            ))
            print(f"DONE {repeat} {method} {elapsed:.6f}s matches={matches}", flush=True)
    guard()
    definition = BenchmarkDefinition(
        benchmark_id=f"glim-complete-continuous-time-{args.baseline_scope}",
        title="Public GLIM Livox complete continuous-time refinement",
        reference_method_id="baseline", bootstrap_samples=1,
        protocol=BenchmarkProtocol(
            protocol_id=f"glim-{args.baseline_scope}-fresh-subprocess-v2",
            dataset_id="glim_versatile",
            dataset_doi="10.5281/zenodo.6864654", dataset_source_sha256=input_digest,
            data_license="CC-BY-4.0 (AIST Zenodo record 6864654)",
            split_policy="Latest capture frames held out; repeats share the same accuracy fixture",
            splits=[BenchmarkSplit(
                split_id=f"repeat-{i:02d}", seed=0,
                fit_count=split_anchor["train_count"],
                holdout_count=split_anchor["holdout_count"],
                fit_ids_sha256=split_anchor["train_ids_sha256"],
                holdout_ids_sha256=split_anchor["holdout_ids_sha256"],
            ) for i in range(1, args.repetitions + 1)],
            initial_estimate_policy=(
                "Same supplied public config, initial estimate and temporal split"
            ),
            tuning_policy=(
                f"Reverse order every other repeat; child OMP_NUM_THREADS={args.omp_threads}"
            ),
            failure_policy="Keep all timings and mismatches; no warmup or excluded outliers",
        ),
        methods=[BenchmarkMethodDefinition(
            method_id=method, label=method, implementation="subprocess", tool_name="Calibrex",
            tool_version="0.5.1", adapter_version=f"continuous-time-{args.baseline_scope}/v2",
            source_commit=baseline if method == "baseline" else full_revision,
        ) for method in ("baseline", "sparse")],
        metrics=[BenchmarkMetricDefinition(
            name=name, label=name,
            unit="bool" if name.endswith("matches") else (
                "calls" if name.endswith(("calls", "calculations")) else "s"
            ),
            direction="higher" if name.endswith("matches") else "lower",
            primary=name == "subprocess_wall_seconds",
            interpretation={
                "complete_result_matches": "Every artifact value and all profile history match",
                "adaptive_query_calls": "Actual adaptive nearest-plane calls in the complete run",
                "uniform_normal_calculations": "Actual uniform geometric fallback calls",
                "interpolated_pose_calculations": "Interior odometry SE3 interpolation calls",
                "adaptive_query_seconds": (
                    "Adaptive nearest-plane call time, including wrapper overhead"
                ),
                "solver_seconds": "Complete clock profile and extrinsic solver call",
                "worker_import_seconds": (
                    "Worker entry through imports, excluding interpreter startup"
                ),
                "pipeline_seconds": "Full evaluation API including replay and provenance hashing",
            }.get(name, "Fresh complete process wall, including artifact saves"),
        ) for name in trials[0].metrics],
        trials=trials,
        limitations=[
            "Single public sensor duplicated as an identity control; "
            "no absolute two-sensor accuracy claim.",
            "Fresh interpreters; no warmup or outlier exclusion; OS caches persist.",
            f"Scope {args.baseline_scope} substitutes {len(modules)} modules. "
            "All candidates, refinements, train/holdout "
            "budgets "
            "and writers are shared. Module filenames and source/input bytes are verified.",
            "Wall includes startup, imports, replay, optimization, diagnostics, input hashes, "
            "YAML output and worker measurement save; parent audits and CLI presentation "
            "are excluded.",
            "Phase timers wrap existing calls without extra work; their wrapper cost is included.",
            "Full artifact equality normalizes only path separators; git revision remains shared.",
            "Raw checksums and conversion lineage are verified before measurement.",
            "Split IDs hash ordered sampled target indices, capture timestamps and XYZ; "
            "the distinct temporal partitions and unchanged sampling caps are verified each run.",
            "Descriptive timings are local observations, not portable confidence estimates.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools.benchmark_adaptive_plane_search", generator_version="v0.2",
            git_commit=source_commit,
            command=subprocess.list2cmdline([sys.executable, __file__, *arguments]),
            data_verified=True,
            source_artifacts={**bindings, **inputs, args.config.as_posix(): config_digest},
        ),
    )
    definition.save(args.output_dir / "pipeline.definition.json")
    aggregate_benchmark_definition(definition).save(args.output_dir / "pipeline.benchmark.json")
    if any(trial.status != "success" for trial in trials):
        raise ValueError("full artifact parity failed; all measurements retained")


if __name__ == "__main__":
    main()
