"""Compare complete native ICP calls against pinned exhaustive implementations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import scipy

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
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.solvers import chen_medioni_point_to_plane_icp_solver as plane
from calibrex.solvers import robust_point_to_point_icp_solver as point
from calibrex.solvers.native_registration_comparison_solver import voxel_downsample_icp_points

REFERENCE_COMMIT = "f959c684c3f4e4ec24a884cd4cea8545d44baa4a"
REFERENCE_SOURCES = (
    "src/calibrex/solvers/robust_point_to_point_icp_solver.py",
    "src/calibrex/solvers/chen_medioni_point_to_plane_icp_solver.py",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _reference(commit: str, output: Path) -> tuple[ModuleType, ModuleType]:
    modules = []
    for index, source in enumerate(REFERENCE_SOURCES):
        path = output / f"baseline.{Path(source).name}.txt"
        path.write_bytes(subprocess.check_output(["git", "show", f"{commit}:{source}"]))
        name = f"_calibrex_icp_reference_{index}"
        loader = importlib.machinery.SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        modules.append(module)
    old_point, old_plane = modules
    old_plane.evaluate_icp_candidate = old_point.evaluate_icp_candidate
    old_plane.split_icp_source_points = old_point.split_icp_source_points
    return old_point, old_plane


def _solve(
    algorithm: str,
    modules: tuple[Any, Any],
    source: Any,
    target: Any,
    options: point.RobustPointToPointIcpOptions,
) -> dict[str, object]:
    point_module, plane_module = modules
    if algorithm == "point":
        return (
            point_module.RobustPointToPointIcpSolver()
            .solve(
                source,
                target,
                initial_transform=SE3.identity(),
                options=options,
            )
            .as_dict()
        )
    return (
        plane_module.ChenMedioniPointToPlaneIcpSolver()
        .solve(
            source,
            target,
            initial_transform=SE3.identity(),
            common_options=options,
        )
        .as_dict()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1500, 3000])
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help="target:source numeric frame IDs (default 0000000000:0000000001)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--baseline-commit", default=REFERENCE_COMMIT)
    args = parser.parse_args()
    pairs = args.pair or ["0000000000:0000000001"]
    if args.repeats < 1 or args.max_iterations < 1 or any(size < 20 for size in args.sizes):
        raise ValueError("invalid repeat, iteration or point count")
    if not re.fullmatch(r"[0-9a-f]{7,40}", args.baseline_commit):
        raise ValueError("baseline must be a Git commit hash")
    if any(not re.fullmatch(r"\d{10}:\d{10}", pair) for pair in pairs):
        raise ValueError("each pair must contain two ten-digit numeric IDs separated by a colon")
    if len(set(args.sizes)) != len(args.sizes) or len(set(pairs)) != len(pairs):
        raise ValueError("duplicate size or pair")
    if args.output_dir.exists():
        raise ValueError("output directory exists; preserve earlier experiments")
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{args.baseline_commit}^{{commit}}"],
        text=True,
    ).strip()
    args.output_dir.mkdir(parents=True)
    reference = _reference(commit, args.output_dir)
    measured_commit = git_commit()
    sources = {
        path: sha256_path(Path(path))
        for path in (
            *REFERENCE_SOURCES,
            "src/calibrex/solvers/_exact_neighbors.py",
            "src/calibrex/solvers/native_registration_comparison_solver.py",
            "src/calibrex/core/geometry.py",
            "src/calibrex/evaluation/holdout.py",
            str(Path(__file__)),
        )
    }
    sources.update(
        {path.name: sha256_path(path) for path in args.output_dir.glob("baseline.*.txt")}
    )
    options = point.RobustPointToPointIcpOptions(
        max_iterations=args.max_iterations,
        correspondence_distance_m=1.5,
        holdout_voxel_size_m=1.0,
    )
    config_digest = _digest(
        {
            "common": asdict(options),
            "plane": asdict(plane.ChenMedioniPointToPlaneOptions()),
            "voxel_size_m": 0.5,
            "caps": args.sizes,
            "initial": SE3.identity().as_dict(),
        }
    )
    cases = []
    preparation_seconds = {}
    for pair in pairs:
        target_id, source_id = pair.split(":")
        prepared_started = time.perf_counter()
        clouds = {}
        for role, frame_id in (("target", target_id), ("source", source_id)):
            path = args.sequence / "velodyne_points" / "data" / f"{frame_id}.bin"
            data = path.read_bytes()
            sources[str(path)] = hashlib.sha256(data).hexdigest()
            clouds[role] = np.frombuffer(data, dtype="<f4").reshape(-1, 4)[:, :3]
        for size in args.sizes:
            source = voxel_downsample_icp_points(clouds["source"], "source", 0.5, size)
            target = voxel_downsample_icp_points(clouds["target"], "target", 0.5, size)
            train, holdout = point.split_icp_source_points(source, options)
            case = f"{target_id}-{source_id}-cap-{size}"
            cases.append((case, source, target, train, holdout))
            sources[f"prepared:{case}"] = _digest(
                [
                    [asdict(item) for item in source],
                    [asdict(item) for item in target],
                ]
            )
        preparation_seconds[pair] = time.perf_counter() - prepared_started
    verified_input = _digest(
        {
            name: digest
            for name, digest in sources.items()
            if name.startswith((str(args.sequence), "prepared:"))
        }
    )
    for algorithm in ("point", "plane"):
        trials, splits = [], []
        anchors = {}
        for case_index, (case, source, target, train, holdout) in enumerate(cases):
            for repetition in range(1, args.repeats + 1):
                split = f"{case}-repeat-{repetition:02d}"
                splits.append(
                    BenchmarkSplit(
                        split_id=split,
                        seed=options.split_seed,
                        fit_count=len(train),
                        holdout_count=len(holdout),
                        fit_ids_sha256=_digest([item.point_id for item in train]),
                        holdout_ids_sha256=_digest([item.point_id for item in holdout]),
                    )
                )
                order = ["exhaustive", "indexed"]
                if (case_index + repetition) % 2 == 0:
                    order.reverse()
                repeated = []
                for method in order:
                    started = time.perf_counter()
                    payload = _solve(
                        algorithm,
                        reference if method == "exhaustive" else (point, plane),
                        source,
                        target,
                        options,
                    )
                    elapsed = time.perf_counter() - started
                    repeated.append((method, elapsed, payload))
                    print(
                        json.dumps(
                            {
                                "algorithm": algorithm,
                                "case": case,
                                "repeat": repetition,
                                "method": method,
                                "seconds": elapsed,
                                "status": payload["status"],
                            }
                        ),
                        flush=True,
                    )
                expected = next(
                    payload for method, _, payload in repeated if method == "exhaustive"
                )
                anchor = anchors.setdefault(case, expected)
                for method, elapsed, payload in repeated:
                    parity = payload == expected == anchor
                    trials.append(
                        BenchmarkTrial(
                            method_id=method,
                            split_id=split,
                            status="success" if parity else "failed",
                            runtime_seconds=elapsed,
                            metrics={"wall_seconds": elapsed, "numerical_parity": float(parity)},
                            failure_reason=None
                            if parity
                            else "Complete numerical payload mismatch",
                            provenance=BenchmarkTrialProvenance(
                                command=" ".join(sys.argv),
                                config_sha256=config_digest,
                                input_sha256=sources[f"prepared:{case}"],
                                output_sha256=_digest(payload),
                                execution_host=platform.platform(),
                                notes=[
                                    f"Order: {order}; repeat {repetition}; shared host; "
                                    "no separate warmup.",
                                    f"Source/target counts: {len(source)}/{len(target)}; "
                                    f"NumPy {np.__version__}; SciPy {scipy.__version__}.",
                                    "Wall includes index/normal preparation, iterations, holdout, "
                                    "all diagnostics and as_dict; excludes "
                                    "input capture/downsampling and benchmark serialization.",
                                    "Complete numerical payload: "
                                    + json.dumps(payload, sort_keys=True),
                                ],
                            ),
                        )
                    )
        definition = BenchmarkDefinition(
            benchmark_id=f"public_kitti_icp_neighbors_{algorithm}",
            title=f"Public KITTI complete {algorithm} ICP neighbor search",
            reference_method_id="exhaustive",
            bootstrap_samples=1,
            protocol=BenchmarkProtocol(
                protocol_id="public_kitti_icp_neighbors/v0.1",
                dataset_id=args.sequence.name,
                dataset_source_sha256=verified_input,
                data_license="KITTI raw dataset terms; raw scans are not redistributed",
                split_policy="Unchanged native spatial source holdout; target rematched per pose",
                splits=splits,
                initial_estimate_policy="Identity for both methods; no reference pose",
                tuning_policy=f"Frozen 0.5 m voxel centroids, caps {args.sizes}, "
                f"options SHA256 {config_digest}; alternate method order",
                failure_policy="Retain mismatches as failures; exit nonzero after saving evidence",
            ),
            methods=[
                BenchmarkMethodDefinition(
                    method_id=method,
                    label=method,
                    implementation="calibrex_native",
                    tool_name="Calibrex",
                    tool_version=__version__,
                    source_commit=source_commit,
                )
                for method, source_commit in (("exhaustive", commit), ("indexed", measured_commit))
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
                    ("wall_seconds", "s", "lower", "Complete solver call with all diagnostics"),
                    ("numerical_parity", "ratio", "higher", "Exact complete pinned solver payload"),
                )
            ],
            trials=trials,
            limitations=[
                "Declared scan pairs from one public sequence; "
                "no reference extrinsic or accuracy claim.",
                "Spatially capped voxel centroids are identical for both methods; this does not "
                "measure whole raw-cloud registration or the full calibrate CLI pipeline.",
                "All iterations and diagnostic settings are unchanged between methods. "
                "Solver convergence/degeneracy status remains visible in every full payload.",
                "Three timing repetitions are not independent accuracy trials or confidence. "
                "Host and process/OS caches are shared; no portable speed guarantee.",
                "Input capture and preparation cost is recorded separately, charged once for "
                "this experiment, not included in solver wall: " + json.dumps(preparation_seconds),
            ],
            provenance=BenchmarkProvenance(
                generator="tools.benchmark_icp_neighbors",
                generator_version="v0.1",
                git_commit=measured_commit,
                command=" ".join(sys.argv),
                source_artifacts=sources.copy(),
                data_verified=all(trial.status == "success" for trial in trials),
            ),
        )
        definition.save(args.output_dir / f"{algorithm}.definition.json")
        aggregate_benchmark_definition(definition).save(
            args.output_dir / f"{algorithm}.benchmark.json"
        )
        print(
            json.dumps(
                {
                    case: {
                        method: statistics.median(
                            trial.runtime_seconds
                            for trial in trials
                            if trial.split_id.startswith(case + "-repeat-")
                            and trial.method_id == method
                        )
                        for method in ("exhaustive", "indexed")
                    }
                    for case, *_ in cases
                }
            ),
            flush=True,
        )
        if not definition.provenance.data_verified:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
