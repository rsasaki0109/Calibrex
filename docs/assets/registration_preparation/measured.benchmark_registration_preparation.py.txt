"""Time public raw-cloud preparation and both complete native ICP solvers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
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
from calibrex.solvers import native_registration_comparison_solver as native
from calibrex.solvers import robust_point_to_point_icp_solver as point

REFERENCE_COMMIT = "9fe0d4f6ff866bed771b62ccbe7b7e57924b12bb"
REFERENCE_SOURCES = (
    "src/calibrex/solvers/robust_point_to_point_icp_solver.py",
    "src/calibrex/solvers/chen_medioni_point_to_plane_icp_solver.py",
    "src/calibrex/solvers/native_registration_comparison_solver.py",
    "src/calibrex/solvers/_exact_neighbors.py",
)
METHODS = ("indexed_baseline", "vectorized_sequential", "vectorized_auto")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _reference(commit: str, output: Path) -> tuple[ModuleType, ModuleType, ModuleType]:
    modules = []
    for index, source in enumerate(REFERENCE_SOURCES):
        path = output / f"baseline.{Path(source).name}.txt"
        path.write_bytes(subprocess.check_output(["git", "show", f"{commit}:{source}"]))
        name = f"_calibrex_registration_reference_{index}"
        loader = importlib.machinery.SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        modules.append(module)
    old_point, old_plane, old_native, old_neighbors = modules
    old_point._ExactNeighbors = old_neighbors._ExactNeighbors
    old_plane._ExactNeighbors = old_neighbors._ExactNeighbors
    old_plane.evaluate_icp_candidate = old_point.evaluate_icp_candidate
    old_plane.split_icp_source_points = old_point.split_icp_source_points
    return old_point, old_plane, old_native


def _run(
    modules: tuple[Any, Any, Any], paths: tuple[Path, Path], cap: int,
    options: point.RobustPointToPointIcpOptions, input_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, float]]:
    point_module, plane_module, native_module = modules
    started = time.perf_counter()
    clouds = []
    for path in paths:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != input_hashes[str(path)]:
            raise ValueError("Raw scan changed during experiment")
        clouds.append(np.frombuffer(data, dtype="<f4").reshape(-1, 4)[:, :3])
    target, source = [native_module.voxel_downsample_icp_points(cloud, role, 0.5, cap)
                      for cloud, role in zip(clouds, ("target", "source"), strict=True)]
    prepared = time.perf_counter()
    point_payload = point_module.RobustPointToPointIcpSolver().solve(
        source, target, SE3.identity(), options,
    ).as_dict()
    point_done = time.perf_counter()
    # Worker controls only affect point ICP's multi-start diagnostics. Keep the
    # plane solver's recorded common options identical in all three methods.
    plane_payload = plane_module.ChenMedioniPointToPlaneIcpSolver().solve(
        source, target, SE3.identity(), replace(options, multi_start_workers=0),
    ).as_dict()
    done = time.perf_counter()
    # Bind prepared positions bit-for-bit, including signed zeros, and stable
    # IDs without duplicating the raw-derived centroids in every report.
    payload = {
        "prepared": [
            {
                "count": len(cloud),
                "ids_sha256": _digest([item.point_id for item in cloud]),
                "positions_float64_le_sha256": hashlib.sha256(np.asarray(
                    [item.position_m for item in cloud], dtype="<f8",
                ).tobytes()).hexdigest(),
            }
            for cloud in (source, target)
        ],
        "point_to_point": point_payload,
        "point_to_plane": plane_payload,
    }
    return payload, {
        "wall_seconds": done - started,
        "preparation_seconds": prepared - started,
        "point_seconds": point_done - prepared,
        "plane_seconds": done - point_done,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1500, 3000])
    parser.add_argument("--pair", action="append", required=True, help="target:source frame IDs")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--baseline-commit", default=REFERENCE_COMMIT)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_iterations < 1 or any(size < 20 for size in args.sizes):
        raise ValueError("invalid repeat, iteration or point count")
    if not re.fullmatch(r"[0-9a-f]{7,40}", args.baseline_commit):
        raise ValueError("baseline must be a Git commit hash")
    if any(not re.fullmatch(r"\d{10}:\d{10}", pair) for pair in args.pair):
        raise ValueError("each pair must contain two ten-digit numeric IDs separated by a colon")
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.pair)) != len(args.pair):
        raise ValueError("duplicate size or pair")
    if args.output_dir.exists():
        raise ValueError("output directory exists; preserve earlier experiments")
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{args.baseline_commit}^{{commit}}"], text=True,
    ).strip()
    paths_by_pair = {
        pair: tuple(args.sequence / "velodyne_points" / "data" / f"{frame}.bin"
                    for frame in pair.split(":"))
        for pair in args.pair
    }
    input_hashes = {
        str(path): sha256_path(path) for paths in paths_by_pair.values() for path in paths
    }
    args.output_dir.mkdir(parents=True)
    reference = _reference(commit, args.output_dir)
    measured_commit = git_commit()
    source_hashes = {
        path: sha256_path(Path(path))
        for path in (
            *REFERENCE_SOURCES, "src/calibrex/core/geometry.py",
            "src/calibrex/evaluation/holdout.py", str(Path(__file__)),
        )
    }
    sources = {**source_hashes, **input_hashes}
    sources.update({path.name: sha256_path(path)
                    for path in args.output_dir.glob("baseline.*.txt")})
    options = point.RobustPointToPointIcpOptions(
        max_iterations=args.max_iterations, correspondence_distance_m=1.5,
        holdout_voxel_size_m=1.0,
    )
    config_digest = _digest({
        "common": asdict(options), "plane": asdict(plane.ChenMedioniPointToPlaneOptions()),
        "voxel_size_m": 0.5, "caps": args.sizes, "pairs": args.pair,
        "initial": SE3.identity().as_dict(), "sequential_workers": 1,
    })
    trials, splits = [], []
    case_names = []
    for pair_index, (pair, paths) in enumerate(paths_by_pair.items()):
        for size_index, cap in enumerate(args.sizes):
            case = f"{pair.replace(':', '-')}-cap-{cap}"
            case_names.append(case)
            anchor = None
            for repetition in range(1, args.repeats + 1):
                offset = (pair_index + size_index + repetition - 1) % len(METHODS)
                order = METHODS[offset:] + METHODS[:offset]
                repeated = []
                for method in order:
                    payload, timing = _run(
                        reference if method == METHODS[0] else (point, plane, native),
                        paths, cap,
                        replace(options, multi_start_workers=1)
                        if method == METHODS[1] else options,
                        input_hashes,
                    )
                    repeated.append((method, payload, timing))
                    print(json.dumps({
                        "case": case, "repeat": repetition, "method": method, **timing,
                        "statuses": [payload[key]["status"]
                                     for key in ("point_to_point", "point_to_plane")],
                    }), flush=True)
                expected = next(payload for method, payload, _ in repeated if method == METHODS[0])
                if anchor is None:
                    anchor = expected
                split = f"{case}-repeat-{repetition:02d}"
                train_ids = expected["point_to_point"]["train_source_ids"]
                holdout_ids = expected["point_to_point"]["holdout_source_ids"]
                splits.append(BenchmarkSplit(
                    split_id=split, seed=options.split_seed,
                    fit_count=len(train_ids), holdout_count=len(holdout_ids),
                    fit_ids_sha256=_digest(train_ids), holdout_ids_sha256=_digest(holdout_ids),
                ))
                for method, payload, timing in repeated:
                    parity = payload == expected == anchor
                    trials.append(BenchmarkTrial(
                        method_id=method, split_id=split, status="success" if parity else "failed",
                        runtime_seconds=timing["wall_seconds"],
                        metrics={**timing, "numerical_parity": float(parity)},
                        failure_reason=None if parity else "Complete numerical payload mismatch",
                        provenance=BenchmarkTrialProvenance(
                            command=" ".join(sys.argv), config_sha256=config_digest,
                            input_sha256=_digest({
                                str(path): input_hashes[str(path)] for path in paths
                            }),
                            output_sha256=_digest(payload), execution_host=platform.platform(),
                            notes=[
                                f"Order: {order}; repetition {repetition}; shared host; no warmup.",
                                f"Python {platform.python_version()}; NumPy {np.__version__}; "
                                f"SciPy {scipy.__version__}; logical CPUs {os.cpu_count()}.",
                                "Wall includes fresh scan reads/hash verification, decoding, "
                                "two-cloud voxel centroids and both solver calls with as_dict. "
                                "All ICP iterations, holdout and diagnostics remain enabled. "
                                "Prepared/output digesting and benchmark serialization "
                                "are excluded.",
                                "Automatic multi-start: up to four workers for "
                                ">=1500 source points, "
                                "otherwise one; explicit sequential method uses one.",
                                "Complete numerical payload: "
                                + json.dumps(payload, sort_keys=True),
                            ],
                        ),
                    ))
    if git_commit() != measured_commit or any(sha256_path(Path(path)) != digest
                                              for path, digest in source_hashes.items()):
        raise RuntimeError("Measured source changed during experiment")
    definition = BenchmarkDefinition(
        benchmark_id="public_kitti_registration_preparation",
        title="Public raw-cloud preparation and complete native registration pair",
        reference_method_id=METHODS[0], bootstrap_samples=1,
        protocol=BenchmarkProtocol(
            protocol_id="public_kitti_registration_preparation/v0.1",
            dataset_id=args.sequence.name, dataset_source_sha256=_digest(input_hashes),
            data_license="KITTI raw dataset terms; raw scans are not redistributed",
            split_policy="Unchanged spatial source holdout; fresh target rematching",
            splits=splits, initial_estimate_policy="Identity for all methods; no reference pose",
            tuning_policy=f"Frozen 0.5 m voxel centroids, caps {args.sizes}, "
            f"config {config_digest}; "
            "rotate method order; only execution workers differ",
            failure_policy="Retain mismatches as failures and exit nonzero after saving evidence",
        ),
        methods=[BenchmarkMethodDefinition(
            method_id=method, label=method, implementation="calibrex_native",
            tool_name="Calibrex", tool_version=__version__,
            source_commit=commit if method == METHODS[0] else measured_commit,
        ) for method in METHODS],
        metrics=[BenchmarkMetricDefinition(
            name=name, label=name, unit="ratio" if name == "numerical_parity" else "s",
            direction="higher" if name == "numerical_parity" else "lower",
            primary=name == "wall_seconds", interpretation=interpretation,
        ) for name, interpretation in (
            ("wall_seconds", "Fresh raw reads/verification, preprocessing, both complete solvers"),
            ("preparation_seconds", "Fresh reads/hash verification, decoding, voxel centroids"),
            ("point_seconds", "Complete point-to-point solver including six multi-start trials"),
            ("plane_seconds", "Complete point-to-plane solver including normals and pose probes"),
            ("numerical_parity", "Bitwise prepared positions and exact complete solver payloads"),
        )],
        trials=trials,
        limitations=[
            "Sampled scan pairs from one public drive; no reference extrinsic or accuracy claim.",
            "Unchanged native voxel/cap policy; full raw scans enter preprocessing on every call.",
            "Timing includes both complete native algorithms, not the full calibrate CLI, "
            "PCD parsing, external GICP/NDT backends, inspection or report serialization.",
            "All convergence, degeneracy and diagnostic outputs remain visible; timing success "
            "means execution/parity rather than calibration recovery.",
            f"{args.repeats} shared-host timing repetitions; process/library/OS caches persist; "
            "not independent accuracy trials, confidence or a portable speed guarantee.",
            "Bootstrap fields are schema summaries only; consult every retained trial.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools.benchmark_registration_preparation", generator_version="v0.1",
            git_commit=measured_commit, command=" ".join(sys.argv), source_artifacts=sources,
            data_verified=all(trial.status == "success" for trial in trials),
        ),
    )
    definition.save(args.output_dir / "registration.definition.json")
    aggregate_benchmark_definition(definition).save(args.output_dir / "registration.benchmark.json")
    print(json.dumps({case: {method: statistics.median(
        trial.runtime_seconds for trial in trials
        if trial.split_id.startswith(case + "-repeat-") and trial.method_id == method
    ) for method in METHODS} for case in case_names}), flush=True)
    if not definition.provenance.data_verified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
