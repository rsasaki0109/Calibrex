"""Compare complete CPU/CUDA odometry replays on a public ROS-independent bag reader.

No preprocessing budget, registration policy or scan is changed between methods.
The optional backend runs real queries and fused robust systems on the device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

SOURCES = (
    "src/calibrex/solvers/scan_to_scan_odometry.py",
    "src/calibrex/solvers/cuda_scan_odometry_adapter.py",
    "src/calibrex/solvers/cuda_lidar_map_adapter.py",
    "src/calibrex/solvers/lidar_lidar_map_solver.py",
    "src/calibrex/data/livox_ros2.py",
    "src/calibrex/core/benchmark.py",
    "tools/benchmark_cuda_scan_odometry.py",
)
_IDENTITY = re.compile(
    r"calibrex\.cuda_scan_odometry/v0\.1;fp64=true;query_fma=false;"
    r"cupy=\d+\.\d+\.\d+;cuda=\d+;driver=\d+;cc=\d+\.\d+;"
    r"query_sha256=[0-9a-f]{64};system_sha256=[0-9a-f]{64}"
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(cpu: Any, gpu: Any, path: str = "") -> list[str]:
    import math

    if path.endswith(".execution_backend"):
        return [] if cpu == "numpy_scipy" and _IDENTITY.fullmatch(gpu) else [path]
    if type(cpu) is not type(gpu):
        return [path]
    if isinstance(cpu, dict):
        if cpu.keys() != gpu.keys():
            return [path]
        return [error for key in cpu for error in _compare(cpu[key], gpu[key], f"{path}.{key}")]
    if isinstance(cpu, list):
        if len(cpu) != len(gpu):
            return [path]
        return [
            error
            for i, (a, b) in enumerate(zip(cpu, gpu, strict=True))
            for error in _compare(a, b, f"{path}[{i}]")
        ]
    if isinstance(cpu, float):
        return [] if math.isclose(cpu, gpu, rel_tol=1e-6, abs_tol=1e-9) else [path]
    return [] if cpu == gpu else [path]


def _inputs(bag: Path) -> list[Path]:
    paths = sorted(bag.glob("*.db3")) + sorted(bag.glob("metadata.yaml"))
    if not paths:
        raise ValueError("bag must contain public SQLite storage")
    return paths


def _worker(args: argparse.Namespace) -> None:
    cpu_start = time.process_time()
    import numpy as np
    from pydantic import Field

    from calibrex import __version__
    from calibrex.core.result import StrictModel
    from calibrex.data.livox_ros2 import LivoxStreamProfile, iter_livox_points
    from calibrex.solvers.scan_to_scan_odometry import IncrementalScanOdometry, ScanOdometryOptions

    class RegistrationRow(StrictModel):
        transform: list[list[float]]
        converged: bool
        iterations: int = Field(ge=0)
        correspondences: int = Field(ge=0)
        rmse_m: float | None
        hessian_min_eigenvalue: float
        execution_backend: str

    class OdometryRow(StrictModel):
        poses: list[list[list[float]]]
        registrations: list[RegistrationRow]
        execution_backend: str
        all_converged: bool

    class Measurement(StrictModel):
        schema_version: str = "calibrex.scan_odometry_measurement/v0.1"
        result: OdometryRow
        statistics: dict[str, int]
        api_seconds: float = Field(ge=0)
        worker_cpu_seconds: float = Field(ge=0)
        scan_rows: list[int]
        scan_sha256: list[str]
        times_s: list[float]
        options: dict[str, Any]
        provenance: dict[str, Any]

    Measurement.model_rebuild(
        _types_namespace={
            "Any": Any,
            "OdometryRow": OdometryRow,
            "RegistrationRow": RegistrationRow,
        }
    )
    options = ScanOdometryOptions(
        voxel_size_m=args.voxel_size,
        local_map_scans=3,
        geometry_quantum_m=args.geometry_quantum_m,
    )
    profile = LivoxStreamProfile("public", args.topic, "", None, "offset_s", "mps2")
    started = time.perf_counter()
    odometry = IncrementalScanOdometry(options, backend=args.worker)
    scans, times, hashes = [], [], []
    for timestamp, points, _ in iter_livox_points(args.bag, profile, max_seconds=args.seconds):
        points = points[np.isfinite(points).all(axis=1)]
        scans.append(len(points))
        times.append(float(timestamp))
        hashes.append(hashlib.sha256(points.tobytes()).hexdigest())
        odometry.add(points, timestamp)
    api_seconds = time.perf_counter() - started
    result = odometry.result()
    if len(result.poses) < 2:
        raise ValueError("at least two scans are required")
    payload = asdict(result)
    payload["poses"] = [pose.tolist() for pose in result.poses]
    for registration in payload["registrations"]:
        registration["transform"] = registration["transform"].tolist()
        if not np.isfinite(registration["rmse_m"]):
            registration["rmse_m"] = None
    statistics = payload.pop("execution_statistics")
    payload["all_converged"] = result.all_converged
    measurement = Measurement(
        result=payload,
        statistics=statistics,
        api_seconds=api_seconds,
        worker_cpu_seconds=time.process_time() - cpu_start,
        scan_rows=scans,
        times_s=times,
        scan_sha256=hashes,
        options=asdict(options),
        provenance={
            "generator": "tools.benchmark_cuda_scan_odometry",
            "version": __version__,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "command": [
                "<PYTHON>",
                "tools/benchmark_cuda_scan_odometry.py",
                "--worker",
                args.worker,
            ],
            "dataset": "public AIST GLIM versatile; DOI 10.5281/zenodo.6864654",
            "topic": args.topic,
            "seconds": args.seconds,
            "input_sha256": {str(p): _digest(p) for p in _inputs(args.bag)},
            "source_sha256": {p: _digest(Path(p)) for p in SOURCES},
            "numpy": np.__version__,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    serialized = measurement.model_dump(mode="json")
    path = args.output_dir / "measurement.json"
    path.write_text(json.dumps(serialized, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if Measurement.model_validate(json.loads(path.read_bytes())) != measurement:
        raise ValueError("saved measurement differs from API result")
    (args.output_dir / "measurement.schema.json").write_text(
        json.dumps(Measurement.model_json_schema(), indent=2) + "\n",
        encoding="utf-8",
    )


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topic", default="/livox/points")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--geometry-quantum-m", type=float, default=0.)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--worker", choices=("cpu", "cuda"))
    args = parser.parse_args(arguments)
    if (not 0 < args.seconds <= 60 or args.repetitions < 1 or not 0 < args.voxel_size < 1
        or not 0 <= args.geometry_quantum_m < 1):
        raise ValueError("seconds must be in (0, 60]; repetitions and voxel size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be new or empty")
    if args.worker:
        _worker(args)
        return
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

    bindings = {p: _digest(Path(p)) for p in SOURCES}
    inputs = {str(p): _digest(p) for p in _inputs(args.bag)}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    input_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "seconds": args.seconds,
                "topic": args.topic,
                "voxel_size_m": args.voxel_size,
                "geometry_quantum_m": args.geometry_quantum_m,
                "local_map_scans": 3,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def guard() -> None:
        if (
            revision != subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            or bindings != {p: _digest(Path(p)) for p in bindings}
            or inputs != {p: _digest(Path(p)) for p in inputs}
        ):
            raise ValueError("source, input bytes or HEAD changed during replay")

    trials, snapshots = [], []
    for repeat in range(1, args.repetitions + 1):
        pair = {}
        order = ("cpu", "cuda") if repeat % 2 else ("cuda", "cpu")
        for method in order:
            guard()
            folder = args.output_dir / f"{method}.{repeat:02d}"
            cache = args.output_dir / "cupy-cache" / method
            before = {p.name: _digest(p) for p in cache.rglob("*") if p.is_file()}
            command = [
                sys.executable,
                __file__,
                "--bag",
                str(args.bag),
                "--topic",
                args.topic,
                "--seconds",
                str(args.seconds),
                "--voxel-size",
                str(args.voxel_size),
                "--output-dir",
                str(folder),
                "--worker",
                method,
            ]
            command.extend(["--geometry-quantum-m", str(args.geometry_quantum_m)])
            print(f"START {repeat} {method}", flush=True)
            started = time.perf_counter()
            subprocess.run(
                command,
                check=True,
                env=dict(
                    os.environ,
                    CUPY_CACHE_DIR=str(cache.resolve()),
                    PYTHONIOENCODING="utf-8",
                ),
            )
            elapsed = time.perf_counter() - started
            after = {p.name: _digest(p) for p in cache.rglob("*") if p.is_file()}
            row = json.loads((folder / "measurement.json").read_bytes())
            if (
                row["provenance"]["source_sha256"] != bindings
                or row["provenance"]["input_sha256"] != inputs
                or row["provenance"]["git_commit"] != revision
            ):
                raise ValueError("worker provenance binding mismatch")
            pair[method] = row
            snapshots.append(
                {
                    "repeat": repeat,
                    "method": method,
                    "row": row,
                    "wall_seconds": elapsed,
                    "cache_before": before,
                    "cache_after": after,
                    "result_sha256": _digest(folder / "measurement.json"),
                }
            )
            print(f"DONE {repeat} {method} {elapsed:.6f}s", flush=True)
        errors = _compare(pair["cpu"]["result"], pair["cuda"]["result"])
        for name in ("scan_rows", "scan_sha256", "times_s", "options"):
            if pair["cpu"][name] != pair["cuda"][name]:
                errors.append(name)
        for snapshot in snapshots[-2:]:
            snapshot["comparison_errors"] = errors
        (args.output_dir / "trials.json").write_text(
            json.dumps(snapshots, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"COMPARISON {repeat} errors={errors[:12]}", flush=True)
    guard()
    for snapshot in snapshots:
        row, method = snapshot["row"], snapshot["method"]
        trials.append(
            BenchmarkTrial(
                method_id=method,
                split_id=f"repeat-{snapshot['repeat']:02d}",
                status="failed" if snapshot["comparison_errors"] else "success",
                failure_reason=(
                    "CPU/CUDA full-result parity failed: "
                    + ", ".join(snapshot["comparison_errors"][:8])
                    if snapshot["comparison_errors"]
                    else None
                ),
                runtime_seconds=snapshot["wall_seconds"],
                metrics={
                    "complete_wall_seconds": snapshot["wall_seconds"],
                    "api_seconds": row["api_seconds"],
                    "worker_cpu_seconds": row["worker_cpu_seconds"],
                    "quality_matches": float(not snapshot["comparison_errors"]),
                    "converged_steps": float(
                        sum(r["converged"] for r in row["result"]["registrations"])
                    ),
                },
                provenance=BenchmarkTrialProvenance(
                    command=f"tools/benchmark_cuda_scan_odometry.py --worker {method}",
                    config_sha256=config_hash,
                    input_sha256=input_hash,
                    output_sha256=snapshot["result_sha256"],
                    notes=[json.dumps(snapshot, sort_keys=True)],
                ),
            )
        )
    representative = snapshots[0]["row"]
    frame_hashes = representative["scan_sha256"]
    definition = BenchmarkDefinition(
        benchmark_id="public-glim-cuda-scan-odometry",
        title="Complete public LiDAR odometry replay",
        reference_method_id="cpu",
        bootstrap_samples=1,
        protocol=BenchmarkProtocol(
            protocol_id="cuda-scan-odometry/v0.1",
            dataset_id="glim_versatile",
            dataset_doi="10.5281/zenodo.6864654",
            dataset_source_sha256=input_hash,
            data_license="Public AIST GLIM dataset terms; raw points are not redistributed",
            split_policy="First frame seeds map; every subsequent frame scored online and "
            "then admitted to the local map. This is not an independent calibration holdout.",
            splits=[
                BenchmarkSplit(
                    split_id=f"repeat-{r:02d}",
                    seed=0,
                    fit_count=1,
                    holdout_count=len(frame_hashes) - 1,
                    fit_ids_sha256=frame_hashes[0],
                    holdout_ids_sha256=hashlib.sha256(
                        "".join(frame_hashes[1:]).encode()
                    ).hexdigest(),
                )
                for r in range(1, args.repetitions + 1)
            ],
            initial_estimate_policy="Identity for first relative step; "
            "native constant velocity thereafter",
            tuning_policy=f"Fixed {args.voxel_size} m voxel, three-scan map; finite rows; "
            f"geometry quantum {args.geometry_quantum_m} m; no tuning per method",
            failure_policy="Retain all cold runs and unconverged registration steps; "
            "never drop trials",
        ),
        methods=[
            BenchmarkMethodDefinition(
                method_id=m,
                label=m,
                implementation="subprocess",
                tool_name="Calibrex",
                tool_version=__version__,
                source_commit=revision,
                adapter_version="cuda-scan-odometry/v0.1",
            )
            for m in ("cpu", "cuda")
        ],
        metrics=[
            BenchmarkMetricDefinition(
                name=name,
                label=name,
                unit="s" if name.endswith("seconds") else "count",
                direction="lower" if name.endswith("seconds") else "higher",
                primary=name == "complete_wall_seconds",
                interpretation={
                    "complete_wall_seconds": "Fresh process, imports, bag decode, preprocessing, "
                    "GPU preparation/compilation, registrations, raw-input/source hashes and saves",
                    "api_seconds": "Stream decode, scan preparation, device startup "
                    "and complete odometry",
                    "worker_cpu_seconds": "All-thread CPU from worker entry through odometry "
                    "and result preparation",
                    "quality_matches": "All poses, transforms, iterations, counts, RMSE "
                    "and eigenvalue fields match",
                    "converged_steps": "Number of converged native registration steps; "
                    "failures retained",
                }[name],
            )
            for name in trials[0].metrics
        ],
        trials=trials,
        limitations=[
            "Real public temporal scans; CPU/GPU agreement is an execution check, "
            "not independent pose accuracy.",
            "Scans are treated as rigid snapshots; separate deskew parity is tested "
            "but not timed here.",
            "CPU/GPU reductions and eigensolves differ; abs 1e-9/rel 1e-6 for every float, "
            "exact counts/iterations/decisions; only verified execution identities differ.",
            "All paired runs, including first GPU compilation, retained. Separate initially empty "
            "compiler caches; order reversed every other pair. Parent guard/aggregation excluded.",
            "Fixed dense preprocessing/map settings; other point budgets and devices "
            "may have different speedups.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools.benchmark_cuda_scan_odometry",
            generator_version="v0.1",
            git_commit=revision,
            data_verified=True,
            source_artifacts={**bindings, **inputs},
            command=subprocess.list2cmdline(
                [sys.executable, __file__, *(arguments or sys.argv[1:])]
            ),
        ),
    )
    definition.save(args.output_dir / "comparison.definition.json")
    aggregate_benchmark_definition(definition).save(args.output_dir / "comparison.benchmark.json")
    if any(trial.status != "success" for trial in trials):
        raise ValueError("numerical parity failed; complete trials retained")


if __name__ == "__main__":
    main()
