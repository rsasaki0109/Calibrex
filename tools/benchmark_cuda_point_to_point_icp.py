"""Fresh-process public-cloud control; native updates, full diagnostics, all trials.

The target is a noisy transformed copy of one KITTI scan. This benchmarks
execution equivalence, not accuracy between independently recorded sensors.
No CuPy import occurs in the parent or native worker.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

SOURCES = (
    "src/calibrex/solvers/robust_point_to_point_icp_solver.py",
    "src/calibrex/solvers/cuda_point_to_point_icp_adapter.py",
    "src/calibrex/solvers/cuda_lidar_map_adapter.py",
    "src/calibrex/solvers/_exact_neighbors.py",
    "src/calibrex/core/geometry.py",
    "src/calibrex/evaluation/holdout.py",
    "tools/benchmark_cuda_point_to_point_icp.py",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity() -> dict[str, object]:
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "sources": {s: _sha(Path(s).read_bytes()) for s in SOURCES}}


def _worker(args: argparse.Namespace) -> None:
    from threadpoolctl import threadpool_info

    from calibrex.core.geometry import SE3
    from calibrex.solvers.robust_point_to_point_icp_solver import (
        IcpPoint,
        RobustPointToPointIcpOptions,
        RobustPointToPointIcpSolver,
        _transform_points,
    )

    output = Path(args.output)
    output.mkdir(parents=True)
    os.environ["CUPY_CACHE_DIR"] = str((output / "cupy-cache").resolve())
    before = _identity()
    raw = Path(args.input).read_bytes()
    source = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    truth = SE3((.03, -.02, .015), (.001, -.002, .0015, 1.))
    target = _transform_points(source, truth) + np.random.default_rng(args.seed).normal(
        0., .002, source.shape)
    inputs = [[IcpPoint(f"{name}-{i:07d}", tuple(row)) for i, row in enumerate(cloud)]
              for name, cloud in (("source", source), ("target", target))]
    options = RobustPointToPointIcpOptions(multi_start_workers=args.workers)
    started = time.perf_counter()
    adapter_provenance = None
    if args.method == "cuda":
        from calibrex.solvers.cuda_point_to_point_icp_adapter import CudaPointToPointIcpAdapter
        accelerated = CudaPointToPointIcpAdapter().solve(*inputs, options=options)
        result = accelerated.result
        adapter_provenance = accelerated.provenance
    else:
        result = RobustPointToPointIcpSolver().solve(*inputs, options=options)
    seconds = time.perf_counter() - started
    after = _identity()
    if before != after or _sha(Path(args.input).read_bytes()) != _sha(raw):
        raise RuntimeError("source, HEAD or input changed during measurement")
    if args.method == "native" and "cupy" in sys.modules:
        raise RuntimeError("native control imported CuPy")
    model = result.as_dict()
    result_bytes = json.dumps(model, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    (output / "result.json").write_bytes(result_bytes)
    record = {"method": args.method, "seconds": seconds, "point_count": len(source),
              "result_sha256": _sha(result_bytes), "provenance": {
                  "source": before, "input_sha256": _sha(raw),
                  "input": "<DATASET_ROOT>/" + Path(args.input).name,
                  "seed": args.seed, "noise_sigma_m": .002, "truth": truth.as_dict(),
                  "options": asdict(options), "python": sys.version, "numpy": np.__version__,
                  "scipy": importlib.metadata.version("scipy"),
                  "blas_environment": {key: os.environ.get(key) for key in (
                      "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                  "blas_runtime": [{key: entry.get(key) for key in (
                      "user_api", "internal_api", "num_threads", "version", "threading_layer")}
                      for entry in threadpool_info()],
                  "cupy_imported": "cupy" in sys.modules,
                  "adapter": adapter_provenance}}
    (output / "measurement.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"method": args.method, "seconds": seconds, "status": result.status}),
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--method", choices=("native", "cuda"))
    args = parser.parse_args()
    if args.method:
        _worker(args)
        return
    if args.repeats < 1 or args.blas_threads < 1:
        parser.error("--repeats and --blas-threads must be positive")
    output = Path(args.output)
    output.mkdir(parents=True)
    identity = _identity()
    source_dir = output / "sources"
    source_dir.mkdir()
    for source in SOURCES:
        (source_dir / (Path(source).name + ".txt")).write_bytes(Path(source).read_bytes())
    trials = []
    anchor = None
    for repeat in range(1, args.repeats + 1):
        order = ("native", "cuda") if repeat % 2 else ("cuda", "native")
        for method in order:
            child = output / f"{method}.{repeat:02d}"
            command = [sys.executable, __file__, "--input", args.input, "--output", str(child),
                       "--method", method, "--workers", str(args.workers), "--seed", str(args.seed)]
            environment = dict(os.environ, PYTHONFAULTHANDLER="1",
                               OPENBLAS_NUM_THREADS=str(args.blas_threads),
                               OMP_NUM_THREADS=str(args.blas_threads),
                               MKL_NUM_THREADS=str(args.blas_threads))
            started = time.perf_counter()
            with (output / f"{method}.{repeat:02d}.log.txt").open("wb") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=environment)
            whole_seconds = time.perf_counter() - started
            measurement = json.loads((child / "measurement.json").read_text(encoding="utf-8"))
            result = (child / "result.json").read_bytes()
            if anchor is None:
                anchor = result
            if anchor != result:
                raise RuntimeError(f"full result differs: {method}, trial {repeat}")
            if measurement["provenance"]["source"] != identity:
                raise RuntimeError("worker source identity differs")
            blas = measurement["provenance"]["blas_runtime"]
            if not blas or any(entry["num_threads"] != args.blas_threads for entry in blas
                               if entry["user_api"] == "blas"):
                raise RuntimeError("worker BLAS thread budget was not applied")
            trial = {"repeat": repeat, "order": list(order),
                     "subprocess_wall_seconds": whole_seconds, **measurement}
            trials.append(trial)
            print(json.dumps({"repeat": repeat, "method": method,
                              "seconds": measurement["seconds"], "whole_seconds": whole_seconds,
                              "full_result_equal": True}),
                  flush=True)
    medians = {method: statistics.median(t["seconds"] for t in trials if t["method"] == method)
               for method in ("native", "cuda")}
    summary = {"provenance": identity, "trials": trials, "medians_s": medians,
               "speedup": medians["native"] / medians["cuda"], "full_result_equal": True,
               "cold_policy": "fresh worker and empty CUDA disk cache for every trial",
               "scope": "complete solve, spatial holdout, rematching, six multi-start diagnostics",
               "accuracy_scope": "copied public scan execution control"}
    whole = {method: statistics.median(t["subprocess_wall_seconds"] for t in trials
                                      if t["method"] == method) for method in ("native", "cuda")}
    summary["whole_medians_s"] = whole
    summary["whole_speedup"] = whole["native"] / whole["cuda"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
