"""Fresh-process public-cloud control; native updates, full diagnostics, all trials.

Inputs can be a noisy copied KITTI scan or a provenance-locked actual frame pair.
Navigation consistency does not establish independent-sensor accuracy.
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
from dataclasses import asdict, replace
from pathlib import Path

import jsonschema
import numpy as np

SOURCES = (
    "src/calibrex/solvers/robust_point_to_point_icp_solver.py",
    "src/calibrex/solvers/cuda_point_to_point_icp_adapter.py",
    "src/calibrex/solvers/cuda_lidar_map_adapter.py",
    "src/calibrex/solvers/_exact_neighbors.py",
    "src/calibrex/core/geometry.py",
    "src/calibrex/evaluation/holdout.py",
    "tools/benchmark_cuda_point_to_point_icp.py",
    "tools/schemas/icp_pair_fixture.schema.json",
)


def _load_pair_fixture(path: Path, root: Path) -> tuple[dict, dict[str, bytes]]:
    fixture = json.loads(path.read_bytes())
    jsonschema.validate(fixture, json.loads(Path(SOURCES[-1]).read_bytes()))
    files = {}
    base = root.resolve()
    for name, digest in fixture["provenance"]["input_sha256"].items():
        target = (base / name).resolve()
        if not target.is_relative_to(base):
            raise ValueError("fixture input is outside its dataset root")
        raw = target.read_bytes()
        if _sha(raw) != digest:
            raise ValueError(f"fixture input digest mismatch: {name}")
        files[name] = raw
    if fixture["source"] not in files or fixture["target"] not in files:
        raise ValueError("fixture pointclouds must be digest-bound")
    return fixture, files


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
    fixture = None
    bound_files = None
    if args.pair_fixture:
        fixture, bound_files = _load_pair_fixture(Path(args.pair_fixture), Path(args.data_root))
        raw = bound_files[fixture["source"]]
    else:
        raw = Path(args.input).read_bytes()
    source = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    truth = SE3((.03, -.02, .015), (.001, -.002, .0015, 1.))
    if fixture is not None:
        target = np.frombuffer(bound_files[fixture["target"]], dtype=np.float32).reshape(-1, 4)[
            :, :3].astype(np.float64)
        initial = SE3.from_lists(**fixture["initial_transform"])
        options = RobustPointToPointIcpOptions(**{**fixture["options"],
                                                "multi_start_workers": args.workers})
    else:
        target = _transform_points(source, truth) + np.random.default_rng(args.seed).normal(
            0., .002, source.shape)
        initial = None
        options = RobustPointToPointIcpOptions(multi_start_workers=args.workers)
    if args.point_budget:
        source = source[np.linspace(0, len(source)-1, min(len(source), args.point_budget),
                                    dtype=np.int64)]
        target = target[np.linspace(0, len(target)-1, min(len(target), args.point_budget),
                                    dtype=np.int64)]
    inputs = [[IcpPoint(f"{name}-{i:07d}", tuple(row)) for i, row in enumerate(cloud)]
              for name, cloud in (("source", source), ("target", target))]
    started = time.perf_counter()
    adapter_provenance = None
    if args.method == "cuda":
        from calibrex.solvers.cuda_point_to_point_icp_adapter import CudaPointToPointIcpAdapter
        accelerated = CudaPointToPointIcpAdapter().solve(*inputs, initial_transform=initial,
                                                       options=options)
        result = accelerated.result
        adapter_provenance = accelerated.provenance
    else:
        result = RobustPointToPointIcpSolver().solve(*inputs, initial_transform=initial,
                                                   options=options)
    seconds = time.perf_counter() - started
    after = _identity()
    unchanged = (_load_pair_fixture(Path(args.pair_fixture), Path(args.data_root)) == (
                     fixture, bound_files)
                 if fixture is not None else _sha(Path(args.input).read_bytes()) == _sha(raw))
    if before != after or not unchanged:
        raise RuntimeError("source, HEAD or input changed during measurement")
    if args.method == "native" and "cupy" in sys.modules:
        raise RuntimeError("native control imported CuPy")
    model = result.as_dict()
    result_bytes = json.dumps(model, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    (output / "result.json").write_bytes(result_bytes)
    record = {"method": args.method, "seconds": seconds, "point_count": len(source),
              "target_point_count": len(target),
              "result_sha256": _sha(result_bytes), "provenance": {
                  "source": before, "input_sha256": _sha(raw),
                  "input": "<DATASET_ROOT>/" + (fixture["source"] if fixture else
                                                   Path(args.input).name),
                  "seed": args.seed, "noise_sigma_m": None if fixture else .002,
                  "truth": None if fixture else truth.as_dict(),
                  "options": asdict(options), "python": sys.version, "numpy": np.__version__,
                  "scipy": importlib.metadata.version("scipy"),
                  "blas_environment": {key: os.environ.get(key) for key in (
                      "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                  "blas_runtime": [{key: entry.get(key) for key in (
                      "user_api", "internal_api", "num_threads", "version", "threading_layer")}
                      for entry in threadpool_info()],
                  "cupy_imported": "cupy" in sys.modules,
                  "adapter": adapter_provenance}}
    if fixture is not None:
        from calibrex.solvers.robust_point_to_point_icp_solver import evaluate_icp_candidate
        reference = SE3.from_lists(**fixture["reference_transform"])
        initial_evaluation = evaluate_icp_candidate(*inputs, initial, options=replace(
            options, effective_diagnostics=False, multi_start_diagnostics=False))
        estimated = result.transform_target_source
        difference = reference.inverse().compose(estimated) if estimated else None
        record["evaluation"] = {
            "initial_holdout_rmse_m": initial_evaluation.holdout_rmse_m,
            "final_holdout_rmse_m": result.holdout_rmse_m,
            "navigation_translation_difference_m": float(np.linalg.norm(
                difference.translation_m)) if difference else None,
            "navigation_rotation_difference_deg": float(np.degrees(2*np.arccos(min(
                1., abs(difference.rotation_quat_xyzw[3]))))) if difference else None,
        }
        record["provenance"].update({"fixture": fixture,
                                    "fixture_sha256": _sha(Path(args.pair_fixture).read_bytes()),
                                    "point_budget": args.point_budget})
    (output / "measurement.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"method": args.method, "seconds": seconds, "status": result.status}),
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--pair-fixture")
    parser.add_argument("--data-root")
    parser.add_argument("--point-budget", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--method", choices=("native", "cuda"))
    args = parser.parse_args()
    if bool(args.input) == bool(args.pair_fixture) or (args.pair_fixture and not args.data_root):
        parser.error("provide --input, or --pair-fixture together with --data-root")
    if args.point_budget < 0:
        parser.error("--point-budget must be non-negative")
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
            command = [sys.executable, __file__, "--output", str(child),
                       "--method", method, "--workers", str(args.workers), "--seed", str(args.seed)]
            command += (["--pair-fixture", args.pair_fixture, "--data-root", args.data_root]
                        if args.pair_fixture else ["--input", args.input])
            command += ["--point-budget", str(args.point_budget)]
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
               "accuracy_scope": ("actual frames with OXTS seed; un-deskewed navigation consistency"
                                  if args.pair_fixture else "copied public scan execution control")}
    whole = {method: statistics.median(t["subprocess_wall_seconds"] for t in trials
                                      if t["method"] == method) for method in ("native", "cuda")}
    summary["whole_medians_s"] = whole
    summary["whole_speedup"] = whole["native"] / whole["cuda"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
