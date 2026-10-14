"""Frozen old/new CUDA comparisons of complete public KITTI map evaluation.

The copied-sensor control includes all fits, holdout, signed controls and
jackknife fits. It does not measure reference odometry or two-sensor accuracy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _quality(prior: dict[str, Any], current: dict[str, Any]) -> bool:
    left, right = copy.deepcopy(prior["evaluation"]), copy.deepcopy(current["evaluation"])
    left["result"].pop("execution_backend")
    right["result"].pop("execution_backend")
    return (
        left == right and prior["prepared_sha256"] == current["prepared_sha256"]
        and prior["partition_sha256"] == current["partition_sha256"]
        and prior["provenance"]["input_sha256"] == current["provenance"]["input_sha256"]
        and prior["map_points"] == current["map_points"]
        and prior["target_points"] == current["target_points"]
    )


def _worker(args: argparse.Namespace) -> None:
    from tools.benchmark_bspline_lidar_imu import _verify_sources

    root = args.source_root.resolve()
    manifest = _verify_sources(root)
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)
    cache = Path(os.environ["CUPY_CACHE_DIR"])
    if cache.exists() and any(cache.iterdir()):
        raise ValueError("worker compilation cache must initially be empty")
    launcher = root / "tools/benchmark_cuda_lidar_map.py"
    loader = importlib.util.spec_from_file_location("_bound_map_benchmark", launcher)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    module._worker(SimpleNamespace(
        sequence=args.sequence, output_dir=args.output, half_window=0, worker="cuda",
        source_commit=args.source_commit,
    ))
    record = json.loads((args.output / "measurement.json").read_bytes())
    executed: dict[str, str] = {}
    for name, imported in list(sys.modules.items()):
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(imported.__file__).resolve()
            if not path.is_relative_to(root / "src"):
                raise ValueError("worker imported outside the frozen source tree")
            relative = path.relative_to(root / "src").as_posix()
            if _sha(path) != manifest[relative]:
                raise ValueError("executed frozen source changed")
            executed[relative] = manifest[relative]
    _verify_sources(root)
    record["frozen_provenance"] = {
        "source_commit": args.source_commit, "source_manifest_sha256": _sha(root / "manifest.json"),
        "executed_sources": executed, "launcher_sha256": _sha(Path(__file__)),
        "map_launcher_sha256": _sha(launcher), "cuda_cache_initially_empty": True,
        "threads": {key: os.environ[key] for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        )},
    }
    record["schema_version"] = "calibrex.cuda_fixed_neighbors/v0.1"
    record["record_type"] = "worker"
    _write(args.output / "measurement.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--baseline-commit", default="efd6ce8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.sequence = args.sequence.resolve()
    if args.worker:
        _worker(args)
        return
    if args.repeats < 1:
        parser.error("repeats must be positive")
    from tools.benchmark_bspline_lidar_imu import _freeze

    root = args.output
    root.mkdir(exist_ok=False)
    launcher = root / "benchmark.py.txt"
    shutil.copyfile(__file__, launcher)
    map_launcher = root / "map-benchmark.py.txt"
    shutil.copyfile("tools/benchmark_cuda_lidar_map.py", map_launcher)
    commits = {
        "prior": subprocess.check_output(
            ["git", "rev-parse", args.baseline_commit], text=True,
        ).strip(),
        "current": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    for variant, commit in commits.items():
        source = root / f"sources-{variant}"
        _freeze(source, commit)
        (source / "tools").mkdir()
        shutil.copyfile(map_launcher, source / "tools/benchmark_cuda_lidar_map.py")
    raw_files = [args.sequence / "velodyne_points/data" / f"{center:010d}.bin"
                 for center in (115, 230, 345)]
    protocol = {
        "schema_version": "calibrex.cuda_fixed_neighbors/v0.1", "record_type": "protocol",
        "repeats": args.repeats,
        "commits": commits, "launcher_sha256": _sha(launcher),
        "map_launcher_sha256": _sha(map_launcher),
        "input_sha256": {str(path): _sha(path) for path in raw_files},
        "scope": ("fresh process through complete copied-sensor map evaluation and saved "
                  "validation; reference odometry excluded"),
        "initially_empty_cupy_caches": True,
    }
    _write(root / "protocol.json", protocol)
    records = []
    for repeat in range(args.repeats):
        for variant in (["prior", "current"] if repeat % 2 == 0 else ["current", "prior"]):
            folder = root / f"{repeat:02d}-{variant}"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path.cwd())
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                env[key] = "1"
            env["CUPY_CACHE_DIR"] = str(root / "caches" / folder.name)
            command = [sys.executable, str(launcher), "--worker", "--output", str(folder),
                       "--sequence", str(args.sequence), "--source-root",
                       str(root / f"sources-{variant}"), "--source-commit", commits[variant]]
            started = time.perf_counter()
            with (root / f"{folder.name}.log").open("w", encoding="utf-8") as output:
                result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - started
            record = {"repeat": repeat, "variant": variant, "worker": folder.name,
                      "exit_code": result.returncode, "whole_seconds": elapsed}
            records.append(record)
            _write(root / "attempts.json", records)
            print(json.dumps(record), flush=True)
            if result.returncode:
                raise RuntimeError("worker failed; inspect retained log")
            if _sha(launcher) != protocol["launcher_sha256"] or any(
                _sha(Path(path)) != digest for path, digest in protocol["input_sha256"].items()
            ):
                raise ValueError("launcher or raw input changed during comparison")
    quality = [_quality(
        json.loads((root / f"{repeat:02d}-prior/measurement.json").read_bytes()),
        json.loads((root / f"{repeat:02d}-current/measurement.json").read_bytes()),
    ) for repeat in range(args.repeats)]
    whole = {variant: statistics.median(record["whole_seconds"] for record in records
                                       if record["variant"] == variant)
             for variant in commits}
    api = {variant: statistics.median(json.loads(
        (root / record["worker"] / "measurement.json").read_bytes())["api_seconds"]
        for record in records if record["variant"] == variant) for variant in commits}
    summary = {
        "schema_version": "calibrex.cuda_fixed_neighbors/v0.1", "record_type": "summary",
        "whole_median_seconds": whole, "whole_speedup": whole["prior"] / whole["current"],
        "api_median_seconds": api, "api_speedup": api["prior"] / api["current"],
        "quality_exact": quality, "quality_passed": all(quality),
        "provenance": {"protocol_sha256": _sha(root / "protocol.json"),
                       "attempts_sha256": _sha(root / "attempts.json")},
    }
    _write(root / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not all(quality):
        raise ValueError("complete public evaluation differs")


if __name__ == "__main__":
    main()
