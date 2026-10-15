"""Compare frozen CUDA map implementations on the complete public KITTI workload.

Each fresh process reads every input point, fits and evaluates the full result,
verifies actual imports and saves the schema-valid artifact. Compiler caches
are independent and empty for every worker unless retained mode is explicit.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools import benchmark_cuda_lidar_map as workload
from tools.benchmark_bspline_lidar_imu import _digest, _freeze, _verify_sources

_LAUNCHERS = (
    "tools/benchmark_cuda_map_preparation.py", "tools/benchmark_cuda_lidar_map.py",
    "tools/benchmark_bspline_lidar_imu.py",
    "tools/schemas/cuda_map_preparation_comparison.schema.json",
)


def _write(path: Path, value: object) -> None:
    if path.name in ("protocol.json", "attempts.json", "summary.json"):
        import jsonschema

        schema = json.loads(Path(__file__).with_name("schemas").joinpath(
            "cuda_map_preparation_comparison.schema.json",
        ).read_bytes())
        schema["$ref"] = "#/$defs/" + path.stem
        jsonschema.validate(value, schema)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Compare every physical evaluation field with the existing map tolerances."""
    prior, current = copy.deepcopy(left), copy.deepcopy(right)
    for value in (prior, current):
        errors = workload._compare(
            {"result": {"execution_backend": "numpy_scipy"}},
            {"result": {"execution_backend": value["result"]["execution_backend"]}},
        )
        if errors:
            return errors
    prior["result"]["execution_backend"] = "numpy_scipy"
    return workload._compare(prior, current)


def _worker(args: argparse.Namespace) -> None:
    root = args.source_root.resolve()
    expected = _verify_sources(root)
    cache = Path(os.environ["CUPY_CACHE_DIR"])
    cache_empty = not any(p.is_file() for p in cache.rglob("*"))
    workload._worker(argparse.Namespace(
        sequence=args.sequence, output_dir=args.output_dir, half_window=args.half_window,
        worker="cuda", source_root=root, source_commit=args.source_commit,
    ))
    imports = {}
    for name, module in list(sys.modules.items()):
        if not (name == "calibrex" or name.startswith("calibrex.")):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if path.suffix != ".py" or not path.is_relative_to(root / "src"):
            raise ValueError(f"calibration import escaped frozen source: {name}")
        relative = path.relative_to(root / "src").as_posix()
        if expected.get(relative) != _digest(path):
            raise ValueError(f"executed source changed: {name}")
        imports[relative] = _digest(path)
    _verify_sources(root)
    record_path = args.output_dir / "measurement.json"
    record = json.loads(record_path.read_bytes())
    record["provenance"].update(
        source_commit=args.source_commit, source_manifest_sha256=_digest(root / "manifest.json"),
        executed_sources_sha256=imports,
        launchers_sha256={p: _digest(Path(p)) for p in _LAUNCHERS},
    )
    record["cache_initially_empty"] = cache_empty
    _write(record_path, record)


def summarize(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute paired and complete-process medians without excluding any run."""
    medians = {
        field: {variant: statistics.median(
            row[field] for row in attempts if row["variant"] == variant
        ) for variant in ("prior", "current")}
        for field in ("wall_seconds", "api_seconds")
    }
    pairs = []
    for repeat in sorted({row["repeat"] for row in attempts}):
        pair = {row["variant"]: row for row in attempts if row["repeat"] == repeat}
        if len(pair) != 2:
            raise ValueError("incomplete CUDA comparison pair")
        pairs.append({"repeat": repeat, "whole_speedup":
                      pair["prior"]["wall_seconds"] / pair["current"]["wall_seconds"]})
    return {
        "workers": len(attempts), "quality_passed": all(not r["errors"] for r in attempts),
        "median_seconds": medians,
        "whole_speedup": medians["wall_seconds"]["prior"] / medians["wall_seconds"]["current"],
        "api_speedup": medians["api_seconds"]["prior"] / medians["api_seconds"]["current"],
        "pairs": pairs,
    }


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--half-window", type=int, default=1)
    parser.add_argument("--compiler-cache", choices=("cold", "retained"), default="cold")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    args = parser.parse_args(arguments)
    if args.repetitions < 1 or not 0 <= args.half_window <= 1:
        raise ValueError("positive repetitions and half-window 0 or 1 are required")
    args.sequence = args.sequence.resolve()
    if args.worker:
        if args.source_root is None or args.source_commit is None:
            raise ValueError("frozen worker source is required")
        _worker(args)
        return
    if not args.baseline_commit:
        raise ValueError("baseline commit is required")
    if args.output_dir.exists():
        raise ValueError("output directory must be new")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    baseline = subprocess.check_output(
        ["git", "rev-parse", args.baseline_commit + "^{commit}"], text=True,
    ).strip()
    # Freeze committed runtime bytes on both sides; working edits cannot silently
    # enter an observation bound to a different source revision.
    changed = subprocess.check_output(["git", "diff", "HEAD", "--", "src/calibrex"], text=True)
    if changed:
        raise ValueError("commit runtime changes before measurement")
    inputs = {str(p): _digest(p) for p in workload._paths(args.sequence, args.half_window)}
    launchers = {p: _digest(Path(p)) for p in _LAUNCHERS}
    args.output_dir.mkdir(parents=True)
    roots = {v: args.output_dir / f"sources-{v}" for v in ("prior", "current")}
    commits = {"prior": baseline, "current": current}
    for variant, root in roots.items():
        _freeze(root, commits[variant])
    protocol = {
        "schema_version": "calibrex.cuda_map_preparation_comparison/v0.1",
        "baseline_commit": baseline, "current_commit": current, "repetitions": args.repetitions,
        "half_window": args.half_window, "compiler_cache": args.compiler_cache,
        "blas_threads": 1, "input_sha256": inputs, "launchers_sha256": launchers,
        "source_manifest_sha256": {v: _digest(r / "manifest.json") for v, r in roots.items()},
        "quality_policy": "Existing full map-evaluation tolerances; exact counts/status/policy; "
        "separately validate both execution identities against their frozen kernel bytes",
        "scope": __doc__,
    }
    _write(args.output_dir / "protocol.json", protocol)

    def guard() -> None:
        if (subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != current
            or {p: _digest(Path(p)) for p in launchers} != launchers
            or {p: _digest(Path(p)) for p in inputs} != inputs):
            raise ValueError("HEAD, launchers or inputs changed during measurement")
        for root in roots.values():
            _verify_sources(root)

    attempts = []
    for repeat in range(args.repetitions):
        order = ("prior", "current") if repeat % 2 == 0 else ("current", "prior")
        pair = {}
        for variant in order:
            guard()
            worker = f"{repeat:02d}-{variant}"
            folder = args.output_dir / worker
            cache = args.output_dir / "cupy-cache" / (
                worker if args.compiler_cache == "cold" else variant
            )
            cache_before = {p.name: _digest(p) for p in cache.rglob("*") if p.is_file()}
            command = [sys.executable, "-m", "tools.benchmark_cuda_map_preparation", "--worker",
                       "--sequence", str(args.sequence), "--output-dir", str(folder),
                       "--source-root", str(roots[variant]), "--source-commit", commits[variant],
                       "--half-window", str(args.half_window)]
            env = dict(os.environ, CUPY_CACHE_DIR=str(cache.resolve()),
                       PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", PYTHONIOENCODING="utf-8",
                       PYTHONPATH=str((roots[variant] / "src").resolve())
                       + os.pathsep + str(Path.cwd()))
            folder.mkdir()
            print(f"START {worker}", flush=True)
            started = time.perf_counter()
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         check=False)
            wall = time.perf_counter() - started
            if process.returncode:
                raise ValueError(f"worker failed; retained log: {folder / 'worker.log'}")
            row = json.loads((folder / "measurement.json").read_bytes())
            if (row["provenance"]["input_sha256"] != inputs
                or row["provenance"]["source_commit"] != commits[variant]
                or row["provenance"]["launchers_sha256"] != launchers
                or row["provenance"]["source_manifest_sha256"]
                    != protocol["source_manifest_sha256"][variant]
                or row["cache_initially_empty"] != (not cache_before)):
                raise ValueError("worker source, input or cache binding mismatch")
            pair[variant] = row
            attempts.append({
                "repeat": repeat, "variant": variant, "worker": worker, "wall_seconds": wall,
                "api_seconds": row["api_seconds"], "cache_before": cache_before,
                "cache_after": {p.name: _digest(p) for p in cache.rglob("*") if p.is_file()},
                "errors": [],
            })
            _write(args.output_dir / "attempts.json", attempts)
            print(f"DONE {worker} wall={wall:.6f}s api={row['api_seconds']:.6f}s", flush=True)
        errors = compare(pair["prior"]["evaluation"], pair["current"]["evaluation"])
        for key in ("prepared_sha256", "partition_sha256", "map_points", "target_points"):
            if pair["prior"][key] != pair["current"][key]:
                errors.append(key)
        for row in attempts[-2:]:
            row["errors"] = errors
        _write(args.output_dir / "attempts.json", attempts)
    guard()
    summary = summarize(attempts)
    summary["provenance"] = {"protocol_sha256": _digest(args.output_dir / "protocol.json")}
    _write(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not summary["quality_passed"]:
        raise ValueError("complete evaluation changed; every observation retained")


if __name__ == "__main__":
    main()
