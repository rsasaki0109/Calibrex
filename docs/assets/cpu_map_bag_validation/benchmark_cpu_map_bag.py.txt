"""Compare CPU map execution through complete copied-topic bag calibration.

Reference odometry is common to all arms. The target copies the same topic with
fixed measurement noise; this is a spent-data execution control, not independent
two-LiDAR accuracy evidence. Run from the repository root with committed sources.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from tools.benchmark_bspline_lidar_imu import _digest, _freeze, _same_float_bits, _verify_sources
from tools.benchmark_cuda_lidar_map import _json

_VARIANTS = ("prior", "lazy", "parallel")
_THREADS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
_HELPERS = (
    "tools/benchmark_cpu_map_bag.py",
    "tools/benchmark_bspline_lidar_imu.py",
    "tools/benchmark_cuda_lidar_map.py",
    "tools/schemas/cpu_map_bag_comparison.schema.json",
)
_SCHEMA = Path(__file__).with_name("schemas") / "cpu_map_bag_comparison.schema.json"


def _read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _write(path: Path, value: dict[str, Any], kind: str) -> None:
    import jsonschema

    schema = _read(_SCHEMA)
    schema["$ref"] = "#/$defs/" + kind
    jsonschema.validate(value, schema)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source(root: Path, variant: str) -> Path:
    return root / ("source-prior" if variant == "prior" else "source-current")


def _physical(record: dict[str, Any]) -> dict[str, Any]:
    artifact = copy.deepcopy(record["artifact"])
    for key in ("created_at", "command", "git_commit"):
        artifact["provenance"].pop(key, None)
    artifact["options"].pop("registration_query_workers", None)
    options = copy.deepcopy(record["options"])
    options["solver"].pop("query_workers", None)
    return {
        "artifact": artifact,
        "options": options,
        "fits": record["fits"],
        "prepared_samples": record["prepared_samples"],
    }


def compare_workers(prior: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Require every physical/evidence bit after narrowly declared metadata."""
    errors = []
    for record in (prior, current):
        budget = record["provenance"]["query_workers"]
        if (
            type(budget) is not int
            or budget < 1
            or record["options"]["solver"].get("query_workers", 1) != budget
            or record["artifact"]["options"].get("registration_query_workers", 1) != budget
        ):
            errors.append("query_workers")
        command = [
            "python",
            "-m",
            "tools.benchmark_cpu_map_bag",
            "--worker",
            record["variant"],
            "--scenario",
            record["scenario"],
            "--repeat",
            str(record["repeat"]),
        ]
        if record["artifact"]["provenance"]["command"] != command:
            errors.append("command")
        if record["artifact"]["provenance"]["git_commit"] != record["provenance"]["source_commit"]:
            errors.append("source_commit")
    left, right = _physical(prior), _physical(current)
    errors.extend(key for key in left if not _same_float_bits(left[key], right[key]))
    for key in ("bag_sha256", "odometry_backend"):
        if prior["provenance"][key] != current["provenance"][key]:
            errors.append(key)
    return errors


def summarize(triads: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep each density cohort separate and include every complete worker."""
    if not triads:
        raise ValueError("no complete triads")
    if len({row["scenario"] for row in triads}) != 1:
        raise ValueError("density cohorts must remain separate")
    if len({row["repeat"] for row in triads}) != len(triads):
        raise ValueError("duplicate triad repeat")
    if sorted(row["repeat"] for row in triads) != list(range(len(triads))):
        raise ValueError("incomplete triad repeats")
    for row in triads:
        if set(row["seconds"]) != set(_VARIANTS) or any(
            not math.isfinite(value) or value <= 0 for value in row["seconds"].values()
        ):
            raise ValueError("incomplete or invalid triad times")
        if set(row["errors"]) != {"lazy", "parallel"}:
            raise ValueError("incomplete triad quality")
    medians = {
        variant: statistics.median(row["seconds"][variant] for row in triads)
        for variant in _VARIANTS
    }
    return {
        "workers": len(triads) * 3,
        "fits": len(triads) * 15,
        "quality_passed": all(not errors for row in triads for errors in row["errors"].values()),
        "median_seconds": medians,
        "prior_to_lazy": medians["prior"] / medians["lazy"],
        "prior_to_parallel": medians["prior"] / medians["parallel"],
        "lazy_to_parallel": medians["lazy"] / medians["parallel"],
        "parallel_faster_pairs": sum(
            row["seconds"]["parallel"] < row["seconds"]["prior"] for row in triads
        ),
    }


def _worker(args: argparse.Namespace) -> None:
    root = args.output_dir.resolve()
    protocol = _read(root / "protocol.json")
    if {name: _digest(Path(name)) for name in _HELPERS} != protocol["helpers_sha256"]:
        raise ValueError("worker benchmark helpers differ")
    variant = args.worker
    source = _source(root, variant)
    expected = _verify_sources(source)
    sys.path.insert(0, str(source / "src"))
    import numpy as np

    from calibrex.core.io import read_mapping
    from calibrex.core.lidar_lidar_extrinsic import LidarLidarExtrinsicArtifact
    from calibrex.evaluation import lidar_lidar_map as runner

    scenario = protocol["scenarios"][args.scenario]
    settings = protocol["settings"]
    budget = protocol["query_workers"][variant]
    folder = root / args.scenario / f"{args.repeat:02d}-{variant}"
    cuda = settings["odometry_backend"] == "cuda"
    cache_empty = None
    if cuda:
        cache = Path(os.environ["CUPY_CACHE_DIR"])
        cache_empty = not cache.exists() or not any(cache.rglob("*"))
        if not cache_empty or os.environ.get("CUDA_CACHE_DISABLE") != "1":
            raise ValueError("worker CUDA compiler cache is not initially empty")
    options = runner.LidarLidarMapOptions(
        sample_every_scans=settings["sample_every_scans"],
        map_half_window_scans=settings["map_half_window_scans"],
        block_duration_s=settings["block_duration_s"],
        jackknife_groups=settings["jackknife_groups"],
    )
    options = replace(
        options,
        preprocess=replace(options.preprocess, voxel_size_m=scenario["map_voxel_size_m"]),
        odometry=replace(options.odometry, **settings["odometry_options"]),
    )
    if variant != "prior":
        options = replace(options, solver=replace(options.solver, query_workers=budget))
    iterator = runner.iter_livox_points
    streams = 0
    rng = np.random.default_rng(settings["target_noise_seed"])

    def copied_target(*arguments: Any, **keywords: Any) -> Any:
        nonlocal streams
        target = streams == 1
        streams += 1
        for stamp, points, offsets in iterator(*arguments, **keywords):
            if target:
                points = points + rng.normal(0.0, settings["target_noise_std_m"], points.shape)
            yield stamp, points, offsets

    runner.iter_livox_points = copied_target
    captured = []
    fits = []
    fit_seconds = []
    stages: dict[str, float] = {}
    sample_objects = []
    original_collect = runner.collect_map_samples

    def collect(*arguments: Any, **keywords: Any) -> Any:
        started = time.perf_counter()
        data, scans = original_collect(*arguments, **keywords)
        stages["collect_seconds"] = time.perf_counter() - started
        sample_objects.extend(data)
        for sample in data:
            entry = {"time_s": sample.time_s, "block": sample.block}
            for name in ("reference_pose", "map_points", "target_points"):
                values = np.ascontiguousarray(getattr(sample, name))
                entry[name] = {
                    "dtype": values.dtype.str,
                    "shape": list(values.shape),
                    "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                }
            captured.append(entry)
        return data, scans

    runner.collect_map_samples = collect
    original_fit = runner.solve_map_extrinsic

    def fit(*arguments: Any, **keywords: Any) -> Any:
        started = time.perf_counter()
        result = original_fit(*arguments, **keywords)
        fit_seconds.append(time.perf_counter() - started)
        fits.append(_json(result))
        return result

    runner.solve_map_extrinsic = fit
    original_evaluate = runner.evaluate_lidar_lidar_map

    def evaluate(*arguments: Any, **keywords: Any) -> Any:
        started = time.perf_counter()
        result = original_evaluate(*arguments, **keywords)
        stages["evaluate_seconds"] = time.perf_counter() - started
        return result

    runner.evaluate_lidar_lidar_map = evaluate
    initial = np.eye(4)
    initial[:3, 3] = settings["initial_translation_m"]
    command = [
        "python",
        "-m",
        "tools.benchmark_cpu_map_bag",
        "--worker",
        variant,
        "--scenario",
        args.scenario,
        "--repeat",
        str(args.repeat),
    ]
    started = time.perf_counter()
    result = runner.run_ros2_lidar_lidar_map(
        protocol["bag"]["path"],
        reference_topic=settings["topic"],
        target_topic=settings["topic"],
        initial=initial,
        point_time_field=settings["point_time_field"],
        options=options,
        max_seconds=settings["max_seconds"],
        backend="cpu",
        odometry_backend=settings["odometry_backend"],
        dataset_family=settings["dataset_family"],
        dataset_license=settings["dataset_license"],
        command=command,
    )
    commit = protocol["commits"]["prior" if variant == "prior" else "current"]
    result.provenance.git_commit = commit
    result.save(folder / "result.json")
    api_seconds = time.perf_counter() - started
    if LidarLidarExtrinsicArtifact.model_validate(read_mapping(folder / "result.json")) != result:
        raise ValueError("saved complete bag artifact differs")
    if (
        streams != 2
        or len(fits) != 5
        or result.jackknife_fits != 4
        or len(result.dofs) != 6
        or any(row.known_bad_control is None for row in result.dofs)
    ):
        raise ValueError("worker did not complete all five fits and six controls")
    executed = {}
    for name, module in list(sys.modules.items()):
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(source / "src"):
                raise ValueError("calibration imported outside frozen sources: " + name)
            key = path.relative_to(source / "src").as_posix()
            if _digest(path) != expected[key]:
                raise ValueError("executed source differs: " + key)
            executed[key] = expected[key]
    _verify_sources(source)
    if not cuda and "cupy" in sys.modules:
        raise ValueError("CPU worker imported CUDA")
    normal_rows = 0
    for sample in sample_objects:
        cache = getattr(sample, "_normal_cache", None)
        normal_rows += int(cache.done.sum()) if cache is not None else len(sample.map_points)
    record = {
        "variant": variant,
        "scenario": args.scenario,
        "repeat": args.repeat,
        "api_seconds": api_seconds,
        "stages": stages,
        "fit_seconds": fit_seconds,
        "artifact": result.model_dump(mode="json", exclude_none=True),
        "fits": fits,
        "options": asdict(options),
        "prepared_samples": captured,
        "work": {
            "normal_rows": normal_rows,
            "map_rows": sum(len(s.map_points) for s in sample_objects),
            "target_rows": sum(len(s.target_points) for s in sample_objects),
        },
        "provenance": {
            "source_commit": commit,
            "source_manifest_sha256": _digest(source / "manifest.json"),
            "executed_sources": executed,
            "bag_sha256": _digest(Path(protocol["bag"]["path"])),
            "result_sha256": _digest(folder / "result.json"),
            "query_workers": budget,
            "odometry_backend": settings["odometry_backend"],
            "cuda_cache_initially_empty": cache_empty,
            "blas_threads": {k: os.environ[k] for k in _THREADS},
            "protocol_sha256": _digest(root / "protocol.json"),
            "helpers_sha256": {name: _digest(Path(name)) for name in _HELPERS},
        },
    }
    if record["provenance"]["bag_sha256"] != protocol["bag"]["sha256"]:
        raise ValueError("input bag changed")
    _write(folder / "measurement.json", record, "worker")


def _run(args: argparse.Namespace) -> None:
    root = args.output_dir.resolve()
    if root.exists():
        raise ValueError("use a new output directory")
    root.mkdir(parents=True)
    commits = {
        key: subprocess.check_output(["git", "rev-parse", value + "^{commit}"], text=True).strip()
        for key, value in (("prior", args.baseline_commit), ("current", args.current_commit))
    }
    for kind, commit in commits.items():
        _freeze(root / ("source-" + kind), commit)
    helpers = {name: _digest(Path(name)) for name in _HELPERS}
    protocol = {
        "schema_version": "calibrex.cpu_map_bag_comparison/v0.1",
        "commits": commits,
        "bag": {
            "path": str(args.bag),
            "bytes": args.bag.stat().st_size,
            "sha256": _digest(args.bag),
        },
        "settings": {
            "max_seconds": args.max_seconds,
            "topic": args.topic,
            "point_time_field": args.point_time_field,
            "dataset_family": args.dataset_family,
            "dataset_license": args.dataset_license,
            "sample_every_scans": 5,
            "map_half_window_scans": 5,
            "block_duration_s": 1.0,
            "jackknife_groups": 4,
            "target_noise_seed": 20261018,
            "target_noise_std_m": 0.01,
            "initial_translation_m": [0.08, -0.07, 0.05],
            "odometry_backend": args.odometry_backend,
            "odometry_options": {
                "voxel_size_m": 0.03,
                "local_map_scans": 3,
                "geometry_quantum_m": 1e-9,
                "motion_translation_quantum_m": 1e-6,
                "motion_rotation_quantum_rad": 1e-7,
            },
        },
        "scenarios": {
            f"map-{index:03d}": {"map_voxel_size_m": value}
            for index, value in enumerate(args.map_voxel_sizes_m)
        },
        "repeats": args.repeats,
        "query_workers": {"prior": 1, "lazy": 1, "parallel": args.query_workers},
        "orders": [list(_VARIANTS[i % 3 :] + _VARIANTS[: i % 3]) for i in range(args.repeats)],
        "source_manifests_sha256": {
            kind: _digest(root / ("source-" + kind) / "manifest.json") for kind in commits
        },
        "helpers_sha256": helpers,
        "scope": "Whole fresh worker: startup, imports, bag streams, reference odometry, "
        "map preparation, five fits, covariance, held-out scoring, six controls, "
        "saves and verification.",
        "limitations": [
            "No independent dual-LiDAR accuracy or SOTA evidence; "
            "spent public copied-topic control.",
            "Unreserved host; one BLAS thread; cold independent CUDA compiler cache when selected.",
            "No timing is discarded, replaced, pooled across densities, "
            "or multiplied with other cohorts.",
        ],
    }
    _write(root / "protocol.json", protocol, "protocol")
    for helper in _HELPERS:
        (root / (Path(helper).name + ".txt")).write_bytes(Path(helper).read_bytes())
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in protocol["scenarios"]}
    for repeat, order in enumerate(protocol["orders"]):
        scenarios = list(protocol["scenarios"])
        if repeat % 2:
            scenarios.reverse()
        for scenario in scenarios:
            measurements = {}
            seconds = {}
            for variant in order:
                folder = root / scenario / f"{repeat:02d}-{variant}"
                folder.mkdir(parents=True)
                env = dict(os.environ, **dict.fromkeys(_THREADS, "1"))
                env.update(
                    PYTHONPATH=str(_source(root, variant) / "src"),
                    PYTHONDONTWRITEBYTECODE="1",
                    CUPY_CACHE_DIR=str(folder / "cupy-cache"),
                    CUDA_CACHE_DISABLE="1",
                )
                started = time.perf_counter()
                with (
                    (folder / "stdout.log").open("w") as stdout,
                    (folder / "stderr.log").open("w") as stderr,
                ):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "tools.benchmark_cpu_map_bag",
                            "--worker",
                            variant,
                            "--scenario",
                            scenario,
                            "--repeat",
                            str(repeat),
                            "--output-dir",
                            str(root),
                        ],
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                    )
                seconds[variant] = time.perf_counter() - started
                _write(
                    folder / "whole.json",
                    {
                        "seconds": seconds[variant],
                        "returncode": completed.returncode,
                        "protocol_sha256": _digest(root / "protocol.json"),
                        "helpers_sha256": helpers,
                    },
                    "whole",
                )
                if completed.returncode:
                    raise RuntimeError(
                        f"worker failed: {scenario}/{repeat:02d}-{variant}; logs retained"
                    )
                measurements[variant] = _read(folder / "measurement.json")
                print(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "repeat": repeat,
                            "variant": variant,
                            "seconds": seconds[variant],
                        }
                    ),
                    flush=True,
                )
            row = {
                "scenario": scenario,
                "repeat": repeat,
                "seconds": seconds,
                "errors": {
                    variant: compare_workers(measurements["prior"], measurements[variant])
                    for variant in _VARIANTS[1:]
                },
                "measurement_sha256": {
                    variant: _digest(root / scenario / f"{repeat:02d}-{variant}/measurement.json")
                    for variant in _VARIANTS
                },
            }
            _write(root / f"{scenario}-triad-{repeat:02d}.json", row, "triad")
            rows[scenario].append(row)
    summary = {
        "schema_version": protocol["schema_version"],
        "scenarios": {name: summarize(triads) for name, triads in rows.items()},
        "protocol_sha256": _digest(root / "protocol.json"),
        "triads_sha256": {
            f"{name}-triad-{repeat:02d}.json": _digest(root / f"{name}-triad-{repeat:02d}.json")
            for name in rows
            for repeat in range(args.repeats)
        },
    }
    for kind in commits:
        _verify_sources(root / ("source-" + kind))
    if {name: _digest(Path(name)) for name in _HELPERS} != helpers:
        raise ValueError("benchmark helpers changed")
    if _digest(args.bag) != protocol["bag"]["sha256"]:
        raise ValueError("input bag changed")
    _write(root / "summary.json", summary, "summary")
    print(json.dumps(summary, indent=2), flush=True)


def main(arguments: list[str] | None = None) -> None:
    """Run committed whole-bag comparisons, retaining all worker evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path)
    parser.add_argument("--topic", default="/livox/points")
    parser.add_argument("--point-time-field", default="t")
    parser.add_argument("--dataset-family")
    parser.add_argument("--dataset-license")
    parser.add_argument("--baseline-commit")
    parser.add_argument("--current-commit", default="HEAD")
    parser.add_argument("--max-seconds", type=float, default=6.0)
    parser.add_argument("--map-voxel-sizes-m", type=float, nargs="+", default=[0.2, 0.03])
    parser.add_argument("--odometry-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--query-workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=_VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--scenario", help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)
    sys.dont_write_bytecode = True
    if args.worker:
        _worker(args)
        return
    if (
        args.bag is None
        or args.baseline_commit is None
        or not args.dataset_family
        or not args.dataset_license
        or args.repeats < 1
        or args.query_workers < 1
        or not math.isfinite(args.max_seconds)
        or args.max_seconds <= 0
        or len(set(args.map_voxel_sizes_m)) != len(args.map_voxel_sizes_m)
        or any(not math.isfinite(value) or value <= 0 for value in args.map_voxel_sizes_m)
    ):
        parser.error(
            "give a bag, a committed baseline, positive unique voxel sizes and positive budgets"
        )
    _run(args)


if __name__ == "__main__":
    main()
