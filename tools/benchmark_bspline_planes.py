"""Compare complete captured spline fits with optional compiled CPU planes.

Use fixed external .npz/.json problem pairs. Freeze the committed source tree,
alternate fresh workers, and report JIT first use separately from warmed fits.
This measures the complete fit, including covariance, rather than a bag pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from tools.benchmark_bspline_lidar_imu import _digest, _freeze, _verify_sources

VERSION = "slac.bspline_plane_fit_benchmark/v0.1"
NUMERIC_FIELDS = (
    "initial_cost",
    "final_cost",
    "residual_variance",
    "trajectory_rotations",
    "trajectory_positions",
    "calibration_covariance",
)
EXACT_FIELDS = (
    "status",
    "iterations",
    "residual_count",
    "parameter_count",
    "calibration_parameter_names",
)
TOLERANCES = {"rtol": 1e-8, "atol": 1e-9}
CPU_TIME_POLICY = "time.process_time: all in-process worker threads; excludes children"


def _write(path: Path, record: dict) -> None:
    import jsonschema

    schema = Path(__file__).with_name("schemas") / "bspline_plane_fit_benchmark.schema.json"
    jsonschema.validate(record, json.loads(schema.read_bytes()))
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load(prefix: Path):
    import numpy as np

    from calibrex.core.continuous_time_bspline import CubicBSplineTrajectory
    from calibrex.core.continuous_time_bspline_fit import (
        BSplineImuSamples,
        BSplineLidarImuProblem,
        BSplinePlaneReturns,
        LidarImuCalibration,
    )

    meta = json.loads(prefix.with_suffix(".json").read_bytes())
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as source:
        z = {key: source[key].copy() for key in source.files}
    calibration = LidarImuCalibration(
        z["imu_rotation"],
        z["gyro_bias"],
        z["lever_arm"],
        float(z["imu_clock_offset"][0]),
        z["accel_bias"],
        z["gravity"],
    )
    return BSplineLidarImuProblem(
        CubicBSplineTrajectory(
            meta["start_time_sec"], meta["knot_spacing_sec"], z["rotations"], z["positions"]
        ),
        BSplinePlaneReturns(
            z["plane_times"],
            z["plane_points"],
            z["plane_anchors"],
            z["plane_normals"],
            meta["plane_sigma"],
        ),
        BSplineImuSamples(z["gyro_times"], z["gyro_values"], meta["gyro_sigma"]),
        BSplineImuSamples(z["accel_times"], z["accel_values"], meta["accel_sigma"]),
        calibration,
        tuple(meta["estimate"]),
        meta["gravity_magnitude_m_s2"],
        meta["gravity_magnitude_sigma_m_s2"],
    ), meta["fit_iterations"]


def _payload(result) -> dict:
    from calibrex.core.continuous_time_bspline_fit import CALIBRATION_BLOCKS

    return {
        **{key: getattr(result, key) for key in (*EXACT_FIELDS[:4], *NUMERIC_FIELDS[:3])},
        "calibration_parameter_names": list(result.calibration_parameter_names),
        "trajectory_rotations": result.trajectory.rotations.tolist(),
        "trajectory_positions": result.trajectory.positions.tolist(),
        "calibration_covariance": result.calibration_covariance.tolist(),
        "calibration": {key: result.calibration.vector(key).tolist() for key in CALIBRATION_BLOCKS},
        "execution_backend": result.execution_backend,
    }


def _compare(prior: dict, current: dict) -> dict:
    import numpy as np

    errors = {}
    passed = all(prior[key] == current[key] for key in EXACT_FIELDS)
    for key in (*NUMERIC_FIELDS, *prior["calibration"]):
        left = prior[key] if key in NUMERIC_FIELDS else prior["calibration"][key]
        right = current[key] if key in NUMERIC_FIELDS else current["calibration"][key]
        a, b = np.asarray(left), np.asarray(right)
        equal = a.shape == b.shape and bool(np.allclose(a, b, **TOLERANCES))
        errors[key] = (
            (float(np.max(np.abs(a - b))) if a.size else 0.0) if a.shape == b.shape else None
        )
        passed = passed and equal
    return {
        "passed": passed,
        "exact_decisions": all(prior[k] == current[k] for k in EXACT_FIELDS),
        "max_absolute_errors": errors,
    }


def _worker(args: argparse.Namespace) -> None:
    root = Path(args.source_root).resolve()
    manifest = _verify_sources(root)
    output = Path(args.output)
    protocol = json.loads(Path(args.protocol).read_bytes())
    for item, prefix in zip(protocol["inputs"], args.problem, strict=True):
        if any(
            _digest(Path(prefix).with_suffix(suffix)) != item[key]
            for suffix, key in ((".npz", "npz_sha256"), (".json", "metadata_sha256"))
        ):
            raise RuntimeError("captured problem changed")
    from calibrex.core.continuous_time_bspline_fit import (
        BSplineLidarImuFitOptions,
        fit_bspline_lidar_imu,
    )

    problems = [_load(Path(prefix)) for prefix in args.problem]
    cache = Path(os.environ["NUMBA_CACHE_DIR"])
    if cache.exists() and any(cache.iterdir()):
        raise RuntimeError("first-use worker cache was not empty")
    runs = []
    for phase in ("first_use", "warm"):
        results = []
        start = time.perf_counter()
        cpu_start = time.process_time()
        for problem, iterations in problems:
            result = fit_bspline_lidar_imu(
                problem,
                BSplineLidarImuFitOptions(
                    max_iterations=iterations,
                    plane_backend=args.variant,
                    plane_workers=args.workers,
                ),
            )
            results.append(_payload(result))
        runs.append(
            {
                "phase": phase,
                "seconds": time.perf_counter() - start,
                "cpu_seconds": time.process_time() - cpu_start,
                "results": results,
            }
        )
    imports = {}
    for name, module in list(sys.modules.items()):
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(root / "src"):
                raise RuntimeError(f"import outside frozen source tree: {name}")
            relative = path.relative_to(root / "src").as_posix()
            if _digest(path) != manifest[relative]:
                raise RuntimeError(f"executed source changed: {name}")
            imports[relative] = manifest[relative]
    if "cupy" in sys.modules or (args.variant == "numpy" and "numba" in sys.modules):
        raise RuntimeError("unexpected accelerator import")
    _write(
        output / "measurement.json",
        {
            "schema_version": VERSION,
            "record_type": "worker",
            "variant": args.variant,
            "runs": runs,
            "provenance": {
                "protocol_sha256": _digest(Path(args.protocol)),
                "source_manifest_sha256": _digest(root / "manifest.json"),
                "executed_sources": imports,
                "launcher_sha256": _digest(Path(__file__)),
                "python": sys.version,
                "numpy": sys.modules["numpy"].__version__,
                "scipy": sys.modules["scipy"].__version__,
                "blas_threads": {
                    key: os.environ[key]
                    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                },
                "numba": sys.modules["numba"].__version__ if args.variant == "numba_cpu" else None,
                "cpu_time_policy": CPU_TIME_POLICY,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem", action="append", required=True, help="external .npz/.json prefix"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--input-origin", default="external fixed captured spline problem; extraction not audited"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=("numpy", "numba_cpu"), help=argparse.SUPPRESS)
    parser.add_argument("--source-root", help=argparse.SUPPRESS)
    parser.add_argument("--protocol", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
        return
    if args.repeats < 1 or args.workers < 1:
        parser.error("repeats and workers must be positive")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src/calibrex"], check=True)
    if subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "src/calibrex"], text=True
    ).strip():
        raise RuntimeError("runtime sources must be committed")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "sources"
    _freeze(source, None)
    prefixes = [str(Path(prefix).resolve()) for prefix in args.problem]
    protocol = {
        "schema_version": VERSION,
        "record_type": "protocol",
        "repeats": args.repeats,
        "workers": args.workers,
        "quality_tolerances": TOLERANCES,
        "inputs": [
            {
                "name": Path(prefix).name,
                "npz_sha256": _digest(Path(prefix).with_suffix(".npz")),
                "metadata_sha256": _digest(Path(prefix).with_suffix(".json")),
            }
            for prefix in prefixes
        ],
        "provenance": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "cpu_time_policy": CPU_TIME_POLICY,
            "source_manifest_sha256": _digest(source / "manifest.json"),
            "launcher_sha256": _digest(Path(__file__)),
            "input_origin": args.input_origin,
            "scope": (
                "complete fixed-input fits including covariance; first-use and warm separately; "
                "excludes bag decoding/front end"
            ),
        },
    }
    _write(out / "protocol.json", protocol)
    attempts = []
    quality = []
    for repeat in range(args.repeats):
        order = ("numpy", "numba_cpu") if repeat % 2 == 0 else ("numba_cpu", "numpy")
        for variant in order:
            worker = out / f"{repeat:02d}-{variant}"
            worker.mkdir()
            env = os.environ.copy()
            env.update(
                dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"), "1")
            )
            env["PYTHONPATH"] = str(source / "src") + os.pathsep + str(Path.cwd())
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["NUMBA_CACHE_DIR"] = str(worker / "numba-cache")
            command = [
                sys.executable,
                "-m",
                "tools.benchmark_bspline_planes",
                "--worker",
                "--variant",
                variant,
                "--source-root",
                str(source),
                "--protocol",
                str(out / "protocol.json"),
                "--output",
                str(worker),
                "--workers",
                str(args.workers),
            ]
            for prefix in prefixes:
                command += ["--problem", prefix]
            start = time.perf_counter()
            with (worker / "worker.log").open("wb") as log:
                result = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            row = {
                "repeat": repeat,
                "variant": variant,
                "worker": worker.name,
                "whole_seconds": time.perf_counter() - start,
                "exit_code": result.returncode,
            }
            attempts.append(row)
            _write(
                out / "attempts.json",
                {
                    "schema_version": VERSION,
                    "record_type": "attempts",
                    "attempts": attempts,
                    "provenance": {"protocol_sha256": _digest(out / "protocol.json")},
                },
            )
            print(json.dumps(row), flush=True)
            if result.returncode:
                raise RuntimeError(f"worker failed; retained {worker}")
        a, b = [
            json.loads((out / f"{repeat:02d}-{variant}" / "measurement.json").read_bytes())
            for variant in ("numpy", "numba_cpu")
        ]
        for left, right in zip(a["runs"], b["runs"], strict=True):
            quality.extend(
                _compare(p, c) for p, c in zip(left["results"], right["results"], strict=True)
            )
    timing_medians = {
        metric: {
            phase: {
                variant: statistics.median(
                    json.loads((out / item["worker"] / "measurement.json").read_bytes())["runs"][
                        index
                    ][metric]
                    for item in attempts
                    if item["variant"] == variant
                )
                for variant in ("numpy", "numba_cpu")
            }
            for index, phase in enumerate(("first_use", "warm"))
        }
        for metric in ("seconds", "cpu_seconds")
    }
    medians = timing_medians["seconds"]
    summary = {
        "schema_version": VERSION,
        "record_type": "summary",
        "medians": medians,
        "cpu_medians": timing_medians["cpu_seconds"],
        "speedups": {
            phase: values["numpy"] / values["numba_cpu"] for phase, values in medians.items()
        },
        "quality_passed": all(item["passed"] for item in quality),
        "quality": quality,
        "provenance": {
            "protocol_sha256": _digest(out / "protocol.json"),
            "attempts_sha256": _digest(out / "attempts.json"),
            "measurements": {
                item["worker"]: _digest(out / item["worker"] / "measurement.json")
                for item in attempts
            },
        },
    }
    _verify_sources(source)
    if _digest(Path(__file__)) != protocol["provenance"]["launcher_sha256"]:
        raise RuntimeError("benchmark helper changed")
    _write(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not summary["quality_passed"]:
        raise RuntimeError("full fitted-output comparison failed")


if __name__ == "__main__":
    main()
