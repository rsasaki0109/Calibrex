"""Compare the complete public-data LiDAR-IMU bag pipeline in fresh CPU workers.

Freeze both source trees before timing. Each worker verifies its actual imports,
uses one BLAS thread, and saves the complete schema-valid calibration evidence.
Startup, bag decoding, initial odometry, map rounds, covariance and controls all
enter the parent wall time; per-round fit times are reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    if isinstance(value, dict) and value.get("schema_version"):
        import jsonschema

        schema = Path(__file__).with_name("schemas") / "bspline_lidar_imu_comparison.schema.json"
        jsonschema.validate(value, json.loads(schema.read_bytes()))
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _manifest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _digest(p) for p in sorted(root.rglob("*.py"))}


def _verify_sources(root: Path) -> dict[str, str]:
    expected = json.loads((root / "manifest.json").read_bytes())
    if _manifest(root / "src") != expected:
        raise ValueError("frozen calibration source tree changed")
    return expected


def _freeze(root: Path, commit: str | None) -> None:
    root.mkdir()
    if commit is None:
        for path in sorted(Path("src/calibrex").rglob("*.py")):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    else:
        archive = subprocess.check_output(["git", "archive", commit, "src/calibrex"])
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            for member in bundle.getmembers():
                name = Path(member.name)
                if not member.isfile() or name.suffix != ".py":
                    continue
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or name.parts[:2] != ("src", "calibrex")
                ):
                    raise ValueError("unexpected Git source archive path")
                contents = bundle.extractfile(member)
                assert contents is not None
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents.read())
    _write(root / "manifest.json", _manifest(root / "src"))


def _worker(args: argparse.Namespace) -> None:
    root = Path(args.source_root).resolve()
    expected = _verify_sources(root)
    launcher_digest = _digest(Path(__file__))
    backend = _backend(args.comparison, args.variant)
    cuda_cache_empty = None
    if args.comparison != "source":
        cache = Path(os.environ["CUPY_CACHE_DIR"])
        cuda_cache_empty = not cache.exists() or not any(cache.iterdir())
        if not cuda_cache_empty:
            raise RuntimeError("worker CUDA compilation cache was not empty")
    from calibrex.data.livox_ros2 import LivoxStreamProfile
    from calibrex.evaluation import lidar_imu_bag_calibration as runner
    from calibrex.solvers import scan_to_scan_odometry as odometry

    controls_timing = _time_controls(runner)
    odometry_timing = _time_odometry(odometry.IncrementalScanOdometry)
    normal_timing = None
    gyro_preprocessing_timing = None
    spatial_timing = preprocessing_timing = registration_timing = None
    if backend == "cuda":
        from calibrex.solvers.cuda_scan_odometry_adapter import CudaOdometryTarget

        normal_timing = _time_normal_preparation(CudaOdometryTarget)
        registration_timing = _time_registration(CudaOdometryTarget)
        from calibrex.solvers import scan_to_scan_odometry as odometry
        from calibrex.solvers.cuda_lidar_map_adapter import _CudaTree

        spatial_timing = _time_method(_CudaTree, "__init__")
        preprocessing_timing = (
            _time_method(CudaOdometryTarget.preprocessor, "__call__")
            if hasattr(CudaOdometryTarget, "preprocessor")
            else _time_method(odometry, "preprocess_scan")
        )
        if (hasattr(CudaOdometryTarget, "preprocessor")
                and hasattr(CudaOdometryTarget.preprocessor, "gyro_deskew")):
            gyro_preprocessing_timing = _time_method(
                CudaOdometryTarget.preprocessor, "gyro_deskew",
            )

    rounds: list[dict[str, object]] = []
    original = runner.fit_bspline_lidar_imu
    fit_backend = _fit_backend(args, args.variant)
    numba_cache_empty = None
    if fit_backend == "numba_cpu":
        cache = Path(os.environ["NUMBA_CACHE_DIR"])
        numba_cache_empty = not cache.exists() or not any(cache.iterdir())
        if not numba_cache_empty:
            raise RuntimeError("worker Numba compilation cache was not empty")

    def measured_fit(problem, options):
        begin = time.perf_counter()
        cpu_begin = time.process_time()
        result = original(problem, options)
        cpu_seconds = time.process_time() - cpu_begin
        rounds.append(
            {
                "seconds": time.perf_counter() - begin,
                "cpu_seconds": cpu_seconds,
                "status": result.status,
                "iterations": result.iterations,
                "initial_cost": result.initial_cost,
                "final_cost": result.final_cost,
                "residual_count": result.residual_count,
                "parameter_count": result.parameter_count,
                "calibration_parameter_names": list(result.calibration_parameter_names),
                "execution_backend": getattr(result, "execution_backend", "numpy"),
                "calibration_covariance": result.calibration_covariance.tolist(),
                "residual_variance": result.residual_variance,
                "trajectory_rotations": result.trajectory.rotations.tolist(),
                "trajectory_positions": result.trajectory.positions.tolist(),
                "calibration": {
                    block: result.calibration.vector(block).tolist()
                    for block in runner.CALIBRATION_BLOCKS
                },
            }
        )
        return result

    runner.fit_bspline_lidar_imu = measured_fit
    begin = time.perf_counter()
    cpu_begin = time.process_time()
    settings = runner.LidarImuBagOptions(max_rounds=args.max_rounds)
    if fit_backend != "numpy":
        settings = replace(settings, fit_backend=fit_backend, fit_workers=args.fit_workers)
    if args.comparison != "source":
        windowing = settings.rotation.windowing
        settings = replace(settings, rotation=replace(
            settings.rotation, windowing=replace(
                windowing, odometry_backend=backend,
                odometry=replace(windowing.odometry, **_odometry_options(args)),
            ),
        ))
    result = runner.calibrate_lidar_imu_bag(
        args.bag,
        LivoxStreamProfile("public-glim-avia", "/livox/points", "/livox/imu", "t", "offset_s", "g"),
        settings,
        dataset_family="glim_versatile",
        dataset_license="upstream AIST GLIM dataset terms",
        max_seconds=args.max_seconds,
        command=[
            "python",
            "-m",
            "tools.benchmark_bspline_lidar_imu",
            "--worker",
            "--variant",
            args.variant,
            "--max-seconds",
            str(args.max_seconds),
            "--max-rounds",
            str(args.max_rounds),
            *_fit_arguments(args),
            *(["--comparison", args.comparison,
               *_odometry_arguments(args)] if args.comparison != "source" else []),
        ],
    )
    result.provenance.git_commit = args.source_commit
    _validate_fit_execution(result.model_dump(mode="json"), rounds, fit_backend)
    out = Path(args.output)
    result.save(out / "result.json")
    pipeline_seconds = time.perf_counter() - begin
    pipeline_cpu_seconds = time.process_time() - cpu_begin
    imports = {}
    for name, module in list(sys.modules.items()):
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(root / "src"):
                raise RuntimeError(f"calibration imported outside frozen tree: {name}")
            relative = path.relative_to(root / "src").as_posix()
            if _digest(path) != expected[relative]:
                raise RuntimeError(f"executed source digest differs: {name}")
            imports[relative] = expected[relative]
    if backend == "cpu" and "cupy" in sys.modules:
        raise RuntimeError("CPU comparison unexpectedly imported CuPy")
    if args.comparison != "source":
        _validate_backend_execution(result.model_dump(mode="json"), backend)
    _verify_sources(root)
    if _digest(Path(__file__)) != launcher_digest:
        raise RuntimeError("benchmark launcher changed during execution")
    import numpy
    import scipy

    _write(
        out / "measurement.json",
        {
            "schema_version": "slac.bspline_lidar_imu_comparison/v0.1",
            "record_type": "worker",
            "variant": args.variant,
            "pipeline_seconds": pipeline_seconds,
            "pipeline_cpu_seconds": pipeline_cpu_seconds,
            "fit_seconds": sum(float(r["seconds"]) for r in rounds),
            "fit_cpu_seconds": sum(float(r["cpu_seconds"]) for r in rounds),
            "odometry_seconds": odometry_timing["seconds"],
            "odometry_calls": odometry_timing["calls"],
            "controls_seconds": controls_timing["seconds"],
            "controls_calls": controls_timing["calls"],
            **({"registration_seconds": registration_timing["seconds"],
                "registration_calls": registration_timing["calls"]}
               if registration_timing is not None else {}),
            **({"normal_preparation_seconds": normal_timing["seconds"],
                "normal_preparation_calls": normal_timing["calls"]}
               if normal_timing is not None else {}),
            **({"spatial_build_seconds": spatial_timing["seconds"],
                "spatial_build_calls": spatial_timing["calls"]}
               if spatial_timing is not None else {}),
            **({"preprocessing_seconds": preprocessing_timing["seconds"],
                "preprocessing_calls": preprocessing_timing["calls"],
                "preprocessing_scope": (
                    "standard __call__ path; excludes device gyro_deskew"
                    if hasattr(CudaOdometryTarget, "preprocessor")
                    else "native preprocess_scan; device preprocessor unavailable")}
               if preprocessing_timing is not None else {}),
            **({"gyro_preprocessing_seconds": gyro_preprocessing_timing["seconds"],
                "gyro_preprocessing_calls": gyro_preprocessing_timing["calls"]}
               if gyro_preprocessing_timing is not None else {}),
            "rounds": rounds,
            "provenance": {
                "bag_sha256": _digest(Path(args.bag)),
                "source_manifest_sha256": _digest(root / "manifest.json"),
                "executed_sources": imports,
                "launcher_sha256": launcher_digest,
                "result_sha256": _digest(out / "result.json"),
                "python": sys.version,
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "fit_backend": fit_backend,
                "fit_workers": args.fit_workers if fit_backend == "numba_cpu" else 1,
                "numba_cache_initially_empty": numba_cache_empty,
                "numba": (sys.modules["numba"].__version__ if fit_backend == "numba_cpu" else None),
                "blas_threads": {
                    name: os.environ[name]
                    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                },
                **({"odometry_backend": backend, "cuda_cache_initially_empty": cuda_cache_empty,
                    "cupy": (sys.modules["cupy"].__version__ if backend == "cuda" else None)}
                   if args.comparison != "source" else {}),
            },
        },
    )


def _compare(prior: dict, current: dict) -> dict:
    """Fixed physical tolerances plus exact decisions; never compare timestamps."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    rotation_error = float(
        np.degrees(
            (
                Rotation.from_quat(prior["t_lidar_imu_rotation_quat_xyzw"]).inv()
                * Rotation.from_quat(current["t_lidar_imu_rotation_quat_xyzw"])
            ).magnitude()
        )
    )
    errors = {
        "rotation_deg": rotation_error,
        "lever_arm_m": float(
            np.max(
                np.abs(
                    np.asarray(prior["t_lidar_imu_translation_m"])
                    - current["t_lidar_imu_translation_m"]
                )
            )
        ),
        "clock_offset_s": abs(prior["imu_clock_offset_s"] - current["imu_clock_offset_s"]),
    }
    decisions = (
        prior["policy_status"] == current["policy_status"]
        and [(c["parameter"], c.get("axis"), c["sign"], c["detected"]) for c in prior["controls"]]
        == [(c["parameter"], c.get("axis"), c["sign"], c["detected"]) for c in current["controls"]]
        and [(b["name"], b["verdict"]) for b in prior["blocks"]]
        == [(b["name"], b["verdict"]) for b in current["blocks"]]
        and [(s["block_a"], s["block_b"], s["entangled"]) for s in prior["separability"]]
        == [(s["block_a"], s["block_b"], s["entangled"]) for s in current["separability"]]
        and prior["converged"] == current["converged"]
    )
    numeric_evidence_equal = (
        all(
            np.allclose(left[field], right[field], rtol=1e-4, atol=1e-8)
            for left, right in zip(prior["blocks"], current["blocks"], strict=True)
            for field in ("estimate", "formal_std", "last_round_change")
        )
        and np.allclose(
            [c["delta_chi2"] for c in prior["controls"]],
            [c["delta_chi2"] for c in current["controls"]],
            rtol=1e-5,
            atol=1e-5,
        )
        and np.allclose(
            [s["max_abs_correlation"] for s in prior["separability"]],
            [s["max_abs_correlation"] for s in current["separability"]],
            atol=1e-5,
        )
        and all(
            np.isclose(prior["noise_sigmas"][k], current["noise_sigmas"][k], rtol=1e-5, atol=1e-8)
            for k in prior["noise_sigmas"]
        )
    )
    holdout = max(
        abs(prior["holdout_whitened_rms"][k] - current["holdout_whitened_rms"][k])
        for k in prior["holdout_whitened_rms"]
    )
    passed = (
        decisions
        and numeric_evidence_equal
        and errors["rotation_deg"] <= 0.001
        and errors["lever_arm_m"] <= 1e-5
        and errors["clock_offset_s"] <= 1e-7
        and holdout <= 1e-5
    )
    return {
        "passed": passed,
        "decisions_equal": decisions,
        "numeric_evidence_equal": numeric_evidence_equal,
        "errors": errors,
        "max_holdout_whitened_rms_difference": holdout,
    }


def _odometry_options(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "voxel_size_m": args.odometry_voxel_size_m,
        "local_map_scans": args.odometry_local_map_scans,
        "geometry_quantum_m": args.odometry_geometry_quantum_m,
        "motion_translation_quantum_m": args.odometry_motion_translation_quantum_m,
        "motion_rotation_quantum_rad": args.odometry_motion_rotation_quantum_rad,
    }


def _odometry_arguments(args: argparse.Namespace) -> list[str]:
    arguments = []
    for name, value in _odometry_options(args).items():
        arguments.extend(["--odometry-" + name.replace("_", "-"), str(value)])
    return arguments


def _validate_backend_execution(result: dict, backend: str) -> None:
    """Require real requested work in every initial and final odometry pass."""
    options = result["options"]
    groups = [options["initial_odometry_execution"], options["segment_odometry_execution"]]
    if options["odometry_backend"] != backend or any(not group for group in groups):
        raise ValueError("missing or incorrect requested odometry execution")
    for report in [r for group in groups for r in group]:
        if report["requested_backend"] != backend:
            raise ValueError("odometry pass requested another backend")
        identities = report["executed_backends"]
        if len(identities) != 1 or (
            identities[0] != "numpy_scipy" if backend == "cpu"
            else not identities[0].startswith("calibrex.cuda_scan_odometry/v0.1;")
        ):
            raise ValueError("odometry pass executed another backend")
        statistics = report["statistics"]
        if any(type(value) is not int or value < 0 for value in statistics.values()):
            raise ValueError("invalid odometry execution counters")
        if backend == "cuda" and any(
            statistics.get(name, 0) <= 0 for name in ("gpu_query_rows", "gpu_fused_system_calls")
        ):
            raise ValueError("CUDA pass did not execute spatial queries and systems")


def _time_method(holder: Any, name: str) -> dict[str, float | int]:
    """Observe wall time without adding synchronization or changing arithmetic."""
    original = getattr(holder, name)
    totals: dict[str, float | int] = {"seconds": 0.0, "calls": 0}

    def measured(*args: Any, **kwargs: Any) -> Any:
        begin = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            totals["seconds"] += time.perf_counter() - begin
            totals["calls"] += 1

    setattr(holder, name, measured)
    return totals


def _time_controls(holder: Any) -> dict[str, float | int]:
    """Observe the complete signed holdout-control suite once per pipeline."""
    return _time_method(holder, "_controls")


def _time_registration(holder: Any) -> dict[str, float | int]:
    """Include normals, all iterations, reductions and native repairs."""
    return _time_method(holder, "register")


def _time_odometry(holder: Any) -> dict[str, float | int]:
    """Include deskew, preprocessing, tree construction and complete registration."""
    return _time_method(holder, "add")


def _time_normal_preparation(holder: Any) -> dict[str, float | int]:
    """Time outermost preparation calls; nested cache completion is counted once.

    Existing preparation synchronizes for its decision flags. No additional GPU
    synchronization is inserted by this observer. The bound launcher identifies it.
    """
    totals: dict[str, float | int] = {"calls": 0, "seconds": 0.0}
    depth = 0
    for name in ("_ensure", "_ensure_all"):
        if not hasattr(holder, name):
            continue
        original = getattr(holder, name)

        def measured(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            nonlocal depth
            depth += 1
            begin = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                depth -= 1
                if depth == 0:
                    totals["calls"] += 1
                    totals["seconds"] += time.perf_counter() - begin

        setattr(holder, name, measured)
    return totals


def _backend(comparison: str, variant: str) -> str:
    """Select actual execution for baseline/current source comparisons."""
    if comparison == "cuda-cuda" or (comparison == "cpu-cuda" and variant == "current"):
        return "cuda"
    return "cpu"


def _fit_backend(args: argparse.Namespace, variant: str) -> str:
    return str(args.prior_fit_backend if variant == "prior" else args.current_fit_backend)


def _fit_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--prior-fit-backend", args.prior_fit_backend,
        "--current-fit-backend", args.current_fit_backend, "--fit-workers", str(args.fit_workers),
    ]


def _validate_fit_execution(result: dict, rounds: list[dict], requested: str) -> None:
    identities = [item["execution_backend"] for item in rounds]
    if not identities:
        raise ValueError("no complete spline fits were observed")
    if requested == "numpy":
        if any(identity != "numpy" for identity in identities):
            raise ValueError("native comparison executed another fit backend")
        return
    if (result["options"].get("fit_backend") != requested
        or result["options"].get("fit_execution") != identities[-1]
        or any(not identity.startswith(("calibrex.numba_bspline_planes/v0.1;",
                                        "calibrex.numba_bspline_planes/v0.2;",
                                        "calibrex.numba_bspline_planes/v0.3;"))
               for identity in identities)):
        raise ValueError("compiled plane comparison did not execute the requested backend")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", default="data/public/glim_versatile/livox.bag")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-commit", default="b89e50bbd9e48a05a2acb6010ac46e50ece9220f")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--prior-fit-backend", choices=("numpy", "numba_cpu"), default="numpy")
    parser.add_argument("--current-fit-backend", choices=("numpy", "numba_cpu"), default="numpy")
    parser.add_argument("--fit-workers", type=int, default=1)
    parser.add_argument(
        "--comparison", choices=("source", "cpu-cuda", "cuda-cuda"), default="source",
    )
    parser.add_argument("--odometry-voxel-size-m", type=float)
    parser.add_argument("--odometry-local-map-scans", type=int)
    parser.add_argument("--odometry-geometry-quantum-m", type=float)
    parser.add_argument("--odometry-motion-translation-quantum-m", type=float)
    parser.add_argument("--odometry-motion-rotation-quantum-rad", type=float)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=("prior", "current"), help=argparse.SUPPRESS)
    parser.add_argument("--source-root", help=argparse.SUPPRESS)
    parser.add_argument("--source-commit", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
        return
    if args.repeats < 1 or args.max_rounds < 2 or args.max_seconds <= 0 or args.fit_workers < 1:
        parser.error("need positive repeats/seconds and at least two map rounds")
    if args.comparison != "source" and (
        any(value is None for value in _odometry_options(args).values())
        or args.odometry_voxel_size_m <= 0 or args.odometry_local_map_scans < 1
        or any(value < 0 for key, value in _odometry_options(args).items()
               if key not in ("voxel_size_m", "local_map_scans"))
    ):
        parser.error("CPU/CUDA comparisons require all five explicit valid odometry settings")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", args.baseline_commit], text=True).strip()
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.comparison == "cpu-cuda":
        commit = current_commit
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "src/calibrex"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    roots = {v: out / f"sources-{v}" for v in ("prior", "current")}
    _freeze(roots["prior"], commit)
    _freeze(roots["current"], None)
    launcher_digest = _digest(Path(__file__))
    bag = Path(args.bag).resolve()
    shutil.copyfile(Path(__file__), out / "benchmark.py.txt")
    shutil.copyfile(
        Path(__file__).with_name("benchmark_bspline_planes.py"), out / "fit-quality.py.txt",
    )
    _write(
        out / "protocol.json",
        {
            "schema_version": "slac.bspline_lidar_imu_comparison/v0.1",
            "record_type": "protocol",
            "baseline_commit": commit,
            "current_commit": current_commit,
            "repeats": args.repeats,
            "max_seconds": args.max_seconds,
            "max_rounds": args.max_rounds,
            "bag_sha256": _digest(bag),
            "launcher_sha256": launcher_digest,
            "scope": (
                "fresh process through complete bag calibration, saved evidence "
                "and import verification"
            ),
            "cpu_time_policy": (
                "time.process_time: all in-process worker threads; excludes children"
            ),
            "quality_tolerances": {
                "rotation_deg": 0.001,
                "lever_arm_m": 1e-5,
                "clock_offset_s": 1e-7,
                "holdout_whitened_rms": 1e-5,
                "block_evidence_relative": 1e-4,
                "block_evidence_absolute": 1e-8,
                "control_chi2_relative": 1e-5,
                "control_chi2_absolute": 1e-5,
            },
            "fit_options": {"prior_backend": args.prior_fit_backend,
                            "current_backend": args.current_fit_backend,
                            "workers": args.fit_workers, "numba_cache_policy": "empty_per_worker",
                            "quality_helper_sha256": _digest(
                                Path(__file__).with_name("benchmark_bspline_planes.py"))},
            **({"comparison": args.comparison, "odometry_options": _odometry_options(args),
                "isolated_cuda_caches": True} if args.comparison != "source" else {}),
        },
    )
    attempts: list[dict] = []
    for repeat in range(args.repeats):
        order = ("prior", "current") if repeat % 2 == 0 else ("current", "prior")
        for variant in order:
            worker_out = out / f"{repeat:02d}-{variant}"
            worker_out.mkdir()
            env = os.environ.copy()
            env.update(
                dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"), "1")
            )
            env["PYTHONPATH"] = str(roots[variant] / "src") + os.pathsep + str(Path.cwd())
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["NUMBA_CACHE_DIR"] = str(worker_out / "numba-cache")
            if args.comparison != "source":
                env["CUPY_CACHE_DIR"] = str(worker_out / "cupy-cache")
            command = [
                sys.executable,
                "-m",
                "tools.benchmark_bspline_lidar_imu",
                "--worker",
                "--variant",
                variant,
                "--source-root",
                str(roots[variant]),
                "--source-commit",
                commit if variant == "prior" else current_commit,
                "--bag",
                str(bag),
                "--max-seconds",
                str(args.max_seconds),
                "--max-rounds",
                str(args.max_rounds),
                "--output",
                str(worker_out),
                *_fit_arguments(args),
                *(["--comparison", args.comparison, *_odometry_arguments(args)]
                  if args.comparison != "source" else []),
            ]
            begin = time.perf_counter()
            with (worker_out / "worker.log").open("wb") as log:
                process = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            attempt = {
                "repeat": repeat,
                "variant": variant,
                "exit_code": process.returncode,
                "whole_seconds": time.perf_counter() - begin,
                "worker": worker_out.name,
            }
            attempts.append(attempt)
            _write(out / "attempts.json", attempts)
            print(json.dumps(attempt), flush=True)
            if process.returncode != 0:
                raise RuntimeError(f"worker failed; retained {worker_out}")
    pairs = []
    for repeat in range(args.repeats):
        prior = json.loads((out / f"{repeat:02d}-prior/result.json").read_bytes())
        current = json.loads((out / f"{repeat:02d}-current/result.json").read_bytes())
        compared = _compare(prior, current)
        if args.prior_fit_backend != args.current_fit_backend:
            from tools.benchmark_bspline_planes import _compare as compare_fit

            records = [json.loads((out / f"{repeat:02d}-{variant}/measurement.json").read_bytes())
                       for variant in ("prior", "current")]
            compared["fits"] = [compare_fit(left, right) for left, right in
                                zip(records[0]["rounds"], records[1]["rounds"], strict=True)]
            compared["passed"] = compared["passed"] and all(
                record["passed"] for record in compared["fits"]
            )
        pairs.append(compared)
    medians = {
        v: statistics.median(a["whole_seconds"] for a in attempts if a["variant"] == v)
        for v in roots
    }
    fit_medians = {
        v: statistics.median(
            json.loads((out / a["worker"] / "measurement.json").read_bytes())["fit_seconds"]
            for a in attempts
            if a["variant"] == v
        )
        for v in roots
    }
    for root in roots.values():
        _verify_sources(root)
    if _digest(Path(__file__)) != launcher_digest:
        raise RuntimeError("benchmark launcher changed during comparison")
    summary = {
        "schema_version": "slac.bspline_lidar_imu_comparison/v0.1",
        "record_type": "summary",
        "whole_median_seconds": medians,
        "whole_speedup": medians["prior"] / medians["current"],
        "fit_median_seconds": fit_medians,
        "fit_speedup": fit_medians["prior"] / fit_medians["current"],
        "quality": pairs,
        "quality_passed": all(p["passed"] for p in pairs),
        "provenance": {
            "protocol_sha256": _digest(out / "protocol.json"),
            "attempts_sha256": _digest(out / "attempts.json"),
        },
    }
    _write(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not summary["quality_passed"]:
        raise RuntimeError("complete result quality comparison failed")


if __name__ == "__main__":
    main()
