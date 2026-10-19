"""Measure committed sparse screw fits and complete joint recovery in fresh workers.

The public geometry case uses eight already-spent GLIM registration captures.
Its assigned timestamps and fixed nearest-plane associations form a performance
fixture, not a reconstruction of the recording's physical motion or IMU data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_bspline_lidar_imu as source  # noqa: E402

ORDER = ("baseline", "candidate", "candidate", "baseline", "baseline", "candidate")
SEEDS = (20261010, 20261011, 20261012)
TOOLS = (
    "tools/benchmark_screw_residual_evaluation.py",
    "tools/benchmark_bspline_lidar_imu.py",
    "tools/benchmark_online_edge_cuda.py",
)
SCHEMA = "schemas/continuous_time_lidar_imu_joint.schema.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path: Path, value: dict) -> None:
    if "provenance" not in value:
        raise ValueError("measurement records require provenance")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance(**values: object) -> dict:
    return {"created_at": datetime.now(timezone.utc).isoformat(), **values}


def quality(value: dict) -> str:
    value = dict(value)
    value.pop("provenance", None)
    return hashlib.sha256(canonical(value)).hexdigest()


def prepare_geometry(folder: Path) -> dict:
    """Keep real point/plane coordinates; explicitly assign performance-only times."""
    import numpy as np
    from scipy.spatial import cKDTree

    measurements, inputs = [], {}
    for index in range(8):
        path = folder / f"registration-{index:02d}.npz"
        receipt = path.with_suffix(".json")
        metadata = json.loads(receipt.read_bytes())
        if sha(path) != metadata["fixture_sha256"]:
            raise ValueError("registration capture receipt differs")
        inputs[path.name] = {"sha256": sha(path), "receipt_sha256": sha(receipt)}
        with np.load(path, allow_pickle=False) as arrays:
            points, target = arrays["source"], arrays["target"]
            selected = np.linspace(0, len(points) - 1, 128, dtype=np.int64)
            points = points[selected].copy()
        tree = cKDTree(target)
        distances, nearest = tree.query(points, k=1, workers=1)
        centers = target[nearest]
        _, neighbours = tree.query(centers, k=10, workers=1)
        local = target[neighbours]
        centered = local - local.mean(axis=1, keepdims=True)
        values, vectors = np.linalg.eigh(centered.transpose(0, 2, 1) @ centered / 10.0)
        keep = (distances <= 1.5) & (values[:, 0] / np.maximum(values.sum(axis=1), 1e-30) <= 0.2)
        for sample in np.flatnonzero(keep):
            timestamp = 4.0 * (index * 128 + int(sample) + 0.5) / (8 * 128)
            measurements.append(
                {
                    "measurement_id": f"glim-{index}-{sample}",
                    "timestamp_sec": timestamp,
                    "point_body_m": points[sample].tolist(),
                    "plane_point_world_m": centers[sample].tolist(),
                    "plane_normal_world": vectors[sample, :, 0].tolist(),
                    "weight": 1.0,
                }
            )
    if len(measurements) < 512:
        raise ValueError("public geometry fixture has insufficient retained planes")
    return {
        "measurements": measurements,
        "inputs": inputs,
        "provenance": provenance(
            dataset="AIST GLIM Livox Avia; already-spent registration captures 00..07",
            input_bag="data/public/glim_versatile/livox.bag",
            scope=(
                "Assigned times, fixed nearest planes; performance regression only; "
                "no motion/accuracy claim"
            ),
            recipe=(
                "128 linspace source indices/capture; 10-neighbour PCA; "
                "distance<=1.5m; planarity<=0.2"
            ),
        ),
    }


def worker(args: argparse.Namespace) -> None:
    folder = args.output.resolve()
    protocol = json.loads((folder / "protocol.json").read_bytes())
    frozen = folder / args.variant
    sys.path.insert(0, str(frozen / "src"))
    expected = source._verify_sources(frozen)
    for path, digest in protocol["tools"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("benchmark tool changed after freezing")
    import numpy as np

    from calibrex.core import continuous_time_sparse as sparse
    from calibrex.core.geometry import SE3
    from calibrex.core.se3_manifold import se3_exp
    from calibrex.evaluation import continuous_time_lidar_imu_joint as joint

    captured = []
    original = sparse.fit_continuous_trajectory

    def fit(problem, options):
        begin = time.perf_counter()
        result = original(problem, options)
        captured.append((problem, options, result, time.perf_counter() - begin))
        return result

    name = f"{args.case}-{args.observation:02d}-{args.variant}"
    output = folder / name
    output.mkdir()
    origin = provenance(
        source_commit=protocol["commits"][args.variant],
        variant=args.variant,
        case=args.case,
        seed=SEEDS[args.observation // 2] if args.case == "joint" else None,
        protocol_sha256=sha(folder / "protocol.json"),
        launcher_sha256=sha(Path(__file__)),
        source_manifest_sha256=sha(frozen / "manifest.json"),
    )
    if args.case == "joint":
        joint.fit_continuous_trajectory = fit
        begin = time.perf_counter()
        artifact = joint.run_synthetic_lidar_imu_joint_recovery(
            seed=SEEDS[args.observation // 2],
            trajectory="screw_linear",
        )
        api_seconds = time.perf_counter() - begin
        artifact.provenance.git_commit = protocol["commits"][args.variant]
        artifact.save(output / "result.json")
        result = json.loads((output / "result.json").read_bytes())
        import jsonschema

        jsonschema.validate(result, json.loads((folder / "result.schema.json").read_bytes()))
    else:
        geometry = json.loads((folder / "geometry.json").read_bytes())
        if sha(folder / "geometry.json") != protocol["geometry_sha256"]:
            raise ValueError("public geometry changed")
        measurements = tuple(
            sparse.TrajectoryPointToPlaneMeasurement(
                **{
                    key: tuple(value) if isinstance(value, list) else value
                    for key, value in item.items()
                }
            )
            for item in geometry["measurements"]
        )
        timestamps = (0.0, 1.0, 2.0, 3.0, 4.0)
        rng = np.random.default_rng(20261020)
        knots = tuple(se3_exp(SE3.identity(), rng.normal(0.0, 0.002, 6)) for _ in timestamps)
        problem = sparse.ContinuousTimeTrajectoryFitProblem(
            timestamps,
            knots,
            point_to_plane_measurements=measurements,
        )
        settings = sparse.ContinuousTimeTrajectoryFitOptions(max_iterations=30)
        begin = time.perf_counter()
        fitted = fit(problem, settings)
        api_seconds = time.perf_counter() - begin
        result = {**asdict(fitted), "provenance": origin}
        write(output / "result.json", result)
    if len(captured) != 1:
        raise ValueError("expected exactly one complete fit")
    problem, settings, fitted, fit_seconds = captured[0]
    inputs = {"problem": asdict(problem), "options": asdict(settings), "provenance": origin}
    write(output / "fit-input.json", inputs)
    write(output / "fit-result.json", {**asdict(fitted), "provenance": origin})
    imported = {}
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("calibrex") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(frozen / "src"):
                raise ValueError("Calibrex import escaped frozen source")
            relative = path.relative_to(frozen / "src").as_posix()
            if sha(path) != expected[relative]:
                raise ValueError("imported source differs")
            imported[relative] = sha(path)
    source._verify_sources(frozen)
    write(
        output / "measurement.json",
        {
            "api_seconds": api_seconds,
            "fit_seconds": fit_seconds,
            "fit_status": fitted.status,
            "iterations": fitted.iterations,
            "point_to_plane_count": len(problem.point_to_plane_measurements),
            "result_quality_sha256": quality(result),
            "input_quality_sha256": quality(inputs),
            "fit_quality_sha256": quality({**asdict(fitted)}),
            "imported": imported,
            "provenance": origin,
        },
    )
    print(
        json.dumps({"worker": name, "api_seconds": api_seconds, "fit_status": fitted.status}),
        flush=True,
    )


def audit(folder: Path, *, verify_git: bool = False) -> dict:
    protocol = json.loads((folder / "protocol.json").read_bytes())
    if protocol["order"] != list(ORDER) or protocol["seeds"] != list(SEEDS):
        raise ValueError("fixed cohort order differs")
    if protocol["gate_api_ratio"] != 2.0 or protocol["blas_threads"] != 1:
        raise ValueError("fixed performance gate differs")
    if sha(folder / "geometry.json") != protocol["geometry_sha256"]:
        raise ValueError("geometry binding differs")
    if sha(folder / "result.schema.json") != protocol["result_schema_sha256"]:
        raise ValueError("result schema binding differs")
    for path, digest in protocol["tools"].items():
        if sha(folder / "toolchain" / path) != digest:
            raise ValueError("frozen toolchain binding differs")
    for variant in ("baseline", "candidate"):
        frozen = folder / variant
        manifest = source._verify_sources(frozen)
        if sha(frozen / "manifest.json") != protocol["manifests"][variant]:
            raise ValueError("source manifest binding differs")
        if verify_git:
            from tools.benchmark_online_edge_cuda import _git_files

            expected = {
                p: b
                for p, b in _git_files(protocol["commits"][variant], ("src/calibrex",)).items()
                if p.endswith(".py")
            }
            actual = {"src/" + p: (frozen / "src" / p).read_bytes() for p in manifest}
            if expected != actual:
                raise ValueError("source capsule differs from committed Git text")
    attempts = json.loads((folder / "attempts.json").read_bytes())
    if len(attempts["observations"]) != 12:
        raise ValueError("cohort is incomplete")
    report = {}
    for case in ("joint", "geometry"):
        rows = [r for r in attempts["observations"] if r["case"] == case]
        if [r["variant"] for r in rows] != list(ORDER):
            raise ValueError("observation order differs")
        records = []
        for observation, row in enumerate(rows):
            directory = folder / f"{case}-{observation:02d}-{row['variant']}"
            if row["exit_code"] != 0:
                raise ValueError("failed observation retained; no performance verdict")
            if set(row["files"]) != {
                "result.json",
                "fit-input.json",
                "fit-result.json",
                "measurement.json",
            }:
                raise ValueError("observation file coverage differs")
            if sha(folder / f"{directory.name}.log") != row["log_sha256"]:
                raise ValueError("worker log binding differs")
            for name, digest in row["files"].items():
                if sha(directory / name) != digest:
                    raise ValueError("observation file binding differs")
            value = json.loads((directory / "measurement.json").read_bytes())
            if value["provenance"]["protocol_sha256"] != sha(folder / "protocol.json"):
                raise ValueError("observation protocol binding differs")
            if value["provenance"]["source_commit"] != protocol["commits"][row["variant"]]:
                raise ValueError("observation source commit differs")
            if value["api_seconds"] <= 0 or row["seconds"] <= value["api_seconds"]:
                raise ValueError("invalid measurement interval")
            if case == "joint":
                import jsonschema

                jsonschema.validate(
                    json.loads((directory / "result.json").read_bytes()),
                    json.loads((folder / "result.schema.json").read_bytes()),
                )
            for filename, key in (
                ("result.json", "result_quality_sha256"),
                ("fit-input.json", "input_quality_sha256"),
                ("fit-result.json", "fit_quality_sha256"),
            ):
                if quality(json.loads((directory / filename).read_bytes())) != value[key]:
                    raise ValueError("complete result/input bits differ from measurement")
            manifest = json.loads((folder / row["variant"] / "manifest.json").read_bytes())
            if not value["imported"] or any(manifest[p] != h for p, h in value["imported"].items()):
                raise ValueError("actual imported sources differ")
            records.append(value)
        pairs = []
        for index in range(0, 6, 2):
            pair = {ORDER[i]: (records[i], rows[i]) for i in (index, index + 1)}
            b, c = pair["baseline"], pair["candidate"]
            for key in ("input_quality_sha256", "result_quality_sha256", "fit_quality_sha256"):
                if b[0][key] != c[0][key]:
                    raise ValueError("complete baseline/candidate input or result bits differ")
            pairs.append(
                {
                    "api_ratio": b[0]["api_seconds"] / c[0]["api_seconds"],
                    "fresh_worker_ratio": b[1]["seconds"] / c[1]["seconds"],
                    "quality_sha256": b[0]["result_quality_sha256"],
                }
            )
        medians = {
            variant: {
                "api_seconds": statistics.median(
                    records[i]["api_seconds"] for i, v in enumerate(ORDER) if v == variant
                ),
                "fresh_worker_seconds": statistics.median(
                    rows[i]["seconds"] for i, v in enumerate(ORDER) if v == variant
                ),
            }
            for variant in ("baseline", "candidate")
        }
        ratio = statistics.median(p["api_ratio"] for p in pairs)
        report[case] = {
            "paired_median_api_ratio": ratio,
            "paired_median_fresh_worker_ratio": statistics.median(
                p["fresh_worker_ratio"] for p in pairs
            ),
            "passes_api_gate": ratio >= 2.0,
            "pairs": pairs,
            "medians_seconds": medians,
            "complete_input_and_result_bits_equal": True,
        }
    return {
        "cases": report,
        "provenance": provenance(
            protocol_sha256=sha(folder / "protocol.json"),
            attempts_sha256=sha(folder / "attempts.json"),
            scope=(
                "Complete sparse fits and synthetic joint recovery; "
                "assigned public-geometry times; no accuracy claim"
            ),
        ),
    }


def run(args: argparse.Namespace) -> dict:
    folder = args.output.resolve()
    if folder.exists():
        raise ValueError("output directory must not exist")
    folder.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder.parent).free < 2 * 1024**3:
        raise ValueError("less than 2 GiB free; do not begin")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "src", *TOOLS], check=True, cwd=ROOT
    )
    commits = {
        "baseline": subprocess.check_output(
            ["git", "rev-parse", args.baseline_commit], cwd=ROOT, text=True
        ).strip(),
        "candidate": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    folder.mkdir()
    geometry = prepare_geometry(args.geometry_folder)
    write(folder / "geometry.json", geometry)
    for path in TOOLS:
        target = folder / "toolchain" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    for variant, commit in commits.items():
        frozen = folder / variant
        source._freeze(frozen, commit)
        # Git may retain CRLF Python text. Pin the executable capsule to the
        # same canonical LF text used by the independent Git archive audit.
        for path in (frozen / "src").rglob("*.py"):
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        (frozen / "manifest.json").write_text(
            json.dumps(source._manifest(frozen / "src"), indent=2) + "\n",
            encoding="utf-8",
        )
    shutil.copyfile(ROOT / SCHEMA, folder / "result.schema.json")
    protocol = {
        "commits": commits,
        "order": list(ORDER),
        "seeds": list(SEEDS),
        "manifests": {v: sha(folder / v / "manifest.json") for v in commits},
        "tools": {p: sha(ROOT / p) for p in TOOLS},
        "geometry_sha256": sha(folder / "geometry.json"),
        "result_schema_sha256": sha(folder / "result.schema.json"),
        "gate_api_ratio": 2.0,
        "blas_threads": 1,
        "host_reserved": False,
        "warmups": 0,
        "provenance": provenance(
            scope="Frozen before timing; no cache/iteration/quality changes; all attempts retained",
        ),
    }
    write(folder / "protocol.json", protocol)
    observations = []
    for case in ("joint", "geometry"):
        for index, variant in enumerate(ORDER):
            name = f"{case}-{index:02d}-{variant}"
            print(json.dumps({"starting": name}), flush=True)
            env = os.environ.copy()
            env.update(
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                PYTHONDONTWRITEBYTECODE="1",
            )
            begin = time.perf_counter()
            with (folder / f"{name}.log").open("wb") as log:
                try:
                    status = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__)),
                            "--worker",
                            "--output",
                            str(folder),
                            "--variant",
                            variant,
                            "--case",
                            case,
                            "--observation",
                            str(index),
                        ],
                        env=env,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                        check=False,
                    ).returncode
                except subprocess.TimeoutExpired:
                    status = -1
            seconds = time.perf_counter() - begin
            observation = {
                "case": case,
                "variant": variant,
                "seconds": seconds,
                "exit_code": status,
                "files": {p.name: sha(p) for p in sorted((folder / name).glob("*.json"))},
                "log_sha256": sha(folder / f"{name}.log"),
            }
            observations.append(observation)
            write(
                folder / "attempts.json",
                {
                    "observations": observations,
                    "provenance": provenance(protocol_sha256=sha(folder / "protocol.json")),
                },
            )
            print(json.dumps({k: v for k, v in observation.items() if k != "files"}), flush=True)
            if status:
                raise RuntimeError("worker failed; retain every attempt and log")
    result = audit(folder, verify_git=True)
    write(folder / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--geometry-folder", type=Path, default=Path("outputs/odometry-search-profile-v3")
    )
    parser.add_argument("--baseline-commit", default="5a80e7470487ceab148f7d49b560f48c1158665e")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "candidate"))
    parser.add_argument("--case", choices=("joint", "geometry"))
    parser.add_argument("--observation", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    result = audit(args.output) if args.audit else run(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if all(v["passes_api_gate"] for v in result["cases"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
