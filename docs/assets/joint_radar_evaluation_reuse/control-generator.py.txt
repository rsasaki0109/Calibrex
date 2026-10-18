"""Compare the complete profiled Radar calibration workflow in fresh processes.

KITTI OXTS supplies the reference kinematics; Radar velocities are generated
execution controls, not independent Radar accuracy evidence. Freeze both source
trees and retain every offset candidate, diagnostic, input and result.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_bspline_lidar_imu import _freeze, _verify_sources  # noqa: E402
from tools.benchmark_lazy_runtime_loading import _git_identity  # noqa: E402

SCHEMA = ROOT / "tools/schemas/radar_reference_benchmark.schema.json"
VERSION = "slac.radar_reference_benchmark/v0.1"
URL = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    "2011_09_30_drive_0018/2011_09_30_drive_0018_sync.zip"
)
RUNTIME = "calibrex/solvers/radar_spatiotemporal_lever_arm_solver.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def validate(value: dict, schema: dict) -> None:
    import jsonschema

    # Validate the serialized JSON representation (dataclass vectors are tuples).
    jsonschema.validate(json.loads(json.dumps(value, allow_nan=False)), schema)


def prepare(oxts: Path, output: Path) -> None:
    """Derive one declared generated-velocity control from all public OXTS samples."""
    from calibrex.core.geometry import quaternion_conjugate_xyzw, rotate_vector_xyzw
    from calibrex.data.kitti import read_timestamps
    from calibrex.solvers.radar_spatiotemporal_lever_arm_solver import (
        RadarSpatiotemporalLeverArmOptions,
        RadarVelocityMeasurement,
        ReferenceKinematicSample,
        _add,
        _cross,
        _interpolate,
    )

    output.mkdir(parents=True, exist_ok=False)
    stamps = read_timestamps(oxts / "timestamps.txt")
    epoch = stamps[0].timestamp_ns
    raw_inputs = {"timestamps.txt": sha(oxts / "timestamps.txt")}
    reference = []
    for stamp in stamps:
        name = f"data/{stamp.index:010d}.txt"
        raw = (oxts / name).read_bytes()
        raw_inputs[name] = hashlib.sha256(raw).hexdigest()
        values = [float(value) for value in raw.split()]
        reference.append(
            ReferenceKinematicSample(
                (stamp.timestamp_ns - epoch) * 1e-9,
                tuple(values[8:11]),
                tuple(values[20:23]),
            )
        )
    if (oxts / "dataformat.txt").is_file():
        raw_inputs["dataformat.txt"] = sha(oxts / "dataformat.txt")
    times = tuple(item.timestamp_sec for item in reference)
    angle = math.radians(17.0)
    rotation = (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))
    inverse = quaternion_conjugate_xyzw(rotation)
    measurements = []
    for index, sample in enumerate(reference):
        timestamp = sample.timestamp_sec
        # The generated timestamp needs the full offset/probe support. No point
        # or measurement budget is applied to the remaining interior samples.
        if timestamp < times[0] + 0.13 or timestamp > times[-1] - 0.13:
            continue
        shifted = timestamp + 0.023
        upper = bisect.bisect_left(times, shifted)
        kinematic = _interpolate(reference[upper - 1 : upper + 1], shifted)
        assert kinematic is not None
        velocity = rotate_vector_xyzw(
            inverse,
            _add(
                kinematic.linear_velocity_body_mps,
                _cross(kinematic.angular_velocity_body_radps, (0.42, -0.18, 0.27)),
            ),
        )
        velocity = tuple(
            value + 0.0001 * math.sin((index + 1) * (axis + 1) * 0.173)
            for axis, value in enumerate(velocity)
        )
        measurements.append(
            RadarVelocityMeasurement(
                f"generated-{index:05d}", timestamp, velocity, 1.0 + (index % 7) * 0.01
            )
        )
    control = {
        "schema_version": VERSION,
        "record_type": "control",
        "reference": [asdict(item) for item in reference],
        "measurements": [asdict(item) for item in measurements],
        "rotation": rotation,
        "options": asdict(RadarSpatiotemporalLeverArmOptions()),
        "provenance": {
            "generator_sha256": sha(Path(__file__)),
            "source_url": URL,
            "dataset_license": "CC BY-NC-SA 3.0",
            "raw_inputs": raw_inputs,
            "description": (
                "Public KITTI kinematics with generated Radar velocities; no Radar accuracy claim."
            ),
            "fields": "zero-based OXTS vf,vl,vu [8:11] and wf,wl,wu [20:23]",
            "truth": {"translation_m": [0.42, -0.18, 0.27], "time_offset_sec": 0.023},
        },
    }
    validate(control, json.loads(SCHEMA.read_bytes()))
    write(output / "control.json", control)


def worker(protocol_path: Path, variant: str, folder: Path) -> None:
    from calibrex.solvers import radar_spatiotemporal_lever_arm_solver as solver

    protocol = json.loads(protocol_path.read_bytes())
    control = json.loads((protocol_path.parent / "control.json").read_bytes())
    reference = [solver.ReferenceKinematicSample(**item) for item in control["reference"]]
    measurements = [solver.RadarVelocityMeasurement(**item) for item in control["measurements"]]
    options = solver.RadarSpatiotemporalLeverArmOptions(**control["options"])
    trace = []
    original = solver._profile_candidate

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        trace.append(
            {
                "offset": args[3],
                "translation": result[0].tolist() if result else None,
                "mse": result[1] if result else None,
                "singular_values": result[2].tolist() if result else None,
            }
        )
        return result

    solver._profile_candidate = record
    start = time.perf_counter()
    result = solver.RadarSpatiotemporalLeverArmSolver().solve(
        measurements, reference, control["rotation"], options
    )
    api_seconds = time.perf_counter() - start
    observed_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if observed_commit != protocol["commits"][variant]:
        raise ValueError("worker Git provenance differs from its pinned source")
    provenance = {
        "git_commit": observed_commit,
        "protocol_sha256": sha(protocol_path),
        "control_sha256": sha(protocol_path.parent / "control.json"),
    }
    write(
        folder / "result.json",
        {
            "schema_version": VERSION,
            "record_type": "result",
            "result": result.as_dict(),
            "candidates": trace,
            "provenance": provenance,
        },
    )
    source = Path(protocol["source_roots"][variant]) / "src"
    imports = {}
    for name, module in sys.modules.copy().items():
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            imports[name] = {"path": path.relative_to(source).as_posix(), "sha256": sha(path)}
    write(
        folder / "measurement.json",
        {
            "api_seconds": api_seconds,
            "imports": imports,
            "result_sha256": sha(folder / "result.json"),
            "provenance": provenance,
        },
    )
    print(json.dumps({"status": result.status, "candidates": len(trace)}), flush=True)


def audit(output: Path) -> dict:
    """Recheck every source, run, profile candidate and diagnostic without calibration."""
    protocol = json.loads((output / "protocol.json").read_bytes())
    schema = json.loads((output / "schema.json").read_bytes())
    validate(protocol, schema)
    digests = protocol["provenance"]
    for name, key in (
        ("launcher.py.txt", "launcher_sha256"),
        ("freezer.py.txt", "freezer_sha256"),
        ("identity.py.txt", "identity_sha256"),
        ("schema.json", "schema_sha256"),
        ("control.json", "control_sha256"),
    ):
        if sha(output / name) != digests[key]:
            raise ValueError(f"benchmark artifact changed: {name}")
    control = json.loads((output / "control.json").read_bytes())
    validate(control, schema)
    if control["provenance"]["generator_sha256"] != digests["launcher_sha256"]:
        raise ValueError("input generator differs from the frozen launcher")
    manifests = {}
    for variant in ("prior", "current"):
        path = output / f"sources-{variant}-manifest.json"
        if sha(path) != digests["manifests"][variant]:
            raise ValueError("source manifest changed")
        manifests[variant] = json.loads(path.read_bytes())
        capsule_path = output / f"sources-{variant}.tar.gz"
        if sha(capsule_path) != digests["capsules"][variant]:
            raise ValueError("source capsule changed")
        with tarfile.open(capsule_path) as capsule:
            contents = {}
            for member in capsule.getmembers():
                if not member.isfile() or not member.name.startswith("src/calibrex/"):
                    raise ValueError("unexpected source capsule entry")
                data = capsule.extractfile(member)
                assert data is not None
                contents[member.name.removeprefix("src/")] = hashlib.sha256(data.read()).hexdigest()
        if contents != manifests[variant]:
            raise ValueError("source capsule and manifest differ")
    if manifests["prior"].keys() != manifests["current"].keys():
        raise ValueError("unexpected runtime source inventory change")
    changed = sorted(
        k for k in manifests["prior"] if manifests["prior"][k] != manifests["current"][k]
    )
    if changed != [RUNTIME]:
        raise ValueError("unexpected runtime source changes")
    attempts = json.loads((output / "attempts.json").read_bytes())
    expected = [
        (repeat, variant)
        for repeat in range(protocol["repetitions"])
        for variant in (("prior", "current") if repeat % 2 == 0 else ("current", "prior"))
    ]
    if [(r["repeat"], r["variant"]) for r in attempts] != expected:
        raise ValueError("missing, duplicate or reordered runs")
    outputs = []
    api = {variant: [] for variant in ("prior", "current")}
    for record in attempts:
        if (
            record["exit_code"] != 0
            or not math.isfinite(record["seconds"])
            or record["seconds"] <= 0
        ):
            raise ValueError("failed run or invalid whole-process timing")
        variant = record["variant"]
        folder = output / f"{record['repeat']:02d}-{variant}"
        measured = json.loads((folder / "measurement.json").read_bytes())
        if sha(folder / "result.json") != measured["result_sha256"]:
            raise ValueError("result changed")
        result = json.loads((folder / "result.json").read_bytes())
        validate(result, schema)
        expected_provenance = {
            "git_commit": protocol["commits"][variant],
            "protocol_sha256": sha(output / "protocol.json"),
            "control_sha256": digests["control_sha256"],
        }
        if (
            result["provenance"] != expected_provenance
            or measured["provenance"] != expected_provenance
        ):
            raise ValueError("run provenance differs")
        value = result["result"]
        if value["status"] != "converged" or len(value["known_bad_probes"]) != 8:
            raise ValueError("control did not complete its calibration and probes")
        assigned = value["train_measurement_ids"] + value["holdout_measurement_ids"]
        if (
            sorted(assigned) != sorted(item["measurement_id"] for item in control["measurements"])
            or len(set(assigned)) != len(assigned)
            or value["solver_options"] != control["options"]
        ):
            raise ValueError("input measurements or solver settings differ")
        if (
            json.loads((folder / "stdout.txt").read_bytes())
            != {"status": value["status"], "candidates": len(result["candidates"])}
            or (folder / "stderr.txt").read_bytes()
        ):
            raise ValueError("worker output differs")
        seconds = measured["api_seconds"]
        if not math.isfinite(seconds) or not 0 < seconds <= record["seconds"]:
            raise ValueError("invalid API timing")
        api[variant].append(seconds)
        for name, item in measured["imports"].items():
            if (
                not name.startswith("calibrex")
                or manifests[variant].get(item["path"]) != item["sha256"]
            ):
                raise ValueError("worker used an unfrozen module")
        if not any(item["path"] == RUNTIME for item in measured["imports"].values()):
            raise ValueError("measured solver import is missing")
        candidates = result["candidates"]
        options = control["options"]
        count = round(2 * options["max_abs_time_offset_sec"] / options["time_offset_step_sec"]) + 1
        offsets = [
            -options["max_abs_time_offset_sec"] + i * options["time_offset_step_sec"]
            for i in range(count)
        ]
        if [item["offset"] for item in candidates] != offsets:
            raise ValueError("profile candidate count or offsets differ")
        outputs.append(json.dumps([result["result"], candidates], allow_nan=False))
    if len(set(outputs)) != 1:
        raise ValueError("complete results or profile candidates differ")
    medians = {
        v: statistics.median(r["seconds"] for r in attempts if r["variant"] == v) for v in api
    }
    api_medians = {v: statistics.median(values) for v, values in api.items()}
    return {
        "whole_seconds_median": medians,
        "api_seconds_median": api_medians,
        "whole_speedup": medians["prior"] / medians["current"],
        "api_speedup": api_medians["prior"] / api_medians["current"],
        "verified_runs": len(attempts),
        "complete_results_exact": True,
        "profile_candidates_per_run": count,
        "reference_count": len(control["reference"]),
        "measurement_count": len(control["measurements"]),
        "runtime_source_changes": changed,
        "provenance": {
            "protocol_sha256": sha(output / "protocol.json"),
            "attempts_sha256": sha(output / "attempts.json"),
            "control_sha256": digests["control_sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-oxts", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker-protocol", type=Path)
    parser.add_argument("--variant", choices=("prior", "current"))
    parser.add_argument("--audit-only", type=Path)
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit(args.audit_only), indent=2))
        return
    if args.worker_protocol:
        worker(args.worker_protocol, args.variant, args.output_dir)
        return
    if args.prepare_oxts:
        prepare(args.prepare_oxts, args.output_dir)
        return
    if not all((args.baseline_commit, args.control, args.output_dir)) or args.repetitions < 2:
        parser.error(
            "baseline commit, control, output directory and at least two repetitions required"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commits = {
        v: subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
        for v, ref in (("prior", args.baseline_commit), ("current", "HEAD"))
    }
    for source, target in (
        (Path(__file__), "launcher.py.txt"),
        (SCHEMA, "schema.json"),
        (ROOT / "tools/benchmark_bspline_lidar_imu.py", "freezer.py.txt"),
        (ROOT / "tools/benchmark_lazy_runtime_loading.py", "identity.py.txt"),
        (args.control, "control.json"),
    ):
        shutil.copyfile(source, output / target)
    manifests, capsules, roots = {}, {}, {}
    for variant, commit in commits.items():
        frozen = output / f"sources-{variant}"
        _freeze(frozen, commit)
        _git_identity(frozen, commit)
        roots[variant] = frozen.as_posix()
        shutil.copyfile(frozen / "manifest.json", output / f"sources-{variant}-manifest.json")
        manifests[variant] = sha(frozen / "manifest.json")
        path = output / f"sources-{variant}.tar.gz"
        with tarfile.open(path, "w:gz") as capsule:
            for source in sorted((frozen / "src").rglob("*.py")):
                capsule.add(source, arcname=source.relative_to(frozen).as_posix())
        capsules[variant] = sha(path)
    protocol = {
        "schema_version": VERSION,
        "record_type": "protocol",
        "commits": commits,
        "source_roots": roots,
        "repetitions": args.repetitions,
        "fresh_workers": True,
        "blas_threads": 1,
        "host_unreserved": True,
        "provenance": {
            "control_sha256": sha(output / "control.json"),
            "launcher_sha256": sha(Path(__file__)),
            "schema_sha256": sha(SCHEMA),
            "freezer_sha256": sha(output / "freezer.py.txt"),
            "identity_sha256": sha(output / "identity.py.txt"),
            "manifests": manifests,
            "capsules": capsules,
            "python": sys.version,
            "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
        },
    }
    validate(protocol, json.loads(SCHEMA.read_bytes()))
    validate(json.loads((output / "control.json").read_bytes()), json.loads(SCHEMA.read_bytes()))
    write(output / "protocol.json", protocol)
    protocol_sha = sha(output / "protocol.json")
    attempts = []
    for repeat in range(args.repetitions):
        for variant in ("prior", "current") if repeat % 2 == 0 else ("current", "prior"):
            frozen = output / f"sources-{variant}"
            _verify_sources(frozen)
            folder = output / f"{repeat:02d}-{variant}"
            folder.mkdir()
            env = os.environ.copy()
            env.update(
                dict.fromkeys(("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"), "1")
            )
            env.update(PYTHONPATH=str(frozen / "src"), PYTHONDONTWRITEBYTECODE="1")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-protocol",
                str(output / "protocol.json"),
                "--variant",
                variant,
                "--output-dir",
                str(folder),
            ]
            print(f"starting {repeat:02d}-{variant}", flush=True)
            with (
                (folder / "stdout.txt").open("w") as stdout,
                (folder / "stderr.txt").open("w") as stderr,
            ):
                start = time.perf_counter()
                done = subprocess.run(command, cwd=frozen, env=env, stdout=stdout, stderr=stderr)
                seconds = time.perf_counter() - start
            attempts.append(
                {
                    "repeat": repeat,
                    "variant": variant,
                    "seconds": seconds,
                    "exit_code": done.returncode,
                }
            )
            write(output / "attempts.json", attempts)
            print(json.dumps(attempts[-1]), flush=True)
            if done.returncode:
                raise RuntimeError(f"worker failed; preserved {folder}")
            _verify_sources(frozen)
            if (
                sha(output / "protocol.json") != protocol_sha
                or sha(Path(__file__)) != protocol["provenance"]["launcher_sha256"]
                or sha(SCHEMA) != protocol["provenance"]["schema_sha256"]
                or sha(output / "control.json") != protocol["provenance"]["control_sha256"]
            ):
                raise ValueError("benchmark protocol, launcher, schema or input changed")
    summary = audit(output)
    write(output / "audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
