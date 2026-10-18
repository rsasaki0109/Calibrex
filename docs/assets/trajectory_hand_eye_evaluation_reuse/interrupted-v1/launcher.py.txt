"""Compare complete prepared-input INS-LiDAR evidence in fresh frozen workers."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tarfile
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_bspline_lidar_imu import _freeze, _verify_sources  # noqa: E402
from tools.benchmark_lazy_runtime_loading import _git_identity  # noqa: E402

VERSION = "slac.trajectory_hand_eye_reuse_benchmark/v0.1"
SCHEMA = ROOT / "tools/schemas/trajectory_hand_eye_reuse_benchmark.schema.json"
RUNTIME = "calibrex/solvers/trajectory_hand_eye_solver.py"
PINNED = {
    "problem.json",
    "control-generator.py.txt",
    "raw-oxts.tar.gz",
    "launcher.py.txt",
    "freezer.py.txt",
    "identity.py.txt",
    "schema.json",
    "result-schema.json",
    "sources-prior-manifest.json",
    "sources-current-manifest.json",
    "sources-prior.tar.gz",
    "sources-current.tar.gz",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    import jsonschema

    jsonschema.validate(value, json.loads(SCHEMA.read_bytes()))
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def comparable(value: dict) -> dict:
    value = json.loads(json.dumps(value, allow_nan=False))
    for name in ("created_at", "git_commit"):
        value["provenance"].pop(name, None)
    return value


def verify_git(full: str, short: str, expected: str, recorded: str | None) -> None:
    if (
        full != expected
        or recorded != short
        or re.fullmatch(r"[a-f0-9]{4,40}", short) is None
        or not full.startswith(short)
    ):
        raise ValueError("actual Git identities differ from the source pin")


def worker(protocol_path: Path, variant: str, folder: Path) -> None:
    import numpy as np

    import calibrex
    from calibrex.core.ins_lidar_hand_eye import InsLidarHandEyeProvenance, InsLidarOdometrySummary
    from calibrex.core.provenance import git_commit
    from calibrex.evaluation import ins_lidar_hand_eye as evaluation
    from calibrex.solvers import trajectory_hand_eye_solver as solver

    protocol = json.loads(protocol_path.read_bytes())
    full = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    short = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    verify_git(full, short, protocol["commits"][variant], git_commit())
    provenance = {
        "protocol_sha256": sha(protocol_path),
        "git_commit": full,
        "git_commit_short": short,
        "problem_sha256": sha(protocol_path.parent / "problem.json"),
    }
    history = {
        "schema_version": VERSION,
        "record_type": "history",
        "fits": [],
        "normalized": [],
        "least_squares": [],
        "provenance": provenance,
    }
    counts = dict.fromkeys(
        (
            "solve_calls",
            "normalized_requests",
            "least_squares_calls",
            "poses_at_calls",
            "motion_sigmas_calls",
        ),
        0,
    )
    active = None
    original_solve = solver.TrajectoryHandEyeSolver.solve
    original_lsq = solver.least_squares
    original_poses = solver.ReferenceTrajectory.poses_at
    original_sigmas = solver.motion_sigmas

    def solve(instance, reference, motions, *args, **kwargs):
        nonlocal active
        active = counts["solve_calls"]
        counts["solve_calls"] += 1
        row = {"fit": active, "input_motion_ids": [m.motion_id for m in motions]}
        history["fits"].append(row)
        result = original_solve(instance, reference, motions, *args, **kwargs)
        row.update(status=result.status, usable_motion_ids=list(result.motion_ids))
        active = None
        return result

    def record(motions, base, params, rows):
        counts["normalized_requests"] += 1
        history["normalized"].append(
            {
                "fit": active,
                "motion_count": len(motions),
                "base": base.reshape(-1).tolist(),
                "params": params.tolist(),
                "residual_count": rows.size,
                "residual_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            }
        )
        return rows

    if hasattr(solver, "_make_normalized_evaluator"):
        original_factory = solver._make_normalized_evaluator

        def factory(reference, motions, options):
            evaluate = original_factory(reference, motions, options)

            def traced(base, params):
                return record(motions, base, params, evaluate(base, params))

            return traced

        solver._make_normalized_evaluator = factory
    else:
        original_normalized = solver._normalized_residuals

        def normalized(reference, motions, base, params, options):
            return record(
                motions,
                base,
                params,
                original_normalized(reference, motions, base, params, options),
            )

        solver._normalized_residuals = normalized

    def lsq(*args, **kwargs):
        result = original_lsq(*args, **kwargs)
        counts["least_squares_calls"] += 1
        row = {
            "fit": active,
            "x": result.x.tolist(),
            "active_mask": result.active_mask.tolist(),
            "fun_shape": list(result.fun.shape),
            "jac_shape": list(result.jac.shape),
            "fun_sha256": hashlib.sha256(result.fun.tobytes()).hexdigest(),
            "jac_sha256": hashlib.sha256(result.jac.tobytes()).hexdigest(),
        }
        row.update(
            {
                key: result[key]
                for key in ("nfev", "njev", "status", "message", "cost", "optimality")
            }
        )
        history["least_squares"].append(row)
        return result

    def poses(*args):
        counts["poses_at_calls"] += 1
        return original_poses(*args)

    def sigmas(*args):
        counts["motion_sigmas_calls"] += 1
        return original_sigmas(*args)

    solver.TrajectoryHandEyeSolver.solve = solve
    solver.least_squares = lsq
    solver.ReferenceTrajectory.poses_at = poses
    solver.motion_sigmas = sigmas
    evaluation.motion_sigmas = sigmas
    start = time.perf_counter()
    problem_path = protocol_path.parent / "problem.json"
    problem = json.loads(problem_path.read_bytes())
    reference = solver.ReferenceTrajectory(
        np.asarray(problem["reference_times_s"]), np.asarray(problem["reference_poses"])
    )
    motions = [
        solver.SensorMotion(
            m["motion_id"], m["block"], m["start_s"], m["end_s"], np.asarray(m["motion"])
        )
        for m in problem["motions"]
    ]
    options = evaluation.InsLidarRunOptions()
    result = evaluation.evaluate_ins_lidar_hand_eye(
        reference, motions, options, reference_transform=np.asarray(problem["reference_transform"])
    )
    first, last = problem["provenance"]["sensor_pose_indices"]
    artifact = evaluation.build_ins_lidar_artifact(
        result,
        options,
        odometry=InsLidarOdometrySummary(
            frame_count=last - first + 1, registration_count=0, converged_registrations=0
        ),
        provenance=InsLidarHandEyeProvenance(
            generator="prepared_hand_eye_control",
            generator_version=calibrex.__version__,
            git_commit=git_commit(),
            command=["complete-hand-eye-control"],
            dataset_family="kitti_oxts_generated_sensor_control",
            drive_ids=["2011_09_30_drive_0018_sync"],
            input_sha256=sha(problem_path),
            dataset_license="CC BY-NC-SA 3.0",
        ),
        reference_calibration="Declared generated sensor-control transform; evaluation only",
        limitations=[
            "Generated sensor motions from public OXTS; performance/parity control, "
            "no raw LiDAR registration or independent accuracy evidence."
        ],
    )
    artifact.save(folder / "result.json")
    api_seconds = time.perf_counter() - start
    verify_git(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
        protocol["commits"][variant],
        artifact.provenance.git_commit,
    )
    (folder / "history.json.gz").write_bytes(
        gzip.compress(
            (json.dumps(history, separators=(",", ":"), allow_nan=False) + "\n").encode(), mtime=0
        )
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
            "schema_version": VERSION,
            "record_type": "measurement",
            "api_seconds": api_seconds,
            "counts": counts,
            "result_sha256": sha(folder / "result.json"),
            "history_sha256": sha(folder / "history.json.gz"),
            "imports": imports,
            "provenance": provenance,
        },
    )
    print(
        json.dumps(
            {
                "solver_status": artifact.solver_status,
                "policy_status": artifact.policy_status,
                "jackknife_fits": artifact.jackknife_fits,
            }
        ),
        flush=True,
    )


def input_lineage(output: Path, problem: dict) -> None:
    import numpy as np
    from scipy.spatial.transform import Rotation

    from calibrex.data.kitti import (
        _packet_from_oxts_values,
        _parse_kitti_time,
        kitti_oxts_pose_world_imu,
    )
    from calibrex.data.kitti_ins_lidar import se3_matrix
    from calibrex.evaluation.ins_lidar_hand_eye import InsLidarRunOptions, build_sensor_motions
    from calibrex.solvers.trajectory_hand_eye_solver import ReferenceTrajectory

    with tarfile.open(output / "raw-oxts.tar.gz") as bundle:
        raw = {}
        for member in bundle.getmembers():
            if (
                not member.isfile()
                or member.name.startswith("/")
                or ".." in Path(member.name).parts
                or member.name in raw
            ):
                raise ValueError("unsafe or duplicate raw input capsule member")
            stream = bundle.extractfile(member)
            assert stream is not None
            raw[member.name] = stream.read()
    provenance = problem["provenance"]
    original = json.loads(raw["download-provenance.json"])
    members = original.get("members", original.get("files"))
    if (
        hashlib.sha256(raw["download-provenance.json"]).hexdigest()
        != provenance["source_provenance_sha256"]
    ):
        raise ValueError("original download provenance differs")
    if set(raw) != set(provenance["source_files"]) | {"download-provenance.json"}:
        raise ValueError("raw input inventory differs")
    for name, digest in provenance["source_files"].items():
        if hashlib.sha256(raw[name]).hexdigest() != digest or members[name]["sha256"] != digest:
            raise ValueError("original OXTS input differs")
    if sha(output / "control-generator.py.txt") != provenance["generator_sha256"]:
        raise ValueError("control generator differs")
    low, high = provenance["reference_pose_indices"]
    if low != 0 or high != 2761 or provenance["sensor_pose_indices"] != [10, 2751]:
        raise ValueError("full public-drive selection differs")
    times = np.asarray(
        [
            _parse_kitti_time(line.strip()) * 1.0e-9
            for line in raw["timestamps.txt"].decode().splitlines()
            if line.strip()
        ][: high + 1]
    )
    packets = [
        _packet_from_oxts_values([float(v) for v in raw[f"data/{i:010d}.txt"].decode().split()])
        for i in range(high + 1)
    ]
    poses = np.stack([se3_matrix(kitti_oxts_pose_world_imu(p, packets[0])) for p in packets])
    if (
        times.tobytes() != np.asarray(problem["reference_times_s"]).tobytes()
        or poses.tobytes() != np.asarray(problem["reference_poses"]).tobytes()
    ):
        raise ValueError("reference trajectory differs from original OXTS")
    reference = ReferenceTrajectory(times, poses)
    first, last = provenance["sensor_pose_indices"]
    sensor_times = times[first : last + 1]
    sensor = reference.poses_at(sensor_times + provenance["declared_clock_offset_s"])
    sensor[:, :3, 3] *= provenance["declared_reference_scale"]
    sensor = sensor @ np.asarray(problem["reference_transform"])
    motions = build_sensor_motions(
        list(sensor_times), list(sensor), InsLidarRunOptions(), id_prefix="control/"
    )
    rng = np.random.default_rng(provenance["noise_seed"])
    for m in motions:
        m.motion[:3, :3] = (
            Rotation.from_rotvec(
                rng.normal(0.0, provenance["declared_rotation_noise_sigma_rad"], 3)
            ).as_matrix()
            @ m.motion[:3, :3]
        )
        m.motion[:3, 3] += rng.normal(0.0, provenance["declared_translation_noise_sigma_m"], 3)
    generated = [
        {
            "motion_id": m.motion_id,
            "block": m.block,
            "start_s": m.start_s,
            "end_s": m.end_s,
            "motion": m.motion.tolist(),
        }
        for m in motions
    ]
    if generated != problem["motions"] or problem["options"] != {}:
        raise ValueError("generated sensor motions or unchanged defaults differ")


def replay_work(history: dict, result: dict, current: bool) -> dict:
    residuals, clocks = {}, {}
    misses = 0
    for event in history["normalized"]:
        fit = event["fit"]
        if fit not in range(len(history["fits"])):
            raise ValueError("unknown fit in evaluation history")
        if event["residual_count"] != 6 * event["motion_count"] or not all(
            math.isfinite(x) for x in event["base"] + event["params"]
        ):
            raise ValueError("invalid normalized evaluation dimensions or parameters")
        cache = residuals.setdefault(fit, OrderedDict())
        clock = clocks.setdefault(fit, OrderedDict())
        key = struct.pack("16d", *event["base"]), struct.pack("8d", *event["params"])
        if key in cache:
            cache.move_to_end(key)
            continue
        cache[key] = None
        if len(cache) > 128:
            cache.popitem(last=False)
        offset = struct.pack("d", event["params"][6])
        if offset in clock:
            clock.move_to_end(offset)
        else:
            misses += 1
            clock[offset] = None
            if len(clock) > 16:
                clock.popitem(last=False)
    controls = sum(dof["known_bad_control"] is not None for dof in result["dofs"])
    fits, requests = len(history["fits"]), len(history["normalized"])
    # One final motion summary per solve, one held-out summary, and two
    # held-out chi-square evaluations for every reported control.
    extra_motions = fits + 1 + 2 * controls
    return {
        "solve_calls": fits,
        "normalized_requests": requests,
        "least_squares_calls": len(history["least_squares"]),
        "poses_at_calls": 2 * (misses if current else requests) + 2 * extra_motions,
        "motion_sigmas_calls": (fits if current else requests) + 2 * controls,
    }


def audit(output: Path) -> dict:
    import jsonschema

    schema = json.loads((output / "schema.json").read_bytes())
    protocol = json.loads((output / "protocol.json").read_bytes())
    jsonschema.validate(protocol, schema)
    if set(protocol["provenance"]["files"]) != PINNED:
        raise ValueError("incomplete pinned artifact inventory")
    for name, digest in protocol["provenance"]["files"].items():
        if sha(output / name) != digest:
            raise ValueError(f"pinned artifact changed: {name}")
    problem = json.loads((output / "problem.json").read_bytes())
    jsonschema.validate(problem, schema)
    input_lineage(output, problem)
    manifests = {}
    for variant in ("prior", "current"):
        manifests[variant] = json.loads((output / f"sources-{variant}-manifest.json").read_bytes())
        contents = {}
        with tarfile.open(output / f"sources-{variant}.tar.gz") as bundle:
            for member in bundle.getmembers():
                if (
                    not member.isfile()
                    or not member.name.startswith("src/calibrex/")
                    or ".." in Path(member.name).parts
                ):
                    raise ValueError("unexpected source capsule member")
                stream = bundle.extractfile(member)
                assert stream is not None
                name = member.name.removeprefix("src/")
                if name in contents:
                    raise ValueError("duplicate source capsule member")
                contents[name] = hashlib.sha256(stream.read()).hexdigest()
        if contents != manifests[variant]:
            raise ValueError("source capsule and manifest differ")
    if manifests["prior"].keys() != manifests["current"].keys() or sorted(
        name
        for name in manifests["prior"]
        if manifests["prior"][name] != manifests["current"][name]
    ) != [RUNTIME]:
        raise ValueError("unexpected core runtime changes")
    attempts = json.loads((output / "attempts.json").read_bytes())
    jsonschema.validate(attempts, schema)
    if attempts["provenance"] != {"protocol_sha256": sha(output / "protocol.json")}:
        raise ValueError("attempt provenance differs")
    runs = attempts["runs"]
    expected = [
        (i, v)
        for i in range(protocol["repetitions"])
        for v in (("prior", "current") if i % 2 == 0 else ("current", "prior"))
    ]
    if [(r["repeat"], r["variant"]) for r in runs] != expected:
        raise ValueError("missing, duplicate or reordered runs")
    outputs, histories = [], []
    work = {v: [] for v in ("prior", "current")}
    api = {v: [] for v in ("prior", "current")}
    blocks = sorted({m["block"] for m in problem["motions"]})
    holdout = [b for i, b in enumerate(blocks) if i % 3 == 2]
    train = [b for b in blocks if b not in holdout]
    fitted = [m for m in problem["motions"] if m["block"] in train]
    fit_ids = [[m["motion_id"] for m in fitted]] + [
        [m["motion_id"] for m in fitted if m["block"] != b] for b in train
    ]
    result_schema = json.loads((output / "result-schema.json").read_bytes())
    for run in runs:
        if run["exit_code"] != 0 or not math.isfinite(run["seconds"]) or run["seconds"] <= 0:
            raise ValueError("failed worker or invalid whole-process timing")
        variant = run["variant"]
        folder = output / f"{run['repeat']:02d}-{variant}"
        measurement = json.loads((folder / "measurement.json").read_bytes())
        jsonschema.validate(measurement, schema)
        if (
            sha(folder / "result.json") != measurement["result_sha256"]
            or sha(folder / "history.json.gz") != measurement["history_sha256"]
        ):
            raise ValueError("result or history changed")
        result = json.loads((folder / "result.json").read_bytes())
        history = json.loads(gzip.decompress((folder / "history.json.gz").read_bytes()))
        jsonschema.validate(result, result_schema)
        jsonschema.validate(history, schema)
        p = measurement["provenance"]
        verify_git(
            p["git_commit"],
            p["git_commit_short"],
            protocol["commits"][variant],
            result["provenance"]["git_commit"],
        )
        if (
            p
            != {
                "protocol_sha256": sha(output / "protocol.json"),
                "git_commit": protocol["commits"][variant],
                "git_commit_short": p["git_commit_short"],
                "problem_sha256": sha(output / "problem.json"),
            }
            or history["provenance"] != p
            or result["provenance"]["input_sha256"] != p["problem_sha256"]
        ):
            raise ValueError("run or input provenance differs")
        if (
            result["solver_status"] != "converged"
            or result["policy_status"] != "pass"
            or result["jackknife_fits"] != len(train)
            or result["train_blocks"] != train
            or result["holdout_blocks"] != holdout
            or result["odometry"]["frame_count"] != 2742
            or result["odometry"]["registration_count"] != 0
        ):
            raise ValueError("completion, full workload or disjoint split differs")
        if len(history["fits"]) != len(fit_ids) or any(
            row["fit"] != index
            or row["status"] != "converged"
            or row["input_motion_ids"] != fit_ids[index]
            or row["usable_motion_ids"] != fit_ids[index]
            for index, row in enumerate(history["fits"])
        ):
            raise ValueError("full fit/jackknife inputs differ")
        for event in history["normalized"]:
            if event["fit"] not in range(len(fit_ids)) or event["motion_count"] != len(
                fit_ids[event["fit"]]
            ):
                raise ValueError("normalized evaluation uses different motions")
        for event in history["least_squares"]:
            if event["fit"] not in range(len(fit_ids)):
                raise ValueError("unknown optimizer fit")
            size = 6 * len(fit_ids[event["fit"]]) + 8
            if event["fun_shape"] != [size] or event["jac_shape"] != [size, 8]:
                raise ValueError("optimizer residual or Jacobian dimensions differ")
        expected_work = replay_work(history, result, variant == "current")
        if measurement["counts"] != expected_work:
            raise ValueError("observed work differs from full history and cache replay")
        seconds = measurement["api_seconds"]
        if not math.isfinite(seconds) or not 0 < seconds <= run["seconds"]:
            raise ValueError("invalid API timing")
        if (folder / "stderr.txt").read_bytes() or json.loads(
            (folder / "stdout.txt").read_bytes()
        ) != {
            "solver_status": result["solver_status"],
            "policy_status": result["policy_status"],
            "jackknife_fits": result["jackknife_fits"],
        }:
            raise ValueError("worker output differs")
        for name, imported in measurement["imports"].items():
            if (
                not name.startswith("calibrex")
                or manifests[variant].get(imported["path"]) != imported["sha256"]
            ):
                raise ValueError("worker imported unfrozen source")
        if not any(imported["path"] == RUNTIME for imported in measurement["imports"].values()):
            raise ValueError("measured solver import is missing")
        outputs.append(json.dumps(comparable(result), allow_nan=False))
        histories.append(json.dumps(comparable(history), allow_nan=False))
        work[variant].append(measurement["counts"])
        api[variant].append(seconds)
    if len(set(outputs)) != 1 or len(set(histories)) != 1:
        raise ValueError("complete results or residual/optimizer histories differ")
    if any(any(row != rows[0] for row in rows) for rows in work.values()):
        raise ValueError("work differs between repeated runs")
    if (
        work["current"][0]["poses_at_calls"] >= work["prior"][0]["poses_at_calls"]
        or work["current"][0]["motion_sigmas_calls"] >= work["prior"][0]["motion_sigmas_calls"]
    ):
        raise ValueError("reuse did not reduce repeated work")
    whole = {v: statistics.median(r["seconds"] for r in runs if r["variant"] == v) for v in api}
    api_medians = {v: statistics.median(seconds) for v, seconds in api.items()}
    report = {
        "schema_version": VERSION,
        "record_type": "audit",
        "verified_runs": len(runs),
        "complete_results_and_histories_exact": True,
        "reference_pose_count": len(problem["reference_times_s"]),
        "sensor_pose_count": 2742,
        "motion_count": len(problem["motions"]),
        "jackknife_fits": len(train),
        "work_per_run": {v: rows[0] for v, rows in work.items()},
        "whole_seconds_median": whole,
        "api_seconds_median": api_medians,
        "whole_speedup": whole["prior"] / whole["current"],
        "api_speedup": api_medians["prior"] / api_medians["current"],
        "runtime_source_changes": [RUNTIME],
        "provenance": {
            "protocol_sha256": sha(output / "protocol.json"),
            "attempts_sha256": sha(output / "attempts.json"),
            "problem_sha256": sha(output / "problem.json"),
        },
    }
    jsonschema.validate(report, schema)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "problem",
        "control-generator",
        "oxts-dir",
        "output-dir",
        "audit-only",
        "worker-protocol",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--variant", choices=("prior", "current"))
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit(args.audit_only), indent=2))
        return
    if args.worker_protocol:
        worker(args.worker_protocol, args.variant, args.output_dir)
        return
    if (
        not all(
            (
                args.problem,
                args.control_generator,
                args.oxts_dir,
                args.output_dir,
                args.baseline_commit,
            )
        )
        or args.repetitions < 2
    ):
        parser.error(
            "problem/generation inputs, baseline, fresh output and two repetitions required"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for source, target in (
        (args.problem, "problem.json"),
        (args.control_generator, "control-generator.py.txt"),
        (Path(__file__), "launcher.py.txt"),
        (SCHEMA, "schema.json"),
        (ROOT / "tools/benchmark_bspline_lidar_imu.py", "freezer.py.txt"),
        (ROOT / "tools/benchmark_lazy_runtime_loading.py", "identity.py.txt"),
        (ROOT / "schemas/ins_lidar_hand_eye.schema.json", "result-schema.json"),
    ):
        shutil.copyfile(source, output / target)
    problem = json.loads((output / "problem.json").read_bytes())
    with tarfile.open(output / "raw-oxts.tar.gz", "w:gz") as bundle:
        for name in sorted(problem["provenance"]["source_files"]):
            bundle.add(args.oxts_dir / name, arcname=name)
        bundle.add(args.oxts_dir / "provenance.json", arcname="download-provenance.json")
    input_lineage(output, problem)
    commits = {
        v: subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
        for v, ref in (("prior", args.baseline_commit), ("current", "HEAD"))
    }
    roots = {}
    for variant, commit in commits.items():
        frozen = output / f"sources-{variant}"
        _freeze(frozen, commit)
        _git_identity(frozen, commit)
        roots[variant] = frozen.as_posix()
        shutil.copyfile(frozen / "manifest.json", output / f"sources-{variant}-manifest.json")
        with tarfile.open(output / f"sources-{variant}.tar.gz", "w:gz") as capsule:
            for source in sorted((frozen / "src").rglob("*.py")):
                capsule.add(source, arcname=source.relative_to(frozen).as_posix())
    from calibrex.evaluation.ins_lidar_hand_eye import InsLidarRunOptions

    protocol = {
        "schema_version": VERSION,
        "record_type": "protocol",
        "commits": commits,
        "source_roots": roots,
        "repetitions": args.repetitions,
        "fresh_workers": True,
        "blas_threads": 1,
        "host_unreserved": True,
        "options": asdict(InsLidarRunOptions()),
        "provenance": {
            "files": {name: sha(output / name) for name in sorted(PINNED)},
            "python": sys.version,
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "pydantic", "PyYAML", "jsonschema")
            },
        },
    }
    write(output / "protocol.json", protocol)
    protocol_sha = sha(output / "protocol.json")
    runs = []
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
            print(f"starting {repeat:02d}-{variant}", flush=True)
            with (
                (folder / "stdout.txt").open("w") as stdout,
                (folder / "stderr.txt").open("w") as stderr,
            ):
                start = time.perf_counter()
                done = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-protocol",
                        str(output / "protocol.json"),
                        "--variant",
                        variant,
                        "--output-dir",
                        str(folder),
                    ],
                    cwd=frozen,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                )
                seconds = time.perf_counter() - start
            runs.append(
                {
                    "repeat": repeat,
                    "variant": variant,
                    "seconds": seconds,
                    "exit_code": done.returncode,
                }
            )
            write(
                output / "attempts.json",
                {
                    "schema_version": VERSION,
                    "record_type": "attempts",
                    "runs": runs,
                    "provenance": {"protocol_sha256": protocol_sha},
                },
            )
            print(json.dumps(runs[-1]), flush=True)
            if done.returncode:
                raise RuntimeError(f"worker failed; preserved {folder}")
            _verify_sources(frozen)
            if (
                any(
                    sha(output / name) != digest
                    for name, digest in protocol["provenance"]["files"].items()
                )
                or sha(Path(__file__)) != protocol["provenance"]["files"]["launcher.py.txt"]
                or sha(SCHEMA) != protocol["provenance"]["files"]["schema.json"]
                or sha(output / "protocol.json") != protocol_sha
            ):
                raise ValueError("pinned inputs, sources or launcher changed during measurement")
    report = audit(output)
    write(output / "audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
