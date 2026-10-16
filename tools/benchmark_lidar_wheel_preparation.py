"""Measure complete LiDAR-wheel evaluation in alternating frozen fresh workers."""

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
from tools.prepare_lidar_wheel_control import SCOPE, VERSION, derive_control  # noqa: E402

SCHEMA = ROOT / "tools/schemas/lidar_wheel_preparation_benchmark.schema.json"
RUNTIME = "calibrex/solvers/lidar_wheel_solver.py"
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
COUNTERS = (
    "solve_calls",
    "residual_requests",
    "least_squares_calls",
    "moving_calls",
    "motion_selection_visits",
    "wheel_at_calls",
    "wheel_samples_queried",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    import jsonschema

    jsonschema.validate(value, json.loads(SCHEMA.read_bytes()))
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def comparable(value: dict) -> dict:
    value = json.loads(json.dumps(value, allow_nan=False))
    for name in ("created_at", "git_commit", "git_commit_short"):
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


def default_options() -> dict:
    from calibrex.evaluation.lidar_wheel import LidarWheelRunOptions

    return json.loads(json.dumps(asdict(LidarWheelRunOptions())))


def inputs(problem: dict) -> tuple:
    import numpy as np

    from calibrex.solvers.lidar_wheel_solver import WheelOdometry
    from calibrex.solvers.vehicle_frame_solver import VehicleMotion

    motions = [
        VehicleMotion(
            m["block"],
            m["start_s"],
            m["end_s"],
            np.asarray(m["velocity_mps"]),
            np.asarray(m["angular_rate_rps"]),
        )
        for m in problem["motions"]
    ]
    wheel = WheelOdometry(
        np.asarray(problem["times_s"]),
        np.asarray(problem["speed_mps"]),
        np.asarray(problem["yaw_rate_rps"]),
    )
    return motions, wheel


def worker(protocol_path: Path, variant: str, folder: Path) -> None:
    import numpy as np

    import calibrex
    from calibrex.core.lidar_wheel_odometry import LidarWheelProvenance
    from calibrex.core.provenance import git_commit
    from calibrex.evaluation import lidar_wheel as evaluation
    from calibrex.solvers import lidar_wheel_solver as solver

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
        "residuals": [],
        "least_squares": [],
        "provenance": provenance,
    }
    counts = dict.fromkeys(COUNTERS, 0)
    active = None
    indices = {}
    original_solve, original_residual = solver.solve_lidar_wheel, solver.lidar_wheel_residuals
    original_lsq, original_moving = solver.least_squares, solver._moving
    original_at, original_scan = solver.WheelOdometry.at, solver._yaw_rate_offset_scan

    def solve(motions, *args, **kwargs):
        nonlocal active
        active = counts["solve_calls"]
        counts["solve_calls"] += 1
        row = {"fit": active, "input_indices": [indices[id(m)] for m in motions]}
        history["fits"].append(row)
        result = original_solve(motions, *args, **kwargs)
        row.update(status=result.status, motion_count=result.motions)
        active = None
        return result

    def moving(motions, opts):
        counts["moving_calls"] += 1
        counts["motion_selection_visits"] += len(motions)
        selected = original_moving(motions, opts)
        if active is not None and "selected_indices" not in history["fits"][active]:
            history["fits"][active]["selected_indices"] = [indices[id(m)] for m in selected]
        return selected

    def scan(motions, wheel, base, opts, step_s=0.002):
        candidates = np.arange(-opts.time_offset_bound_s, opts.time_offset_bound_s + step_s, step_s)
        history["fits"][active]["scan_candidates"] = len(candidates)
        return original_scan(motions, wheel, base, opts, step_s)

    def at(wheel, times):
        counts["wheel_at_calls"] += 1
        counts["wheel_samples_queried"] += len(times)
        return original_at(wheel, times)

    def record(motions, params, base, rows):
        counts["residual_requests"] += 1
        history["residuals"].append(
            {
                "fit": active,
                "input_count": len(motions),
                "selected_count": rows.size // 6,
                "params": params.tolist(),
                "base": base.ravel().tolist(),
                "residual_count": rows.size,
                "residual_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            }
        )
        return rows

    def residual(motions, wheel, params, base, opts=None):
        return record(motions, params, base, original_residual(motions, wheel, params, base, opts))

    if hasattr(solver, "_make_residual_evaluator"):
        original_factory = solver._make_residual_evaluator

        def factory(motions, wheel, base, opts):
            evaluate = original_factory(motions, wheel, base, opts)
            return lambda params: record(motions, params, base, evaluate(params))

        solver._make_residual_evaluator = factory

    def lsq(*args, **kwargs):
        result = original_lsq(*args, **kwargs)
        counts["least_squares_calls"] += 1
        row = {
            "fit": active,
            "x": result.x.tolist(),
            "active_mask": result.active_mask.tolist(),
            "fun_shape": list(result.fun.shape),
            "jac_shape": list(result.jac.shape),
        }
        for key in ("x", "active_mask", "fun", "jac"):
            row[key + "_sha256"] = hashlib.sha256(result[key].tobytes()).hexdigest()
        row.update(
            {
                key: result[key]
                for key in ("nfev", "njev", "status", "message", "cost", "optimality")
            }
        )
        history["least_squares"].append(row)
        return result

    solver.solve_lidar_wheel = evaluation.solve_lidar_wheel = solve
    solver.lidar_wheel_residuals = evaluation.lidar_wheel_residuals = residual
    solver._moving, solver.least_squares = moving, lsq
    solver._yaw_rate_offset_scan, solver.WheelOdometry.at = scan, at
    start = time.perf_counter()
    problem_path = protocol_path.parent / "problem.json"
    problem = json.loads(problem_path.read_bytes())
    motions, wheel = inputs(problem)
    indices.update({id(m): i for i, m in enumerate(motions)})
    options = evaluation.LidarWheelRunOptions()
    result = evaluation.evaluate_lidar_wheel(motions, wheel, options)
    artifact = evaluation.build_artifact(
        result,
        options,
        motions=len(motions),
        references=[],
        provenance=LidarWheelProvenance(
            generator="prepared_public_kitti_wheel_proxy",
            generator_version=calibrex.__version__,
            git_commit=git_commit(),
            command=["complete-wheel-control"],
            dataset_family="kitti_oxts_generated_sensor_control",
            sequence_ids=["2011_09_30_drive_0018_sync"],
            wheel_source="OXTS vf and wu proxy; KITTI has no wheel odometry",
            input_sha256=sha(problem_path),
            dataset_license="CC BY-NC-SA 3.0",
        ),
        limitations=[SCOPE],
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
    imports = {
        name: {
            "path": Path(m.__file__).resolve().relative_to(source).as_posix(),
            "sha256": sha(Path(m.__file__)),
        }
        for name, m in sys.modules.copy().items()
        if name == "calibrex" or name.startswith("calibrex.")
    }
    write(
        folder / "measurement.json",
        {
            "schema_version": VERSION,
            "record_type": "measurement",
            "api_seconds": api_seconds,
            "counts": counts,
            "imports": imports,
            "result_sha256": sha(folder / "result.json"),
            "history_sha256": sha(folder / "history.json.gz"),
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


def _same_generator_source(current: bytes, pinned: bytes) -> bool:
    return current.replace(b"\r\n", b"\n") == pinned.replace(b"\r\n", b"\n")


def _same_control_geometry(generated: dict, expected: dict) -> bool:
    """Only offline regeneration permits bounded platform roundoff in motions."""

    import numpy as np

    if {k: v for k, v in generated.items() if k != "motions"} != {
        k: v for k, v in expected.items() if k != "motions"
    } or len(generated["motions"]) != len(expected["motions"]):
        return False
    for left, right in zip(generated["motions"], expected["motions"], strict=True):
        if {k: v for k, v in left.items() if k not in {"velocity_mps", "angular_rate_rps"}} != {
            k: v for k, v in right.items() if k not in {"velocity_mps", "angular_rate_rps"}
        }:
            return False
        for key, tolerance in (("velocity_mps", 2e-7), ("angular_rate_rps", 1e-12)):
            a, b = np.asarray(left[key]), np.asarray(right[key])
            if (
                a.shape != (3,)
                or b.shape != (3,)
                or not np.isfinite(a).all()
                or not np.isfinite(b).all()
                or not np.allclose(a, b, atol=tolerance, rtol=0.0)
            ):
                return False
    return True


def input_lineage(output: Path, problem: dict) -> None:
    import numpy as np

    from calibrex.data.kitti import (
        _packet_from_oxts_values,
        _parse_kitti_time,
        kitti_oxts_pose_world_imu,
    )
    from calibrex.data.kitti_ins_lidar import se3_matrix

    raw = {}
    with tarfile.open(output / "raw-oxts.tar.gz") as bundle:
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
    expected_names = {"timestamps.txt", *[f"data/{i:010d}.txt" for i in range(2762)]}
    if set(provenance["source_files"]) != expected_names or set(raw) != expected_names | {
        "download-provenance.json"
    }:
        raise ValueError("raw input inventory differs")
    for name, digest in provenance["source_files"].items():
        if hashlib.sha256(raw[name]).hexdigest() != digest or members[name]["sha256"] != digest:
            raise ValueError("original OXTS input differs")
    if sha(output / "control-generator.py.txt") != provenance[
        "generator_sha256"
    ] or not _same_generator_source(
        (ROOT / "tools/prepare_lidar_wheel_control.py").read_bytes(),
        (output / "control-generator.py.txt").read_bytes(),
    ):
        raise ValueError("control generator differs")
    for key, value in {
        "reference_pose_indices": [0, 2761],
        "sensor_rotation_xyz_deg": [1.0, -2.0, 3.0],
        "sensor_translation_m": [0.8, -0.3, 0.9],
        "wheel_proxy_columns": {"speed_mps": 8, "yaw_rate_rps": 22},
        "scope": SCOPE,
        "dataset": "KITTI 2011_09_30_drive_0018_sync OXTS",
        "license": "CC BY-NC-SA 3.0",
    }.items():
        if provenance[key] != value:
            raise ValueError("full public-drive geometry or proxy definition differs")
    times = np.asarray(
        [
            _parse_kitti_time(line.strip()) * 1e-9
            for line in raw["timestamps.txt"].decode().splitlines()
            if line.strip()
        ][:2762]
    )
    values = np.array(
        [[float(v) for v in raw[f"data/{i:010d}.txt"].decode().split()] for i in range(2762)]
    )
    packets = [_packet_from_oxts_values(row.tolist()) for row in values]
    body = np.stack([se3_matrix(kitti_oxts_pose_world_imu(p, packets[0])) for p in packets])
    if not _same_control_geometry(
        derive_control(times, body, values), {k: v for k, v in problem.items() if k != "provenance"}
    ):
        raise ValueError("generated control geometry, proxy samples or unchanged defaults differ")


def replay_work(history: dict, current: bool) -> dict:
    """Count selection visits and exact eight-state clock-cache misses from the trace."""

    counts = dict.fromkeys(COUNTERS, 0)
    counts.update(
        solve_calls=len(history["fits"]),
        residual_requests=len(history["residuals"]),
        least_squares_calls=len(history["least_squares"]),
    )
    clocks = {}
    for row in history["fits"]:
        counts["moving_calls"] += 1
        counts["motion_selection_visits"] += len(row["input_indices"])
        counts["wheel_at_calls"] += row["scan_candidates"]
        counts["wheel_samples_queried"] += row["scan_candidates"] * len(row["selected_indices"])
        clocks[row["fit"]] = OrderedDict()
    for row in history["residuals"]:
        fit = row["fit"]
        if not current or fit is None:
            counts["moving_calls"] += 1
            counts["motion_selection_visits"] += row["input_count"]
        interpolate = True
        if current and fit is not None:
            cache, key = clocks[fit], struct.pack("d", row["params"][5])
            interpolate = key not in cache
            cache[key] = None
            cache.move_to_end(key)
            if len(cache) > 8:
                cache.popitem(last=False)
        if interpolate:
            counts["wheel_at_calls"] += 1
            counts["wheel_samples_queried"] += row["selected_count"]
    return counts


def audit(output: Path) -> dict:
    import jsonschema
    import numpy as np

    from calibrex.evaluation.lidar_wheel import LidarWheelRunOptions

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
    options = LidarWheelRunOptions()
    if protocol["options"] != default_options():
        raise ValueError("complete default workload differs")
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
                    or member.name.removeprefix("src/") in contents
                ):
                    raise ValueError("unsafe or duplicate source capsule member")
                stream = bundle.extractfile(member)
                assert stream is not None
                contents[member.name.removeprefix("src/")] = hashlib.sha256(
                    stream.read()
                ).hexdigest()
        if contents != manifests[variant]:
            raise ValueError("source capsule and manifest differ")
    if manifests["prior"].keys() != manifests["current"].keys() or sorted(
        n for n in manifests["prior"] if manifests["prior"][n] != manifests["current"][n]
    ) != [RUNTIME]:
        raise ValueError("unexpected core runtime changes")
    attempts = json.loads((output / "attempts.json").read_bytes())
    jsonschema.validate(attempts, schema)
    if attempts["provenance"] != {"protocol_sha256": sha(output / "protocol.json")}:
        raise ValueError("attempt provenance differs")
    expected = [
        (i, v)
        for i in range(protocol["repetitions"])
        for v in (("prior", "current") if i % 2 == 0 else ("current", "prior"))
    ]
    if [(r["repeat"], r["variant"]) for r in attempts["runs"]] != expected:
        raise ValueError("missing, duplicate or reordered runs")
    motions = problem["motions"]
    blocks = sorted({m["block"] for m in motions})
    held_blocks = {b for i, b in enumerate(blocks) if i % options.holdout_every == 1}
    train = [i for i, m in enumerate(motions) if m["block"] not in held_blocks]
    holdout = [i for i, m in enumerate(motions) if m["block"] in held_blocks]
    train_blocks = sorted(set(blocks) - held_blocks)
    assignment = np.array_split(
        np.asarray(train_blocks), min(options.jackknife_groups, len(train_blocks))
    )
    subsets = [train] + [
        [i for i in train if motions[i]["block"] not in set(group.tolist())] for group in assignment
    ]

    def selected(subset):
        return [
            i
            for i in subset
            if float(np.linalg.norm(motions[i]["velocity_mps"])) >= options.solver.min_speed_mps
        ]

    moving = [selected(subset) for subset in subsets]
    held_moving = selected(holdout)
    scan_count = len(
        np.arange(
            -options.solver.time_offset_bound_s, options.solver.time_offset_bound_s + 0.002, 0.002
        )
    )
    outputs, histories = [], []
    work, api = {v: [] for v in ("prior", "current")}, {v: [] for v in ("prior", "current")}
    for run in attempts["runs"]:
        if run["exit_code"] != 0 or not math.isfinite(run["seconds"]) or run["seconds"] <= 0:
            raise ValueError("failed worker or invalid whole-process timing")
        variant = run["variant"]
        folder = output / f"{run['repeat']:02d}-{variant}"
        measured = json.loads((folder / "measurement.json").read_bytes())
        jsonschema.validate(measured, schema)
        if (
            sha(folder / "result.json") != measured["result_sha256"]
            or sha(folder / "history.json.gz") != measured["history_sha256"]
        ):
            raise ValueError("result or history changed")
        result = json.loads((folder / "result.json").read_bytes())
        history = json.loads(gzip.decompress((folder / "history.json.gz").read_bytes()))
        jsonschema.validate(result, json.loads((output / "result-schema.json").read_bytes()))
        if not histories:
            jsonschema.validate(history, schema)
        p = measured["provenance"]
        verify_git(
            p["git_commit"],
            p["git_commit_short"],
            protocol["commits"][variant],
            result["provenance"].get("git_commit"),
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
            or result["jackknife_fits"] != len(assignment)
            or result["motions"] != len(motions)
            or result["train_motions"] != len(train)
            or result["holdout_motions"] != len(holdout)
        ):
            raise ValueError("completion or full workload differs")
        expected_fits = [
            {
                "fit": i,
                "input_indices": subset,
                "selected_indices": moving[i],
                "status": "converged",
                "motion_count": len(moving[i]),
                "scan_candidates": scan_count,
            }
            for i, subset in enumerate(subsets)
        ]
        if history["fits"] != expected_fits:
            raise ValueError("fit/jackknife inputs or clock scan budget differ")
        outside = []
        for event in history["residuals"]:
            fit = event["fit"]
            if fit is None:
                outside.append(event)
                subset, chosen = (train, moving[0]) if len(outside) == 7 else (holdout, held_moving)
            elif isinstance(fit, bool) or fit not in range(len(subsets)):
                raise ValueError("unknown residual fit")
            else:
                subset = chosen = moving[fit]
            if (
                event["input_count"] != len(subset)
                or event["selected_count"] != len(chosen)
                or event["residual_count"] != 6 * len(chosen)
            ):
                raise ValueError("residual uses different motions or row count")
        if len(outside) != 8:
            raise ValueError("held-out controls or median summaries differ")
        if len(history["least_squares"]) != len(subsets):
            raise ValueError("optimizer fit count differs")
        for i, event in enumerate(history["least_squares"]):
            if (
                event["fit"] != i
                or event["fun_shape"] != [6 * len(moving[i])]
                or event["jac_shape"] != [6 * len(moving[i]), 6]
            ):
                raise ValueError("optimizer residual/Jacobian dimensions differ")
        if measured["counts"] != replay_work(history, variant == "current"):
            raise ValueError("observed work differs from cache replay")
        seconds = measured["api_seconds"]
        if not math.isfinite(seconds) or not 0 < seconds <= run["seconds"]:
            raise ValueError("invalid API timing")
        if (folder / "stderr.txt").read_bytes() or json.loads(
            (folder / "stdout.txt").read_bytes()
        ) != {k: result[k] for k in ("solver_status", "policy_status", "jackknife_fits")}:
            raise ValueError("worker output differs")
        for name, imported in measured["imports"].items():
            if (
                not (name == "calibrex" or name.startswith("calibrex."))
                or manifests[variant].get(imported["path"]) != imported["sha256"]
            ):
                raise ValueError("worker imported unfrozen source")
        if not any(row["path"] == RUNTIME for row in measured["imports"].values()):
            raise ValueError("measured solver import is missing")
        outputs.append(json.dumps(comparable(result), allow_nan=False))
        histories.append(json.dumps(comparable(history), allow_nan=False))
        work[variant].append(measured["counts"])
        api[variant].append(seconds)
    if len(set(outputs)) != 1 or len(set(histories)) != 1:
        raise ValueError("complete results or residual/optimizer histories differ")
    if any(any(r != rows[0] for r in rows) for rows in work.values()):
        raise ValueError("work differs between repeated runs")
    whole = {
        v: statistics.median(r["seconds"] for r in attempts["runs"] if r["variant"] == v)
        for v in api
    }
    medians = {v: statistics.median(seconds) for v, seconds in api.items()}
    report = {
        "schema_version": VERSION,
        "record_type": "audit",
        "verified_runs": len(attempts["runs"]),
        "complete_results_and_histories_exact": True,
        "reference_pose_count": 2762,
        "motion_count": len(motions),
        "block_count": len(blocks),
        "train_motions": len(train),
        "holdout_motions": len(holdout),
        "jackknife_fits": len(assignment),
        "work_per_run": {v: rows[0] for v, rows in work.items()},
        "whole_seconds_median": whole,
        "api_seconds_median": medians,
        "whole_speedup": whole["prior"] / whole["current"],
        "api_speedup": medians["prior"] / medians["current"],
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
    for name in ("problem", "oxts-dir", "output-dir", "audit-only", "worker-protocol"):
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
        not all((args.problem, args.oxts_dir, args.output_dir, args.baseline_commit))
        or args.repetitions < 2
    ):
        parser.error("problem, original OXTS, baseline, fresh output and two repetitions required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for source, target in (
        (args.problem, "problem.json"),
        (ROOT / "tools/prepare_lidar_wheel_control.py", "control-generator.py.txt"),
        (Path(__file__), "launcher.py.txt"),
        (SCHEMA, "schema.json"),
        (ROOT / "tools/benchmark_bspline_lidar_imu.py", "freezer.py.txt"),
        (ROOT / "tools/benchmark_lazy_runtime_loading.py", "identity.py.txt"),
        (ROOT / "schemas/lidar_wheel_odometry.schema.json", "result-schema.json"),
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
    protocol = {
        "schema_version": VERSION,
        "record_type": "protocol",
        "commits": commits,
        "source_roots": roots,
        "repetitions": args.repetitions,
        "fresh_workers": True,
        "blas_threads": 1,
        "host_unreserved": True,
        "options": default_options(),
        "provenance": {
            "files": {n: sha(output / n) for n in sorted(PINNED)},
            "python": sys.version,
            "versions": {
                n: importlib.metadata.version(n)
                for n in ("numpy", "scipy", "pydantic", "PyYAML", "jsonschema")
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
                    sha(output / n) != digest
                    for n, digest in protocol["provenance"]["files"].items()
                )
                or sha(Path(__file__)) != protocol["provenance"]["files"]["launcher.py.txt"]
                or sha(SCHEMA) != protocol["provenance"]["files"]["schema.json"]
                or sha(ROOT / "tools/prepare_lidar_wheel_control.py")
                != protocol["provenance"]["files"]["control-generator.py.txt"]
                or sha(output / "protocol.json") != protocol_sha
            ):
                raise ValueError("pinned sources, inputs or launcher changed during measurement")
    report = audit(output)
    write(output / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
