"""Measure complete prepared-input GNSS-LiDAR evidence in frozen fresh workers."""

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
from tools.prepare_gnss_lever_arm_control import SCOPE, VERSION, derive_control  # noqa: E402

SCHEMA = ROOT / "tools/schemas/gnss_lever_arm_reuse_benchmark.schema.json"
RUNTIME = "calibrex/solvers/gnss_lever_arm_solver.py"
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


def inputs(problem: dict) -> tuple:
    import numpy as np

    from calibrex.solvers.gnss_lever_arm_solver import GnssTrackModel, OdometryWindow

    track = GnssTrackModel(
        np.asarray(problem["times_s"]), np.asarray(problem["enu_m"]), np.asarray(problem["sigma_m"])
    )
    windows = [
        OdometryWindow(w["window_id"], w["block"], np.asarray(w["times_s"]), np.asarray(w["poses"]))
        for w in problem["windows"]
    ]
    return track, windows


def default_options() -> dict:
    from calibrex.evaluation.gnss_lidar_lever_arm import GnssLidarRunOptions

    return json.loads(json.dumps(asdict(GnssLidarRunOptions())))


def worker(protocol_path: Path, variant: str, folder: Path) -> None:
    import numpy as np

    import calibrex
    from calibrex.core.gnss_lidar_lever_arm import (
        GnssLidarLeverArmProvenance,
        GnssLidarSegmentSummary,
    )
    from calibrex.core.provenance import git_commit
    from calibrex.evaluation import gnss_lidar_lever_arm as evaluation
    from calibrex.solvers import gnss_lever_arm_solver as solver

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
        "excitation": [],
        "provenance": provenance,
    }
    counts = dict.fromkeys(
        (
            "solve_calls",
            "residual_requests",
            "least_squares_calls",
            "lidar_poses_calls",
            "svd_calls",
            "profiled_windows",
        ),
        0,
    )
    active = None
    original_solve = solver.GnssLeverArmSolver.solve
    original_residuals = solver.window_residuals
    original_lsq = solver.least_squares
    original_poses = solver.WindowPairs.lidar_poses
    original_procrustes = solver.weighted_procrustes
    original_excitation = solver.pair_excitation

    def solve(instance, track, windows, *args, **kwargs):
        nonlocal active
        active = counts["solve_calls"]
        counts["solve_calls"] += 1
        row = {"fit": active, "input_window_ids": [w.window_id for w in windows]}
        history["fits"].append(row)
        result = original_solve(instance, track, windows, *args, **kwargs)
        row.update(
            status=result.status,
            usable_window_ids=list(result.window_ids),
            pair_count=result.pair_count,
        )
        active = None
        return result

    def record(pairs, lever, offset, normalize, rows):
        counts["residual_requests"] += 1
        history["residuals"].append(
            {
                "fit": active,
                "window_ids": [p.window_id for p in pairs],
                "pair_count": sum(len(p.first) for p in pairs),
                "lever": lever.tolist(),
                "offset": offset,
                "normalize": normalize,
                "residual_count": rows.size,
                "residual_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
            }
        )
        return rows

    def residuals(track, pairs, lever, offset, opts, *, normalize=True):
        return record(
            pairs,
            lever,
            offset,
            normalize,
            original_residuals(track, pairs, lever, offset, opts, normalize=normalize),
        )

    if hasattr(solver, "_make_residual_evaluator"):
        original_factory = solver._make_residual_evaluator
        original_alignments = solver._alignment_rotations

        def factory(track, pairs, opts):
            evaluate = original_factory(track, pairs, opts)
            return lambda lever, offset: record(pairs, lever, offset, True, evaluate(lever, offset))

        def alignments(covariances):
            counts["svd_calls"] += 1
            counts["profiled_windows"] += len(covariances)
            return original_alignments(covariances)

        solver._make_residual_evaluator = factory
        solver._alignment_rotations = alignments

    def procrustes(*args):
        counts["svd_calls"] += 1
        counts["profiled_windows"] += 1
        return original_procrustes(*args)

    def poses(*args):
        counts["lidar_poses_calls"] += 1
        return original_poses(*args)

    def excitation(pairs, *args):
        history["excitation"].append(
            {
                "window_ids": [p.window_id for p in pairs],
                "pair_count": sum(len(p.first) for p in pairs),
            }
        )
        return original_excitation(pairs, *args)

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

    solver.GnssLeverArmSolver.solve = solve
    solver.window_residuals = residuals
    evaluation.window_residuals = residuals
    solver.least_squares = lsq
    solver.WindowPairs.lidar_poses = poses
    solver.weighted_procrustes = procrustes
    evaluation.pair_excitation = excitation
    start = time.perf_counter()
    problem_path = protocol_path.parent / "problem.json"
    problem = json.loads(problem_path.read_bytes())
    track, windows = inputs(problem)
    options = evaluation.GnssLidarRunOptions()
    result = evaluation.evaluate_gnss_lidar_lever_arm(
        track, windows, options, reference_lever_arm=np.asarray(problem["reference_lever_arm"])
    )
    artifact = evaluation.build_gnss_lidar_artifact(
        result,
        options,
        segments=GnssLidarSegmentSummary(
            scans_read=2762,
            scans_in_gnss_coverage=2762,
            odometry_segments=1,
            unreliable_registrations=0,
            windows=len(windows),
            gnss_epochs_used=len(track.times_s),
            gnss_epochs_rejected=0,
        ),
        provenance=GnssLidarLeverArmProvenance(
            generator="prepared_gnss_control",
            generator_version=calibrex.__version__,
            git_commit=git_commit(),
            command=["complete-gnss-control"],
            dataset_family="kitti_oxts_generated_gnss_odometry_control",
            sequence_ids=["2011_09_30_drive_0018_sync"],
            input_sha256=sha(problem_path),
            input_digest_scope="Complete prepared generated control",
            dataset_license="CC BY-NC-SA 3.0",
        ),
        reference="Declared generated antenna lever arm; evaluation only",
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


def _same_control_geometry(generated: dict, expected: dict) -> bool:
    """Bound platform roundoff during regeneration; measured payloads stay exact."""

    import numpy as np

    def poses_match(left: object, right: object) -> bool:
        a, b = np.asarray(left), np.asarray(right)
        return bool(
            a.shape == b.shape
            and a.shape[-2:] == (4, 4)
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.array_equal(a[..., 3, :], b[..., 3, :])
            and np.allclose(a[..., :3, :3], b[..., :3, :3], rtol=0.0, atol=1e-13)
            and np.allclose(a[..., :3, 3], b[..., :3, 3], rtol=0.0, atol=2e-8)
        )

    exact = {k: v for k, v in generated.items() if k not in {"enu_m", "windows"}}
    if exact != {k: v for k, v in expected.items() if k not in {"enu_m", "windows"}}:
        return False
    a, b = np.asarray(generated["enu_m"]), np.asarray(expected["enu_m"])
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.allclose(a, b, rtol=0.0, atol=2e-8)
    ):
        return False
    if len(generated["windows"]) != len(expected["windows"]):
        return False
    return all(
        {k: v for k, v in a.items() if k != "poses"} == {k: v for k, v in b.items() if k != "poses"}
        and poses_match(a["poses"], b["poses"])
        for a, b in zip(generated["windows"], expected["windows"], strict=True)
    )


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
    if set(raw) != set(provenance["source_files"]) | {"download-provenance.json"}:
        raise ValueError("raw input inventory differs")
    for name, digest in provenance["source_files"].items():
        if hashlib.sha256(raw[name]).hexdigest() != digest or members[name]["sha256"] != digest:
            raise ValueError("original OXTS input differs")
    if (
        sha(output / "control-generator.py.txt") != provenance["generator_sha256"]
        or sha(ROOT / "tools/prepare_gnss_lever_arm_control.py") != provenance["generator_sha256"]
    ):
        raise ValueError("control generator differs")
    if provenance["reference_pose_indices"] != [0, 2761]:
        raise ValueError("full public-drive selection differs")
    times = np.asarray(
        [
            _parse_kitti_time(line.strip()) * 1.0e-9
            for line in raw["timestamps.txt"].decode().splitlines()
            if line.strip()
        ][:2762]
    )
    packets = [
        _packet_from_oxts_values([float(v) for v in raw[f"data/{i:010d}.txt"].decode().split()])
        for i in range(2762)
    ]
    body = np.stack([se3_matrix(kitti_oxts_pose_world_imu(p, packets[0])) for p in packets])
    generated = derive_control(times, body)
    if (
        not _same_control_geometry(
            generated, {k: v for k, v in problem.items() if k != "provenance"}
        )
        or provenance["scope"] != SCOPE
        or any(
            provenance[k] != v
            for k, v in {
                "noise_seed": 37,
                "clock_offset_s": 0.012,
                "gnss_noise_sigma_m": 0.02,
                "odometry_walk_sigma_m": 0.002,
            }.items()
        )
    ):
        raise ValueError("generated control geometry, noise or unchanged defaults differ")


def replay_work(history: dict, current: bool) -> dict:
    caches, clocks = {}, {}
    counts = dict.fromkeys(
        (
            "solve_calls",
            "residual_requests",
            "least_squares_calls",
            "lidar_poses_calls",
            "svd_calls",
            "profiled_windows",
        ),
        0,
    )
    counts.update(
        solve_calls=len(history["fits"]),
        residual_requests=len(history["residuals"]),
        least_squares_calls=len(history["least_squares"]),
    )
    for event in history["residuals"]:
        fit, size = event["fit"], len(event["window_ids"])
        if event["residual_count"] != 3 * event["pair_count"] or not all(
            math.isfinite(x) for x in [*event["lever"], event["offset"]]
        ):
            raise ValueError("invalid residual dimensions or parameters")
        if fit is None or not current:
            counts["lidar_poses_calls"] += size
            counts["svd_calls"] += size
            counts["profiled_windows"] += size
            continue
        if (
            isinstance(fit, bool)
            or fit not in range(len(history["fits"]))
            or not event["normalize"]
        ):
            raise ValueError("invalid cached fit")
        cache, clock = caches.setdefault(fit, OrderedDict()), clocks.setdefault(fit, OrderedDict())
        key = struct.pack("3d", *event["lever"]), struct.pack("d", event["offset"])
        if key in cache:
            cache.move_to_end(key)
            continue
        cache[key] = None
        if len(cache) > 256:
            cache.popitem(last=False)
        counts["svd_calls"] += 1
        counts["profiled_windows"] += size
        if key[1] in clock:
            clock.move_to_end(key[1])
        else:
            counts["lidar_poses_calls"] += size
            clock[key[1]] = None
            if len(clock) > 64:
                clock.popitem(last=False)
    counts["lidar_poses_calls"] += sum(len(e["window_ids"]) for e in history["excitation"])
    return counts


def audit(output: Path) -> dict:
    import jsonschema
    import numpy as np

    from calibrex.evaluation.gnss_lidar_lever_arm import GnssLidarRunOptions
    from calibrex.solvers.gnss_lever_arm_solver import GnssTrackModel, build_pairs

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
    options = GnssLidarRunOptions()
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
                    or member.name in contents
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
    track, windows = inputs(problem)
    train = [w for i, w in enumerate(windows) if i % options.holdout_every != 1]
    holdout = [w for i, w in enumerate(windows) if i % options.holdout_every == 1]
    assignment = np.array_split(np.arange(len(train)), min(options.jackknife_groups, len(train)))
    subsets = [train] + [
        [w for i, w in enumerate(train) if i not in set(members.tolist())] for members in assignment
    ]
    epoch = float(track.times_s[0])
    rebased = GnssTrackModel(track.times_s - epoch, track.enu_m, track.sigma_m, track.max_gap_s)
    fit_pairs = [build_pairs(rebased, subset, options.solver, epoch=epoch) for subset in subsets]
    held_pairs = build_pairs(rebased, holdout, options.solver, epoch=epoch)

    def ids(pairs):
        return [p.window_id for p in pairs]

    def pair_count(pairs):
        return sum(len(p.first) for p in pairs)

    expected_excitation = [
        {"window_ids": ids(p), "pair_count": pair_count(p)} for p in (fit_pairs[0], held_pairs)
    ]
    work, api = {v: [] for v in ("prior", "current")}, {v: [] for v in ("prior", "current")}
    outputs, histories = [], []
    for run in attempts["runs"]:
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
        jsonschema.validate(result, json.loads((output / "result-schema.json").read_bytes()))
        # Independent run provenance is checked below; every complete payload
        # must equal the first schema-validated history after removing Git IDs.
        if not histories:
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
            or result["jackknife_fits"] != len(assignment)
            or result["train_windows"] != len(train)
            or result["holdout_windows"] != len(holdout)
            or result["segments"]
            != {
                "scans_read": 2762,
                "scans_in_gnss_coverage": 2762,
                "odometry_segments": 1,
                "unreliable_registrations": 0,
                "windows": len(windows),
                "gnss_epochs_used": 2762,
                "gnss_epochs_rejected": 0,
            }
        ):
            raise ValueError("completion or full workload differs")
        if len(history["fits"]) != len(subsets) or any(
            row
            != {
                "fit": i,
                "input_window_ids": ids(subsets[i]),
                "status": "converged",
                "usable_window_ids": ids(fit_pairs[i]),
                "pair_count": pair_count(fit_pairs[i]),
            }
            for i, row in enumerate(history["fits"])
        ):
            raise ValueError("fit/jackknife inputs differ")
        if history["excitation"] != expected_excitation:
            raise ValueError("excitation inputs differ")
        outside = []
        for event in history["residuals"]:
            fit = event["fit"]
            if fit is None:
                outside.append(event)
                pairs = fit_pairs[0] if event["window_ids"] == ids(fit_pairs[0]) else held_pairs
            elif isinstance(fit, bool) or fit not in range(len(subsets)):
                raise ValueError("unknown residual fit")
            else:
                pairs = fit_pairs[fit]
            if event["window_ids"] != ids(pairs) or event["pair_count"] != pair_count(pairs):
                raise ValueError("residual uses different windows/pairs")
        if len(outside) != 10 or sum(not e["normalize"] for e in outside) != 2:
            raise ValueError("held-out controls or metre summaries differ")
        for i, pairs in enumerate(fit_pairs):
            calls = [e for e in history["least_squares"] if e["fit"] == i]
            if len(calls) != options.solver.irls_iterations:
                raise ValueError("IRLS budget differs")
            for event in calls:
                size = 3 * pair_count(pairs) + 4
                if event["fun_shape"] != [size] or event["jac_shape"] != [size, 4]:
                    raise ValueError("optimizer residual/Jacobian dimensions differ")
        if len(history["least_squares"]) != len(subsets) * options.solver.irls_iterations:
            raise ValueError("optimizer fit count differs")
        if measurement["counts"] != replay_work(history, variant == "current"):
            raise ValueError("observed work differs from cache replay")
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
        if not any(row["path"] == RUNTIME for row in measurement["imports"].values()):
            raise ValueError("measured solver import is missing")
        outputs.append(json.dumps(comparable(result), allow_nan=False))
        histories.append(json.dumps(comparable(history), allow_nan=False))
        work[variant].append(measurement["counts"])
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
        "window_count": len(windows),
        "train_windows": len(train),
        "holdout_windows": len(holdout),
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
        (ROOT / "tools/prepare_gnss_lever_arm_control.py", "control-generator.py.txt"),
        (Path(__file__), "launcher.py.txt"),
        (SCHEMA, "schema.json"),
        (ROOT / "tools/benchmark_bspline_lidar_imu.py", "freezer.py.txt"),
        (ROOT / "tools/benchmark_lazy_runtime_loading.py", "identity.py.txt"),
        (ROOT / "schemas/gnss_lidar_lever_arm.schema.json", "result-schema.json"),
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
                or sha(ROOT / "tools/prepare_gnss_lever_arm_control.py")
                != protocol["provenance"]["files"]["control-generator.py.txt"]
                or sha(output / "protocol.json") != protocol_sha
            ):
                raise ValueError("pinned sources, inputs or launcher changed during measurement")
    report = audit(output)
    write(output / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
