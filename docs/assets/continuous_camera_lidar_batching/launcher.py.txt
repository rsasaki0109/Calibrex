"""Compare complete artifact-backed timed Camera-LiDAR calibration in fresh workers."""

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

VERSION = "slac.continuous_camera_lidar_reuse_benchmark/v0.1"
SCHEMA = ROOT / "tools/schemas/continuous_camera_lidar_reuse_benchmark.schema.json"
BATCH_VERSION = "slac.continuous_camera_lidar_reuse_benchmark/v0.2"
BATCH_SCHEMA = ROOT / "tools/schemas/continuous_camera_lidar_batch_benchmark.schema.json"
RUNTIME = "calibrex/solvers/continuous_time_camera_lidar_solver.py"
PINNED = {
    "problem.json",
    "input-provenance.json",
    "control-generator.py.txt",
    "source-points.json.gz",
    "motion-control.json.gz",
    "launcher.py.txt",
    "freezer.py.txt",
    "identity.py.txt",
    "schema.json",
    "problem-schema.json",
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

    schema = BATCH_SCHEMA if value["schema_version"] == BATCH_VERSION else SCHEMA
    jsonschema.validate(value, json.loads(schema.read_bytes()))
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def comparable(value: dict) -> dict:
    """Exclude only the two declared run/source provenance fields."""
    value = json.loads(json.dumps(value, allow_nan=False))
    for name in ("created_at", "git_commit"):
        value["provenance"].pop(name)
    return value


def _verify_git_provenance(
    full: str,
    short: str,
    expected: str,
    recorded: str | None,
) -> None:
    if (
        full != expected
        or recorded != short
        or re.fullmatch(r"[a-f0-9]{4,40}", short) is None
        or not full.startswith(short)
    ):
        raise ValueError("worker Git provenance differs from its source pin")


def worker(protocol_path: Path, variant: str, folder: Path) -> None:
    from calibrex.core import capture_time
    from calibrex.core.continuous_trajectory import PiecewiseSE3Trajectory
    from calibrex.core.provenance import git_commit
    from calibrex.evaluation.continuous_time_camera_lidar_run import (
        run_continuous_time_camera_lidar_problem,
    )
    from calibrex.solvers import continuous_time_camera_lidar_solver as solver

    protocol = json.loads(protocol_path.read_bytes())
    batched = protocol["schema_version"] == BATCH_VERSION
    # Check the existing core's real (abbreviated) provenance contract before
    # spending time on calibration; retain both actual Git forms in the receipt.
    _verify_git_provenance(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
        protocol["commits"][variant],
        git_commit(),
    )
    computations = 0
    work = dict.fromkeys(
        (
            "scalar_deskew_points",
            "batched_deskew_points",
            "batched_deskew_calls",
            "scalar_projection_points",
            "batched_projection_points",
            "batched_projection_calls",
        ),
        0,
    )
    original_twist = capture_time._static_point_at_later_body_frame
    original_trajectory = PiecewiseSE3Trajectory.transform_static_point_between_body_frames

    def counted_twist(*args):
        nonlocal computations
        computations += 1
        work["scalar_deskew_points"] += 1
        return original_twist(*args)

    def counted_trajectory(*args, **kwargs):
        nonlocal computations
        computations += 1
        work["scalar_deskew_points"] += 1
        return original_trajectory(*args, **kwargs)

    capture_time._static_point_at_later_body_frame = counted_twist
    if hasattr(solver, "_static_point_at_later_body_frame"):
        solver._static_point_at_later_body_frame = counted_twist
    PiecewiseSE3Trajectory.transform_static_point_between_body_frames = counted_trajectory
    if batched:
        original_prepare = solver._prepare_point_transform

        def counted_prepare(*args):
            transform = original_prepare(*args)

            def counted_point(*point_args):
                work["scalar_projection_points"] += 1
                return transform(*point_args)

            return counted_point

        solver._prepare_point_transform = counted_prepare
        if hasattr(solver, "_constant_twist_points"):
            original_batch = solver._constant_twist_points
            original_project = solver._CaptureBatch.project

            def counted_batch(points, *args):
                nonlocal computations
                computations += len(points)
                work["batched_deskew_points"] += len(points)
                work["batched_deskew_calls"] += 1
                return original_batch(points, *args)

            def counted_project(batch, *args):
                work["batched_projection_points"] += len(batch.selected)
                work["batched_projection_calls"] += 1
                return original_project(batch, *args)

            solver._constant_twist_points = counted_batch
            solver._CaptureBatch.project = counted_project
    start = time.perf_counter()
    result = run_continuous_time_camera_lidar_problem(
        protocol_path.parent / "problem.json",
        result_id="timed-projection-control",
        command=["complete-timed-control"],
    )
    api_seconds = time.perf_counter() - start
    observed_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    observed_short = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    _verify_git_provenance(
        observed_commit,
        observed_short,
        protocol["commits"][variant],
        result.provenance.git_commit,
    )
    result.save(folder / "result.json")
    source = Path(protocol["source_roots"][variant]) / "src"
    imports = {}
    for name, module in sys.modules.copy().items():
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            imports[name] = {"path": path.relative_to(source).as_posix(), "sha256": sha(path)}
    write(
        folder / "measurement.json",
        {
            "schema_version": protocol["schema_version"],
            "record_type": "measurement",
            "api_seconds": api_seconds,
            "deskew_computations": computations,
            **({"work_counts": work} if batched else {}),
            "imports": imports,
            "result_sha256": sha(folder / "result.json"),
            "provenance": {
                "protocol_sha256": sha(protocol_path),
                "git_commit": observed_commit,
                "git_commit_short": observed_short,
                "problem_sha256": sha(protocol_path.parent / "problem.json"),
            },
        },
    )
    print(json.dumps({"status": result.status, "evaluations": len(result.trace)}), flush=True)


def _input_lineage(output: Path, problem: dict) -> None:
    receipt = json.loads((output / "input-provenance.json").read_bytes())
    source = receipt["provenance"]
    if (
        sha(output / "control-generator.py.txt") != source["generator_sha256"]
        or sha(output / "problem.json") != source["problem_sha256"]
    ):
        raise ValueError("input generator or problem differs from its lineage")
    identities = {
        k: source[k]
        for k in ("selected_points_sha256", "public_kinematics_control_sha256", "generator_sha256")
    }
    lineage = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    if lineage != source["lineage_sha256"] or lineage != problem["provenance"]["source_sha256"]:
        raise ValueError("input lineage digest differs")
    points_data = gzip.decompress((output / "source-points.json.gz").read_bytes())
    motion_data = gzip.decompress((output / "motion-control.json.gz").read_bytes())
    if (
        hashlib.sha256(points_data).hexdigest() != source["selected_points_sha256"]
        or hashlib.sha256(motion_data).hexdigest() != source["public_kinematics_control_sha256"]
    ):
        raise ValueError("original public-derived source input differs")
    frames = json.loads(points_data)["frames"]
    motion = json.loads(motion_data)
    if len(frames) != len(problem["captures"]) or receipt["capture_count"] != len(frames):
        raise ValueError("input frame count differs")
    for capture, frame in zip(problem["captures"], frames, strict=True):
        sample = motion["reference"][int(frame["frame_id"])]
        if (
            capture["capture_id"] != frame["frame_id"]
            or capture["twist"]
            != {
                "linear_velocity_body_mps": sample["linear_velocity_body_mps"],
                "angular_velocity_body_radps": sample["angular_velocity_body_radps"],
            }
            or capture["lidar_message_stamp_sec"] != sample["timestamp_sec"]
            or capture["camera_exposure_time_sec"] != sample["timestamp_sec"]
            or len(capture["correspondences"]) != len(frame["correspondences"])
        ):
            raise ValueError("public points, kinematics or timestamps differ")
        for i, (point, original) in enumerate(
            zip(capture["correspondences"], frame["correspondences"], strict=True)
        ):
            if any(
                point[k] != original[k]
                for k in (
                    "correspondence_id",
                    "point_lidar_m",
                    "image_covariance_px2",
                    "outlier_probability",
                    "reliability",
                )
            ) or point["capture_offset_sec"] != -0.1 + 0.1 * i / (
                len(frame["correspondences"]) - 1
            ):
                raise ValueError("selected original points or declared point times differ")
    if receipt["correspondences"] != sum(len(c["correspondences"]) for c in problem["captures"]):
        raise ValueError("input correspondence count differs")


def _replay_clock_work(problem: dict, result: dict, options: dict) -> dict:
    """Derive point work and two-clock cache misses from the unchanged trace."""
    if problem.get("body_trajectory") is not None:
        raise ValueError("ordered-batch counters require the constant-twist control")
    sizes = {
        c["capture_id"]: sum(
            p["reliability"] * (1.0 - p["outlier_probability"]) >= options["minimum_confidence"]
            for p in c["correspondences"]
        )
        for c in problem["captures"]
    }
    train, holdout = result["train_capture_ids"], result["holdout_capture_ids"]
    initial = result["initial_time_offset_sec"]
    probe = options["time_sensitivity_probe_sec"]
    visits = [
        (train, initial),
        (holdout, initial),
        (train, initial - probe),
        (train, initial + probe),
    ]
    visits += [(train, item["time_offset_sec"]) for item in result["trace"][1:]]
    visits += [(holdout, result["estimated_time_offset_sec"])]
    caches = {name: [] for name in sizes}
    output = dict.fromkeys(
        (
            "deskew_points",
            "deskew_batch_calls",
            "projection_points",
            "projection_batch_calls",
        ),
        0,
    )
    import struct

    for captures, offset in visits:
        key = struct.pack("d", offset)
        for capture in captures:
            output["projection_points"] += sizes[capture]
            output["projection_batch_calls"] += 1
            cache = caches[capture]
            if key in cache:
                cache.remove(key)
            else:
                output["deskew_points"] += sizes[capture]
                output["deskew_batch_calls"] += 1
            cache.append(key)
            del cache[:-2]
    return output


def audit(output: Path) -> dict:
    """Verify the complete comparison offline, without calibration or raw downloads."""
    import jsonschema

    protocol = json.loads((output / "protocol.json").read_bytes())
    batched = protocol["schema_version"] == BATCH_VERSION
    if protocol["schema_version"] not in (VERSION, BATCH_VERSION):
        raise ValueError("unknown benchmark version")
    schema = json.loads((output / "schema.json").read_bytes())
    jsonschema.validate(protocol, schema)
    if set(protocol["provenance"]["files"]) != PINNED:
        raise ValueError("incomplete pinned artifact inventory")
    for name, digest in protocol["provenance"]["files"].items():
        if sha(output / name) != digest:
            raise ValueError(f"pinned artifact changed: {name}")
    problem = json.loads((output / "problem.json").read_bytes())
    jsonschema.validate(problem, json.loads((output / "problem-schema.json").read_bytes()))
    _input_lineage(output, problem)
    manifests = {}
    for variant in ("prior", "current"):
        manifests[variant] = json.loads((output / f"sources-{variant}-manifest.json").read_bytes())
        with tarfile.open(output / f"sources-{variant}.tar.gz") as capsule:
            contents = {}
            for member in capsule.getmembers():
                if not member.isfile() or not member.name.startswith("src/calibrex/"):
                    raise ValueError("unexpected source capsule entry")
                stream = capsule.extractfile(member)
                assert stream is not None
                contents[member.name.removeprefix("src/")] = hashlib.sha256(
                    stream.read()
                ).hexdigest()
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
    jsonschema.validate(attempts, schema)
    if attempts["provenance"] != {"protocol_sha256": sha(output / "protocol.json")}:
        raise ValueError("attempt provenance differs")
    expected = [
        (i, v)
        for i in range(protocol["repetitions"])
        for v in (("prior", "current") if i % 2 == 0 else ("current", "prior"))
    ]
    records = attempts["runs"]
    if [(r["repeat"], r["variant"]) for r in records] != expected:
        raise ValueError("missing, duplicate or reordered runs")
    outputs, counts = [], {v: [] for v in ("prior", "current")}
    observed_work = {v: [] for v in ("prior", "current")}
    api = {v: [] for v in ("prior", "current")}
    result_schema = json.loads((output / "result-schema.json").read_bytes())
    for record in records:
        if (
            record["exit_code"] != 0
            or not math.isfinite(record["seconds"])
            or record["seconds"] <= 0
        ):
            raise ValueError("failed run or invalid whole-process timing")
        variant = record["variant"]
        folder = output / f"{record['repeat']:02d}-{variant}"
        measured = json.loads((folder / "measurement.json").read_bytes())
        jsonschema.validate(measured, schema)
        if sha(folder / "result.json") != measured["result_sha256"]:
            raise ValueError("result changed")
        result = json.loads((folder / "result.json").read_bytes())
        jsonschema.validate(result, result_schema)
        provenance = {
            "protocol_sha256": sha(output / "protocol.json"),
            "git_commit": protocol["commits"][variant],
            "git_commit_short": measured["provenance"]["git_commit_short"],
            "problem_sha256": sha(output / "problem.json"),
        }
        _verify_git_provenance(
            measured["provenance"]["git_commit"],
            measured["provenance"]["git_commit_short"],
            protocol["commits"][variant],
            result["provenance"]["git_commit"],
        )
        if (
            measured["provenance"] != provenance
            or result["provenance"]["source_sha256"] != provenance["problem_sha256"]
        ):
            raise ValueError("run provenance differs")
        if (
            result["status"] != "converged"
            or result["options"] != protocol["options"]
            or not result["trace"]
            or result["time_observability_rank"] != 1
            or result["initial_transform_camera_lidar"] != problem["initial_transform_camera_lidar"]
            or result["initial_time_offset_sec"]
            != problem["capture_time_policy"]["sensor_time_offset_sec"]
        ):
            raise ValueError("solver settings, initial state or completion differs")
        if [r["evaluation"] for r in result["trace"]] != list(
            range(1, len(result["trace"]) + 1)
        ) or len(result["trace"]) > protocol["options"]["max_evaluations"]:
            raise ValueError("objective evaluation history differs")
        train, holdout = result["train_capture_ids"], result["holdout_capture_ids"]
        if sorted(train + holdout) != sorted(c["capture_id"] for c in problem["captures"]) or len(
            set(train + holdout)
        ) != len(train + holdout):
            raise ValueError("input captures or disjoint split differs")
        sizes = {
            c["capture_id"]: sum(
                p["reliability"] * (1 - p["outlier_probability"])
                >= protocol["options"]["minimum_confidence"]
                for p in c["correspondences"]
            )
            for c in problem["captures"]
        }
        queries = sum(sizes[k] for k in train) * (len(result["trace"]) + 2) + 2 * sum(
            sizes[k] for k in holdout
        )
        counts[variant].append(measured["deskew_computations"])
        if batched:
            expected_work = _replay_clock_work(problem, result, protocol["options"])
            expected_counts = {
                "scalar_deskew_points": expected_work["deskew_points"] if variant == "prior" else 0,
                "batched_deskew_points": expected_work["deskew_points"]
                if variant == "current"
                else 0,
                "batched_deskew_calls": expected_work["deskew_batch_calls"]
                if variant == "current"
                else 0,
                "scalar_projection_points": queries if variant == "prior" else 0,
                "batched_projection_points": queries if variant == "current" else 0,
                "batched_projection_calls": expected_work["projection_batch_calls"]
                if variant == "current"
                else 0,
            }
            if (
                measured["work_counts"] != expected_counts
                or measured["deskew_computations"] != expected_work["deskew_points"]
                or expected_work["projection_points"] != queries
            ):
                raise ValueError("observed work differs from complete trace and two-clock replay")
            observed_work[variant].append(measured["work_counts"])
        elif variant == "prior" and measured["deskew_computations"] != queries:
            raise ValueError("baseline deskew count differs from unchanged trace workload")
        if (
            json.loads((folder / "stdout.txt").read_bytes())
            != {"status": result["status"], "evaluations": len(result["trace"])}
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
        outputs.append(json.dumps(comparable(result), allow_nan=False))
    if len(set(outputs)) != 1:
        raise ValueError("complete results or objective traces differ")
    if any(len(set(values)) != 1 for values in counts.values()):
        raise ValueError("deskew counts differ between repeated runs")
    if not batched and not 0 < counts["current"][0] < counts["prior"][0]:
        raise ValueError("reuse did not reduce deskew computations")
    medians = {
        v: statistics.median(r["seconds"] for r in records if r["variant"] == v) for v in api
    }
    api_medians = {v: statistics.median(values) for v, values in api.items()}
    report = {
        "schema_version": protocol["schema_version"],
        "record_type": "audit",
        "verified_runs": len(records),
        "complete_results_exact": True,
        "capture_count": len(problem["captures"]),
        "correspondence_count": sum(len(c["correspondences"]) for c in problem["captures"]),
        "objective_evaluations_per_run": len(result["trace"]),
        "projection_requests_per_run": queries,
        "deskew_computations_per_run": {v: values[0] for v, values in counts.items()},
        **(
            {"work_counts_per_run": {v: values[0] for v, values in observed_work.items()}}
            if batched
            else {}
        ),
        "whole_seconds_median": medians,
        "api_seconds_median": api_medians,
        "whole_speedup": medians["prior"] / medians["current"],
        "api_speedup": api_medians["prior"] / api_medians["current"],
        "runtime_source_changes": changed,
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
        "input-provenance",
        "control-generator",
        "source-points",
        "motion-control",
        "output-dir",
        "audit-only",
        "worker-protocol",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--variant", choices=("prior", "current"))
    parser.add_argument("--ordered-batches", action="store_true")
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
                args.input_provenance,
                args.control_generator,
                args.source_points,
                args.motion_control,
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
    version = BATCH_VERSION if args.ordered_batches else VERSION
    schema_path = BATCH_SCHEMA if args.ordered_batches else SCHEMA
    for source, target in (
        (args.problem, "problem.json"),
        (args.input_provenance, "input-provenance.json"),
        (args.control_generator, "control-generator.py.txt"),
        (Path(__file__), "launcher.py.txt"),
        (schema_path, "schema.json"),
        (ROOT / "tools/benchmark_bspline_lidar_imu.py", "freezer.py.txt"),
        (ROOT / "tools/benchmark_lazy_runtime_loading.py", "identity.py.txt"),
        (ROOT / "schemas/continuous_time_camera_lidar_problem.schema.json", "problem-schema.json"),
        (ROOT / "schemas/continuous_time_camera_lidar_result.schema.json", "result-schema.json"),
    ):
        shutil.copyfile(source, output / target)
    for path, name in (
        (args.source_points, "source-points.json.gz"),
        (args.motion_control, "motion-control.json.gz"),
    ):
        (output / name).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    problem = json.loads((output / "problem.json").read_bytes())
    _input_lineage(output, problem)
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
    from calibrex.solvers.continuous_time_camera_lidar_solver import (
        ContinuousTimeCameraLidarOptions,
    )

    protocol = {
        "schema_version": version,
        "record_type": "protocol",
        "commits": commits,
        "source_roots": roots,
        "repetitions": args.repetitions,
        "fresh_workers": True,
        "blas_threads": 1,
        "host_unreserved": True,
        "options": asdict(ContinuousTimeCameraLidarOptions(**problem["options"])),
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
                    "schema_version": version,
                    "record_type": "attempts",
                    "runs": runs,
                    "provenance": {"protocol_sha256": protocol_sha},
                },
            )
            print(json.dumps(runs[-1]), flush=True)
            if done.returncode:
                raise RuntimeError(f"worker failed; preserved {folder}")
            _verify_sources(frozen)
            if any(
                sha(output / name) != digest
                for name, digest in protocol["provenance"]["files"].items()
            ):
                raise ValueError("benchmark pinned artifact changed during measurement")
            if (
                sha(Path(__file__)) != protocol["provenance"]["files"]["launcher.py.txt"]
                or sha(schema_path) != protocol["provenance"]["files"]["schema.json"]
                or sha(output / "protocol.json") != protocol_sha
            ):
                raise ValueError("benchmark launcher or schema changed during measurement")
    report = audit(output)
    write(output / "audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
