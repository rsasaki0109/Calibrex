"""Compare complete, frozen public TUM RGB-D pipelines in alternating processes."""

from __future__ import annotations

import argparse
import gzip
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_bspline_lidar_imu import _freeze, _verify_sources  # noqa: E402
from tools.benchmark_lazy_runtime_loading import _git_identity  # noqa: E402
from tools.benchmark_radar_doppler_batches import sources  # noqa: E402

VERSION = "slac.tum_window_processes_benchmark/v0.1"
SCHEMA = ROOT / "tools/schemas/tum_window_processes_benchmark.schema.json"
CONFIG = ROOT / "examples/public_datasets/tum_rgbd_freiburg1_xyz/native_joint_slac_config.yaml"
DATA = ROOT / "data/public/rgbd_dataset_freiburg1_xyz"
INPUTS_SHA256 = "f1ec8837db3c70ddb173869a4362dd54a5a9da3dc8e6f9cc7ac59a7b14e2a108"
CHANGES = {
    "calibrex/solvers/_tum_window_transport.py",
    "calibrex/solvers/_tum_window_processes.py",
    "calibrex/solvers/_tum_window_worker.py",
    "calibrex/solvers/native_tum_joint_slac_solver.py",
}
ADDED = {
    "calibrex/solvers/_tum_window_transport.py",
    "calibrex/solvers/_tum_window_processes.py",
    "calibrex/solvers/_tum_window_worker.py",
}
SCOPE = (
    "Complete native_tum_joint_slac pipeline on the public development TUM fr1/xyz sequence: "
    "unchanged primary and replication windows, scalar and full-XYZ lattice ablations, "
    "two-sided pairs, reassociation, curvature, multistart, holdout, probes and saved reports. "
    "Three bounded native window processes preserve every depth pixel, sampled point, "
    "original evaluation population and per-window evaluation order. Whole-worker time "
    "includes child startup, imports, native factor transport, reconstruction and aggregation; "
    "optimizer, diagnostic and frontend totals are summed process time and may overlap; "
    "no accuracy or SOTA claim."
)
STAGES = (
    "_xyz_pair_reassociation_runs",
    "_iterative_reassociation_runs",
    "_rematch_spatial_holdouts",
)
FRONTEND = ("read_tum_depth_png", "sample_tum_depth_points")
PINNED = {
    "config.yaml",
    "upstream-manifest.yaml",
    "launcher.py.txt",
    "freezer.py.txt",
    "identity.py.txt",
    "capsule-auditor.py.txt",
    "schema.json",
    "result-schema.json",
    "sources-prior-manifest.json",
    "sources-current-manifest.json",
    "sources-prior.tar.gz",
    "sources-current.tar.gz",
    "inputs.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def record(kind: str, payload: dict, provenance: dict) -> dict:
    return {
        "schema_version": VERSION,
        "record_type": kind,
        "payload": payload,
        "provenance": provenance,
    }


def comparable(value: dict) -> dict:
    value = json.loads(json.dumps(value, allow_nan=False))
    for key in ("id", "created_at", "git_commit"):
        value["run"].pop(key)
    for key in ("generated_at", "git_commit"):
        value["run"]["provenance"].pop(key)
    if value["solid_state"] is not None:
        value["solid_state"]["provenance"].pop("git_commit")
    provenance = value["run"]["provenance"]
    provenance.pop("native_tum_joint_slac_window_execution", None)
    provenance["native_tum_joint_slac_options"].pop("window_workers", None)
    return value


def worker(
    protocol_path: Path, variant: str, folder: Path,
    child_request: Path | None = None, child_response: Path | None = None,
) -> None:
    import numpy as np

    from calibrex.core.provenance import git_commit
    from calibrex.graph import joint_optimization as graph
    from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration
    from calibrex.solvers import native_tum_joint_slac_solver as adapter

    protocol = json.loads(protocol_path.read_bytes())
    source = protocol_path.parent / ("sources-" + variant)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    short = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    if commit != protocol["payload"]["commits"][variant] or git_commit() != short:
        raise ValueError("worker Git identity differs")
    provenance = {
        "protocol_sha256": sha(protocol_path),
        "git_commit": commit,
        "git_commit_short": short,
    }
    import yaml

    from calibrex.data.tum_rgbd import (
        associate_depth_groundtruth,
        read_groundtruth,
        read_image_index,
    )
    config = yaml.safe_load((protocol_path.parent / "config.yaml").read_bytes())
    values = config["pipeline"]["factors"][adapter._FACTOR_NAME]["options"]
    paired = associate_depth_groundtruth(
        read_image_index(DATA / "depth.txt"), read_groundtruth(DATA / "groundtruth.txt"),
        max_difference_sec=values["max_pose_time_delta_sec"],
    )
    frame_windows, path_windows = {}, {}
    for start in values["replication_start_indices"]:
        selected = [paired[start + i * values["frame_stride"]]
                    for i in range(values["map_frame_count"] + values["query_frame_count"])]
        frame_windows[adapter._pose_block(0, selected[values["map_frame_count"]])] = start
        for frame in selected:
            if frame.depth.path in path_windows:
                raise ValueError("benchmark requires disjoint frontend window paths")
            path_windows[frame.depth.path] = start

    def window(parameters):
        matches = [start for name, start in frame_windows.items() if name in parameters]
        if len(matches) != 1:
            raise ValueError("numeric event has no unique window identity")
        return matches[0]

    frontend_window = None
    events, solves, solve_seconds = [], [], []
    diagnostic_events, stage_seconds, factory_calls = [], {}, {}
    frontend_events, frontend_seconds = [], dict.fromkeys(FRONTEND, 0.0)

    for name in FRONTEND:
        original = getattr(adapter, name)

        def frontend(*args, _name=name, _original=original, **kwargs):
            nonlocal frontend_window
            if _name == "read_tum_depth_png":
                frontend_window = path_windows[Path(args[0]).relative_to(DATA).as_posix()]
            begin = time.perf_counter()
            result = _original(*args, **kwargs)
            frontend_seconds[_name] += time.perf_counter() - begin
            image = result if _name == "read_tum_depth_png" else args[0]
            item = {
                "kind": _name,
                "window": frontend_window,
                "width": image.width,
                "height": image.height,
                "pixel_count": len(image.raw_values),
                "pixels_sha256": hashlib.sha256(
                    np.asarray(image.raw_values, dtype="<u2").tobytes()
                ).hexdigest(),
            }
            if _name == "read_tum_depth_png":
                item["path"] = Path(args[0]).relative_to(DATA).as_posix()
            else:
                item["point_count"] = len(result)
                item["points_sha256"] = hashlib.sha256(
                    np.asarray(result, dtype="<f8").reshape(-1, 3).tobytes()
                ).hexdigest()
            frontend_events.append(item)
            return result

        setattr(adapter, name, frontend)

    for name in ("_spatial_factor", "make_joint_trilinear_xyz_pair_factor"):
        original = getattr(adapter, name)
        factory_calls[name] = 0

        def factory(*args, _name=name, _original=original, **kwargs):
            factory_calls[_name] += 1
            return _original(*args, **kwargs)

        setattr(adapter, name, factory)

    for name in STAGES:
        original = getattr(adapter, name)

        def stage(*args, _name=name, _original=original, **kwargs):
            begin = time.perf_counter()
            result = _original(*args, **kwargs)
            stage_seconds[_name] = stage_seconds.get(_name, 0.0) + time.perf_counter() - begin
            return result

        setattr(adapter, name, stage)

    for name in ("_xyz_pair_reassociation_state", "_reassociated_spatial_state"):
        original = getattr(adapter, name)

        def state(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            values = args[1] if _name == "_xyz_pair_reassociation_state" else args[3]
            diagnostic_events.append(
                {
                    "kind": _name,
                    "window": window(values),
                    "values_sha256": digest(values),
                    "state_sha256": digest(result.as_dict()),
                    "factor_count": len(result.factors),
                }
            )
            return result

        setattr(adapter, name, state)

    original_objective = adapter._shared_holdout_objective

    def objective(*args, **kwargs):
        result = original_objective(*args, **kwargs)
        diagnostic_events.append(
            {
                "kind": "_shared_holdout_objective",
                "window": args[0].start_index,
                "measurement_count": len(args[1]),
                "candidate_sha256": digest(args[4]),
                "rematch": kwargs["rematch"],
                "value_hex": result.hex(),
            }
        )
        return result

    adapter._shared_holdout_objective = objective
    original_linearize = graph._linearize
    original_residuals = graph._factor_residuals
    original_solve = graph.BackendNeutralJointOptimizer.solve
    plan_requests, plan_accepts, plan_builds = 0, 0, 0
    rgbd = sys.modules.get("calibrex.graph._rgbd_batches")
    if rgbd is not None:
        original_prepare = rgbd._prepare_rgbd_plan
        original_build = getattr(rgbd, "_build_rgbd_plan", None)
        if original_build is not None:

            def build(*args):
                nonlocal plan_builds
                plan = original_build(*args)
                plan_builds += plan is not None
                return plan

            rgbd._build_rgbd_plan = build

        def prepare(*args):
            nonlocal plan_requests, plan_accepts, plan_builds
            plan_requests += 1
            plan = original_prepare(*args)
            plan_accepts += plan is not None
            if original_build is None:
                plan_builds += plan is not None
            return plan

        rgbd._prepare_rgbd_plan = prepare

    def array(value):
        return {
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest(),
        }

    def event(kind, factors, values, residual):
        return {
            "kind": kind,
            "window": window(values),
            "factor_count": len(factors),
            "values_sha256": digest(values),
            "factor_ids_sha256": digest([f.factor_id for f in factors]),
            "residual": array(residual),
        }

    def linearize(factors, values, blocks, layout, options):
        residual, jacobian = original_linearize(factors, values, blocks, layout, options)
        item = event("linearize", factors, values, residual)
        item["jacobian"] = array(jacobian)
        events.append(item)
        return residual, jacobian

    def residuals(factors, values):
        result = original_residuals(factors, values)
        events.append(event("residuals", factors, values, result))
        return result

    def solve(self, blocks, factors, options):
        begin = time.perf_counter()
        result = original_solve(self, blocks, factors, options)
        solve_seconds.append(time.perf_counter() - begin)
        solves.append(
            {
                "window": window({block.name for block in blocks}),
                "factor_count": len(factors),
                "factor_ids_sha256": digest([f.factor_id for f in factors]),
                "parameter_dimension": sum(block.dimension for block in blocks),
                "active_dimension": sum(block.dimension for block in blocks if not block.fixed),
                "iterations": len(result.history),
                "probes": len(result.probes),
                "status": result.status,
                "result_sha256": digest(result.as_dict()),
            }
        )
        write(
            folder / "progress.json",
            record("progress", {"solves": solves, "solve_seconds": solve_seconds}, provenance),
        )
        return result

    graph._linearize, graph._factor_residuals = linearize, residuals
    graph.BackendNeutralJointOptimizer.solve = solve
    if child_request is None and variant == "current":
        from calibrex.solvers import _tum_window_processes as processes

        def command(request, response):
            destination = folder / "children" / request.stem.removesuffix("-request")
            destination.mkdir(parents=True)
            return [
                sys.executable, str(Path(__file__).resolve()),
                "--worker-protocol", str(protocol_path), "--variant", variant,
                "--output", str(destination),
                "--window-request", str(request), "--window-response", str(response),
            ]

        processes._worker_command = command
    begin = time.perf_counter()
    if child_request is None:
        result = run_calibration(protocol_path.parent / "config.yaml", CalibrationRunOptions())
        seconds = time.perf_counter() - begin
        assert result is not None
        value = result.model_dump(mode="json")
        (folder / "result.json.gz").write_bytes(
            gzip.compress((json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), mtime=0)
        )
    else:
        from calibrex.solvers._tum_window_processes import _execute_request
        _execute_request(child_request, child_response)
        seconds = time.perf_counter() - begin
    imports = {}
    for name, module in sys.modules.copy().items():
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            imports[name] = {
                "path": path.relative_to(source / "src").as_posix(), "sha256": sha(path),
            }
    raw_history = record(
        "history", {"events": events, "solves": solves,
                    "diagnostics": diagnostic_events, "frontend": frontend_events}, provenance,
    )
    history_name = "history.json.gz" if child_request else "parent-history.json.gz"
    (folder / history_name).write_bytes(gzip.compress(
        (json.dumps(raw_history, indent=2, allow_nan=False) + "\n").encode(), mtime=0,
    ))
    info = {
        "api_seconds": seconds, "optimizer_seconds": sum(solve_seconds), "imports": imports,
        "history_sha256": sha(folder / history_name),
        "solve_count": len(solves),
        "jacobian_calls": sum(e["kind"] == "linearize" for e in events),
        "residual_calls": sum(e["kind"] == "residuals" for e in events),
        "rgbd_plan_requests": plan_requests, "rgbd_plan_accepts": plan_accepts,
        "rgbd_plan_builds": plan_builds, "stage_seconds": stage_seconds,
        "frontend_seconds": frontend_seconds, "factory_calls": factory_calls,
    }
    process_info = record("process", {
        **info, "pid": os.getpid(), "parent_pid": os.getppid(),
        "role": "child" if child_request else "parent",
        "request_sha256": sha(child_request) if child_request else None,
        "response_sha256": sha(child_response) if child_response else None,
    }, provenance)
    write(folder / "process.json", process_info)
    if child_request is not None:
        return
    process_paths = [
        *sorted((folder / "children").glob("*/process.json"), key=lambda p: int(p.parent.name)),
        folder / "process.json",
    ]
    process_records = [json.loads(path.read_bytes()) for path in process_paths]
    raw_histories = [json.loads(gzip.decompress((path.parent / (
        "parent-history.json.gz" if path.parent == folder else "history.json.gz"
    )).read_bytes())) for path in process_paths]
    history_payload = {
        key: sorted((event for raw in raw_histories for event in raw["payload"][key]),
                    key=lambda event: event["window"])
        for key in ("events", "solves", "diagnostics", "frontend")
    }
    combined_seconds = []
    for path, raw in zip(process_paths, raw_histories, strict=True):
        progress = json.loads((path.parent / "progress.json").read_bytes())
        combined_seconds.extend(zip(raw["payload"]["solves"], progress["payload"]["solve_seconds"],
                                    strict=True))
    combined_seconds.sort(key=lambda item: item[0]["window"])
    shutil.copyfile(folder / "progress.json", folder / "parent-progress.json")
    write(folder / "progress.json", record("progress", {
        "solves": history_payload["solves"], "solve_seconds": [s for _, s in combined_seconds],
    }, provenance))
    histories = record("history", history_payload, provenance)
    (folder / "history.json.gz").write_bytes(gzip.compress(
        (json.dumps(histories, indent=2, allow_nan=False) + "\n").encode(), mtime=0,
    ))
    for key in ("optimizer_seconds", "solve_count", "jacobian_calls", "residual_calls",
                "rgbd_plan_requests", "rgbd_plan_accepts", "rgbd_plan_builds"):
        info[key] = sum(record["payload"][key] for record in process_records)
    for key in ("stage_seconds", "frontend_seconds", "factory_calls"):
        names = set().union(*(record["payload"][key] for record in process_records))
        info[key] = {name: sum(record["payload"][key].get(name, 0) for record in process_records)
                     for name in names}
    info.update(
        result_sha256=sha(folder / "result.json.gz"),
        history_sha256=sha(folder / "history.json.gz"),
        processes=[{"path":path.relative_to(folder).as_posix(), "sha256":sha(path)}
                   for path in process_paths],
    )
    write(folder / "measurement.json", record("measurement", info, provenance))


def audit(output: Path) -> dict:
    """Verify complete numeric histories, results, source lineage and work scope."""
    import jsonschema
    import yaml

    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    schema = json.loads((output / "schema.json").read_bytes())
    if schema != json.loads(SCHEMA.read_bytes()):
        raise ValueError("archive benchmark schema differs")
    validate = jsonschema.Draft202012Validator(schema).validate
    validate(protocol)
    payload = protocol["payload"]
    pins = protocol["provenance"]["files"]
    if set(pins) != PINNED or any(sha(output / name) != value for name, value in pins.items()):
        raise ValueError("pinned config, tools, inputs or source capsule changed")
    if payload["scope"] != SCOPE or payload["blas_threads"] != 1:
        raise ValueError("benchmark scope differs")
    config = yaml.safe_load((output / "config.yaml").read_bytes())
    reference = yaml.safe_load(CONFIG.read_bytes())
    reference["dataset"]["path"] = config["dataset"]["path"]
    reference["project"]["output_dir"] = config["project"]["output_dir"]
    reference["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"].update(window_workers=3)
    if config != reference:
        raise ValueError("complete pipeline options differ")
    inputs = json.loads((output / "inputs.json").read_bytes())
    if (
        len(inputs) != 802
        or len({p["name"] for p in inputs}) != len(inputs)
        or sha(output / "inputs.json") != INPUTS_SHA256
    ):
        raise ValueError("public depth input inventory differs")
    if payload["inputs_sha256"] != sha(output / "inputs.json"):
        raise ValueError("input fingerprint differs")
    manifests = {v: sources(output, v) for v in ("prior", "current")}
    prior, current = manifests["prior"], manifests["current"]
    changed = {k for k in prior.keys() | current.keys() if prior.get(k) != current.get(k)}
    if changed != CHANGES or set(current) != set(prior) | ADDED:
        raise ValueError("unexpected runtime source changes")
    index_path = output / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_bytes())
        actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        if set(index["files"]) != actual - {"index.json"} or index["provenance"] != {
            "protocol_sha256": sha(protocol_path)
        }:
            raise ValueError("published file inventory differs")
        for name, value in index["files"].items():
            path = (output / name).resolve()
            if not path.is_relative_to(output.resolve()) or value != {
                "sha256": sha(path),
                "size_bytes": path.stat().st_size,
            }:
                raise ValueError("published bytes differ")
    attempts = json.loads((output / "attempts.json").read_bytes())
    validate(attempts)
    if attempts["provenance"] != {"protocol_sha256": sha(protocol_path)}:
        raise ValueError("attempt provenance differs")
    attempts = attempts["payload"]["attempts"]
    order = [
        v
        for i in range(payload["repetitions"])
        for v in (("current", "prior") if i % 2 == 0 else ("prior", "current"))
    ]
    if payload["repetitions"] < 2 or [a["variant"] for a in attempts] != order:
        raise ValueError("alternating run inventory differs")
    result_schema = json.loads((output / "result-schema.json").read_bytes())
    if result_schema != json.loads((ROOT / "schemas/result.schema.json").read_bytes()):
        raise ValueError("calibration result schema differs")
    timings = {
        kind: {v: [] for v in manifests}
        for kind in ("whole", "api", "optimizer", *STAGES, *FRONTEND)
    }
    results, histories = [], []
    plan_counts = {}
    factories, diagnostic_counts, frontend_counts = {}, None, None
    for index, attempt in enumerate(attempts):
        variant = attempt["variant"]
        if attempt["folder"] != f"{index // 2 + 1:02d}-{variant}" or attempt["exit_code"] != 0:
            raise ValueError("worker identity or exit status differs")
        folder = output / attempt["folder"]
        measurement = json.loads((folder / "measurement.json").read_bytes())
        validate(measurement)
        info, lineage = measurement["payload"], measurement["provenance"]
        if (
            lineage["protocol_sha256"] != sha(protocol_path)
            or lineage["git_commit"] != payload["commits"][variant]
            or not lineage["git_commit"].startswith(lineage["git_commit_short"])
        ):
            raise ValueError("actual Git identity or protocol binding differs")
        if info["result_sha256"] != sha(folder / "result.json.gz") or info["history_sha256"] != sha(
            folder / "history.json.gz"
        ):
            raise ValueError("result or history bytes changed")
        process_paths = [folder / entry["path"] for entry in info["processes"]]
        expected = ["children/60/process.json", "children/300/process.json", "process.json"]
        if [entry["path"] for entry in info["processes"]] != (
            expected if variant == "current" else ["process.json"]
        ):
            raise ValueError("complete process inventory differs")
        raw_processes, raw_histories = [], []
        for path, entry in zip(process_paths, info["processes"], strict=True):
            if not path.resolve().is_relative_to(folder.resolve()) or sha(path) != entry["sha256"]:
                raise ValueError("process bytes differ")
            process = json.loads(path.read_bytes())
            validate(process)
            raw = process["payload"]
            if process["provenance"] != lineage or raw["role"] != (
                "parent" if path.parent == folder else "child"
            ):
                raise ValueError("process lineage differs")
            for imported in raw["imports"].values():
                if manifests[variant].get(imported["path"]) != imported["sha256"]:
                    raise ValueError("child imported an unfrozen source module")
            if "calibrex.graph.joint_factors" not in raw["imports"]:
                raise ValueError("process graph imports missing")
            history_path = path.parent / (
                "parent-history.json.gz" if path.parent == folder else "history.json.gz"
            )
            if sha(history_path) != raw["history_sha256"]:
                raise ValueError("raw process history bytes differ")
            raw_history = json.loads(gzip.decompress(history_path.read_bytes()))
            validate(raw_history)
            if raw_history["provenance"] != lineage:
                raise ValueError("raw process history lineage differs")
            h = raw_history["payload"]
            if (raw["solve_count"] != len(h["solves"])
                or raw["rgbd_plan_requests"] != len(h["events"])
                or raw["jacobian_calls"] != sum(e["kind"] == "linearize" for e in h["events"])
                or raw["residual_calls"] != sum(e["kind"] == "residuals" for e in h["events"])
                or not 0 < raw["optimizer_seconds"] <= raw["api_seconds"]
                or raw["rgbd_plan_accepts"] != raw["rgbd_plan_requests"]):
                raise ValueError("raw process work differs")
            if any(not 0 < s <= raw["api_seconds"] for s in raw["stage_seconds"].values()):
                raise ValueError("raw process diagnostic time differs")
            if any(not 0 < s <= raw["api_seconds"] for s in raw["frontend_seconds"].values()):
                raise ValueError("raw process frontend time differs")
            if raw["role"] == "child":
                start = int(path.parent.name)
                if any(e["window"] != start for events in h.values() for e in events):
                    raise ValueError("child window identity differs")
                if not raw["request_sha256"] or not raw["response_sha256"]:
                    raise ValueError("native transport fingerprints missing")
            progress = json.loads((path.parent / (
                "parent-progress.json" if raw["role"] == "parent" else "progress.json"
            )).read_bytes())
            validate(progress)
            if (progress["provenance"] != lineage
                or progress["payload"]["solves"] != h["solves"]
                or sum(progress["payload"]["solve_seconds"]) != raw["optimizer_seconds"]):
                raise ValueError("raw process solve timing differs")
            raw_processes.append(raw)
            raw_histories.append(h)
        parent = raw_processes[-1]
        if parent["pid"] in {p["pid"] for p in raw_processes[:-1]} or any(
            p["parent_pid"] != parent["pid"] for p in raw_processes[:-1]
        ):
            raise ValueError("child process ownership differs")
        for key in ("optimizer_seconds", "solve_count", "jacobian_calls", "residual_calls",
                    "rgbd_plan_requests", "rgbd_plan_accepts", "rgbd_plan_builds"):
            if info[key] != sum(p[key] for p in raw_processes):
                raise ValueError("aggregate process work or timing differs")
        for key in ("stage_seconds", "frontend_seconds", "factory_calls"):
            names = set().union(*(p[key] for p in raw_processes))
            if info[key] != {n:sum(p[key].get(n, 0) for p in raw_processes) for n in names}:
                raise ValueError("aggregate process stage work or timing differs")
        if info["api_seconds"] != parent["api_seconds"] or info["imports"] != parent["imports"]:
            raise ValueError("parent API timing or imports differ")
        result = json.loads(gzip.decompress((folder / "result.json.gz").read_bytes()))
        jsonschema.validate(result, result_schema)
        p = result["run"]["provenance"]
        execution = p.get("native_tum_joint_slac_window_execution")
        if variant == "current" and execution != {
            "method":"native_tum_window_execution/v0.1", "backend":"process",
            "requested_workers":3, "effective_workers":3,
            "window_start_indices":[60,180,300],
            "aggregation":"configured replication order; primary problem is reused",
        }:
            raise ValueError("bounded window execution policy differs")
        identities = [p["git_commit"], result["run"]["git_commit"]]
        if result["solid_state"] is not None:
            identities.append(result["solid_state"]["provenance"]["git_commit"])
        if any(value != lineage["git_commit_short"] for value in identities):
            raise ValueError("pipeline Git identity differs")
        if (
            p["solver_backend"] != "native_tum_joint_slac"
            or p["max_iterations"] != 20
            or p["native_tum_joint_slac_paired_frame_count"] != 796
            or p["native_tum_joint_slac_solver"]["status"] != "converged"
            or p["native_tum_joint_slac_solver"]["information_rank"] != 56
            or len(p["native_tum_joint_slac_solver"]["history"]) != 9
        ):
            raise ValueError("full primary workload differs")
        for key in (
            "replication_windows",
            "spatial_ablation_windows",
            "xyz_lattice_windows",
            "xyz_pair_windows",
            "xyz_pair_reassociation",
            "iterative_reassociation",
            "rematching",
        ):
            if [item["start_index"] for item in p["native_tum_joint_slac_" + key]] != [
                60,
                180,
                300,
            ]:
                raise ValueError("full three-window evaluation differs")
        prefix = config["dataset"]["path"].replace("\\", "/").rstrip("/") + "/"
        input_hashes = {prefix + item["name"]: item["sha256"] for item in inputs}
        for item in p["native_tum_joint_slac_selected_depth_files"]:
            if input_hashes.get(item["path"].replace("\\", "/")) != item["sha256"]:
                raise ValueError("pipeline depth fingerprint differs")
        history = json.loads(gzip.decompress((folder / "history.json.gz").read_bytes()))
        validate(history)
        if history["provenance"] != lineage:
            raise ValueError("history lineage differs")
        events, solves = history["payload"]["events"], history["payload"]["solves"]
        expected_history = {
            key: sorted((e for h in raw_histories for e in h[key]), key=lambda e: e["window"])
            for key in ("events", "solves", "diagnostics", "frontend")
        }
        if history["payload"] != expected_history:
            raise ValueError("canonical history differs from raw process events")
        diagnostics = history["payload"]["diagnostics"]
        frontend = history["payload"]["frontend"]
        counts_frontend = {
            name: sum(event["kind"] == name for event in frontend) for name in FRONTEND
        }
        if counts_frontend != dict.fromkeys(FRONTEND, 75) or (
            frontend_counts is not None and frontend_counts != counts_frontend
        ):
            raise ValueError("complete frontend evaluation work differs")
        frontend_counts = counts_frontend
        for event in frontend:
            if event["width"] != 640 or event["height"] != 480 or event["pixel_count"] != 640 * 480:
                raise ValueError("complete frontend pixel dimensions differ")
        for name in FRONTEND:
            seconds = info["frontend_seconds"][name]
            if not 0 < seconds < math.inf:
                raise ValueError("frontend timing differs")
            timings[name][variant].append(seconds)


        counts = {
            name: sum(event["kind"] == name for event in diagnostics)
            for name in (
                "_xyz_pair_reassociation_state",
                "_reassociated_spatial_state",
                "_shared_holdout_objective",
            )
        }
        if any(count == 0 for count in counts.values()) or (
            diagnostic_counts is not None and counts != diagnostic_counts
        ):
            raise ValueError("complete diagnostic evaluation work differs")
        diagnostic_counts = counts
        for name in STAGES:
            seconds = info["stage_seconds"][name]
            if not 0 < seconds < math.inf:
                raise ValueError("diagnostic timing differs")
            timings[name][variant].append(seconds)

        if variant in factories and factories[variant] != info["factory_calls"]:
            raise ValueError("repeated factory work differs")
        factories[variant] = info["factory_calls"]
        requests, accepts = info["rgbd_plan_requests"], info["rgbd_plan_accepts"]
        if requests != len(events) or not info["jacobian_calls"] <= accepts <= requests:
            raise ValueError("array evaluation work differs")
        if variant in plan_counts and plan_counts[variant][:2] != (requests, accepts):
            raise ValueError("repeated array evaluation work differs")
        builds = info["rgbd_plan_builds"]
        if not 0 < builds <= accepts or (
            variant in plan_counts and plan_counts[variant][2] != builds
        ):
            raise ValueError("array plan preparation work differs")
        plan_counts[variant] = (requests, accepts, builds)
        if (
            len(solves) != 24
            or len(solves) != info["solve_count"]
            or sum(e["kind"] == "linearize" for e in events) != info["jacobian_calls"]
            or sum(e["kind"] == "residuals" for e in events) != info["residual_calls"]
            or len(events) < 1000
        ):
            raise ValueError("complete evaluation inventory differs")
        primary_trace = next(s for s in solves if s["window"] == 180)
        if (
            primary_trace["result_sha256"] != digest(p["native_tum_joint_slac_solver"])
            or primary_trace["iterations"] != 9
            or primary_trace["parameter_dimension"] != 56
            or primary_trace["active_dimension"] != 56
            or primary_trace["factor_count"] != p["native_tum_joint_slac_correspondence_count"] + 8
        ):
            raise ValueError("primary optimizer trace differs from saved result")
        for event in events:
            if len(event["residual"]["shape"]) != 1 or (
                event["kind"] == "linearize"
                and (
                    len(event["jacobian"]["shape"]) != 2
                    or event["jacobian"]["shape"][0] != event["residual"]["shape"][0]
                )
            ):
                raise ValueError("numeric array dimensions differ")
        progress = json.loads((folder / "progress.json").read_bytes())
        validate(progress)
        if (
            progress["provenance"] != lineage
            or progress["payload"]["solves"] != solves
            or len(progress["payload"]["solve_seconds"]) != len(solves)
            or not math.isclose(sum(progress["payload"]["solve_seconds"]),
                                info["optimizer_seconds"], rel_tol=1e-14, abs_tol=0.0)
        ):
            raise ValueError("optimizer timing or work differs")
        if (
            any(manifests[variant].get(v["path"]) != v["sha256"] for v in info["imports"].values())
            or "calibrex.graph.joint_factors" not in info["imports"]
        ):
            raise ValueError("worker used an unfrozen module")
        if (folder / "stdout.txt").read_bytes() or (folder / "stderr.txt").read_bytes():
            raise ValueError("unexpected worker logs")
        if (
            not 0
            < info["api_seconds"]
            <= attempt["seconds"]
            < math.inf
        ):
            raise ValueError("invalid timing scope")
        timings["whole"][variant].append(attempt["seconds"])
        for kind in ("api", "optimizer"):
            timings[kind][variant].append(info[kind + "_seconds"])
        results.append(digest(comparable(result)))
        histories.append(digest(history["payload"]))
    if len(set(results)) != 1 or len(set(histories)) != 1:
        raise ValueError("complete results or every numeric-array digest differ")
    if (plan_counts["current"][:2] != plan_counts["prior"][:2]
        or factories["current"] != factories["prior"]):
        raise ValueError("unchanged graph preparation or correspondence factories differ")
    medians = {
        kind: {v: statistics.median(t) for v, t in items.items()} for kind, items in timings.items()
    }
    report = record(
        "report",
        {
            "seconds_median": medians,
            "speedup": {kind: items["prior"] / items["current"] for kind, items in medians.items()},
            "verified_runs": len(attempts),
            "solve_count": info["solve_count"],
            "jacobian_calls": info["jacobian_calls"],
            "residual_calls": info["residual_calls"],
            "complete_results_exact": True,
            "all_array_digests_exact": True,
            "runtime_source_changes": sorted(changed),
            "array_plan_requests_per_run": {v: counts[0] for v, counts in plan_counts.items()},
            "array_plan_accepts_per_run": {v: counts[1] for v, counts in plan_counts.items()},
            "array_plan_builds_per_run": {v: counts[2] for v, counts in plan_counts.items()},
            "diagnostic_evaluations_per_run": diagnostic_counts,
            "frontend_evaluations_per_run": frontend_counts,
            "factory_calls_per_run": factories,
        },
        {"protocol_sha256": sha(protocol_path), "attempts_sha256": sha(output / "attempts.json")},
    )
    if (output / "summary.json").exists() and json.loads(
        (output / "summary.json").read_bytes()
    ) != report:
        raise ValueError("saved summary differs from recomputed evidence")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--audit-only", type=Path)
    parser.add_argument("--worker-protocol", type=Path)
    parser.add_argument("--variant", choices=("prior", "current"))
    parser.add_argument("--window-request", type=Path)
    parser.add_argument("--window-response", type=Path)
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit(args.audit_only), indent=2))
        return
    if args.worker_protocol:
        worker(args.worker_protocol, args.variant, args.output,
               args.window_request, args.window_response)
        return
    if not args.baseline_commit or args.output is None or args.repetitions < 2:
        parser.error("baseline commit, output and at least two repetitions are required")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=ROOT):
        raise ValueError("commit runtime changes before measuring")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import yaml

    inputs = [
        {"name": p.relative_to(DATA).as_posix(), "sha256": sha(p), "size_bytes": p.stat().st_size}
        for p in sorted(DATA.rglob("*"))
        if p.is_file() and (p.parent.name == "depth" or p.suffix == ".txt")
    ]
    write(output / "inputs.json", inputs)
    if sha(output / "inputs.json") != INPUTS_SHA256:
        raise ValueError("public depth input inventory differs")
    config = yaml.safe_load(CONFIG.read_bytes())
    config["dataset"]["path"] = str(DATA)
    config["project"]["output_dir"] = str(output / "shared-output")
    config["pipeline"]["factors"]["tum_rgbd_joint_point_to_plane"]["options"]["window_workers"] = 3
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    commits = {
        v: subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
        for v, ref in (("prior", args.baseline_commit), ("current", "HEAD"))
    }
    for path, name in (
        (Path(__file__), "launcher.py.txt"),
        (SCHEMA, "schema.json"),
        (CONFIG.with_name("manifest.yaml"), "upstream-manifest.yaml"),
        (ROOT / "schemas/result.schema.json", "result-schema.json"),
        (ROOT / "tools/benchmark_bspline_lidar_imu.py", "freezer.py.txt"),
        (ROOT / "tools/benchmark_lazy_runtime_loading.py", "identity.py.txt"),
        (ROOT / "tools/benchmark_radar_doppler_batches.py", "capsule-auditor.py.txt"),
    ):
        shutil.copyfile(path, output / name)
    for variant, commit in commits.items():
        source = output / ("sources-" + variant)
        _freeze(source, commit)
        _git_identity(source, commit)
        shutil.copyfile(source / "manifest.json", output / f"sources-{variant}-manifest.json")
        with tarfile.open(output / f"sources-{variant}.tar.gz", "w:gz") as bundle:
            bundle.add(source / "src", arcname="src")
    protocol = record(
        "protocol",
        {
            "scope": SCOPE,
            "commits": commits,
            "inputs_sha256": sha(output / "inputs.json"),
            "repetitions": args.repetitions,
            "blas_threads": 1,
            "environment": {
                "python": sys.version,
                "packages": {
                    n: importlib.metadata.version(n)
                    for n in ("numpy", "scipy", "pydantic", "PyYAML")
                },
            },
        },
        {"files": {name: sha(output / name) for name in sorted(PINNED)}},
    )
    protocol_path = output / "protocol.json"
    write(protocol_path, protocol)
    attempts = []
    for repeat in range(args.repetitions):
        for variant in ("current", "prior") if repeat % 2 == 0 else ("prior", "current"):
            folder = output / f"{repeat + 1:02d}-{variant}"
            folder.mkdir()
            source = output / ("sources-" + variant)
            _verify_sources(source)
            env = os.environ.copy()
            env.update(
                dict.fromkeys(("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"), "1")
            )
            env.update(
                PYTHONPATH=str(source / "src"), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1"
            )
            with (
                (folder / "stdout.txt").open("w") as stdout,
                (folder / "stderr.txt").open("w") as stderr,
            ):
                begin = time.perf_counter()
                done = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker-protocol",
                        str(protocol_path),
                        "--variant",
                        variant,
                        "--output",
                        str(folder),
                    ],
                    cwd=source,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                )
                seconds = time.perf_counter() - begin
            attempts.append(
                {
                    "folder": folder.name,
                    "variant": variant,
                    "seconds": seconds,
                    "exit_code": done.returncode,
                }
            )
            write(
                output / "attempts.json",
                record("attempts", {"attempts": attempts}, {"protocol_sha256": sha(protocol_path)}),
            )
            print(json.dumps(attempts[-1]), flush=True)
            if done.returncode:
                raise RuntimeError(f"worker failed; see {folder / 'stderr.txt'}")
            _verify_sources(source)
            if sha(Path(__file__)) != protocol["provenance"]["files"]["launcher.py.txt"]:
                raise ValueError("benchmark launcher changed during measurement")
    report = audit(output)
    write(output / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
