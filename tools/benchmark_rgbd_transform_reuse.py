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

VERSION = "slac.rgbd_transform_reuse_benchmark/v0.1"
SCHEMA = ROOT / "tools/schemas/rgbd_transform_reuse_benchmark.schema.json"
CONFIG = ROOT / "examples/public_datasets/tum_rgbd_freiburg1_xyz/native_joint_slac_config.yaml"
DATA = ROOT / "data/public/rgbd_dataset_freiburg1_xyz"
INPUTS_SHA256 = "f1ec8837db3c70ddb173869a4362dd54a5a9da3dc8e6f9cc7ac59a7b14e2a108"
CHANGES = {"calibrex/graph/joint_factors.py"}
SCOPE = (
    "Complete native_tum_joint_slac pipeline on the public development TUM fr1/xyz sequence: "
    "unchanged primary and replication windows, scalar and full-XYZ lattice ablations, "
    "two-sided pairs, reassociation, curvature, multistart, holdout, probes and saved reports. "
    "Only bounded evaluation-local transform and calibrated-point reuse changes; "
    "no accuracy or SOTA claim."
)
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
    return value


def worker(protocol_path: Path, variant: str, folder: Path) -> None:
    import numpy as np

    from calibrex.core.provenance import git_commit
    from calibrex.graph import joint_optimization as graph
    from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration

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
    events, solves, solve_seconds = [], [], []
    original_linearize = graph._linearize
    original_residuals = graph._factor_residuals
    original_solve = graph.BackendNeutralJointOptimizer.solve

    def array(value):
        return {
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest(),
        }

    def event(kind, factors, values, residual):
        return {
            "kind": kind,
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
    begin = time.perf_counter()
    result = run_calibration(protocol_path.parent / "config.yaml", CalibrationRunOptions())
    seconds = time.perf_counter() - begin
    assert result is not None
    value = result.model_dump(mode="json")
    (folder / "result.json.gz").write_bytes(
        gzip.compress((json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), mtime=0)
    )
    history = record("history", {"events": events, "solves": solves}, provenance)
    (folder / "history.json.gz").write_bytes(
        gzip.compress((json.dumps(history, indent=2, allow_nan=False) + "\n").encode(), mtime=0)
    )
    imports = {}
    for name, module in sys.modules.copy().items():
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            imports[name] = {
                "path": path.relative_to(source / "src").as_posix(),
                "sha256": sha(path),
            }
    write(
        folder / "measurement.json",
        record(
            "measurement",
            {
                "api_seconds": seconds,
                "optimizer_seconds": sum(solve_seconds),
                "imports": imports,
                "result_sha256": sha(folder / "result.json.gz"),
                "history_sha256": sha(folder / "history.json.gz"),
                "solve_count": len(solves),
                "jacobian_calls": sum(e["kind"] == "linearize" for e in events),
                "residual_calls": sum(e["kind"] == "residuals" for e in events),
            },
            provenance,
        ),
    )


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
    if changed != CHANGES or set(prior) != set(current):
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
        for v in (("prior", "current") if i % 2 == 0 else ("current", "prior"))
    ]
    if payload["repetitions"] < 2 or [a["variant"] for a in attempts] != order:
        raise ValueError("alternating run inventory differs")
    result_schema = json.loads((output / "result-schema.json").read_bytes())
    if result_schema != json.loads((ROOT / "schemas/result.schema.json").read_bytes()):
        raise ValueError("calibration result schema differs")
    timings = {kind: {v: [] for v in manifests} for kind in ("whole", "api", "optimizer")}
    results, histories = [], []
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
        result = json.loads(gzip.decompress((folder / "result.json.gz").read_bytes()))
        jsonschema.validate(result, result_schema)
        p = result["run"]["provenance"]
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
        if (
            len(solves) != 24
            or len(solves) != info["solve_count"]
            or sum(e["kind"] == "linearize" for e in events) != info["jacobian_calls"]
            or sum(e["kind"] == "residuals" for e in events) != info["residual_calls"]
            or len(events) < 1000
        ):
            raise ValueError("complete evaluation inventory differs")
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
            or sum(progress["payload"]["solve_seconds"]) != info["optimizer_seconds"]
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
            < info["optimizer_seconds"]
            <= info["api_seconds"]
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
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit(args.audit_only), indent=2))
        return
    if args.worker_protocol:
        worker(args.worker_protocol, args.variant, args.output)
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
        for variant in ("prior", "current") if repeat % 2 == 0 else ("current", "prior"):
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
