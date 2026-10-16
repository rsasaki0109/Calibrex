"""Compare complete public Livox joint SLAC pipelines with frozen CPU workers."""

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

VERSION = "slac.joint_lidar_transform_reuse_benchmark/v0.1"
SCHEMA = ROOT / "tools/schemas/joint_lidar_transform_reuse_benchmark.schema.json"
CONFIG = (
    ROOT
    / "examples/public_datasets/livox_horizon_horizon_pcd_sample/native_full_points_config.yaml"
)
DATA = ROOT / "data/public/livox_horizon_horizon_pair"
INPUTS = {
    "base_horizon_100432.pcd": (
        "19b3864cbee76b785d207fcb3fe48f6f506d142d8233f842d0a87fe4330fe407",
        23921,
    ),
    "target_horizon_100538.pcd": (
        "61a74eb8b8ff4eeba4ca0ff824a4046ae220e947e8156f27d93da452538f5b6f",
        23666,
    ),
}
CHANGES = {
    "calibrex/graph/_evaluation_scope.py",
    "calibrex/graph/joint_factors.py",
    "calibrex/graph/joint_optimization.py",
}
SCOPE = (
    "Complete native_joint_slac pipeline on all finite points of the official public Livox "
    "Horizon pair, including inspection, frontend, joint LM, holdout, probes, fixed baseline "
    "and report saving. Existing development geometry is degenerate; no accuracy claim."
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
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(kind: str, payload: dict, provenance: dict) -> dict:
    return {
        "schema_version": VERSION,
        "record_type": kind,
        "payload": payload,
        "provenance": provenance,
    }


def write(path: Path, value: dict) -> None:
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def comparable(value: dict) -> dict:
    """Remove only declared run identity, wall-clock dates and source identity."""
    value = json.loads(json.dumps(value, allow_nan=False))
    value["run"].pop("id")
    value["run"].pop("created_at")
    value["run"]["provenance"].pop("generated_at")
    value["run"]["provenance"].pop("git_commit")
    return value


def ids_digest(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def worker(protocol_path: Path, variant: str, folder: Path) -> None:
    import numpy as np

    from calibrex.core.provenance import git_commit
    from calibrex.graph import joint_optimization as graph
    from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration

    protocol = json.loads(protocol_path.read_bytes())
    source = protocol_path.parent / ("sources-" + variant)
    full = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    short = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    if full != protocol["payload"]["commits"][variant] or git_commit() != short:
        raise ValueError("actual Git identity differs from frozen sources")
    provenance = {
        "protocol_sha256": sha(protocol_path),
        "git_commit": full,
        "git_commit_short": short,
    }
    events = []
    original_linearize, original_residuals = graph._linearize, graph._factor_residuals

    def array(value):
        return {
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.asarray(value, dtype="<f8").tobytes(order="C")).hexdigest(),
        }

    def event(kind, factors, values, residual):
        return {
            "kind": kind,
            "factor_count": len(factors),
            "values": values,
            "factor_ids_sha256": ids_digest([factor.factor_id for factor in factors]),
            "residual": array(residual),
        }

    def linearize(factors, values, blocks, layout, options):
        residual, jacobian = original_linearize(factors, values, blocks, layout, options)
        item = event("linearize", factors, values, residual)
        item["jacobian"] = array(jacobian)
        events.append(item)
        # Retain completed work if a long full-point worker is interrupted.
        write(folder / "history.json", record("history", {"events": events}, provenance))
        return residual, jacobian

    def residuals(factors, values):
        result = original_residuals(factors, values)
        events.append(event("residuals", factors, values, result))
        return result

    graph._linearize, graph._factor_residuals = linearize, residuals
    begin = time.perf_counter()
    result = run_calibration(protocol_path.parent / "config.yaml", CalibrationRunOptions())
    seconds = time.perf_counter() - begin
    assert result is not None
    value = result.model_dump(mode="json")
    if value["run"]["provenance"]["git_commit"] != short:
        raise ValueError("pipeline source provenance differs from the actual worker")
    (folder / "result.json.gz").write_bytes(
        gzip.compress((json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), mtime=0)
    )
    write(folder / "history.json", record("history", {"events": events}, provenance))
    imports = {}
    for name, module in sys.modules.copy().items():
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            imports[name] = {
                "path": path.relative_to(source / "src").as_posix(),
                "sha256": sha(path),
            }
    joint = value["run"]["provenance"]["native_joint_slac_solver"]
    write(
        folder / "measurement.json",
        record(
            "measurement",
            {
                "api_seconds": seconds,
                "imports": imports,
                "result_sha256": sha(folder / "result.json.gz"),
                "history_sha256": sha(folder / "history.json"),
                "iterations": len(joint["history"]),
                "observation_count": value["run"]["provenance"][
                    "native_joint_slac_observation_count"
                ],
                "status": joint["status"],
            },
            provenance,
        ),
    )


def audit(output: Path) -> dict:
    """Verify complete results, all numeric-array digests and declared work."""
    import jsonschema
    import yaml

    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    schema = json.loads((output / "schema.json").read_bytes())
    if schema != json.loads(SCHEMA.read_bytes()):
        raise ValueError("archive tool schema differs")
    validate = jsonschema.Draft202012Validator(schema).validate
    validate(protocol)
    payload = protocol["payload"]
    pins = protocol["provenance"]["files"]
    if set(pins) != PINNED or any(sha(output / k) != v for k, v in pins.items()):
        raise ValueError("pinned config, tools or source capsule changed")
    if payload["scope"] != SCOPE or payload["blas_threads"] != 1:
        raise ValueError("benchmark scope differs")
    if {x["name"]: (x["sha256"], x["point_count"]) for x in payload["inputs"]} != INPUTS:
        raise ValueError("full public PCD inventory differs")
    if len(payload["inputs"]) != 2:
        raise ValueError("duplicate PCD input")
    config = yaml.safe_load((output / "config.yaml").read_bytes())
    reference = yaml.safe_load(CONFIG.read_bytes())
    reference["dataset"]["path"] = config["dataset"]["path"]
    reference["project"]["output_dir"] = config["project"]["output_dir"]
    reference["solver"]["backend"] = "native_joint_slac"
    if config != reference:
        raise ValueError("full-point pipeline options differ")
    manifests = {v: sources(output, v) for v in ("prior", "current")}
    prior, current = manifests["prior"], manifests["current"]
    changed = {k for k in prior.keys() | current.keys() if prior.get(k) != current.get(k)}
    if changed != CHANGES or set(current) - set(prior) != {"calibrex/graph/_evaluation_scope.py"}:
        raise ValueError("unexpected runtime source changes")
    index_path = output / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_bytes())
        files = index["files"]
        actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        if set(files) != actual - {"index.json"} or index["provenance"] != {
            "protocol_sha256": sha(protocol_path)
        }:
            raise ValueError("published file inventory differs")
        for name, expected in files.items():
            path = (output / name).resolve()
            if not path.is_relative_to(output.resolve()) or expected != {
                "sha256": sha(path),
                "size_bytes": path.stat().st_size,
            }:
                raise ValueError("published file bytes changed")
    attempts_record = json.loads((output / "attempts.json").read_bytes())
    validate(attempts_record)
    if attempts_record["provenance"] != {"protocol_sha256": sha(protocol_path)}:
        raise ValueError("attempt protocol binding differs")
    attempts = attempts_record["payload"]["attempts"]
    repetitions = payload["repetitions"]
    order = [
        v
        for i in range(repetitions)
        for v in (("prior", "current") if i % 2 == 0 else ("current", "prior"))
    ]
    if repetitions < 2 or [r["variant"] for r in attempts] != order:
        raise ValueError("alternating run inventory differs")
    result_schema = json.loads((output / "result-schema.json").read_bytes())
    if result_schema != json.loads((ROOT / "schemas/result.schema.json").read_bytes()):
        raise ValueError("archive calibration schema differs")
    whole, api = {v: [] for v in manifests}, {v: [] for v in manifests}
    results, histories = [], []
    for index, attempt in enumerate(attempts):
        variant = attempt["variant"]
        name = f"{index // 2 + 1:02d}-{variant}"
        if attempt["folder"] != name or attempt["exit_code"] != 0:
            raise ValueError("worker identity or exit status differs")
        folder = output / name
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
            folder / "history.json"
        ):
            raise ValueError("result or history bytes changed")
        result = json.loads(gzip.decompress((folder / "result.json.gz").read_bytes()))
        jsonschema.validate(result, result_schema)
        p = result["run"]["provenance"]
        if p["git_commit"] != lineage["git_commit_short"]:
            raise ValueError("pipeline Git identity differs")
        joint = p["native_joint_slac_solver"]
        if (
            p["solver_backend"] != "native_joint_slac"
            or p["max_iterations"] != 40
            or joint["status"] != "degenerate"
            or joint["information_rank"] != 9
            or p["native_joint_slac_observation_count"] != 16884
            or len(joint["train_factor_ids"]) != 13600
            or len(joint["holdout_factor_ids"]) != 3285
            or len(joint["known_bad_probes"]) != 24
            or len(joint["history"]) != 40
            or (info["observation_count"], info["iterations"], info["status"])
            != (16884, 40, joint["status"])
        ):
            raise ValueError("full joint work counts differ")
        if p["native_joint_slac_frontend"] != {
            "voxel_size_m": 1.0,
            "correspondence_gate_m": 1.5,
            "max_source_points": None,
            "max_target_points": None,
        }:
            raise ValueError("point cap or frontend changed")
        if (p["native_joint_slac_source_sha256"], p["native_joint_slac_target_sha256"]) != tuple(
            value[0] for value in INPUTS.values()
        ):
            raise ValueError("pipeline input hashes differ")
        history = json.loads((folder / "history.json").read_bytes())
        validate(history)
        if history["provenance"] != lineage:
            raise ValueError("history lineage differs")
        events = history["payload"]["events"]
        jacobians = [e for e in events if e["kind"] == "linearize"]
        residuals = [e for e in events if e["kind"] == "residuals"]
        if [e["kind"] for e in events] != ["linearize", "residuals"] * 40 + ["linearize"] + [
            "residuals"
        ] * 25:
            raise ValueError("joint evaluation inventory differs")
        if any(
            e["jacobian"]["shape"] != [13605, 12]
            or e["factor_count"] != 13600
            or e["residual"]["shape"] != [13605]
            for e in jacobians
        ):
            raise ValueError("full Jacobian dimensions differ")
        if any(
            e["factor_count"] not in (13600, 3285)
            or e["residual"]["shape"] != [13605 if e["factor_count"] == 13600 else 3285]
            for e in residuals
        ):
            raise ValueError("full residual dimensions differ")
        for event in events:
            ids = (
                joint["train_factor_ids"]
                if event["factor_count"] == 13600
                else joint["holdout_factor_ids"]
            )
            if event["factor_ids_sha256"] != ids_digest(ids):
                raise ValueError("evaluation factor IDs or order differ")
        final_values = joint["optimized_values"]
        if jacobians[-1]["values"] != final_values or residuals[40]["values"] != final_values:
            raise ValueError("final evaluation values differ")
        for event, probe in zip(residuals[41:], joint["known_bad_probes"], strict=True):
            expected_values = {name: list(vector) for name, vector in final_values.items()}
            expected_values[probe["block"]][probe["dimension"]] += probe["amount"]
            if event["values"] != expected_values:
                raise ValueError("known-bad probe values differ")
        if (
            any(manifests[variant].get(v["path"]) != v["sha256"] for v in info["imports"].values())
            or "calibrex.graph.joint_factors" not in info["imports"]
        ):
            raise ValueError("worker used an unfrozen module")
        if (folder / "stdout.txt").read_bytes() or (folder / "stderr.txt").read_bytes():
            raise ValueError("unexpected worker logs")
        if not 0 < info["api_seconds"] <= attempt["seconds"] < math.inf:
            raise ValueError("invalid timing scope")
        whole[variant].append(attempt["seconds"])
        api[variant].append(info["api_seconds"])
        results.append(json.dumps(comparable(result), allow_nan=False))
        histories.append(json.dumps(history["payload"], allow_nan=False))
    if len(set(results)) != 1 or len(set(histories)) != 1:
        raise ValueError("complete results or every numeric-array digest differ")
    whole_medians = {v: statistics.median(x) for v, x in whole.items()}
    api_medians = {v: statistics.median(x) for v, x in api.items()}
    report = record(
        "report",
        {
            "whole_seconds_median": whole_medians,
            "api_seconds_median": api_medians,
            "whole_speedup": whole_medians["prior"] / whole_medians["current"],
            "api_speedup": api_medians["prior"] / api_medians["current"],
            "verified_runs": len(attempts),
            "observations": 16884,
            "joint_iterations": 40,
            "jacobian_calls": 41,
            "residual_calls": 65,
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
    inputs = []
    for name, (digest, count) in INPUTS.items():
        path = DATA / name
        if sha(path) != digest:
            raise ValueError("public input bytes differ")
        inputs.append(
            {
                "name": name,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "point_count": count,
            }
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import yaml

    config = yaml.safe_load(CONFIG.read_bytes())
    config["dataset"]["path"] = str(DATA)
    config["project"]["output_dir"] = str(output / "shared-output")
    config["solver"]["backend"] = "native_joint_slac"
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
            "inputs": inputs,
            "repetitions": args.repetitions,
            "blas_threads": 1,
            "environment": {
                "python": sys.version,
                "packages": {
                    n: importlib.metadata.version(n) for n in ("numpy", "scipy", "pydantic")
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
