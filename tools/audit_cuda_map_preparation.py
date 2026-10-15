"""Recompute the public CUDA map comparison without a GPU or raw dataset."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any

import jsonschema
from tools.benchmark_cuda_map_preparation import compare, summarize

from calibrex.core.lidar_lidar_extrinsic import LidarLidarExtrinsicArtifact


def _sha(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _read(root: Path, name: str) -> Any:
    return json.loads((root / name).read_bytes())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _constants(source: bytes) -> dict[str, str]:
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    values[target.id] = node.value.value
    return values


def _identity(value: str, constants: dict[str, str]) -> str:
    match = re.fullmatch(
        r"(calibrex\.cuda_lidar_map/v0\.1;fp64=true;query_fma=false;"
        r"cupy=[0-9.]+;cuda=[0-9]+;driver=[0-9]+;cc=[0-9.]+;)"
        r"query_sha256=([0-9a-f]{64})(?:;normal_sha256=([0-9a-f]{64});normal_cache=lazy"
        r"(?:;registration_sha256=([0-9a-f]{64}))?)?", value,
    )
    _require(match is not None, "invalid CUDA execution identity")
    assert match is not None
    _require(match[2] == _sha(constants["_SOURCE"].encode()), "query kernel identity mismatch")
    normal = constants.get("_NORMAL_SOURCE")
    _require(match[3] == (None if normal is None else _sha(normal.encode())),
             "normal kernel identity mismatch")
    registration = constants.get("_REGISTRATION_SOURCE")
    _require(match[4] == (None if registration is None else _sha(registration.encode())),
             "registration kernel identity mismatch")
    return match[1]


def audit(root: Path) -> dict[str, Any]:
    """Verify byte bindings, kernel work, complete quality and every timing ratio."""
    index = _read(root, "publication-index.json")
    files = {p.relative_to(root).as_posix(): _sha(p.read_bytes())
             for p in root.rglob("*") if p.is_file() and p.name != "publication-index.json"}
    _require(files == index["files"], "publication byte binding mismatch")
    protocol, attempts, summary = (_read(root, f"{name}.json")
                                   for name in ("protocol", "attempts", "summary"))
    schema = _read(root, "comparison.schema.json")
    for name, value in (("protocol", protocol), ("attempts", attempts), ("summary", summary)):
        jsonschema.validate(value, {**schema, "$ref": "#/$defs/" + name})
    _require(summary["provenance"]["protocol_sha256"]
             == _sha((root / "protocol.json").read_bytes()),
             "summary protocol binding mismatch")
    for name, digest in protocol["launchers_sha256"].items():
        _require(files["launchers/" + Path(name).name + ".txt"] == digest,
                 "measured launcher binding mismatch")
    manifests, constants = {}, {}
    for variant in ("prior", "current"):
        name = f"sources-{variant}-manifest.json"
        _require(files[name] == protocol["source_manifest_sha256"][variant],
                 "source manifest binding mismatch")
        manifests[variant] = _read(root, name)
        captured = {}
        with tarfile.open(root / f"sources-{variant}.tar.gz", "r:gz") as bundle:
            for member in bundle:
                _require(member.isfile(), "source capsule contains non-file member")
                if member.name == "LICENSE":
                    continue
                _require(member.name.startswith("src/calibrex/")
                         and ".." not in Path(member.name).parts,
                         "source capsule contains unsafe member")
                relative = member.name.removeprefix("src/")
                _require(relative not in captured, "duplicate source capsule member")
                stream = bundle.extractfile(member)
                assert stream is not None
                raw = stream.read()
                captured[relative] = _sha(raw)
                if relative == "calibrex/solvers/cuda_lidar_map_adapter.py":
                    constants[variant] = _constants(raw)
        _require(captured == manifests[variant], "source capsule differs from manifest")
    _require(len(attempts) == 2 * protocol["repetitions"], "worker count mismatch")
    measurements, previous_cache = {}, {}
    for position, attempt in enumerate(attempts):
        repeat, within = divmod(position, 2)
        order = ("prior", "current") if repeat % 2 == 0 else ("current", "prior")
        variant = order[within]
        worker = f"{repeat:02d}-{variant}"
        _require((attempt["repeat"], attempt["variant"], attempt["worker"])
                 == (repeat, variant, worker), "worker order mismatch")
        row = _read(root, worker + "/measurement.json")
        artifact_path = root / worker / "result.json"
        artifact = json.loads(artifact_path.read_bytes())
        LidarLidarExtrinsicArtifact.model_validate(artifact)
        _require(row["artifact"] == artifact
                 and row["result_sha256"] == _sha(artifact_path.read_bytes()),
                 "saved full artifact mismatch")
        provenance = row["provenance"]
        _require(provenance["source_commit"] == protocol[
            "baseline_commit" if variant == "prior" else "current_commit"
        ] == artifact["provenance"]["git_commit"], "source commit mismatch")
        _require(provenance["input_sha256"] == protocol["input_sha256"], "raw input mismatch")
        _require(provenance["launchers_sha256"] == protocol["launchers_sha256"],
                 "worker launcher mismatch")
        _require(provenance["source_manifest_sha256"]
                 == protocol["source_manifest_sha256"][variant], "worker manifest mismatch")
        executed = provenance["executed_sources_sha256"]
        _require(executed and all(manifests[variant].get(p) == h for p, h in executed.items()),
                 "executed imports mismatch")
        for required in ("calibrex/solvers/cuda_lidar_map_adapter.py",
                         "calibrex/evaluation/lidar_lidar_map.py",
                         "calibrex/solvers/lidar_lidar_map_solver.py"):
            _require(required in executed, "required CUDA import missing")
        for name, digest in provenance["source_sha256"].items():
            expected = (manifests[variant][name.removeprefix("src/")]
                        if name.startswith("src/") else protocol["launchers_sha256"][name])
            _require(digest == expected, "worker source binding mismatch")
        backend = row["evaluation"]["result"]["execution_backend"]
        _identity(backend, constants[variant])
        _require(backend == artifact["options"]["execution_backend"], "artifact identity mismatch")
        _require(0 < row["api_seconds"] == attempt["api_seconds"] < attempt["wall_seconds"],
                 "worker timing mismatch")
        cold = protocol["compiler_cache"] == "cold" or repeat == 0
        _require(row["cache_initially_empty"] == cold and bool(attempt["cache_before"]) != cold,
                 "compiler cache scope mismatch")
        if not cold:
            _require(attempt["cache_before"] == previous_cache[variant], "retained cache changed")
        _require(bool(attempt["cache_after"]), "compiled cache evidence missing")
        previous_cache[variant] = attempt["cache_after"]
        stats = row["statistics"]
        _require(stats["prepared_samples"] == len(row["map_points"])
                 and stats["gpu_query_rows"] > 0, "missing map query work")
        if "_NORMAL_SOURCE" in constants[variant]:
            normal_rows, calls = stats["gpu_normal_rows"], stats["gpu_normal_kernel_calls"]
            _require(0 < normal_rows <= sum(row["map_points"])
                     and 0 < calls <= normal_rows <= calls * 65536
                     and 0 <= stats["scalar_normal_repairs"] <= normal_rows,
                     "invalid lazy normal kernel work")
        if "_REGISTRATION_SOURCE" in constants[variant]:
            _require(stats["gpu_projection_query_calls"] + stats["gpu_normal_kernel_calls"]
                     == stats["gpu_query_calls"]
                     and stats["gpu_projection_query_rows"] + stats["gpu_normal_rows"]
                     == stats["gpu_query_rows"], "fused query work mismatch")
            _require(0 < stats["gpu_jacobian_kernel_calls"] <= stats["gpu_residual_kernel_calls"]
                     <= stats["gpu_projection_query_calls"]
                     and 0 < stats["gpu_jacobian_kernel_rows"] <= stats["gpu_residual_kernel_rows"]
                     <= stats["gpu_projection_query_rows"], "fused registration work missing")
        measurements[worker] = row
    for repeat in range(protocol["repetitions"]):
        prior, current = (measurements[f"{repeat:02d}-{v}"] for v in ("prior", "current"))
        errors = compare(prior["evaluation"], current["evaluation"])
        for key in ("prepared_sha256", "partition_sha256", "map_points", "target_points"):
            _require(prior[key] == current[key], "prepared public workload changed")
        _require(_identity(prior["evaluation"]["result"]["execution_backend"], constants["prior"])
                 == _identity(current["evaluation"]["result"]["execution_backend"],
                              constants["current"]), "CUDA environment changed")
        artifacts = [copy.deepcopy(row["artifact"]) for row in (prior, current)]
        for artifact in artifacts:
            artifact["options"]["execution_backend"] = "<DECLARED_CUDA>"
            for key in ("created_at", "command", "git_commit"):
                artifact["provenance"].pop(key, None)
        from tools.benchmark_cuda_lidar_map import _compare

        _require(not errors and not _compare(*artifacts), "full evaluation or artifact changed")
        _require(all(not r["errors"] for r in attempts if r["repeat"] == repeat),
                 "retained quality errors")
        if "_NORMAL_SOURCE" in constants["current"] and "_NORMAL_SOURCE" not in constants["prior"]:
            saved = sum(current["map_points"]) - current["statistics"]["gpu_normal_rows"]
            _require(prior["statistics"]["gpu_query_rows"]
                     - current["statistics"]["gpu_query_rows"] == saved,
                     "normal query reduction disagrees with unchanged registration work")
        if ("_REGISTRATION_SOURCE" in constants["current"]
            and "_REGISTRATION_SOURCE" not in constants["prior"]):
            counters = {k: v for k, v in current["statistics"].items()
                        if k in prior["statistics"]}
            _require(counters == prior["statistics"], "fusion changed retained registration work")
    expected = summarize(attempts)
    expected["provenance"] = summary["provenance"]
    _require(summary == expected, "summary differs from complete observations")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))


if __name__ == "__main__":
    main()
