"""Audit the CPU-readable evidence for rejected CUDA odometry experiments."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

import numpy as np
from jsonschema import validate

_VARIANTS = {
    "voxels": ["prior", "current"],
    "layout": ["prior", "fixed", "network", "packed"],
    "radix": ["prior", "radix"],
    "ordering": ["prior", "normal", "morton", "native_order"],
    "batching": ["prior", "4", "8"],
}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_bytes())


def _physical(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "execution_backend"}


def audit(root: Path) -> dict:
    """Check byte bindings, every observation, numerical fields and warm scopes."""
    index = _read(root, "publication-index.json")
    files = {p.relative_to(root).as_posix() for p in root.rglob("*")
             if p.is_file() and p != root / "publication-index.json"}
    if files != set(index["files"]):
        raise ValueError("screen index does not bind every file")
    if any(_digest((root / name).read_bytes()) != digest
           for name, digest in index["files"].items()):
        raise ValueError("screen file digest differs")
    screen = _read(root, "screen.json")
    validate(screen, _read(root, "screen.schema.json"))
    if (screen["schema_version"] != "slac.odometry_candidate_screen/v0.1"
            or screen["decision"] != "no_candidate_adopted"
            or index["provenance"]["screen_sha256"]
            != _digest((root / "screen.json").read_bytes())):
        raise ValueError("screen decision or binding differs")
    manifest = _read(root, "source-manifest.json")
    with tarfile.open(root / "sources.tar.gz") as bundle:
        sources = {}
        for item in bundle.getmembers():
            if not item.isfile() or not item.name.endswith(".py"):
                continue
            if not item.name.startswith("src/calibrex/") or ".." in Path(item.name).parts:
                raise ValueError("unexpected source archive path")
            stream = bundle.extractfile(item)
            assert stream is not None
            sources[item.name.removeprefix("src/")] = _digest(stream.read())
    if sources != manifest:
        raise ValueError("screen source capsule differs")
    provenance = screen["provenance"]
    if (provenance["source_manifest_sha256"]
            != _digest((root / "source-manifest.json").read_bytes())
            or provenance["publisher_sha256"] != _digest((root / "publisher.py.txt").read_bytes())
            or provenance["profile_parent_sha256"]
            != _digest((root / "profile/parent.json").read_bytes())):
        raise ValueError("screen provenance differs")
    profile = _read(root, "profile/parent.json")
    measurement = _read(root, "profile/measurement.json")
    result = _read(root, "profile/result.json")
    if (profile["exit_code"] != 0 or not profile["diagnostic_only"]
            or not profile["profiling_overhead"]
            or profile["source_manifest_sha256"]
            != _digest((root / "source-manifest.json").read_bytes())
            or profile["launcher_sha256"]
            != _digest((root / "profile/launcher.py.txt").read_bytes())
            or measurement["provenance"]["bag_sha256"] != screen["bag_sha256"]
            or measurement["provenance"]["source_manifest_sha256"]
            != profile["source_manifest_sha256"]
            or result["provenance"]["git_commit"] != screen["source_commit"]
            or measurement["provenance"]["launcher_sha256"]
            != _digest((root / "profile/benchmark.py.txt").read_bytes())):
        raise ValueError("diagnostic profile source/input binding differs")
    if any(sources.get(name) != digest
           for name, digest in measurement["provenance"]["executed_sources"].items()):
        raise ValueError("profile executed source differs")
    observations = 0
    rows = []
    common_fixtures = None
    if [c["name"] for c in screen["cohorts"]] != list(_VARIANTS):
        raise ValueError("screen cohort set or order differs")
    for cohort in screen["cohorts"]:
        name = cohort["name"]
        voxel = name == "voxels"
        expected_cases = ([f"{prefix}-{i}" for prefix in ("raw", "processed") for i in range(8)]
                          if voxel else list(range(8)))
        expected_scopes = ([{"name": prefix, "first_repeat": 1, "case_prefix": prefix}
                            for prefix in ("raw", "processed")] if voxel else
                           [{"name": "target_and_registration", "first_repeat": 0}])
        if (cohort["variants"] != _VARIANTS[name] or cohort["cases"] != expected_cases
                or cohort["repeats"] != (7 if voxel else 5)
                or cohort["scopes"] != expected_scopes
                or cohort["quality"] != ("numeric" if name == "batching" else "exact")):
            raise ValueError("candidate scope or sampling contract differs")
        protocol = _read(root, f"{name}/protocol.json")
        summary = _read(root, f"{name}/summary.json")
        runs = json.loads((root / name / "runs.json").read_bytes())
        if (protocol["launcher_sha256"] != _digest((root / name / "launcher.py.txt").read_bytes())
                or summary["provenance"]["protocol_sha256"]
                != _digest((root / name / "protocol.json").read_bytes())
                or protocol["blas_threads"] != 1):
            raise ValueError("candidate launcher or protocol binding differs")
        if "sources" in protocol:
            if protocol["sources"] != {"src/" + k: v for k, v in sources.items()}:
                raise ValueError("candidate runtime source differs")
        elif protocol["source_sha256"] != sources["calibrex/solvers/scan_to_scan_odometry.py"]:
            raise ValueError("voxel runtime source differs")
        fixture = protocol["fixtures"]
        if common_fixtures is None:
            common_fixtures = fixture
        elif fixture != common_fixtures:
            raise ValueError("candidate registration inputs differ")
        for i in range(8):
            filename = f"registration-{i:02d}"
            raw = (root / "fixtures" / (filename + ".json")).read_bytes()
            meta = json.loads(raw)
            if (_digest(raw) != fixture[filename + ".json"]
                    or meta["fixture_sha256"] != fixture[filename + ".npz"]
                    or meta["source_manifest_sha256"] != profile["source_manifest_sha256"]):
                raise ValueError("captured registration input binding differs")
        if "bag_sha256" in protocol and protocol["bag_sha256"] != screen["bag_sha256"]:
            raise ValueError("voxel raw input differs")
        if "candidate_sha256" in protocol and (
            protocol["candidate_sha256"] != _digest((root / name / "candidate.py.txt").read_bytes())
        ):
            raise ValueError("candidate module source differs")
        if "kernel_helper_sha256" in protocol and (
            protocol["kernel_helper_sha256"]
            != _digest((root / name / "kernels.py.txt").read_bytes())
        ):
            raise ValueError("candidate kernel helper differs")
        if "archived_source_sha256" in protocol:
            if set(protocol["archived_source_sha256"]) != {
                "cuda_lidar_map_adapter.py", "cuda_scan_odometry_adapter.py",
            }:
                raise ValueError("archived radix input set differs")
            for key, value in protocol["archived_source_sha256"].items():
                if value != _digest((root / "radix-inputs" / (key + ".txt")).read_bytes()):
                    raise ValueError("archived radix source differs")
        if "morton_source_sha256" in protocol and (
            protocol["morton_source_sha256"]
            != _digest((root / "radix-inputs/cuda_lidar_map_adapter.py.txt").read_bytes())
        ):
            raise ValueError("archived Morton source differs")
        # Parse all launchers without executing experimental code or importing CUDA.
        ast.parse((root / name / "launcher.py.txt").read_bytes())
        repeats = cohort["repeats"]
        cases = cohort["cases"]
        variants = cohort["variants"]
        expected = {(i, v, c) for i in range(repeats) for v in variants for c in cases}
        actual = [(r["repeat"], r["variant"], r["case"]) for r in runs]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("candidate observations are missing or duplicated")
        references = {r["case"]: r for r in runs if r["variant"] == "prior" and r["repeat"] == 0}
        for record in runs:
            seconds = record["seconds"]
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("candidate timing is not positive and finite")
            reference = references[record["case"]]
            if "initialization_seconds" in record and not (
                0 <= record["initialization_seconds"] <= seconds
            ):
                raise ValueError("candidate initialization timing differs")
            if "result" in record:
                a, b = _physical(reference["result"]), _physical(record["result"])
                if (np.asarray(b["transform"]).shape != (4, 4)
                        or any(not np.all(np.isfinite(b[k])) for k in (
                            "transform", "rmse_m", "hessian_min_eigenvalue",
                        ))):
                    raise ValueError("candidate registration fields are invalid")
                if cohort["quality"] == "exact":
                    if a != b:
                        raise ValueError("candidate physical registration fields differ")
                else:
                    if any(a[k] != b[k] for k in ("converged", "iterations", "correspondences")):
                        raise ValueError("batched registration decisions or dimensions differ")
                    if any(not np.allclose(a[k], b[k], rtol=1e-6, atol=1e-9)
                           for k in ("transform", "rmse_m", "hessian_min_eigenvalue")):
                        raise ValueError("batched registration numerical fields differ")
                    if record["physical_exact"] != (a == b):
                        raise ValueError("batched equality claim differs")
                work = record["work"]
                if any(type(v) is not int or v < 0 for v in work.values()):
                    raise ValueError("invalid candidate work counter")
                if any(work[k] != reference["work"][k] for k in (
                    "gpu_query_calls", "gpu_query_rows", "gpu_normal_rows",
                    "scalar_query_repairs", "scalar_normal_repairs",
                )):
                    raise ValueError("candidate measurement/iteration work differs")
            elif any(record[k] != reference[k] for k in ("input_rows", "output_rows")):
                raise ValueError("voxel input/output point counts differ")
        observations += len(runs)
        for scope in cohort["scopes"]:
            first = scope["first_repeat"]
            prefix = scope.get("case_prefix")
            medians = {v: statistics.median(sum(
                r["seconds"] for r in runs if r["repeat"] == i and r["variant"] == v
                and (prefix is None or str(r["case"]).startswith(prefix))
            ) for i in range(first, repeats)) for v in variants}
            if prefix is None:
                declared, ratios = summary["median_seconds"], summary["speedups"]
            else:
                declared, ratios = summary["medians"][prefix], summary["speedups"]
            if declared != medians:
                raise ValueError("candidate median differs")
            for variant in variants:
                if variant == "prior":
                    continue
                ratio = medians["prior"] / medians[variant]
                claimed = ratios[prefix] if prefix else ratios[variant]
                if ratio != claimed:
                    raise ValueError("candidate speed ratio differs")
                rows.append({"cohort": name, "scope": scope["name"], "variant": variant,
                             "prior_seconds": medians["prior"],
                             "candidate_seconds": medians[variant], "speedup": ratio})
    if rows != screen["comparisons"] or observations != screen["observations"]:
        raise ValueError("screen aggregate differs")
    return {"decision": screen["decision"], "cohorts": len(screen["cohorts"]),
            "observations": observations, "comparisons": rows, "files": len(files),
            "provenance": {"screen_sha256": _digest((root / "screen.json").read_bytes()),
                           "source_commit": screen["source_commit"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?",
                        default=Path("docs/assets/odometry_candidate_screen"))
    print(json.dumps(audit(parser.parse_args().archive), indent=2))


if __name__ == "__main__":
    main()
