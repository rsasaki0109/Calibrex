"""Audit rejected traversal, normal scheduling, bounds and placement experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path
from typing import Any

from jsonschema import validate

_SOURCE_ARCHIVE = Path(__file__).resolve().parents[1] / "docs/assets/incremental_local_map_screen"
_COHORTS = {
    "traversal": (["prior", "parent", "heap"], 5, 8, "target_and_registration"),
    "normals": (["prior", "lazy", "prefetch"], 5, 8, "target_and_registration"),
    "bounds": (["prior", "soa", "columns"], 7, 8, "bounds_capture"),
    "placement": (["prior", "pose", "edges"], 7, 4, "placement"),
}
_TIMING_SCOPES = {
    "traversal": "target construction and registration, five rotated-order repeats, "
                 "one untimed first-case warmup per variant; no cold/whole-bag claim",
    "normals": "target construction and registration, five rotated-order repeats, "
               "one untimed first-case warmup per variant; no cold/whole-bag claim",
    "bounds": "complete bounds capture and validation only; cKDTree build and topology "
              "capture excluded; seven rotated-order repeats, eight fixed targets",
    "placement": "complete placement, native full derivatives versus pose-only and "
                 "referenced-edge reuse; seven rotated-order repeats over four fixed inputs",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(root: Path, name: str) -> Any:
    return json.loads((root / name).read_bytes())


def _physical(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != "execution_backend"}


def audit(root: Path, source_archive: Path = _SOURCE_ARCHIVE) -> dict[str, Any]:
    """Check every bound observation without running experimental CPU/GPU code."""
    index = _read(root, "publication-index.json")
    validate(index, _read(root, "publication.schema.json"))
    files = {p.relative_to(root).as_posix(): _sha(p) for p in root.rglob("*")
             if p.is_file() and p != root / "publication-index.json"}
    if files != index["files"]:
        raise ValueError("scheduling archive inventory or bytes differ")
    screen = _read(root, "screen.json")
    validate(screen, _read(root, "screen.schema.json"))
    if index["provenance"]["screen_sha256"] != _sha(root / "screen.json"):
        raise ValueError("scheduling screen binding differs")
    provenance = screen["provenance"]
    if (provenance["source_manifest_sha256"] != _sha(source_archive / "source-manifest.json")
            or provenance["source_capsule_sha256"] != _sha(source_archive / "sources.tar.gz")
            or provenance["publisher_sha256"] != _sha(root / "publisher.py.txt")
            or provenance["auditor_sha256"] != _sha(root / "audit.py.txt")):
        raise ValueError("scheduling source/provenance binding differs")
    manifest = _read(source_archive, "source-manifest.json")
    sources = {}
    with tarfile.open(source_archive / "sources.tar.gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not member.name.endswith(".py"):
                continue
            if not member.name.startswith("src/calibrex/") or ".." in Path(member.name).parts:
                raise ValueError("unexpected scheduling runtime source")
            stream = bundle.extractfile(member)
            assert stream is not None
            sources[member.name.removeprefix("src/")] = hashlib.sha256(stream.read()).hexdigest()
    if sources != manifest:
        raise ValueError("scheduling runtime capsule differs")
    if [c["name"] for c in screen["cohorts"]] != list(_COHORTS):
        raise ValueError("scheduling cohort set or order differs")
    observations, comparisons = 0, []
    registration_inputs = None
    for cohort in screen["cohorts"]:
        name = cohort["name"]
        variants, repeats, cases, scope = _COHORTS[name]
        if (cohort["variants"], cohort["repeats"], cohort["cases"], cohort["scope"]) != (
            variants, repeats, cases, scope,
        ):
            raise ValueError("scheduling timing/sampling contract differs")
        folder = root / name
        protocol = _read(folder, "protocol.json")
        if (protocol["launcher_sha256"] != _sha(folder / "launcher.py.txt")
                or protocol["blas_threads"] != 1
                or protocol["timing_scope"] != _TIMING_SCOPES[name]):
            raise ValueError("scheduling launcher/protocol differs")
        if name == "traversal":
            if any(_sha(folder / f"{kind}_{variant}.cu.txt") != digest
                   for variant, kinds in protocol["kernels"].items()
                   for kind, digest in kinds.items()):
                raise ValueError("stackless traversal kernel binding differs")
            if _sha(folder / "prepare.cu.txt") != protocol["preparation_kernel_sha256"]:
                raise ValueError("stackless navigation preparation differs")
        if name == "bounds":
            if protocol["source_sha256"] != sources["calibrex/solvers/scipy_spatial_snapshot.py"]:
                raise ValueError("spatial bounds runtime differs")
            if any(_sha(folder / (k + ".py.txt")) != v
                   for k, v in protocol["candidates"].items()):
                raise ValueError("spatial bounds candidate source differs")
        elif protocol["sources"] != {"src/" + k: v for k, v in sources.items()}:
            raise ValueError("scheduling runtime source differs")
        if (name in ("traversal", "normals")
                and protocol["source_commit"] != screen["source_commit"]):
            raise ValueError("scheduling source commit differs")
        fixtures = protocol["fixtures"]
        if name != "placement":
            if registration_inputs is None:
                registration_inputs = fixtures
            elif fixtures != registration_inputs:
                raise ValueError("scheduling registration inputs differ")
        for filename, digest in fixtures.items():
            if filename.endswith(".json"):
                fixture = root / "fixtures" / name / filename
                if _sha(fixture) != digest:
                    raise ValueError("scheduling fixture metadata differs")
                metadata = json.loads(fixture.read_bytes())
                if "fixture_sha256" in metadata and (
                    metadata["fixture_sha256"] != fixtures[filename[:-5] + ".npz"]
                ):
                    raise ValueError("scheduling coordinate input binding differs")
        runs = _read(folder, "runs.json")
        expected = {(i, v, c) for i in range(repeats) for v in variants for c in range(cases)}
        actual = [(r["repeat"], r["variant"], r["case"]) for r in runs]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("scheduling observations missing or duplicated")
        references = {r["case"]: r for r in runs if r["repeat"] == 0 and r["variant"] == "prior"}
        variant_work = {(r["variant"], r["case"]): r.get("work") for r in runs if r["repeat"] == 0}
        for record in runs:
            if not math.isfinite(record["seconds"]) or record["seconds"] <= 0:
                raise ValueError("scheduling timing is not positive and finite")
            reference = references[record["case"]]
            if "result" in record:
                result = record["result"]
                values = [result["rmse_m"], result["hessian_min_eigenvalue"],
                          *(x for row in result["transform"] for x in row)]
                if (not all(math.isfinite(v) for v in values)
                        or _physical(result) != _physical(reference["result"])):
                    raise ValueError("scheduling physical registration fields differ")
                work = record["work"]
                if (any(type(v) is not int or v < 0 for v in work.values())
                        or work != variant_work[record["variant"], record["case"]]
                        or (name == "traversal" and work != reference["work"])
                        or not 0 <= record["initialization_seconds"] <= record["seconds"]):
                    raise ValueError("scheduling execution work differs")
            elif any(record[k] != reference[k] for k in (
                "output_sha256", "nodes" if name == "bounds" else "rows",
            )):
                raise ValueError("scheduling output identities differ")
        medians = {v: statistics.median(sum(
            r["seconds"] for r in runs if r["variant"] == v and r["repeat"] == i
        ) for i in range(repeats)) for v in variants}
        ratios = {v: medians["prior"] / medians[v] for v in variants}
        summary = _read(folder, "summary.json")
        if (summary["median_seconds"] != medians or summary["speedups"] != ratios
                or not summary["exact"] or summary["provenance"]["protocol_sha256"]
                != _sha(folder / "protocol.json")):
            raise ValueError("scheduling timing aggregate differs")
        comparisons.extend({"cohort": name, "variant": v, "speedup": ratios[v]}
                           for v in variants[1:])
        observations += len(runs)
    if comparisons != screen["comparisons"] or observations != screen["observations"]:
        raise ValueError("scheduling screen aggregate differs")
    return {"decision": screen["decision"], "observations": observations,
            "comparisons": comparisons, "files": len(files),
            "provenance": {"publication_index_sha256": _sha(root / "publication-index.json")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?",
                        default=Path("docs/assets/odometry_scheduling_screen"))
    print(json.dumps(audit(parser.parse_args().archive), indent=2))


if __name__ == "__main__":
    main()
