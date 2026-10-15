"""CPU-only audit of experimental incremental local-map screening evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path
from typing import Any

from jsonschema import validate
from tools.benchmark_bspline_lidar_imu import _compare, _validate_exact_pair


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(root: Path, name: str) -> Any:
    return json.loads((root / name).read_bytes())


def audit(root: Path) -> dict[str, Any]:
    """Check byte/source bindings, all observations, and complete fit equality."""
    index = _read(root, "publication-index.json")
    validate(index, _read(root, "publication.schema.json"))
    files = {p.relative_to(root).as_posix(): _sha(p) for p in root.rglob("*")
             if p.is_file() and p.name != "publication-index.json"}
    if files != index["files"]:
        raise ValueError("local-map publication inventory or bytes differ")
    protocol = _read(root, "whole/protocol.json")
    if index["provenance"]["input_protocol_sha256"] != _sha(root / "whole/protocol.json"):
        raise ValueError("local-map publication protocol differs")
    if (protocol["repeats"] != 2 or protocol["max_seconds"] != 6.
            or protocol["max_rounds"] != 4 or protocol["blas_threads"] != 1
            or not all(protocol[k] for k in (
                "fresh_workers", "empty_separate_cuda_caches", "host_unreserved",
            )) or protocol["fit_plane_backend"] != "cuda"
            or protocol["launcher_sha256"] != _sha(root / "whole/launcher.py.txt")
            or protocol["candidate_sha256"] != _sha(root / "whole/candidate.py.txt")
            or protocol["source_manifest_sha256"] != _sha(root / "source-manifest.json")):
        raise ValueError("local-map complete-worker scope or source differs")
    manifest = _read(root, "source-manifest.json")
    with tarfile.open(root / "sources.tar.gz") as bundle:
        sources = {}
        for member in bundle.getmembers():
            if not member.isfile() or not member.name.endswith(".py"):
                continue
            if not member.name.startswith("src/calibrex/") or ".." in Path(member.name).parts:
                raise ValueError("unexpected local-map capsule source")
            stream = bundle.extractfile(member)
            assert stream is not None
            sources[member.name.removeprefix("src/")] = hashlib.sha256(stream.read()).hexdigest()
    if sources != manifest:
        raise ValueError("local-map runtime capsule differs")
    observations = 0
    ratios = {}
    common_outputs = None
    for cohort, variants, first in (
        ("cpu", ["prior", "cached_keys", "cached_unique"], 0),
        ("gpu", ["prior", "hash", "hash_cached"], 1),
    ):
        folder = root / cohort
        scope = _read(folder, "protocol.json")
        if (scope["source_sha256"] != sources["calibrex/solvers/scan_to_scan_odometry.py"]
                or scope["launcher_sha256"] != _sha(folder / "launcher.py.txt")
                or scope["blas_threads"] != 1):
            raise ValueError("local-map pilot launcher/runtime differs")
        inputs = _read(root, "inputs.json")
        if len(inputs["windows"]) != 59:
            raise ValueError("local-map input window count differs")
        if cohort == "gpu" and (
            scope["inputs_sha256"] != _sha(root / "inputs.json")
            or scope["inputs_npz_sha256"] != inputs["npz_sha256"]
        ):
            raise ValueError("local-map pilot captured inputs differ")
        if cohort == "cpu" and scope["bag_sha256"] != protocol["bag_sha256"]:
            raise ValueError("local-map pilot public input differs")
        if cohort == "gpu":
            parsed = ast.parse((folder / "launcher.py.txt").read_bytes())
            kernel = next(ast.literal_eval(n.value) for n in parsed.body
                          if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "SOURCE"
                                  for t in n.targets))
            if hashlib.sha256(kernel.encode()).hexdigest() != scope["kernel_sha256"]:
                raise ValueError("local-map pilot kernel identity differs")
        runs = _read(folder, "runs.json")
        expected = {(i, v, c) for i in range(7) for v in variants for c in range(59)}
        actual = [(r["repeat"], r["variant"], r["case"]) for r in runs]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("local-map observations missing or duplicated")
        outputs = {r["case"]: (r["rows"], r["output_sha256"]) for r in runs
                   if r["variant"] == "prior" and r["repeat"] == 0}
        if common_outputs is None:
            common_outputs = outputs
        elif outputs != common_outputs:
            raise ValueError("local-map cohorts use different reference outputs")
        for record in runs:
            if (outputs[record["case"]] != (record["rows"], record["output_sha256"])
                    or not math.isfinite(record["seconds"]) or record["seconds"] <= 0
                    or record.get("fallbacks", 0) != 0):
                raise ValueError("local-map output identity, timing or work differs")
        summary = _read(folder, "summary.json")
        medians = {v: statistics.median(sum(
            r["seconds"] for r in runs if r["repeat"] == i and r["variant"] == v
        ) for i in range(first, 7)) for v in variants}
        speedups = {v: medians["prior"] / medians[v] for v in variants[1:]}
        if (summary["medians" if cohort == "cpu" else "warm_medians"] != medians
                or summary["speedups"] != speedups or not summary["exact"]
                or summary["provenance"]["protocol_sha256"] != _sha(folder / "protocol.json")):
            raise ValueError("local-map pilot aggregate differs")
        if cohort == "cpu" and (
            summary["provenance"]["inputs_sha256"] != _sha(root / "inputs.json")
            or scope["source_commit"] != protocol["source_commit"]
        ):
            raise ValueError("local-map pilot capture provenance differs")
        observations += len(runs)
        ratios[cohort] = speedups
    attempts = _read(root, "whole/attempts.json")
    if [(a["repeat"], a["variant"]) for a in attempts] != [
        (0, "prior"), (0, "current"), (1, "current"), (1, "prior"),
    ] or any(a["exit_code"] != 0 or not math.isfinite(a["whole_seconds"])
             or a["whole_seconds"] <= 0 for a in attempts):
        raise ValueError("local-map complete-worker cohort differs")
    quality = []
    for repeat in range(2):
        results, measurements = [], []
        for variant in ("prior", "current"):
            folder = root / "whole" / f"{repeat:02d}-{variant}"
            result = _read(folder, "result.json")
            measurement = _read(folder, "measurement.json")
            validate(measurement, _read(root, "comparison.schema.json"))
            provenance = measurement["provenance"]
            if (provenance["bag_sha256"] != protocol["bag_sha256"]
                    or provenance["source_manifest_sha256"]
                    != protocol["source_manifest_sha256"]
                    or provenance["launcher_sha256"] != _sha(root / "benchmark.py.txt")
                    or result["provenance"]["git_commit"] != protocol["source_commit"]
                    or any(sources.get(k) != v for k, v in provenance["executed_sources"].items())):
                raise ValueError("local-map executed runtime or input differs")
            work = _read(folder, "candidate-work.json")
            if (work["variant"] != variant
                    or work["provenance"]["protocol_sha256"] != _sha(root / "whole/protocol.json")
                    or work["provenance"]["candidate_sha256"] != protocol["candidate_sha256"]
                    or work["work"]["calls"] != (448 if variant == "current" else 0)
                    or work["work"]["input_rows"] != (25427273 if variant == "current" else 0)
                    or work["work"]["fallbacks"] != 0):
                raise ValueError("local-map candidate execution work differs")
            results.append(result)
            measurements.append(measurement)
        _validate_exact_pair(*results, *[m["rounds"] for m in measurements])
        quality.append(_compare(*results))
    summary = _read(root, "whole/summary.json")
    medians = {v: statistics.median(a["whole_seconds"] for a in attempts if a["variant"] == v)
               for v in ("prior", "current")}
    if (summary["whole_median_seconds"] != medians
            or summary["whole_speedup"] != medians["prior"] / medians["current"]
            or summary["quality"] != quality or not all(q["passed"] for q in quality)
            or not summary["exact_full_results_and_fits"]
            or summary["provenance"]["protocol_sha256"] != _sha(root / "whole/protocol.json")
            or summary["provenance"]["attempts_sha256"] != _sha(root / "whole/attempts.json")):
        raise ValueError("local-map complete-worker aggregate differs")
    return {"observations": observations, "fresh_workers": len(attempts),
            "component_speedups": ratios, "whole_speedup": summary["whole_speedup"],
            "exact_full_results_and_fits": True,
            "provenance": {"publication_index_sha256": _sha(root / "publication-index.json")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?",
                        default=Path("docs/assets/incremental_local_map_screen"))
    print(json.dumps(audit(parser.parse_args().archive), indent=2))


if __name__ == "__main__":
    main()
