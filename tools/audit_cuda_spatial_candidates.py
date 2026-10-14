"""CPU-only audit of exact-query candidate diagnostics and their source bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import jsonschema
from tools.audit_bspline_lidar_imu_benchmark import _cuda_kernel_hashes
from tools.benchmark_cuda_spatial_candidates import validate_measurement


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path) -> dict[str, Any]:
    """Verify every file, candidate, timing sample and imported runtime source."""
    index_path = root / "publication-index.json"
    index = json.loads(index_path.read_bytes())
    schema = Path(__file__).with_name("schemas") / "cuda_spatial_candidates_publication.schema.json"
    jsonschema.validate(index, json.loads(schema.read_bytes()))
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and p != index_path}
    if actual != set(index["files"]):
        raise ValueError("publication does not bind every file")
    for relative, digest in index["files"].items():
        if _sha(root / relative) != digest:
            raise ValueError(f"archive digest differs: {relative}")
    if _sha(root / "measurement.json") != index["provenance"]["measurement_sha256"]:
        raise ValueError("measurement binding differs")
    record = json.loads((root / "measurement.json").read_bytes())
    jsonschema.validate(record, json.loads((root / "measurement.schema.json").read_bytes()))
    provenance = record["provenance"]
    if _sha(root / "benchmark.py.txt") != provenance["generator_sha256"]:
        raise ValueError("generator binding differs")
    helper = ModuleType("_bound_spatial_candidates")
    helper.__file__ = str(root / "benchmark.py.txt")
    exec(compile((root / "benchmark.py.txt").read_bytes(), helper.__file__, "exec"),
         helper.__dict__)
    validate_measurement(record)
    expected = json.loads((root / "sources-manifest.json").read_bytes())
    source_bytes = {}
    with tarfile.open(root / "sources.tar.gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile() or not member.name.endswith(".py"):
                continue
            if not member.name.startswith("src/calibrex/") or ".." in Path(member.name).parts:
                raise ValueError("unexpected source archive path")
            stream = bundle.extractfile(member)
            assert stream is not None
            source_bytes[member.name.removeprefix("src/")] = stream.read()
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in source_bytes.items()}
    if hashes != expected or hashes != provenance["executed_sources"]:
        raise ValueError("imported source binding differs")
    query, system = _cuda_kernel_hashes(source_bytes)
    if re.fullmatch(
        r"calibrex\.cuda_scan_odometry/v0\.1;fp64=true;query_fma=false;"
        r"cupy=[^;]+;cuda=\d+;driver=\d+;cc=\d+\.\d+;"
        rf"query_sha256={query};system_sha256={system}",
        provenance["baseline_execution_identity"],
    ) is None:
        raise ValueError("baseline kernel identity differs")
    sorted_source = (root / "sorted.cu.txt").read_text(encoding="utf-8")
    heap_source = (root / "heap.cu.txt").read_text(encoding="utf-8")
    if (hashlib.sha256(sorted_source.encode()).hexdigest() != query
            or heap_source != helper._heap_source(sorted_source)):
        raise ValueError("candidate shader differs from its bound definition")
    for method in ("sorted", "heap"):
        if _sha(root / f"{method}.cu.txt") != provenance["shader_sha256"][method]:
            raise ValueError("candidate shader binding differs")
    timings = {(r["leafsize"], r["neighbors"], r["method"]): r["kernel_median_seconds"]
               for r in record["records"]}
    return {"candidates": len(timings), "exact_query_parity": True,
            "heap_17_speedup_at_leaf16": timings[(16, 17, "sorted")] / timings[(16, 17, "heap")],
            "scope": record["scope"], "files": len(index["files"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(audit(parser.parse_args().archive), indent=2))


if __name__ == "__main__":
    main()
