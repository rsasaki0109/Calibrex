"""Audit published complete CPU spline-fit timing and quality without input data."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

from tools.benchmark_bspline_planes import TOLERANCES, VERSION, _compare


def audit(root: Path) -> dict:
    import jsonschema

    index = json.loads((root / "index.json").read_bytes())
    index_schema = Path(__file__).with_name("schemas") / "bspline_plane_fit_publication.schema.json"
    jsonschema.validate(index, json.loads(index_schema.read_bytes()))
    for relative, expected in index["files"].items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError("publication path leaves the archive")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"publication file changed: {relative}")
    schema = json.loads((root / "benchmark.schema.json").read_bytes())
    jsonschema.Draft202012Validator.check_schema(schema)
    records = {}
    for relative in index["files"]:
        if relative in ("protocol.json", "attempts.json", "summary.json") or relative.endswith(
            "/measurement.json"
        ):
            record = json.loads((root / relative).read_bytes())
            jsonschema.validate(record, schema)
            if record["schema_version"] != VERSION:
                raise ValueError("unexpected benchmark schema version")
            records[relative] = record
    protocol = records["protocol.json"]
    protocol_digest = index["files"]["protocol.json"]
    if index["provenance"]["publisher_sha256"] != index["files"]["publish.py"]:
        raise ValueError("publication helper binding differs")
    if (
        protocol["quality_tolerances"] != TOLERANCES
        or index["provenance"]["protocol_sha256"] != protocol_digest
        or index["provenance"]["source_commit"] != protocol["provenance"]["source_commit"]
    ):
        raise ValueError("publication protocol binding differs")
    if protocol["provenance"]["launcher_sha256"] != index["files"]["benchmark.py"]:
        raise ValueError("executed launcher differs from published snapshot")
    source = {}
    with tarfile.open(root / "sources.tar.gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            if member.name in source:
                raise ValueError("duplicate source archive member")
            content = archive.extractfile(member)
            assert content is not None
            source[member.name] = content.read()
    manifest = json.loads(source["manifest.json"])
    if (
        hashlib.sha256(source["manifest.json"]).hexdigest()
        != protocol["provenance"]["source_manifest_sha256"]
    ):
        raise ValueError("frozen source manifest changed")
    actual = {
        key.removeprefix("src/"): hashlib.sha256(value).hexdigest()
        for key, value in source.items()
        if key.startswith("src/") and key.endswith(".py")
    }
    if actual != manifest:
        raise ValueError("frozen source bytes differ")
    attempts = records["attempts.json"]
    if attempts["provenance"]["protocol_sha256"] != protocol_digest:
        raise ValueError("attempts refer to another protocol")
    expected_pairs = {
        (repeat, variant)
        for repeat in range(protocol["repeats"])
        for variant in ("numpy", "numba_cpu")
    }
    actual_pairs = [(item["repeat"], item["variant"]) for item in attempts["attempts"]]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("worker cohort is incomplete or duplicated")
    measurements = {}
    for item in attempts["attempts"]:
        if item["exit_code"] != 0:
            raise ValueError("worker did not complete")
        relative = item["worker"] + "/measurement.json"
        worker = records[relative]
        prov = worker["provenance"]
        if (
            worker["variant"] != item["variant"]
            or prov["protocol_sha256"] != protocol_digest
            or prov["source_manifest_sha256"] != protocol["provenance"]["source_manifest_sha256"]
            or prov["launcher_sha256"] != index["files"]["benchmark.py"]
            or any(manifest.get(key) != value for key, value in prov["executed_sources"].items())
            or set(prov["blas_threads"].values()) != {"1"}
            or [run["phase"] for run in worker["runs"]] != ["first_use", "warm"]
            or any(len(run["results"]) != len(protocol["inputs"]) for run in worker["runs"])
        ):
            raise ValueError("worker execution binding differs")
        for run in worker["runs"]:
            for result in run["results"]:
                identity = result["execution_backend"]
                if (item["variant"] == "numpy" and identity != "numpy") or (
                    item["variant"] == "numba_cpu"
                    and not identity.startswith("calibrex.numba_bspline_planes/v0.1;")
                ):
                    raise ValueError("requested CPU backend was not executed")
        measurements[item["worker"]] = worker
    quality = []
    for repeat in range(protocol["repeats"]):
        a, b = [measurements[f"{repeat:02d}-{variant}"] for variant in ("numpy", "numba_cpu")]
        for left, right in zip(a["runs"], b["runs"], strict=True):
            quality.extend(
                _compare(p, c) for p, c in zip(left["results"], right["results"], strict=True)
            )
    medians = {
        phase: {
            variant: statistics.median(
                worker["runs"][position]["seconds"]
                for worker in measurements.values()
                if worker["variant"] == variant
            )
            for variant in ("numpy", "numba_cpu")
        }
        for position, phase in enumerate(("first_use", "warm"))
    }
    summary = records["summary.json"]
    prov = summary["provenance"]
    bindings = {worker: index["files"][worker + "/measurement.json"] for worker in measurements}
    if (
        summary["quality"] != quality
        or not summary["quality_passed"]
        or not all(item["passed"] for item in quality)
        or summary["medians"] != medians
        or summary["speedups"]
        != {phase: times["numpy"] / times["numba_cpu"] for phase, times in medians.items()}
        or prov["protocol_sha256"] != protocol_digest
        or prov["attempts_sha256"] != index["files"]["attempts.json"]
        or prov["measurements"] != bindings
    ):
        raise ValueError("summary cannot be recomputed from complete workers")
    return {
        "verified_pairs": protocol["repeats"],
        "verified_full_fit_comparisons": len(quality),
        "source_files": len(manifest),
        "speedups": summary["speedups"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", nargs="?", type=Path, default=Path("docs/assets/numba_bspline_planes")
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.archive), indent=2))


if __name__ == "__main__":
    main()
