"""CPU-only integrity, complete-result and timing audit for pose batching."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

from jsonschema import validate

from calibrex.core.validation import validate_file


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_bytes())


def comparable(value: dict) -> dict:
    value = json.loads(json.dumps(value))
    for key in ("created_at", "git_commit"):
        value["provenance"].pop(key)
    return value


def audit(root: Path) -> dict:
    """Verify all saved workers, source capsules, decisions and trace values."""
    index = read(root, "publication-index.json")
    validate(index, read(root, "publication.schema.json"))
    files = {
        p.relative_to(root).as_posix(): sha(p)
        for p in root.rglob("*")
        if p.is_file() and p != root / "publication-index.json"
    }
    if files != index["files"]:
        raise ValueError("publication inventory or bytes differ")
    protocol, summary = read(root, "protocol.json"), read(root, "summary.json")
    attempts = json.loads((root / "attempts.json").read_bytes())
    schema = read(root, "comparison.schema.json")
    for record in (protocol, summary, attempts):
        validate(record, schema)
    if sha(root / "protocol.json") != index["provenance"]["input_protocol_sha256"] or summary[
        "provenance"
    ] != {
        "protocol_sha256": sha(root / "protocol.json"),
        "attempts_sha256": sha(root / "attempts.json"),
    }:
        raise ValueError("comparison protocol binding differs")
    if (
        sha(root / "benchmark.py.txt") != protocol["launcher_sha256"]
        or sha(root / "source-freezer.py.txt") != protocol["source_freezer_sha256"]
    ):
        raise ValueError("benchmark launcher or freezer binding differs")
    evaluator_sources = {}
    for variant in ("prior", "current"):
        manifest = read(root, "sources-" + variant + "-manifest.json")
        if (
            sha(root / ("sources-" + variant + "-manifest.json"))
            != protocol["source_manifests"][variant]
        ):
            raise ValueError("source manifest binding differs")
        actual = {}
        with tarfile.open(root / ("sources-" + variant + ".tar.gz")) as capsule:
            for member in capsule.getmembers():
                name = Path(member.name)
                if (
                    not member.isfile()
                    or ".." in name.parts
                    or name.is_absolute()
                    or name.parts[:2] != ("src", "calibrex")
                    or name.suffix != ".py"
                ):
                    raise ValueError("unexpected source capsule entry")
                data = capsule.extractfile(member)
                assert data is not None
                contents = data.read()
                actual[name.relative_to("src").as_posix()] = hashlib.sha256(contents).hexdigest()
                if name.name == "probabilistic_camera_lidar_refiner.py":
                    evaluator_sources[variant] = ast.parse(contents)
        if actual != manifest:
            raise ValueError("source capsule differs from its manifest")
    prior_function = next(
        n
        for n in evaluator_sources["prior"].body
        if isinstance(n, ast.FunctionDef) and n.name == "evaluate_probabilistic_camera_lidar_pose"
    )
    scalar_function = next(
        n
        for n in evaluator_sources["current"].body
        if isinstance(n, ast.FunctionDef) and n.name == "_evaluate_scalar_pose"
    )
    if [ast.dump(n) for n in prior_function.body[2:]] != [
        ast.dump(n) for n in scalar_function.body[1:]
    ]:
        raise ValueError("original scalar fallback arithmetic differs")
    receipt = read(root, "validation/receipt.json")
    parity = read(root, "schema-parity-receipt.json")
    if (
        not parity["all_schema_exports_identical"]
        or len(parity["exports"]) != 130
        or parity["provenance"]["protocol_sha256"] != sha(root / "protocol.json")
        or parity["provenance"]["checker_sha256"] != sha(root / "schema-checker.py.txt")
    ):
        raise ValueError("schema parity receipt binding differs")
    if (
        not receipt["canonical_runtime_matches_measured_source_after_tests"]
        or receipt["source_commit"] != protocol["commits"]["current"]
        or receipt["source_manifest_sha256"] != protocol["source_manifests"]["current"]
        or receipt["pytest_log_sha256"] != sha(root / "validation/pytest.log")
        or f"{receipt['tests_passed']} passed" not in (root / "validation/pytest.log").read_text()
        or any(
            sha(root / "validation" / name) != digest
            for name, digest in receipt["test_sources"].items()
        )
    ):
        raise ValueError("correctness receipt or test source differs")
    expected_order = [
        (i, v)
        for i in range(protocol["repetitions"])
        for v in (("prior", "current") if i % 2 == 0 else ("current", "prior"))
    ]
    if [(r["repeat"], r["variant"]) for r in attempts] != expected_order or any(
        r["exit_code"] != 0 for r in attempts
    ):
        raise ValueError("incomplete or non-alternating worker attempts")
    api = {"prior": [], "current": []}
    for repeat in range(protocol["repetitions"]):
        results = []
        for variant in ("prior", "current"):
            folder = f"{repeat:02d}-{variant}"
            result, record = (
                read(root, folder + "/result.json"),
                read(root, folder + "/measurement.json"),
            )
            validate(record, schema)
            provenance = record["provenance"]
            if (
                provenance["source_commit"] != protocol["commits"][variant]
                or provenance["source_manifest_sha256"] != protocol["source_manifests"][variant]
                or provenance["launcher_sha256"] != protocol["launcher_sha256"]
                or provenance["result_sha256"] != sha(root / folder / "result.json")
                or result["provenance"]["git_commit"] != protocol["commits"][variant]
            ):
                raise ValueError("worker execution provenance differs")
            for name, key in (
                ("correspondence", "correspondence_artifact_sha256"),
                ("problem", "initialization_problem_sha256"),
            ):
                if result["provenance"][key] != protocol["inputs"][name]:
                    raise ValueError("refinement input binding differs")
            if (
                not validate_file(root / folder / "result.json").valid
                or len(result["trace"]) != record["trace_evaluations"]
                or result["options"]["max_evaluations"] != protocol["max_evaluations"]
            ):
                raise ValueError("worker result or evaluation count differs")
            results.append(comparable(result))
            api[variant].append(record["api_seconds"])
        if results[0] != results[1]:
            raise ValueError("complete refinement metrics, traces or decisions differ")
    whole = {
        v: statistics.median(r["whole_seconds"] for r in attempts if r["variant"] == v) for v in api
    }
    api_median = {v: statistics.median(times) for v, times in api.items()}
    if (
        summary["workers"] != len(attempts)
        or summary["whole_median_seconds"] != whole
        or summary["api_median_seconds"] != api_median
        or summary["whole_speedup"] != whole["prior"] / whole["current"]
        or summary["api_speedup"] != api_median["prior"] / api_median["current"]
    ):
        raise ValueError("published timing ratios differ from all workers")
    failed = root / "failed-validation-v1"
    if failed.exists():
        failed_protocol = read(failed, "protocol.json")
        failed_attempts = json.loads((failed / "attempts.json").read_bytes())
        validate(failed_protocol, schema)
        validate(failed_attempts, schema)
        if (
            len(failed_attempts) != 1
            or failed_attempts[0]["exit_code"] == 0
            or failed_protocol["source_manifests"] != protocol["source_manifests"]
            or failed_protocol["inputs"] != protocol["inputs"]
            or sha(failed / "benchmark.py.txt") != failed_protocol["launcher_sha256"]
            or comparable(read(failed, "00-prior/result.json"))
            != comparable(read(root, "00-prior/result.json"))
        ):
            raise ValueError("failed validation attempt binding differs")
    result = {
        "workers": len(attempts),
        "exact_complete_results": True,
        "whole_speedup": summary["whole_speedup"],
        "api_speedup": summary["api_speedup"],
    }
    if (root / "earlier-dense-v2").exists():
        result["earlier_dense"] = audit(root / "earlier-dense-v2")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root), indent=2))
