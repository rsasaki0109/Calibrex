"""Verify source-bound CLI evidence, complete outputs and all reported timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from jsonschema import validate
from tools.benchmark_lazy_runtime_loading import comparable_result

from calibrex.core.validation import validate_file


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_bytes())


def audit(root: Path, *, expected_runtime_changes: frozenset[str] | None = None) -> dict:
    index = read(root, "index.json")
    actual_files = {
        path.relative_to(root).as_posix(): sha(path)
        for path in root.rglob("*")
        if path.is_file() and path != root / "index.json"
    }
    if actual_files != index["files"]:
        raise ValueError("publication inventory or bytes differ")
    protocol = read(root, "protocol.json")
    validate(protocol, read(root, "protocol.schema.json"))
    if index["provenance"]["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("protocol binding differs")
    for name, field in (
        ("launcher.py.txt", "launcher_sha256"),
        ("freezer.py.txt", "freezer_sha256"),
    ):
        if sha(root / name) != protocol["provenance"][field]:
            raise ValueError("launcher or source freezer differs")
    manifests = {}
    for variant in protocol["commits"]:
        name = "sources-" + variant
        manifest = read(root, name + "-manifest.json")
        if (
            sha(root / (name + "-manifest.json"))
            != protocol["provenance"]["source_manifests"][variant]
        ):
            raise ValueError("source manifest binding differs")
        actual = {}
        with tarfile.open(root / (name + ".tar.gz")) as capsule:
            for member in capsule.getmembers():
                path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or path.parts[:2] != ("src", "calibrex")
                    or path.suffix != ".py"
                    or path.relative_to("src").as_posix() in actual
                ):
                    raise ValueError("unexpected source capsule member")
                stream = capsule.extractfile(member)
                assert stream is not None
                actual[path.relative_to("src").as_posix()] = hashlib.sha256(
                    stream.read()
                ).hexdigest()
        if actual != manifest:
            raise ValueError("capsule contents differ from source manifest")
        manifests[variant] = manifest
    changed = {
        name
        for name in manifests["prior"].keys() | manifests["current"].keys()
        if manifests["prior"].get(name) != manifests["current"].get(name)
    }
    allowed = (
        expected_runtime_changes
        if expected_runtime_changes is not None
        else {
            "calibrex/cli/main.py",
            "calibrex/core/schema_registry.py",
            "calibrex/solvers/__init__.py",
        }
    )
    if changed != allowed:
        raise ValueError("changes extend beyond runtime loading")
    attempts = json.loads((root / "attempts.json").read_bytes())
    expected_order = [
        (case, repeat, variant)
        for case in ("help", "validate", "refine")
        for repeat in range(protocol["repetitions"])
        for variant in (("prior", "current") if repeat % 2 == 0 else ("current", "prior"))
    ]
    if [(r["case"], r["repeat"], r["variant"]) for r in attempts] != expected_order or any(
        r["exit_code"] != 0 or not math.isfinite(r["seconds"]) or r["seconds"] <= 0
        for r in attempts
    ):
        raise ValueError("incomplete, failed or reordered attempts")
    summary = read(root, "summary.json")
    if summary["workers"] != len(attempts) or summary["provenance"] != {
        "protocol_sha256": sha(root / "protocol.json"),
        "attempts_sha256": sha(root / "attempts.json"),
    }:
        raise ValueError("summary provenance differs")
    for case in ("help", "validate", "refine"):
        outputs = []
        records = [r for r in attempts if r["case"] == case]
        for record in records:
            folder = root / f"{case}-{record['repeat']:02d}-{record['variant']}"
            imports = read(folder, "imports.json")
            # Evidence captured on Windows must also be auditable on Linux.
            path_type = (
                PureWindowsPath if PureWindowsPath(imports["source"]).drive else PurePosixPath
            )
            source = path_type(imports["source"]) / "src"
            if source.parent.name != "sources-" + record["variant"]:
                raise ValueError("worker source variant differs")
            manifest = manifests[record["variant"]]
            if "calibrex.cli.main" not in imports["modules"]:
                raise ValueError("worker did not execute the CLI")
            for name in imports["modules"].values():
                relative = path_type(name).relative_to(source).as_posix()
                if relative not in manifest:
                    raise ValueError("worker imported a module outside its capsule")
            if (folder / "stderr.txt").read_text():
                raise ValueError("worker reported unexpected stderr")
            stdout = (folder / "stdout.txt").read_text()
            if case == "refine":
                if not validate_file(folder / "result.json").valid:
                    raise ValueError("invalid result")
                result = read(folder, "result.json")
                command = result["provenance"]["command"]
                expected_command = ["calibrex", *protocol["commands"]["refine"][:-1]]
                expected_command[expected_command.index("--output") + 1] = command[
                    command.index("--output") + 1
                ]
                # CLI explicitly records its split seed in semantic provenance.
                expected_command.extend(["--split-seed", "0"])
                if command != expected_command:
                    raise ValueError("refinement command provenance differs")
                for name, field in (
                    ("correspondence", "correspondence_artifact_sha256"),
                    ("problem", "initialization_problem_sha256"),
                ):
                    if result["provenance"][field] != protocol["provenance"]["inputs"][name]:
                        raise ValueError("result input binding differs")
                display = json.loads(stdout)
                if display["result"] != command[command.index("--output") + 1]:
                    raise ValueError("result display path differs")
                display["result"] = "<OUTPUT>"
                outputs.append([comparable_result(result), display])
            elif case == "validate":
                value = json.loads(stdout)
                if value["valid"] is not True or value["path"] != protocol["commands"]["validate"][
                    1
                ].replace("\\", "/"):
                    raise ValueError("validation command did not validate its input")
                outputs.append(value)
            else:
                outputs.append(stdout)
        if any(value != outputs[0] for value in outputs[1:]):
            raise ValueError("complete CLI outputs differ")
        medians = {
            variant: statistics.median(r["seconds"] for r in records if r["variant"] == variant)
            for variant in protocol["commits"]
        }
        expected = {
            "median_seconds": medians,
            "speedup": medians["prior"] / medians["current"],
            "exact_outputs": True,
        }
        if summary["cases"][case] != expected:
            raise ValueError("reported timing differs from all workers")
    parity = read(root, "schema-parity.json")
    if (
        parity["provenance"]
        != {
            "source_manifests": protocol["provenance"]["source_manifests"],
            "checker_sha256": sha(root / "schema-checker.py.txt"),
        }
        or len(parity["exports"]) != 130
        or any(v["prior"] != v["current"] for v in parity["exports"].values())
    ):
        raise ValueError("schema parity receipt differs")
    receipt = read(root, "validation/receipt.json")
    if (
        not receipt["canonical_runtime_matches_measured_source_after_tests"]
        or receipt["source_commit"] != protocol["commits"]["current"]
        or receipt["source_manifest_sha256"]
        != protocol["provenance"]["source_manifests"]["current"]
        or receipt["provenance"]
        != {
            "pytest_log_sha256": sha(root / "validation/pytest.log"),
            "checker_sha256": sha(root / "validation/checker.py.txt"),
        }
        or f"{receipt['tests_passed']} passed" not in (root / "validation/pytest.log").read_text()
        or any(
            sha(root / "validation" / name) != digest
            for name, digest in receipt["test_sources"].items()
        )
    ):
        raise ValueError("correctness receipt differs")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root), indent=2))
