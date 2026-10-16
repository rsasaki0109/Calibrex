"""Compare complete deterministic uncertainty CLI runs in frozen fresh processes.

Uses the existing public-KITTI-geometry execution control, whose image
correspondences are generated. This measures performance and numerical parity,
not uncertainty coverage or calibration accuracy on real camera observations.
"""

from __future__ import annotations

import argparse
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
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calibrex.evaluation.empirical_se3_uncertainty import _RefitContext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_bspline_lidar_imu import _freeze, _verify_sources  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def import_receipt() -> dict[str, str]:
    return {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.copy().items()
        if name == "calibrex" or name.startswith("calibrex.")
    }


def observed_initialize(context: _RefitContext) -> None:
    """Observe each spawned process without changing the refit or its inputs."""
    from calibrex.evaluation import empirical_se3_uncertainty as uncertainty

    uncertainty._initialize_refits(context)
    write(context.output_root.parent / f"process-{os.getpid()}-imports.json", import_receipt())


def worker(protocol_path: Path, variant: str, folder: Path) -> None:
    protocol = json.loads(protocol_path.read_bytes())
    if sha(Path(__file__)) != protocol["provenance"]["launcher_sha256"]:
        raise ValueError("launcher changed")
    if variant == "current":
        from calibrex.evaluation import empirical_se3_uncertainty as uncertainty

        uncertainty._initialize_refits = observed_initialize
    from calibrex.cli.main import main

    command = [
        "camera-lidar",
        "empirical-uncertainty",
        str(protocol_path.parent / "correspondence.json"),
        str(protocol_path.parent / "problem.json"),
        "--result-dir",
        str(folder / "resamples"),
        "--output",
        str(folder / "uncertainty.yaml"),
        "--block-length",
        "3",
        "--resample-count",
        "20",
        "--seed",
        "0",
        "--stability-only",
        "--json",
    ]
    if variant == "current":
        command.extend(["--workers", str(protocol["workers"][variant])])
    status = main(command)
    write(folder / "imports.json", import_receipt())
    raise SystemExit(status)


def comparable(value: dict) -> dict:
    """Normalize only execution metadata; bound child hashes are checked first."""
    value = json.loads(json.dumps(value))
    for field in ("created_at", "git_commit"):
        value["provenance"][field] = "<" + field + ">"
    command = value["provenance"]["command"]
    for option in ("--result-dir", "--output"):
        if option in command:
            command[command.index(option) + 1] = "<" + option + ">"
    if "--workers" in command:
        position = command.index("--workers")
        del command[position : position + 2]
    if "iterations" in value:
        for item in value["iterations"]:
            item["output_sha256"] = "<BOUND_CHILD_HASH>"
        value["provenance"]["resample_output_sha256"] = {}
    return value


def _read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_bytes())


def _verify_imports(paths: dict, source: str, manifest: dict, *, frozen: bool) -> None:
    path_type = PureWindowsPath if PureWindowsPath(source).drive else PurePosixPath
    base = path_type(source) / "src"
    if "calibrex.solvers.probabilistic_camera_lidar_refiner" not in paths:
        raise ValueError("worker did not import the numerical refiner")
    for filename in paths.values():
        relative = path_type(filename).relative_to(base).as_posix()
        if relative not in manifest or (frozen and sha(Path(filename)) != manifest[relative]):
            raise ValueError("worker imported an unfrozen module")


def summarize(root: Path, *, frozen: bool = False) -> dict:
    from jsonschema import validate

    from calibrex.core.io import read_mapping
    from calibrex.core.validation import validate_file

    protocol = _read(root, "protocol.json")
    validate(protocol, _read(root, "protocol.schema.json"))
    records = json.loads((root / "attempts.json").read_bytes())
    expected = [
        (repeat, variant)
        for repeat in range(protocol["repetitions"])
        for variant in (("prior", "current") if repeat % 2 == 0 else ("current", "prior"))
    ]
    if [(r["repeat"], r["variant"]) for r in records] != expected or any(
        r["exit_code"] != 0 or not math.isfinite(r["seconds"]) or r["seconds"] <= 0 for r in records
    ):
        raise ValueError("missing, failed or reordered attempts")
    reference = None
    evaluations = []
    for record in records:
        variant = record["variant"]
        folder = root / f"{record['repeat']:02d}-{variant}"
        source = protocol["sources"][variant]
        manifest = _read(root, "sources-" + variant + "-manifest.json")
        _verify_imports(_read(folder, "imports.json"), source, manifest, frozen=frozen)
        children = sorted(folder.glob("process-*-imports.json"))
        if len(children) != (protocol["workers"][variant] if variant == "current" else 0):
            raise ValueError("spawned process receipt count differs")
        for child in children:
            _verify_imports(_read(folder, child.name), source, manifest, frozen=frozen)
        if (folder / "stderr.txt").read_text():
            raise ValueError("unexpected worker stderr")
        path = folder / "uncertainty.yaml"
        if not validate_file(path).valid:
            raise ValueError("schema-invalid uncertainty")
        result = read_mapping(path)
        if (
            result["resample_count"] != 20
            or result["block_length"] != 3
            or result["policy_status"] != "inconclusive"
            or result["reference_transform"] is not None
        ):
            raise ValueError("changed refit workload or accuracy claim")
        values = {}
        count = 0
        ids = [f"resample-{i:04d}" for i in range(20)]
        if [item["iteration_id"] for item in result["iterations"]] != ids:
            raise ValueError("refit iteration order differs")
        if {p.stem for p in (folder / "resamples").glob("*.yaml")} != set(ids):
            raise ValueError("missing or extra refit files")
        for iteration in result["iterations"]:
            name = iteration["iteration_id"]
            child = folder / "resamples" / (name + ".yaml")
            if (
                sha(child) != iteration["output_sha256"]
                or sha(child) != result["provenance"]["resample_output_sha256"][name]
            ):
                raise ValueError("child output hash binding differs")
            if not validate_file(child).valid:
                raise ValueError("schema-invalid refit")
            value = read_mapping(child)
            p = value["provenance"]
            if (
                not isinstance(p["git_commit"], str)
                or len(p["git_commit"]) < 7
                or not protocol["commits"][variant].startswith(p["git_commit"])
                or p["correspondence_artifact_sha256"]
                != protocol["provenance"]["inputs"]["correspondence.json"]
                or p["initialization_problem_sha256"]
                != protocol["provenance"]["inputs"]["problem.json"]
            ):
                raise ValueError("refit input/source binding differs")
            if value["options"]["max_evaluations"] != 400:
                raise ValueError("refit evaluation budget differs")
            count += len(value["trace"])
            values[name] = comparable(value)
        source_commit = result["provenance"]["git_commit"]
        if (
            not isinstance(source_commit, str)
            or len(source_commit) < 7
            or not protocol["commits"][variant].startswith(source_commit)
        ):
            raise ValueError("uncertainty source metadata differs")
        values["uncertainty"] = comparable(result)
        if reference is None:
            reference = values
        if values != reference:
            raise ValueError("complete refit or uncertainty result differs")
        evaluations.append(count)
    medians = {
        v: statistics.median(r["seconds"] for r in records if r["variant"] == v)
        for v in ("prior", "current")
    }
    return {
        "workers": len(records),
        "refits_per_worker": 20,
        "evaluations_per_worker": evaluations,
        "complete_results_exact": True,
        "median_seconds": medians,
        "speedup": medians["prior"] / medians["current"],
        "provenance": {
            "protocol_sha256": sha(root / "protocol.json"),
            "attempts_sha256": sha(root / "attempts.json"),
        },
    }


def audit(root: Path) -> dict:
    """Audit all retained output bytes, source capsules, imports and timings on CPU."""
    index = _read(root, "index.json")
    actual = {
        p.relative_to(root).as_posix(): sha(p)
        for p in root.rglob("*")
        if p.is_file() and p != root / "index.json"
    }
    if actual != index["files"]:
        raise ValueError("archive inventory or bytes differ")
    protocol = _read(root, "protocol.json")
    for name, field in (
        ("launcher.py.txt", "launcher_sha256"),
        ("freezer.py.txt", "freezer_sha256"),
        ("protocol.schema.json", "schema_sha256"),
    ):
        if sha(root / name) != protocol["provenance"][field]:
            raise ValueError("launcher, freezer or schema binding differs")
    manifests = {}
    for variant in ("prior", "current"):
        name = "sources-" + variant
        manifest = _read(root, name + "-manifest.json")
        if sha(root / (name + "-manifest.json")) != protocol["provenance"]["manifests"][variant]:
            raise ValueError("source manifest binding differs")
        files = {}
        with tarfile.open(root / (name + ".tar.gz")) as capsule:
            for member in capsule.getmembers():
                path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or path.is_absolute()
                    or ".." in path.parts
                    or path.parts[:2] != ("src", "calibrex")
                    or path.suffix != ".py"
                    or path.relative_to("src").as_posix() in files
                ):
                    raise ValueError("unexpected source capsule member")
                stream = capsule.extractfile(member)
                assert stream is not None
                files[path.relative_to("src").as_posix()] = hashlib.sha256(
                    stream.read()
                ).hexdigest()
        if files != manifest:
            raise ValueError("source capsule differs")
        manifests[variant] = manifest
    changed = {
        name
        for name in manifests["prior"].keys() | manifests["current"].keys()
        if manifests["prior"].get(name) != manifests["current"].get(name)
    }
    if changed != {
        "calibrex/cli/main.py",
        "calibrex/evaluation/empirical_se3_uncertainty.py",
        "calibrex/solvers/probabilistic_camera_lidar_refiner.py",
    }:
        raise ValueError("runtime changes extend beyond refit acceleration")
    if (
        sha(root / "fixture-protocol.json")
        != protocol["provenance"]["inputs"]["fixture-protocol.json"]
    ):
        raise ValueError("fixture provenance differs")
    summary = summarize(root)
    if summary != _read(root, "summary.json"):
        raise ValueError("reported summary differs")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-protocol", type=Path)
    parser.add_argument("--variant", choices=("prior", "current"))
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.worker_protocol:
        worker(args.worker_protocol, args.variant, args.output_dir)
        return
    if args.audit:
        print(json.dumps(audit(args.audit), indent=2))
        return
    if (
        not all((args.baseline_commit, args.fixture_dir, args.output_dir))
        or args.repetitions < 2
        or not 2 <= args.workers <= 20
    ):
        parser.error(
            "baseline, fixture, output, at least two pairs and 2-20 processes are required"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for name in ("correspondence.json", "problem.json", "fixture-protocol.json"):
        shutil.copyfile(args.fixture_dir / name, output / name)
        inputs[name] = sha(output / name)
    fixture = _read(output, "fixture-protocol.json")
    if (
        fixture["correspondence_sha256"] != inputs["correspondence.json"]
        or fixture["problem_sha256"] != inputs["problem.json"]
    ):
        raise ValueError("fixture input binding differs")
    commits = {
        v: subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
        for v, ref in (("prior", args.baseline_commit), ("current", "HEAD"))
    }
    manifests, sources = {}, {}
    for variant, commit in commits.items():
        source = output / ("sources-" + variant)
        _freeze(source, commit)
        # Give each capsule a real Git identity. Use the existing local object
        # store read-only, so normal CLI provenance (and its Git subprocess
        # cost) is retained rather than patched during the timing.
        subprocess.run(
            [
                "git",
                "-c",
                "init.templateDir=",
                "-c",
                "init.defaultBranch=benchmark",
                "init",
                "--bare",
                "--quiet",
                str(source / ".git"),
            ],
            check=True,
        )
        common = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--git-common-dir"],
                cwd=ROOT,
                text=True,
            ).strip()
        )
        objects = (ROOT / common).resolve() / "objects"
        if not objects.is_dir():
            raise ValueError("local Git object store is unavailable")
        (source / ".git/objects/info/alternates").write_bytes((objects.as_posix() + "\n").encode())
        subprocess.run(["git", "-C", str(source), "update-ref", "HEAD", commit], check=True)
        sources[variant] = str(source)
        manifests[variant] = sha(source / "manifest.json")
        shutil.copyfile(source / "manifest.json", output / (source.name + "-manifest.json"))
        with tarfile.open(output / (source.name + ".tar.gz"), "w:gz") as capsule:
            for path in sorted((source / "src").rglob("*.py")):
                capsule.add(path, arcname=path.relative_to(source).as_posix())
    helper = ROOT / "tools/benchmark_bspline_lidar_imu.py"
    schema = ROOT / "tools/schemas/empirical_refit_benchmark.schema.json"
    for path, name in (
        (Path(__file__), "launcher.py.txt"),
        (helper, "freezer.py.txt"),
        (schema, "protocol.schema.json"),
    ):
        shutil.copyfile(path, output / name)
    protocol = {
        "schema_version": "slac.empirical_refit_benchmark/v0.1",
        "commits": commits,
        "repetitions": args.repetitions,
        "fresh_workers": True,
        "host_unreserved": True,
        "blas_threads": 1,
        "sources": sources,
        "workers": {"prior": 1, "current": args.workers},
        "provenance": {
            "inputs": inputs,
            "manifests": manifests,
            "launcher_sha256": sha(Path(__file__)),
            "freezer_sha256": sha(helper),
            "schema_sha256": sha(schema),
            "python": sys.version,
            "versions": {
                n: importlib.metadata.version(n) for n in ("numpy", "scipy", "pydantic", "PyYAML")
            },
        },
    }
    write(output / "protocol.json", protocol)
    from jsonschema import validate

    validate(protocol, json.loads(schema.read_bytes()))
    protocol_digest = sha(output / "protocol.json")
    records = []

    def verify(variant):
        _verify_sources(Path(sources[variant]))
        if (
            subprocess.check_output(
                ["git", "-C", sources[variant], "rev-parse", "HEAD"],
                text=True,
            ).strip()
            != commits[variant]
        ):
            raise ValueError("capsule Git identity changed")
        if (
            sha(output / "protocol.json") != protocol_digest
            or sha(Path(__file__)) != protocol["provenance"]["launcher_sha256"]
            or sha(helper) != protocol["provenance"]["freezer_sha256"]
            or sha(schema) != protocol["provenance"]["schema_sha256"]
            or inputs != {name: sha(output / name) for name in inputs}
        ):
            raise ValueError("frozen benchmark source or input changed")

    for repeat in range(args.repetitions):
        for variant in ("prior", "current") if repeat % 2 == 0 else ("current", "prior"):
            verify(variant)
            folder = output / f"{repeat:02d}-{variant}"
            folder.mkdir()
            env = os.environ.copy()
            env.update(
                dict.fromkeys(("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"), "1")
            )
            env.update(PYTHONPATH=str(Path(sources[variant]) / "src"), PYTHONDONTWRITEBYTECODE="1")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-protocol",
                str(output / "protocol.json"),
                "--variant",
                variant,
                "--output-dir",
                str(folder),
            ]
            with (
                (folder / "stdout.txt").open("w") as stdout,
                (folder / "stderr.txt").open("w") as stderr,
            ):
                start = time.perf_counter()
                done = subprocess.run(
                    command,
                    cwd=sources[variant],
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
                seconds = time.perf_counter() - start
            records.append(
                {
                    "repeat": repeat,
                    "variant": variant,
                    "seconds": seconds,
                    "exit_code": done.returncode,
                }
            )
            write(output / "attempts.json", records)
            print(json.dumps(records[-1]), flush=True)
            if done.returncode:
                raise RuntimeError(f"worker failed; every attempt retained at {folder}")
            verify(variant)
    write(output / "summary.json", summarize(output, frozen=True))
    print((output / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
