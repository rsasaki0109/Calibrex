"""Compare complete CLI commands in alternating fresh, source-bound processes.

Use an existing probabilistic execution control, never an accuracy holdout.
The timed commands retain their normal parsing, validation, solving and saving.
Source checks, external result validation and schema export parity are untimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _git_identity(source: Path, commit: str) -> None:
    """Retain normal Git provenance for extracted sources using local objects."""
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
    (source / ".git/objects/info/alternates").write_bytes((objects.as_posix() + "\n").encode())
    subprocess.run(["git", "-C", str(source), "update-ref", "HEAD", commit], check=True)


def comparable_result(result: dict) -> dict:
    """Normalize only run time, source commit and the declared CLI output path."""
    result = json.loads(json.dumps(result))
    for key in ("created_at", "git_commit"):
        result["provenance"].pop(key)
    command = result["provenance"]["command"]
    if command.count("--output") != 1:
        raise ValueError("unexpected result command")
    command[command.index("--output") + 1] = "<OUTPUT>"
    return result


def worker(protocol_path: Path, variant: str, case: str, folder: Path) -> None:
    protocol = json.loads(protocol_path.read_bytes())
    source = protocol_path.parent / ("sources-" + variant)
    if sha(Path(__file__)) != protocol["provenance"]["launcher_sha256"]:
        raise ValueError("launcher changed")
    command = [
        s.replace("<OUTPUT>", str(folder / "result.json")) for s in protocol["commands"][case]
    ]
    from calibrex.cli.main import main

    try:
        status = main(command)
    except SystemExit as exc:
        if not isinstance(exc.code, int):
            raise
        status = exc.code
    # Hash verification occurs in the parent after the timed command. Save
    # the actual import paths, so installed or live-checkout imports fail.
    imports = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.copy().items()
        if name == "calibrex" or name.startswith("calibrex.")
    }
    write(folder / "imports.json", {"source": str(source), "modules": imports})
    raise SystemExit(status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--correspondence", type=Path)
    parser.add_argument("--problem", type=Path)
    parser.add_argument("--input-format", choices=("json", "yaml"), default="json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--worker-protocol", type=Path)
    parser.add_argument("--variant", choices=["prior", "current"])
    parser.add_argument("--case", choices=["help", "validate", "refine"])
    args = parser.parse_args()
    if args.worker_protocol:
        worker(args.worker_protocol, args.variant, args.case, args.output_dir)
        return
    if not all((args.baseline_commit, args.correspondence, args.problem, args.output_dir)):
        parser.error("baseline commit, correspondence, problem and output directory are required")
    if args.repetitions < 2:
        parser.error("at least two repetitions are required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commits = {
        name: subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
        for name, ref in (("prior", args.baseline_commit), ("current", "HEAD"))
    }
    suffix = "." + args.input_format
    origins = {}
    for name in ("correspondence", "problem"):
        origin = getattr(args, name)
        origins[name] = sha(origin)
        if args.input_format == "yaml":
            from calibrex.core.io import read_mapping, write_mapping

            value = read_mapping(origin)
            write_mapping(output / (name + suffix), value)
            if read_mapping(output / (name + suffix)) != value:
                raise ValueError("input conversion changed the mapping")
        else:
            shutil.copyfile(origin, output / (name + suffix))
    inputs = {name: sha(output / (name + suffix)) for name in ("correspondence", "problem")}
    shutil.copyfile(__file__, output / "launcher.py.txt")
    helper = ROOT / "tools/benchmark_bspline_lidar_imu.py"
    shutil.copyfile(helper, output / "freezer.py.txt")
    manifests = {}
    for variant, commit in commits.items():
        frozen = output / ("sources-" + variant)
        _freeze(frozen, commit)
        _git_identity(frozen, commit)
        manifests[variant] = sha(frozen / "manifest.json")
        shutil.copyfile(
            frozen / "manifest.json", output / ("sources-" + variant + "-manifest.json")
        )
        with tarfile.open(output / ("sources-" + variant + ".tar.gz"), "w:gz") as capsule:
            for path in sorted((frozen / "src").rglob("*.py")):
                capsule.add(path, arcname=path.relative_to(frozen).as_posix())
    protocol = {
        "schema_version": "slac.lazy_runtime_benchmark/v0.1",
        "commits": commits,
        "repetitions": args.repetitions,
        "fresh_workers": True,
        "host_unreserved": True,
        "blas_threads": 1,
        "commands": {
            "help": ["--help"],
            "validate": ["validate", str(output / ("correspondence" + suffix)), "--json"],
            "refine": [
                "camera-lidar",
                "refine-probabilistic-multiframe",
                str(output / ("correspondence" + suffix)),
                str(output / ("problem" + suffix)),
                "--output",
                "<OUTPUT>",
                "--json",
            ],
        },
        "provenance": {
            "inputs": inputs,
            "source_manifests": manifests,
            "launcher_sha256": sha(Path(__file__)),
            "freezer_sha256": sha(helper),
            "python": sys.version,
        },
    }
    write(output / "protocol.json", protocol)
    protocol_digest = sha(output / "protocol.json")
    if args.input_format == "yaml":
        import importlib.metadata

        import yaml

        if not getattr(yaml, "__with_libyaml__", False):
            raise RuntimeError("compiled YAML timing requires LibYAML; no worker started")
        write(
            output / "input-transcoding.json",
            {
                "format": "yaml",
                "original_sha256": origins,
                "input_sha256": inputs,
                "pyyaml": yaml.__version__,
                "libyaml": True,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "pydantic", "jsonschema", "PyYAML")
                },
                "provenance": {
                    "launcher_sha256": sha(Path(__file__)),
                    "protocol_sha256": sha(output / "protocol.json"),
                },
            },
        )
    import jsonschema

    jsonschema.validate(
        protocol,
        json.loads((ROOT / "tools/schemas/lazy_runtime_benchmark.schema.json").read_bytes()),
    )
    attempts = []
    for case in protocol["commands"]:
        for repeat in range(args.repetitions):
            order = ("prior", "current") if repeat % 2 == 0 else ("current", "prior")
            for variant in order:
                frozen = output / ("sources-" + variant)
                expected = _verify_sources(frozen)
                if inputs != {name: sha(output / (name + suffix)) for name in inputs}:
                    raise ValueError("benchmark input changed before execution")
                folder = output / f"{case}-{repeat:02d}-{variant}"
                folder.mkdir()
                env = os.environ.copy()
                env.update(
                    dict.fromkeys(
                        ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"), "1"
                    )
                )
                env["PYTHONPATH"] = str(frozen / "src")
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-protocol",
                    str(output / "protocol.json"),
                    "--variant",
                    variant,
                    "--case",
                    case,
                    "--output-dir",
                    str(folder),
                ]
                with (
                    (folder / "stdout.txt").open("w") as stdout,
                    (folder / "stderr.txt").open("w") as stderr,
                ):
                    start = time.perf_counter()
                    completed = subprocess.run(
                        command, env=env, cwd=frozen, stdout=stdout, stderr=stderr, check=False
                    )
                    seconds = time.perf_counter() - start
                attempts.append(
                    {
                        "case": case,
                        "repeat": repeat,
                        "variant": variant,
                        "seconds": seconds,
                        "exit_code": completed.returncode,
                    }
                )
                write(output / "attempts.json", attempts)
                print(json.dumps(attempts[-1]), flush=True)
                if completed.returncode:
                    raise RuntimeError(f"worker failed; preserved {folder}")
                imports = json.loads((folder / "imports.json").read_bytes())
                for name in imports["modules"].values():
                    path = Path(name)
                    if (
                        not path.is_relative_to(frozen / "src")
                        or sha(path) != expected[path.relative_to(frozen / "src").as_posix()]
                    ):
                        raise ValueError("worker used an unfrozen module")
                _verify_sources(frozen)
                if inputs != {name: sha(output / (name + suffix)) for name in inputs}:
                    raise ValueError("benchmark input changed")
                if (
                    sha(Path(__file__)) != protocol["provenance"]["launcher_sha256"]
                    or sha(helper) != protocol["provenance"]["freezer_sha256"]
                    or sha(output / "protocol.json") != protocol_digest
                    or subprocess.check_output(
                        ["git", "-C", str(frozen), "rev-parse", "HEAD"],
                        text=True,
                    ).strip()
                    != commits[variant]
                ):
                    raise ValueError("benchmark source changed")
    # Validate and compare every result, not only selected summary statistics.
    from calibrex.core.validation import validate_file

    summaries = {}
    for case in protocol["commands"]:
        outputs = []
        for record in (r for r in attempts if r["case"] == case):
            folder = output / f"{case}-{record['repeat']:02d}-{record['variant']}"
            stdout = (folder / "stdout.txt").read_text()
            if case == "refine":
                if not validate_file(folder / "result.json").valid:
                    raise ValueError("invalid refinement result")
                result = json.loads((folder / "result.json").read_bytes())
                command = result["provenance"]["command"]
                if command[command.index("--output") + 1] != str(folder / "result.json"):
                    raise ValueError("unexpected output provenance")
                outputs.append(comparable_result(result))
                value = json.loads(stdout)
                value["result"] = "<OUTPUT>"
                # The complete result and the CLI display must both agree.
                outputs[-1] = [outputs[-1], value]
            elif case == "validate":
                outputs.append(json.loads(stdout))
            else:
                outputs.append(stdout)
        if any(value != outputs[0] for value in outputs[1:]):
            raise ValueError(f"complete {case} outputs differ")
        medians = {
            v: statistics.median(
                r["seconds"] for r in attempts if r["case"] == case and r["variant"] == v
            )
            for v in commits
        }
        summaries[case] = {
            "median_seconds": medians,
            "speedup": medians["prior"] / medians["current"],
            "exact_outputs": True,
        }
    write(
        output / "summary.json",
        {
            "cases": summaries,
            "workers": len(attempts),
            "provenance": {
                "protocol_sha256": sha(output / "protocol.json"),
                "attempts_sha256": sha(output / "attempts.json"),
            },
        },
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
