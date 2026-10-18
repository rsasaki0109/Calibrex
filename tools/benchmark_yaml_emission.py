"""Compare complete safe-YAML exports of an unchanged public TUM result.

This measures result export, including fresh interpreter startup and two full
writes, not calibration latency. Both public writers retain the complete result.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
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
from tools.benchmark_lazy_runtime_loading import _git_identity  # noqa: E402
from tools.benchmark_radar_doppler_batches import sources  # noqa: E402

VERSION = "slac.yaml_emission_benchmark/v0.1"
SCHEMA = ROOT / "tools/schemas/yaml_emission_benchmark.schema.json"
INPUT = ROOT / "docs/assets/tum_window_processes/01-current/result.json.gz"
INPUT_SHA256 = "98a0f67ecad130e246c1733274f84ed4669d0ba9c0a53fd1232be78eeb476f66"
METHODS = ("write", "atomic")
SCOPE = (
    "Complete export of the unchanged 5,339,931-byte public TUM development result "
    "through write_mapping and write_mapping_atomic, once each in every fresh worker. "
    "Startup, input loading, both complete writes, atomic fsync and worker evidence "
    "enter whole-worker time; no complete-calibration speed or accuracy claim."
)
PINNED = {
    "input.json.gz", "input-provenance.json", "result-schema.json", "schema.json",
    "launcher.py.txt", "freezer.py.txt", "identity.py.txt", "capsule-auditor.py.txt",
    "sources-prior-manifest.json", "sources-current-manifest.json",
    "sources-prior.tar.gz", "sources-current.tar.gz",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    """Retain mapping/list order, numeric JSON types and signed zero."""
    raw = json.dumps(value, ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def record(kind: str, payload: dict, provenance: dict) -> dict:
    return {"schema_version": VERSION, "record_type": kind,
            "payload": payload, "provenance": provenance}


def write(path: Path, value: object) -> None:
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def _load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def worker(protocol_path: Path, variant: str, folder: Path) -> None:
    import yaml

    from calibrex.core import io

    source = protocol_path.parent / ("sources-" + variant)
    protocol = _load(protocol_path)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != protocol["payload"]["commits"][variant]:
        raise ValueError("worker Git identity differs")
    data = json.loads(gzip.decompress((protocol_path.parent / "input.json.gz").read_bytes()))
    native_calls = []
    original = getattr(io, "_C_DUMP", None)
    if original is not None:
        def native(*args, **kwargs):
            result = original(*args, **kwargs)
            native_calls.append(1)
            return result
        io._C_DUMP = native
    measurements = {}
    for name, emitter in (("write", io.write_mapping), ("atomic", io.write_mapping_atomic)):
        path = folder / (name + ".yaml")
        begin = time.perf_counter()
        emitter(path, data)
        seconds = time.perf_counter() - begin
        measurements[name] = {"seconds": seconds, "sha256": sha(path),
                              "size_bytes": path.stat().st_size}
    imports = {}
    for name, module in sys.modules.copy().items():
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            imports[name] = {"path": path.relative_to(source / "src").as_posix(),
                             "sha256": sha(path)}
    write(folder / "worker.json", record("worker", {
        "methods": measurements, "native_emissions": len(native_calls),
        "input_fingerprint": digest(data), "imports": imports,
        "environment": {"python": sys.version, "pyyaml": yaml.__version__,
                        "libyaml": bool(getattr(yaml, "__with_libyaml__", False))},
    }, {"protocol_sha256": sha(protocol_path), "git_commit": commit,
        "variant": variant, "pid": os.getpid()}))


def _yaml_bytes(folder: Path, method: str) -> bytes:
    path = folder / (method + ".yaml")
    packed = path.with_suffix(".yaml.gz")
    if path.exists() == packed.exists():
        raise ValueError("YAML output inventory is missing or ambiguous")
    return path.read_bytes() if path.exists() else gzip.decompress(packed.read_bytes())


def audit(output: Path) -> dict:
    """Verify all output content, source capsules, work and reported timings."""
    import jsonschema
    import yaml

    protocol_path = output / "protocol.json"
    protocol = _load(protocol_path)
    schema = _load(output / "schema.json")
    if schema != _load(SCHEMA):
        raise ValueError("tool schema differs")
    validate = jsonschema.Draft202012Validator(schema).validate
    validate(protocol)
    pins = protocol["provenance"]["files"]
    if set(pins) != PINNED or any(sha(output / name) != value for name, value in pins.items()):
        raise ValueError("pinned input, tools or source capsule changed")
    payload = protocol["payload"]
    if payload["scope"] != SCOPE or pins["input.json.gz"] != INPUT_SHA256:
        raise ValueError("public result or export scope differs")
    if _load(output / "input-provenance.json") != {
        "source": INPUT.relative_to(ROOT).as_posix(), "sha256": INPUT_SHA256,
        "scope": "Already computed public TUM development result, preserved without recalibration.",
        "provenance": {"source_protocol_sha256": sha(INPUT.parent.parent / "protocol.json")},
    }:
        raise ValueError("public result provenance differs")
    manifests = {v: sources(output, v) for v in ("prior", "current")}
    before, after = manifests["prior"], manifests["current"]
    if set(before) != set(after) or {
        name for name in before if before[name] != after[name]
    } != {"calibrex/core/io.py"}:
        raise ValueError("unexpected runtime source changes")
    seed = json.loads(gzip.decompress((output / "input.json.gz").read_bytes()))
    result_schema = _load(output / "result-schema.json")
    if result_schema != _load(ROOT / "schemas/result.schema.json"):
        raise ValueError("result schema differs")
    jsonschema.validate(seed, result_schema)
    fingerprint = digest(seed)
    attempts_record = _load(output / "attempts.json")
    validate(attempts_record)
    if attempts_record["provenance"] != {"protocol_sha256": sha(protocol_path)}:
        raise ValueError("attempt provenance differs")
    attempts = attempts_record["payload"]["attempts"]
    order = [v for i in range(payload["repetitions"])
             for v in (("current", "prior") if i % 2 == 0 else ("prior", "current"))]
    if [a["variant"] for a in attempts] != order or any(a["exit_code"] for a in attempts):
        raise ValueError("incomplete, failed or reordered attempts")
    timings = {name: {v: [] for v in manifests} for name in ("whole", *METHODS)}
    for index, attempt in enumerate(attempts):
        variant = attempt["variant"]
        if attempt["folder"] != f"{index // 2 + 1:02d}-{variant}":
            raise ValueError("worker folder differs")
        folder = output / attempt["folder"]
        info = _load(folder / "worker.json")
        validate(info)
        if info["provenance"] != {
            "protocol_sha256": sha(protocol_path), "git_commit": payload["commits"][variant],
            "variant": variant, "pid": info["provenance"]["pid"],
        }:
            raise ValueError("worker source or protocol differs")
        data = info["payload"]
        if (data["input_fingerprint"] != fingerprint
            or data["native_emissions"] != (2 if variant == "current" else 0)
            or data["environment"] != payload["environment"]):
            raise ValueError("worker input, backend or environment differs")
        if "calibrex.core.io" not in data["imports"]:
            raise ValueError("writer import missing")
        if any(manifests[variant].get(p["path"]) != p["sha256"]
               for p in data["imports"].values()):
            raise ValueError("worker imported an unfrozen source")
        whole = attempt["seconds"]
        if not math.isfinite(whole) or whole <= 0:
            raise ValueError("invalid whole-worker time")
        timings["whole"][variant].append(whole)
        for name in METHODS:
            measurement = data["methods"][name]
            raw = _yaml_bytes(folder, name)
            if (hashlib.sha256(raw).hexdigest() != measurement["sha256"]
                or len(raw) != measurement["size_bytes"]):
                raise ValueError("YAML output bytes differ")
            loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
            loaded = yaml.load(raw.decode("utf-8"), Loader=loader)
            if digest(loaded) != fingerprint:
                raise ValueError("complete YAML content, types or ordering differ")
            seconds = measurement["seconds"]
            if not math.isfinite(seconds) or not 0 < seconds <= whole:
                raise ValueError("invalid writer time")
            timings[name][variant].append(seconds)
        if sum(data["methods"][name]["seconds"] for name in METHODS) > whole:
            raise ValueError("writer times exceed their worker")
    medians = {name: {v: statistics.median(values) for v, values in variants.items()}
               for name, variants in timings.items()}
    report = record("report", {
        "seconds_median": medians,
        "speedup": {name: v["prior"] / v["current"] for name, v in medians.items()},
        "verified_workers": len(attempts), "full_exports": 2 * len(attempts),
        "complete_values_types_order_exact": True,
        "runtime_source_changes": ["calibrex/core/io.py"],
    }, {"protocol_sha256": sha(protocol_path), "attempts_sha256": sha(output / "attempts.json")})
    validate(report)
    if (output / "summary.json").exists() and _load(output / "summary.json") != report:
        raise ValueError("reported timing differs from all workers")
    if (output / "index.json").exists():
        index = _load(output / "index.json")
        actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        if (set(index["files"]) != actual - {"index.json"}
            or index["provenance"] != {"protocol_sha256": sha(protocol_path)}):
            raise ValueError("published file inventory differs")
        for name, item in index["files"].items():
            path = (output / name).resolve()
            if not path.is_relative_to(output.resolve()) or item != {
                "sha256": sha(path), "size_bytes": path.stat().st_size,
            }:
                raise ValueError("published file bytes differ")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--audit-only", type=Path)
    parser.add_argument("--worker-protocol", type=Path)
    parser.add_argument("--variant", choices=("prior", "current"))
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit(args.audit_only), indent=2))
        return
    if args.worker_protocol:
        worker(args.worker_protocol, args.variant, args.output)
        return
    if not args.baseline_commit or args.output is None or args.repetitions < 2:
        parser.error("baseline commit, output and at least two repetitions are required")
    import yaml
    if not getattr(yaml, "__with_libyaml__", False):
        raise ValueError("LibYAML is required to measure the native emitter")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=ROOT):
        raise ValueError("commit runtime changes before measuring")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if sha(INPUT) != INPUT_SHA256:
        raise ValueError("public input differs")
    copies = (
        (INPUT, "input.json.gz"), (SCHEMA, "schema.json"), (Path(__file__), "launcher.py.txt"),
        (ROOT / "schemas/result.schema.json", "result-schema.json"),
        (ROOT / "tools/benchmark_bspline_lidar_imu.py", "freezer.py.txt"),
        (ROOT / "tools/benchmark_lazy_runtime_loading.py", "identity.py.txt"),
        (ROOT / "tools/benchmark_radar_doppler_batches.py", "capsule-auditor.py.txt"),
    )
    for path, name in copies:
        shutil.copyfile(path, output / name)
    write(output / "input-provenance.json", {
        "source": INPUT.relative_to(ROOT).as_posix(), "sha256": INPUT_SHA256,
        "scope": "Already computed public TUM development result, preserved without recalibration.",
        "provenance": {"source_protocol_sha256": sha(INPUT.parent.parent / "protocol.json")},
    })
    commits = {v: subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
               for v, ref in (("prior", args.baseline_commit), ("current", "HEAD"))}
    for variant, commit in commits.items():
        source = output / ("sources-" + variant)
        _freeze(source, commit)
        _git_identity(source, commit)
        shutil.copyfile(source / "manifest.json", output / f"sources-{variant}-manifest.json")
        with tarfile.open(output / f"sources-{variant}.tar.gz", "w:gz") as bundle:
            bundle.add(source / "src", arcname="src")
    protocol = record("protocol", {
        "scope": SCOPE, "commits": commits, "repetitions": args.repetitions,
        "environment": {"python": sys.version, "pyyaml": yaml.__version__, "libyaml": True},
    }, {"files": {name: sha(output / name) for name in sorted(PINNED)}})
    protocol_path = output / "protocol.json"
    write(protocol_path, protocol)
    attempts = []
    for repeat in range(args.repetitions):
        for variant in ("current", "prior") if repeat % 2 == 0 else ("prior", "current"):
            folder = output / f"{repeat + 1:02d}-{variant}"
            folder.mkdir()
            source = output / ("sources-" + variant)
            _verify_sources(source)
            env = dict(os.environ, PYTHONPATH=str(source / "src"))
            begin = time.perf_counter()
            with ((folder / "stdout.log").open("wb") as out,
                  (folder / "stderr.log").open("wb") as err):
                done = subprocess.run([
                    sys.executable, str(Path(__file__)), "--worker-protocol", str(protocol_path),
                    "--variant", variant, "--output", str(folder),
                ], cwd=source, env=env, stdout=out, stderr=err)
            attempts.append({"folder": folder.name, "variant": variant,
                             "seconds": time.perf_counter() - begin, "exit_code": done.returncode})
            write(output / "attempts.json", record("attempts", {"attempts": attempts},
                                                   {"protocol_sha256": sha(protocol_path)}))
            print(json.dumps(attempts[-1]), flush=True)
            if done.returncode:
                raise RuntimeError(f"worker failed; see {folder / 'stderr.log'}")
            _verify_sources(source)
            if sha(Path(__file__)) != protocol["provenance"]["files"]["launcher.py.txt"]:
                raise ValueError("launcher changed during measurement")
    report = audit(output)
    write(output / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
