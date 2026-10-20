"""Compare complete joint recovery after removing repeated IMU evaluation.

Use the maintained joint worker and its already-spent three synthetic seeds.
The 2x whole-API target is reported even when exact execution cleanup misses it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_screw_residual_evaluation as previous  # noqa: E402

TOOLS = (*previous.TOOLS, "tools/benchmark_screw_imu_evaluation.py")
FILES = {"result.json", "fit-input.json", "fit-result.json", "measurement.json"}
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
             *(f"LPT{i}" for i in range(10))}


def _unpack_archive(path: Path, directory: Path) -> None:
    """Read regular indexed files only; never import or execute archived code."""
    import hashlib

    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or "index.json" not in names:
            raise ValueError("duplicate member or missing index")
        if sum(member.size for member in members) > 64 * 1024**2:
            raise ValueError("archive exceeds unpacked size limit")
        for member in members:
            name = PurePosixPath(member.name)
            if (not member.isfile() or name.is_absolute() or ".." in name.parts
                    or "\\" in member.name or ":" in member.name
                    or name.as_posix() != member.name
                    or any(part.endswith((".", " ")) for part in name.parts)
                    or any(part.split(".")[0].upper() in _RESERVED for part in name.parts)
                    or member.size > 4 * 1024**2):
                raise ValueError("unsafe archive member")
        index_file = archive.extractfile("index.json")
        assert index_file is not None
        index = json.loads(index_file.read())
        if set(index["files"]) != set(names) - {"index.json"}:
            raise ValueError("index coverage differs")
        for member in members:
            file = archive.extractfile(member)
            assert file is not None
            raw = file.read()
            if member.name != "index.json" and index["files"][member.name] != {
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            }:
                raise ValueError("archive member differs from index")
            target = directory / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)


def audit_archive(path: Path, *, temporary_directory: Path | None = None) -> dict[str, Any]:
    """Replay bindings and whole-result equality on a CPU without calibration runs."""
    with tempfile.TemporaryDirectory(dir=temporary_directory) as temporary:
        directory = Path(temporary)
        _unpack_archive(path, directory)
        import jsonschema

        for index, variant in enumerate(previous.ORDER):
            worker = directory / f"joint-{index:02d}-{variant}"
            inputs = json.loads((worker / "fit-input.json").read_bytes())
            fitted = json.loads((worker / "fit-result.json").read_bytes())
            origin = fitted.pop("provenance")
            jsonschema.validate(inputs, json.loads(
                (directory / "input-payload.schema.json").read_bytes()))
            jsonschema.validate({"payload": fitted, "provenance": origin}, json.loads(
                (directory / "fit-payload.schema.json").read_bytes()))
        result = audit(directory)
        retained = json.loads((directory / "summary.json").read_bytes())
        jsonschema.validate(retained, json.loads((directory / "summary.schema.json").read_bytes()))
        if previous.quality(result) != previous.quality(retained):
            raise ValueError("retained summary differs from complete replay")
        return result


def audit(folder: Path, *, verify_git: bool = False) -> dict[str, Any]:
    """Check the entire six-worker cohort without running calibration code."""
    import jsonschema

    protocol = json.loads((folder / "protocol.json").read_bytes())
    if (protocol["order"] != list(previous.ORDER)
            or protocol["seeds"] != list(previous.SEEDS)
            or protocol["target_api_ratio"] != 2.0 or protocol["blas_threads"] != 1):
        raise ValueError("fixed cohort protocol differs")
    if previous.sha(folder / "result.schema.json") != protocol["result_schema_sha256"]:
        raise ValueError("result schema binding differs")
    for name, digest in protocol["tools"].items():
        if previous.sha(folder / "toolchain" / name) != digest:
            raise ValueError("toolchain binding differs")
    manifests = {}
    for variant in ("baseline", "candidate"):
        frozen = folder / variant
        manifests[variant] = previous.source._verify_sources(frozen)
        if previous.sha(frozen / "manifest.json") != protocol["manifests"][variant]:
            raise ValueError("source manifest binding differs")
        if verify_git:
            from tools.benchmark_online_edge_cuda import _git_files

            committed = {p: b for p, b in _git_files(
                protocol["commits"][variant], ("src/calibrex",),
            ).items() if p.endswith(".py")}
            actual = {"src/" + p: (frozen / "src" / p).read_bytes()
                      for p in manifests[variant]}
            if actual != committed:
                raise ValueError("source capsule differs from committed Git text")
    attempts = json.loads((folder / "attempts.json").read_bytes())
    rows = attempts["observations"]
    if [row["variant"] for row in rows] != list(previous.ORDER):
        raise ValueError("cohort incomplete or out of order")
    measurements = []
    for index, row in enumerate(rows):
        name = f"joint-{index:02d}-{row['variant']}"
        directory = folder / name
        if row["exit_code"] != 0 or set(row["files"]) != FILES:
            raise ValueError("failed or incomplete worker; retain all attempts")
        if previous.sha(folder / f"{name}.log") != row["log_sha256"]:
            raise ValueError("worker log binding differs")
        for filename, digest in row["files"].items():
            if previous.sha(directory / filename) != digest:
                raise ValueError("worker file binding differs")
        value = json.loads((directory / "measurement.json").read_bytes())
        origin = value["provenance"]
        if (origin["protocol_sha256"] != previous.sha(folder / "protocol.json")
                or origin["source_commit"] != protocol["commits"][row["variant"]]
                or origin["seed"] != previous.SEEDS[index // 2]):
            raise ValueError("worker provenance differs")
        if not 0 < value["fit_seconds"] <= value["api_seconds"] < row["seconds"]:
            raise ValueError("invalid measurement intervals")
        result = json.loads((directory / "result.json").read_bytes())
        jsonschema.validate(result, json.loads((folder / "result.schema.json").read_bytes()))
        for filename, key in (
            ("result.json", "result_quality_sha256"),
            ("fit-input.json", "input_quality_sha256"),
            ("fit-result.json", "fit_quality_sha256"),
        ):
            if previous.quality(json.loads((directory / filename).read_bytes())) != value[key]:
                raise ValueError("complete input or outcome differs from measurement")
        if not value["imported"] or any(
            manifests[row["variant"]][p] != digest for p, digest in value["imported"].items()
        ):
            raise ValueError("imported source differs")
        measurements.append(value)
    pairs = []
    for index in range(0, 6, 2):
        pair = {previous.ORDER[i]: (measurements[i], rows[i]) for i in (index, index + 1)}
        b, c = pair["baseline"], pair["candidate"]
        for key in ("input_quality_sha256", "result_quality_sha256", "fit_quality_sha256"):
            if b[0][key] != c[0][key]:
                raise ValueError("complete baseline/candidate input or outcome differs")
        pairs.append({"seed": previous.SEEDS[index // 2],
                      "api_ratio": b[0]["api_seconds"] / c[0]["api_seconds"],
                      "fit_ratio": b[0]["fit_seconds"] / c[0]["fit_seconds"],
                      "fresh_worker_ratio": b[1]["seconds"] / c[1]["seconds"]})
    api_ratio = statistics.median(p["api_ratio"] for p in pairs)
    return {
        "pairs": pairs,
        "paired_median_api_ratio": api_ratio,
        "paired_median_fit_ratio": statistics.median(p["fit_ratio"] for p in pairs),
        "paired_median_fresh_worker_ratio": statistics.median(
            p["fresh_worker_ratio"] for p in pairs),
        "median_seconds": {variant: {
            key: statistics.median(m[key] for m, row in zip(measurements, rows, strict=True)
                                   if row["variant"] == variant)
            for key in ("api_seconds", "fit_seconds")
        } for variant in ("baseline", "candidate")},
        "all_complete_inputs_and_outcomes_equal": True,
        "meets_2x_api_target": api_ratio >= 2.0,
        "provenance": previous.provenance(
            protocol_sha256=previous.sha(folder / "protocol.json"),
            attempts_sha256=previous.sha(folder / "attempts.json"),
            scope="Already-spent synthetic development; no independent accuracy claim",
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze sources, run all workers once, and retain misses of the speed target."""
    folder = args.output.resolve()
    if folder.exists() or shutil.disk_usage(folder.parent).free < 2 * 1024**3:
        raise ValueError("require a new output directory and at least 2 GiB free")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", *TOOLS],
                   check=True, cwd=ROOT)
    commits = {"baseline": subprocess.check_output(
        ["git", "rev-parse", args.baseline_commit], cwd=ROOT, text=True).strip(),
        "candidate": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    folder.mkdir()
    for path in TOOLS:
        target = folder / "toolchain" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    for variant, commit in commits.items():
        frozen = folder / variant
        previous.source._freeze(frozen, commit)
        for path in (frozen / "src").rglob("*.py"):
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        (frozen / "manifest.json").write_text(
            json.dumps(previous.source._manifest(frozen / "src"), indent=2) + "\n",
            encoding="utf-8")
    shutil.copyfile(ROOT / previous.SCHEMA, folder / "result.schema.json")
    previous.write(folder / "protocol.json", {
        "commits": commits, "order": list(previous.ORDER), "seeds": list(previous.SEEDS),
        "manifests": {v: previous.sha(folder / v / "manifest.json") for v in commits},
        "tools": {p: previous.sha(ROOT / p) for p in TOOLS},
        "result_schema_sha256": previous.sha(folder / "result.schema.json"),
        "target_api_ratio": 2.0, "blas_threads": 1, "host_reserved": False, "warmups": 0,
        "provenance": previous.provenance(
            scope="Frozen before timing; unchanged iterations, quality and selection"),
    })
    rows = []
    for index, variant in enumerate(previous.ORDER):
        name = f"joint-{index:02d}-{variant}"
        print(json.dumps({"starting": name}), flush=True)
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1")
        begin = time.perf_counter()
        with (folder / f"{name}.log").open("wb") as log:
            try:
                status = subprocess.run([
                    sys.executable, str(ROOT / previous.TOOLS[0]), "--worker",
                    "--output", str(folder), "--variant", variant, "--case", "joint",
                    "--observation", str(index),
                ], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    timeout=600, check=False).returncode
            except subprocess.TimeoutExpired:
                status = -1
        rows.append({"variant": variant, "seconds": time.perf_counter() - begin,
                     "exit_code": status,
                     "files": {p.name: previous.sha(p)
                               for p in sorted((folder / name).glob("*.json"))},
                     "log_sha256": previous.sha(folder / f"{name}.log")})
        previous.write(folder / "attempts.json", {
            "observations": rows, "provenance": previous.provenance(
                protocol_sha256=previous.sha(folder / "protocol.json")),
        })
        if status:
            raise RuntimeError("worker failed; retain all attempts")
        print(json.dumps({"finished": name, "seconds": rows[-1]["seconds"]}), flush=True)
    result = audit(folder, verify_git=True)
    previous.write(folder / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--archive", type=Path)
    parser.add_argument("--baseline-commit", default="0f76b0ec6bda2d975ea58d4b4bbd52d174892dae")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    result = (audit_archive(args.archive) if args.archive else
              audit(args.output) if args.audit else run(args))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["meets_2x_api_target"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
