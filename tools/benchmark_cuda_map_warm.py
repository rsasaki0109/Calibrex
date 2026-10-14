"""Compare complete warm map APIs using an existing frozen source comparison.

Each independent worker primes its own empty compilation cache before timed
calls. Foreign or unknown GPU compute observations stop only the owned worker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
from tools.benchmark_cuda_map_pipeline import _quality


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    if isinstance(value, dict) and value.get("schema_version"):
        import jsonschema

        schema = Path(__file__).with_name("warm.schema.json")
        if not schema.exists():
            schema = Path(__file__).with_name("schemas") / "cuda_map_warm_comparison.schema.json"
        jsonschema.validate(value, json.loads(schema.read_bytes()))
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class _WorkerProcesses:
    """Keep process identities for the worker and its launcher descendants."""

    def __init__(self, process: subprocess.Popen[Any]) -> None:
        self.process = process
        self.root = psutil.Process(process.pid)
        self.known = {self.root.pid: self.root}

    def capture(self) -> set[int]:
        try:
            for child in self.root.children(recursive=True):
                self.known[child.pid] = child
        except psutil.NoSuchProcess:
            pass
        return {pid for pid, process in self.known.items() if process.is_running()}

    def stop(self) -> None:
        """Terminate only the process objects captured from this worker tree."""
        children = [process for pid, process in self.known.items() if pid != self.root.pid]
        for child in reversed(children):
            with suppress(psutil.NoSuchProcess):
                child.terminate()
        if self.process.poll() is None:
            self.process.terminate()
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            with suppress(psutil.NoSuchProcess):
                child.kill()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def finish(self) -> None:
        """Wait for launcher descendants before starting the next measurement."""
        _, alive = psutil.wait_procs(list(self.known.values()), timeout=10)
        if alive:
            self.stop()
            raise RuntimeError("owned worker descendants did not finish")


def _observation(owned_pids: set[int] | None) -> dict[str, int]:
    try:
        response = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"own": 0, "foreign": 0, "unknown": 1}
    row = {"own": 0, "foreign": 0, "unknown": int(response.returncode != 0)}
    for line in response.stdout.splitlines():
        if not line.strip():
            continue
        try:
            pid = int(line.strip())
        except ValueError:
            row["unknown"] += 1
        else:
            if pid <= 0:
                row["unknown"] += 1
                continue
            try:
                if not psutil.Process(pid).is_running():
                    continue
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                row["unknown"] += 1
                continue
            row["own" if owned_pids is not None and pid in owned_pids else "foreign"] += 1
    return row


def _worker(args: argparse.Namespace) -> None:
    from tools.benchmark_bspline_lidar_imu import _verify_sources

    root = args.source_root.resolve()
    expected = _verify_sources(root)
    cache = Path(os.environ["CUPY_CACHE_DIR"])
    if cache.exists() and any(cache.iterdir()):
        raise ValueError("worker cache must initially be empty")
    sys.path.insert(0, str(root / "src"))
    os.chdir(root)
    launcher = root / "tools/benchmark_cuda_lidar_map.py"
    launcher_digest = _sha(launcher)
    spec = importlib.util.spec_from_file_location("_bound_warm_map", launcher)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    import cupy as cp

    original = cp.cuda.nvrtc.compileProgram
    compilation: dict[str, int | float] = {"calls": 0, "seconds": 0.0}

    def measured_compile(*values: Any, **keywords: Any) -> Any:
        begin = time.perf_counter()
        try:
            return original(*values, **keywords)
        finally:
            compilation["calls"] += 1
            compilation["seconds"] += time.perf_counter() - begin

    cp.cuda.nvrtc.compileProgram = measured_compile
    rows = []
    try:
        for call in range(args.warm_calls + 1):
            folder = args.output / f"{call:02d}"
            before = dict(compilation)
            begin = time.perf_counter()
            bench._worker(
                SimpleNamespace(
                    sequence=args.sequence,
                    output_dir=folder,
                    half_window=0,
                    worker="cuda",
                    source_commit=args.source_commit,
                )
            )
            record = json.loads((folder / "measurement.json").read_bytes())
            row = {
                "call": call,
                "phase": "prime" if call == 0 else "warm",
                "api_seconds": record["api_seconds"],
                "whole_call_seconds": time.perf_counter() - begin,
                "nvrtc_calls": compilation["calls"] - before["calls"],
                "nvrtc_seconds": compilation["seconds"] - before["seconds"],
                "measurement_sha256": _sha(folder / "measurement.json"),
            }
            rows.append(row)
            if call and row["nvrtc_calls"]:
                raise ValueError("warm call compiled a new program")
    finally:
        cp.cuda.nvrtc.compileProgram = original
    executed = {}
    for name, module in list(sys.modules.items()):
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(root / "src"):
                raise ValueError("worker imported outside frozen sources")
            relative = path.relative_to(root / "src").as_posix()
            digest = _sha(path)
            if expected[relative] != digest:
                raise ValueError("executed source changed")
            executed[relative] = digest
    _verify_sources(root)
    if _sha(launcher) != launcher_digest:
        raise ValueError("frozen map launcher changed during warm calls")
    _write(
        args.output / "phase-summary.json",
        {
            "schema_version": "calibrex.cuda_map_warm_pipeline/v0.1",
            "record_type": "worker",
            "rows": rows,
            "warm_api_median_seconds": statistics.median(row["api_seconds"] for row in rows[1:]),
            "provenance": {
                "source_commit": args.source_commit,
                "source_manifest_sha256": _sha(root / "manifest.json"),
                "executed_sources": executed,
                "launcher_sha256": _sha(Path(__file__)),
                "map_launcher_sha256": _sha(launcher),
                "cuda_cache_initially_empty": True,
                "threads": {
                    name: os.environ[name]
                    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                },
                "scope": (
                    "complete warm calibration/evidence APIs after an owned prime; imports excluded"
                ),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warm-calls", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source-commit", help=argparse.SUPPRESS)
    parser.add_argument("--sequence", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.warm_calls < 1:
        parser.error("repeats and warm-calls must be positive")
    args.output = args.output.resolve()
    if args.worker:
        if args.source_root is None or args.sequence is None or not args.source_commit:
            parser.error("worker requires frozen sources, commit and sequence")
        args.sequence = args.sequence.resolve()
        _worker(args)
        return
    if args.comparison_dir is None:
        parser.error("comparison-dir is required")
    source = args.comparison_dir.resolve()
    protocol = json.loads((source / "protocol.json").read_bytes())
    if _sha(source / "map-benchmark.py.txt") != protocol["map_launcher_sha256"]:
        raise ValueError("frozen map launcher differs")
    for variant in ("prior", "current"):
        if (
            _sha(source / f"sources-{variant}/tools/benchmark_cuda_lidar_map.py")
            != protocol["map_launcher_sha256"]
        ):
            raise ValueError("worker map launcher differs")
    sequence = Path(next(iter(protocol["input_sha256"]))).parent.parent.parent
    for path, digest in protocol["input_sha256"].items():
        if _sha(Path(path)) != digest:
            raise ValueError("raw input differs from frozen comparison")
    args.output.mkdir(exist_ok=False)
    launcher = args.output / "benchmark.py.txt"
    launcher.write_bytes(Path(__file__).read_bytes())
    schema = Path(__file__).with_name("schemas") / "cuda_map_warm_comparison.schema.json"
    (args.output / "warm.schema.json").write_bytes(schema.read_bytes())
    _write(
        args.output / "protocol.json",
        {
            "schema_version": "calibrex.cuda_map_warm_pipeline/v0.1",
            "record_type": "protocol",
            "commits": protocol["commits"],
            "repeats": args.repeats,
            "warm_calls_per_worker": args.warm_calls,
            "provenance": {
                "scope": (
                    "alternating frozen sources with one fresh worker and own prime per variant"
                ),
                "comparison_protocol_sha256": _sha(source / "protocol.json"),
                "input_sha256": protocol["input_sha256"],
                "launcher_sha256": _sha(launcher),
                "schema_sha256": _sha(schema),
                "source_manifest_sha256": {
                    variant: _sha(source / f"sources-{variant}/manifest.json")
                    for variant in ("prior", "current")
                },
            },
        },
    )
    attempts, observations, medians = [], [], {"prior": [], "current": []}
    owners: list[_WorkerProcesses] = []
    started = time.perf_counter()
    for repeat in range(args.repeats):
        for variant in ("prior", "current") if repeat % 2 == 0 else ("current", "prior"):
            previous_owned: set[int] = set()
            for owner in owners:
                previous_owned.update(owner.capture())
            initial = _observation(previous_owned)
            observations.append({"offset_seconds": time.perf_counter() - started, **initial})
            if initial["foreign"] or initial["unknown"]:
                _write(args.output / "observations.json", observations)
                raise RuntimeError("foreign or unknown GPU process observed before worker")
            if initial["own"]:
                raise RuntimeError("previous owned GPU context remained after worker completion")
            folder = args.output / f"{repeat:02d}-{variant}"
            folder.mkdir()
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path.cwd())
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                env[name] = "1"
            env["CUPY_CACHE_DIR"] = str(args.output / "caches" / folder.name)
            command = [
                sys.executable,
                str(launcher),
                "--worker",
                "--output",
                str(folder),
                "--source-root",
                str(source / f"sources-{variant}"),
                "--source-commit",
                protocol["commits"][variant],
                "--sequence",
                str(sequence),
                "--warm-calls",
                str(args.warm_calls),
            ]
            begin = time.perf_counter()
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                ownership = _WorkerProcesses(process)
                owners.append(ownership)
                interrupted = False
                while process.poll() is None:
                    try:
                        owned = ownership.capture()
                    except psutil.AccessDenied:
                        row = {"own": 0, "foreign": 0, "unknown": 1}
                    else:
                        row = _observation(owned)
                    observations.append({"offset_seconds": time.perf_counter() - started, **row})
                    _write(args.output / "observations.json", observations)
                    if row["foreign"] or row["unknown"]:
                        ownership.stop()
                        interrupted = True
                        break
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=2)
                code = process.wait()
                try:
                    ownership.finish()
                except RuntimeError:
                    code = -1
                    interrupted = True
            attempts.append(
                {
                    "repeat": repeat,
                    "variant": variant,
                    "exit_code": code,
                    "whole_seconds": time.perf_counter() - begin,
                    "worker": folder.name,
                    "interrupted": interrupted,
                }
            )
            _write(args.output / "attempts.json", attempts)
            _write(args.output / "observations.json", observations)
            if code or interrupted:
                raise RuntimeError(
                    "worker failed or was stopped; retained complete attempt and log"
                )
            report = json.loads((folder / "phase-summary.json").read_bytes())
            medians[variant].append(report["warm_api_median_seconds"])
            print(json.dumps(attempts[-1]), flush=True)
        for call in range(args.warm_calls + 1):
            pair = [
                json.loads(
                    (args.output / f"{repeat:02d}-{v}/{call:02d}/measurement.json").read_bytes()
                )
                for v in ("prior", "current")
            ]
            if any(row["provenance"]["input_sha256"] != protocol["input_sha256"] for row in pair):
                raise ValueError("warm input differs from frozen protocol")
            if not _quality(*pair):
                raise ValueError("complete prime or warm evaluation differs")
    median = {v: statistics.median(values) for v, values in medians.items()}
    summary = {
        "schema_version": "calibrex.cuda_map_warm_pipeline/v0.1",
        "record_type": "summary",
        "warm_api_median_seconds": median,
        "warm_api_speedup": median["prior"] / median["current"],
        "quality_exact": True,
        "pairs": args.repeats,
        "warm_calls_per_worker": args.warm_calls,
        "provenance": {
            "protocol_sha256": _sha(source / "protocol.json"),
            "launcher_sha256": _sha(launcher),
            "attempts_sha256": _sha(args.output / "attempts.json"),
            "observations_sha256": _sha(args.output / "observations.json"),
            "scope": (
                "alternating independent process pairs; own empty cache and prime per worker; "
                "complete API timings"
            ),
            "limitation": (
                "periodic GPU process observation; does not reserve CPU or provide continuous "
                "GPU isolation"
            ),
        },
    }
    _write(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
