"""Compare complete ICP against frozen sources in alternating fresh workers.

Uses provenance-locked actual KITTI pairs from prepare_kitti_icp_pairs. Native
workers import no CuPy. Every startup, cold compiler cache and saved result is
retained. Baseline import paths and bytes are verified, not just Git labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, model: object) -> None:
    if isinstance(model, dict) and model.get("schema_version"):
        import jsonschema

        schema_path = (
            Path(__file__).with_name("schemas") / "icp_neighbor_order_comparison.schema.json"
        )
        jsonschema.validate(model, json.loads(schema_path.read_bytes()))
    path.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source_mapping(root: Path) -> tuple[dict, dict[str, tuple[Path, str]]]:
    from tools.benchmark_cuda_point_to_point_icp import SOURCES

    manifest = json.loads((root / "manifest.json").read_bytes())
    if set(manifest["files"]) != set(SOURCES):
        raise ValueError("baseline manifest must bind every declared ICP source")
    mapping = {}
    for relative, digest in manifest["files"].items():
        path = root / Path(relative).name
        if _sha(path.read_bytes()) != digest:
            raise ValueError(f"baseline source digest mismatch: {relative}")
        if relative.startswith("src/"):
            name = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
            mapping[name] = (path.resolve(), digest)
    return manifest, mapping


class _HistoricalSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, mapping: dict[str, tuple[Path, str]]) -> None:
        self.mapping = mapping

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in self.mapping:
            return None
        if fullname in sys.modules:
            raise RuntimeError(f"historical source was imported before activation: {fullname}")
        return importlib.util.spec_from_file_location(fullname, self.mapping[fullname][0])


def _worker(args: argparse.Namespace) -> None:
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    started = time.perf_counter()
    launcher_digest = _sha(Path(__file__).read_bytes())
    baseline_root = Path(args.baseline_root).resolve()
    manifest, mapping = _source_mapping(baseline_root)
    if args.variant == "prior":
        if any(name in sys.modules for name in mapping):
            raise RuntimeError("baseline activation must precede calibration imports")
        sys.meta_path.insert(0, _HistoricalSourceFinder(mapping))
    from tools import benchmark_cuda_point_to_point_icp as benchmark

    benchmark._worker(
        argparse.Namespace(
            output=args.output,
            method=args.backend,
            workers=0,
            seed=20261013,
            pair_fixture=args.fixture,
            data_root=args.data_root,
            point_budget=args.point_budget,
        )
    )
    execution = {}
    for module_name, (prior_path, digest) in mapping.items():
        module = sys.modules.get(module_name)
        if module is None:
            if args.backend == "cuda" or "cuda" not in module_name:
                raise RuntimeError(f"expected calibration module was not loaded: {module_name}")
            continue
        actual = Path(module.__file__).resolve()
        actual_digest = _sha(actual.read_bytes())
        if args.variant == "prior":
            if actual != prior_path or actual_digest != digest:
                raise RuntimeError(f"baseline source binding differs: {module_name}")
        elif actual.is_relative_to(baseline_root):
            raise RuntimeError(f"candidate loaded historical source: {module_name}")
        execution["src/" + module_name.replace(".", "/") + ".py"] = actual_digest
    _source_mapping(baseline_root)
    if _sha(Path(__file__).read_bytes()) != launcher_digest:
        raise RuntimeError("comparison launcher changed during measurement")
    output = Path(args.output)
    record = json.loads((output / "measurement.json").read_bytes())
    record["variant"] = args.variant
    record["provenance"]["checkout_source_guard"] = record["provenance"]["source"]
    record["provenance"].update(
        {
            "source": {
                "commit": manifest["commit"]
                if args.variant == "prior"
                else record["provenance"]["source"]["commit"],
                "sources": execution,
            },
            "comparison_launcher_sha256": launcher_digest,
            "baseline_manifest_sha256": _sha((baseline_root / "manifest.json").read_bytes()),
            "source_binding": "frozen_prior" if args.variant == "prior" else "current_checkout",
            "scope": "full ICP, spatial holdout, rematching and all six initial-value diagnostics",
        }
    )
    _write(output / "measurement.json", record)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "backend": args.backend,
                "solver_seconds": record["seconds"],
                "internal_whole_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def _compare(args: argparse.Namespace) -> None:
    from tools.benchmark_cuda_point_to_point_icp import SOURCES, _identity

    root = Path(args.output)
    root.mkdir(parents=True)
    baseline = root / "sources/prior"
    current = root / "sources/current"
    baseline.mkdir(parents=True)
    current.mkdir(parents=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.baseline_commit + "^{commit}"],
        text=True,
    ).strip()
    prior_files = {}
    for source in SOURCES:
        raw = subprocess.check_output(["git", "show", commit + ":" + source])
        (baseline / Path(source).name).write_bytes(raw)
        (current / Path(source).name).write_bytes(Path(source).read_bytes())
        prior_files[source] = _sha(raw)
    _write(baseline / "manifest.json", {"commit": commit, "files": prior_files})
    launcher_raw = Path(__file__).read_bytes()
    (root / "launcher.py.txt").write_bytes(launcher_raw)
    fixture_raw = Path(args.fixture).read_bytes()
    (root / "fixture.json").write_bytes(fixture_raw)
    identity = _identity()
    protocol = {
        "schema_version": "calibrex.icp_neighbor_order_comparison/v0.1",
        "kind": "protocol",
        "provenance": {
            "generator": "calibrex.icp_neighbor_order_comparison/v0.1",
            "current": identity,
            "prior_commit": commit,
            "launcher_sha256": _sha(launcher_raw),
            "fixture_sha256": _sha(fixture_raw),
        },
        "backend": args.backend,
        "repeats": args.repeats,
        "point_budget": args.point_budget,
        "planned_orders": [
            ["prior", "candidate"] if i % 2 else ["candidate", "prior"]
            for i in range(1, args.repeats + 1)
        ],
        "scope": "fresh subprocess startup, guards, full solve, evaluation and saves",
        "cold_policy": "one BLAS thread; separate initially empty CUDA disk cache per worker",
    }
    _write(root / "protocol.json", protocol)
    trials = []
    anchor = None
    for repeat, order in enumerate(protocol["planned_orders"], 1):
        for variant in order:
            output = root / f"{variant}.{repeat:02d}"
            command = [
                sys.executable,
                "-m",
                "tools.benchmark_icp_neighbor_order",
                "--worker",
                "--variant",
                variant,
                "--baseline-root",
                str(baseline),
                "--backend",
                args.backend,
                "--fixture",
                args.fixture,
                "--data-root",
                args.data_root,
                "--point-budget",
                str(args.point_budget),
                "--output",
                str(output),
            ]
            environment = dict(
                os.environ,
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                PYTHONFAULTHANDLER="1",
            )
            start = time.perf_counter()
            with (root / f"{variant}.{repeat:02d}.log.txt").open("wb") as log:
                process = subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            whole = time.perf_counter() - start
            trial = {
                "repeat": repeat,
                "variant": variant,
                "order": order,
                "whole_seconds": whole,
                "exit_code": process.returncode,
            }
            trials.append(trial)
            _write(
                root / "attempts.json",
                {
                    "schema_version": protocol["schema_version"],
                    "kind": "attempts",
                    "provenance": protocol["provenance"],
                    "trials": trials,
                },
            )
            if process.returncode:
                raise RuntimeError(f"worker failed; retained {output} and its log")
            record = json.loads((output / "measurement.json").read_bytes())
            trial["measurement"] = record
            raw = (output / "result.json").read_bytes()
            if anchor is None:
                anchor = raw
            if raw != anchor:
                raise RuntimeError(f"complete calibration result differs: {variant}, {repeat}")
            runtime = record["provenance"]["blas_runtime"]
            if not runtime or any(
                item["num_threads"] != 1 for item in runtime if item["user_api"] == "blas"
            ):
                raise RuntimeError("worker did not apply the BLAS budget")
            if args.backend == "native" and record["provenance"]["cupy_imported"]:
                raise RuntimeError("native worker imported CuPy")
            if _identity() != identity or _sha(Path(__file__).read_bytes()) != _sha(launcher_raw):
                raise RuntimeError("HEAD or measured source changed during comparison")
            _write(
                root / "attempts.json",
                {
                    "schema_version": protocol["schema_version"],
                    "kind": "attempts",
                    "provenance": protocol["provenance"],
                    "trials": trials,
                },
            )
            print(
                json.dumps(
                    {
                        "repeat": repeat,
                        "variant": variant,
                        "whole_seconds": whole,
                        "solver_seconds": record["seconds"],
                        "full_result_equal": True,
                    }
                ),
                flush=True,
            )
    whole = {
        variant: statistics.median(t["whole_seconds"] for t in trials if t["variant"] == variant)
        for variant in ("prior", "candidate")
    }
    solver = {
        variant: statistics.median(
            t["measurement"]["seconds"] for t in trials if t["variant"] == variant
        )
        for variant in ("prior", "candidate")
    }
    _write(
        root / "summary.json",
        {
            "schema_version": protocol["schema_version"],
            "kind": "summary",
            "provenance": protocol["provenance"],
            "trials": trials,
            "whole_medians_s": whole,
            "solver_medians_s": solver,
            "whole_speedup": whole["prior"] / whole["candidate"],
            "solver_speedup": solver["prior"] / solver["candidate"],
            "full_result_equal": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--baseline-root")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("native", "cuda"), default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--point-budget", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=("prior", "candidate"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.point_budget < 0:
        parser.error("repeats must be positive and point-budget nonnegative")
    if args.worker:
        if not args.baseline_root or not args.variant:
            parser.error("worker requires baseline-root and variant")
        _worker(args)
    else:
        if not args.baseline_commit:
            parser.error("comparison requires baseline-commit")
        _compare(args)


if __name__ == "__main__":
    main()
