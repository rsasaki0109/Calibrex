"""Compare two maintained CPU commits on all 48 existing KITTI pairs.

Read-only input reuse, three pairs of fresh workers, complete result equality,
per-update/API/cold-worker clocks, and a CPU-only audit. No accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_online_edge_cpu as cpu  # noqa: E402

base = cpu.base
ORDER = ("baseline", "candidate", "candidate", "baseline", "baseline", "candidate")
TOOLCHAIN = (
    *cpu.TOOLCHAIN,
    "tools/online_grid_reuse_worker.py",
    "tools/benchmark_online_grid_reuse.py",
)
METRICS = ("solve", "median_update", "first_update", "all_updates", "whole_worker")


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def record_schemas() -> dict[str, dict]:
    """Closed records bind timings, exact work, source, tools and every attempt."""
    positive = {"type": "number", "exclusiveMinimum": 0}
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    commit = {"type": "string", "pattern": "^[0-9a-f]{40}$"}

    def obj(properties):
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        }

    hashes = {"type": "object", "additionalProperties": digest}
    provenance = {"type": "object"}
    metrics = obj(dict.fromkeys(METRICS, positive))
    return {
        "protocol": obj(
            {
                "commits": obj(dict.fromkeys(("baseline", "candidate"), commit)),
                "source_protocol_sha256": obj(dict.fromkeys(("baseline", "candidate"), digest)),
                "toolchain_sha256": obj(dict.fromkeys(TOOLCHAIN, digest)),
                "schema_sha256": hashes,
                "receipt_sha256": digest,
                "order": {"const": list(ORDER)},
                "blas_threads": {"const": 1},
                "preparation_workers": {"const": 3},
                "host_reserved": {"const": False},
                "warmups": {"const": 0},
                "solve_gate": {"const": 2.0},
                "provenance": provenance,
            }
        ),
        "source": obj(
            {
                "source_commit": commit,
                "base_git_commit": commit,
                "source_kind": {"const": "git-commit"},
                "source_manifest_sha256": digest,
                "toolchain_sha256": obj(dict.fromkeys(TOOLCHAIN, digest)),
                "receipt_sha256": digest,
                "provenance": provenance,
            }
        ),
        "attempts": obj(
            {
                "observations": {
                    "type": "array",
                    "maxItems": len(ORDER),
                    "items": obj(
                        {
                            "label": {"enum": ["baseline", "candidate"]},
                            "worker": {"type": "string", "pattern": "^[0-9]{2}-numba$"},
                            "seconds": positive,
                            "exit_code": {"type": "integer"},
                            "artifact_sha256": hashes,
                        }
                    ),
                },
                "provenance": provenance,
            }
        ),
        "timing": obj(
            {
                "solve_seconds": {"type": "array", "minItems": 1, "maxItems": 1, "items": positive},
                "updates": {
                    "type": "array",
                    "minItems": 30,
                    "maxItems": 30,
                    "items": obj(
                        {
                            "frame_id": {"type": "string"},
                            "seconds": positive,
                        }
                    ),
                },
                "measurement_sha256": digest,
                "provenance": provenance,
            }
        ),
        "summary": obj(
            {
                "medians_seconds": obj(dict.fromkeys(("baseline", "candidate"), metrics)),
                "ratios": metrics,
                "passes_solve_gate": {"type": "boolean"},
                "observations": {"const": 6},
                "complete_quality_bits_equal": {"const": True},
                "prepared_bits_equal": {"const": True},
                "quality_sha256": digest,
                "provenance": provenance,
            }
        ),
    }


def _validate(value: dict, name: str) -> None:
    import jsonschema

    jsonschema.validate(value, record_schemas()[name])


def _finite(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("nonfinite or nonpositive timing")
    return value


def _quality_digest(value: dict) -> str:
    # JSON bytes distinguish signed zero and every serialized binary64 value.
    return hashlib.sha256(
        json.dumps(
            base._comparable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def audit(folder: Path, *, verify_git: bool = False) -> dict:
    """Validate retained records without importing accelerators or running snapshots."""
    import jsonschema

    protocol, attempts = base.read(folder / "protocol.json"), base.read(folder / "attempts.json")
    _validate(protocol, "protocol")
    _validate(attempts, "attempts")
    if attempts["provenance"]["protocol_sha256"] != base.sha(folder / "protocol.json"):
        raise ValueError("attempt lineage differs")
    if [item["label"] for item in attempts["observations"]] != list(ORDER):
        raise ValueError("incomplete counterbalance")
    for name, digest in protocol["toolchain_sha256"].items():
        if base.sha(folder / "toolchain" / name) != digest:
            raise ValueError("frozen toolchain changed")
    for name, digest in protocol["schema_sha256"].items():
        if base.sha(folder / name) != digest:
            raise ValueError("frozen schema changed")
    manifests, sources = {}, {}
    for label, commit in protocol["commits"].items():
        root = folder / label
        source = base.read(root / "protocol.json")
        _validate(source, "source")
        if base.sha(root / "protocol.json") != protocol["source_protocol_sha256"][label]:
            raise ValueError("source protocol changed")
        if source["source_commit"] != commit or source["base_git_commit"] != commit:
            raise ValueError("source commit differs")
        if source["toolchain_sha256"] != protocol["toolchain_sha256"]:
            raise ValueError("source toolchain differs")
        if base.sha(root / "receipt.json") != protocol["receipt_sha256"]:
            raise ValueError("receipt changed")
        if source["receipt_sha256"] != protocol["receipt_sha256"]:
            raise ValueError("source receipt differs")
        manifests[label] = base._verify_sources(root / "source")
        if base.sha(root / "source/manifest.json") != source["source_manifest_sha256"]:
            raise ValueError("source manifest changed")
        for name, schema in cpu.schemas(folder / "toolchain").items():
            if base.read(root / f"schemas/{name}.schema.json") != schema:
                raise ValueError("derived worker schema differs")
        if verify_git:
            frozen = base._git_files(commit, ("src/calibrex",))
            expected = {
                name.removeprefix("src/"): hashlib.sha256(data).hexdigest()
                for name, data in frozen.items()
                if name.endswith(".py")
            }
            if expected != manifests[label]:
                raise ValueError("committed source differs")
        sources[label] = source
    if verify_git:
        frozen = base._git_files(protocol["commits"]["candidate"], TOOLCHAIN)
        if {name: hashlib.sha256(data).hexdigest() for name, data in frozen.items()} != protocol[
            "toolchain_sha256"
        ]:
            raise ValueError("committed toolchain differs")
    times = {label: {metric: [] for metric in METRICS} for label in ("baseline", "candidate")}
    qualities, captures, configs = [], [], []
    for index, observation in enumerate(attempts["observations"]):
        label, worker = observation["label"], observation["worker"]
        if worker != f"{index:02d}-numba" or observation["exit_code"] != 0:
            raise ValueError("failed or substituted worker")
        root = folder / label
        path = root / worker
        expected_files = {f"{name}.json" for name in ("result", "config", "measurement", "timing")}
        expected_files |= {"stdout.txt", "stderr.txt"}
        if set(observation["artifact_sha256"]) != expected_files:
            raise ValueError("worker artifact set differs")
        if any(
            base.sha(path / name) != digest
            for name, digest in observation["artifact_sha256"].items()
        ):
            raise ValueError("worker artifact changed")
        measured, result, config, timing = [
            base.read(path / f"{name}.json")
            for name in ("measurement", "result", "config", "timing")
        ]
        for name, value in (("records", measured), ("result", result), ("config", config)):
            jsonschema.validate(value, base.read(root / f"schemas/{name}.schema.json"))
            cpu._check_lineage(value, sources[label], root, "numba")
        _validate(timing, "timing")
        if timing["measurement_sha256"] != base.sha(path / "measurement.json"):
            raise ValueError("timing measurement differs")
        expected_provenance = {
            **measured["provenance"],
            "timing_wrapper_sha256": protocol["toolchain_sha256"][
                "tools/online_grid_reuse_worker.py"
            ],
        }
        if any(
            timing["provenance"][key] != value
            for key, value in expected_provenance.items()
            if key != "scope"
        ):
            raise ValueError("timing lineage differs")
        if any(
            measured[f"{name}_sha256"] != base.sha(path / f"{name}.json")
            for name in ("result", "config")
        ):
            raise ValueError("saved output differs")
        if any(
            manifests[label].get(name) != digest
            for name, digest in measured["imported_source_sha256"].items()
        ):
            raise ValueError("executed source differs")
        quality = result["result"]
        kernels = cpu.kernel_digests(root / "source")
        software = measured["software"]
        execution = {
            "backend": "numba",
            "kernel_sha256": kernels["_interpolate"],
            "policy": cpu.POLICY,
            "runtime": (
                f"{cpu.runtime_prefix(root / 'source')}{software['numba']}"
                f";numpy={software['numpy']}"
            ),
            "objective_evaluations": 21958,
            "point_rows": 395476000,
            "grid_calls": 30,
            "grid_pose_evaluations": 21840,
            "numba_frame_pose_evaluations": 197738,
            "native_frame_pose_evaluations": 0,
        }
        if (
            quality.get("execution") != execution
            or measured["image_kernel_sha256"] != kernels["_image_response"]
        ):
            raise ValueError("execution identity or work differs")
        if (
            measured["evaluations"],
            measured["point_rows"],
            measured["timeline_steps"],
            measured["native_raw_grid_pose_evaluations"],
            len(measured["frame_points"]),
            len(quality["holdout_frame_ids"]),
        ) != (21958, 395476000, 30, 0, 48, 10):
            raise ValueError("complete calibration work differs")
        if [value["frame_id"] for value in timing["updates"]] != [
            value["frame_id"] for value in quality["timeline"]
        ]:
            raise ValueError("timed update population differs")
        for key in ("edge_evaluation_backend", "grid_evaluation_backend"):
            if config.pop(key) != "numba":
                raise ValueError("worker backend differs")
        config.pop("provenance")
        configs.append(config)
        qualities.append(_quality_digest(quality))
        captures.append(measured["prepared_capture_sha256"])
        solve = _finite(timing["solve_seconds"][0])
        updates = [_finite(value["seconds"]) for value in timing["updates"]]
        whole = _finite(observation["seconds"])
        for name in ("elapsed_seconds", "preparation_seconds", "solve_and_save_seconds"):
            _finite(measured[name])
        if (
            sum(updates) > solve
            or solve > measured["solve_and_save_seconds"]
            or measured["elapsed_seconds"] > whole
        ):
            raise ValueError("timing boundaries differ")
        for metric, value in zip(
            METRICS,
            (solve, statistics.median(updates), updates[0], sum(updates), whole),
            strict=True,
        ):
            times[label][metric].append(value)
    if (
        len(set(qualities)) != 1
        or len(set(captures)) != 1
        or any(value != configs[0] for value in configs)
    ):
        raise ValueError("complete quality, configuration or prepared bits differ")
    medians = {
        label: {name: statistics.median(values) for name, values in group.items()}
        for label, group in times.items()
    }
    ratios = {name: medians["baseline"][name] / medians["candidate"][name] for name in METRICS}
    value = {
        "medians_seconds": medians,
        "ratios": ratios,
        "passes_solve_gate": ratios["solve"] >= protocol["solve_gate"],
        "observations": len(ORDER),
        "complete_quality_bits_equal": True,
        "prepared_bits_equal": True,
        "quality_sha256": qualities[0],
        "provenance": {
            **base._provenance(sources["candidate"]),
            "protocol_sha256": base.sha(folder / "protocol.json"),
            "attempts_sha256": base.sha(folder / "attempts.json"),
            "git_verified": verify_git,
            "scope": "Spent-data performance; solve gate and cold-worker ratio are separate",
        },
    }
    _validate(value, "summary")
    if (folder / "summary.json").exists():
        saved = base.read(folder / "summary.json")
        _validate(saved, "summary")
        if any(saved[key] != item for key, item in value.items() if key != "provenance"):
            raise ValueError("saved comparison differs")
        for key in ("protocol_sha256", "attempts_sha256", "source_commit"):
            if saved["provenance"][key] != value["provenance"][key]:
                raise ValueError("saved comparison lineage differs")
    return value


def execute_worker(command: list[str], env: dict[str, str], path: Path) -> int:
    """Retain failed launches/timeouts as attempts, with their original logs."""
    with (path / "stdout.txt").open("wb") as stdout, (path / "stderr.txt").open("wb") as stderr:
        try:
            return subprocess.run(
                command, env=env, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=600, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            return -1
        except OSError:
            return -2


def run(args: argparse.Namespace) -> dict:
    parent = args.output.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(parent).free
    if free < cpu.MIN_FREE:
        raise ValueError("comparison requires 20 GiB free")
    commits = {
        label: subprocess.check_output(["git", "rev-parse", commit], text=True).strip()
        for label, commit in (("baseline", args.baseline), ("candidate", args.commit))
    }
    if commits["baseline"] == commits["candidate"]:
        raise ValueError("comparison requires two different committed sources")
    frozen = base._git_files(commits["candidate"], TOOLCHAIN)
    if any(
        (ROOT / name).read_text(encoding="utf-8").encode() != frozen[name] for name in TOOLCHAIN
    ):
        raise ValueError("commit the timing toolchain before measuring")
    receipt = base.read(args.receipt)
    base.verify_fixture(args.fixture, receipt)
    folder = args.output.resolve()
    folder.mkdir()
    for name, data in frozen.items():
        target = folder / "toolchain" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in frozen.items()}
    source_hashes, schema_hashes = {}, {}
    for label, commit in commits.items():
        root = folder / label
        root.mkdir()
        base._freeze(root / "source", commit)
        for path in (root / "source/src").rglob("*.py"):
            path.write_bytes(path.read_text(encoding="utf-8").encode())
        write(root / "source/manifest.json", base._manifest(root / "source/src"))
        (root / "schemas").mkdir()
        for name, schema in cpu.schemas(folder / "toolchain").items():
            path = root / f"schemas/{name}.schema.json"
            write(path, schema)
            schema_hashes[path.relative_to(folder).as_posix()] = base.sha(path)
        (root / "receipt.json").write_bytes(args.receipt.read_bytes())
        source = {
            "source_commit": commit,
            "base_git_commit": commit,
            "source_kind": "git-commit",
            "source_manifest_sha256": base.sha(root / "source/manifest.json"),
            "toolchain_sha256": hashes,
            "receipt_sha256": base.sha(root / "receipt.json"),
        }
        source["provenance"] = base._provenance(source)
        _validate(source, "source")
        write(root / "protocol.json", source)
        source_hashes[label] = base.sha(root / "protocol.json")
    (folder / "schemas").mkdir()
    for name, schema in record_schemas().items():
        path = folder / f"schemas/{name}.schema.json"
        write(path, schema)
        schema_hashes[path.relative_to(folder).as_posix()] = base.sha(path)
    protocol = {
        "commits": commits,
        "source_protocol_sha256": source_hashes,
        "toolchain_sha256": hashes,
        "schema_sha256": schema_hashes,
        "receipt_sha256": base.sha(args.receipt),
        "order": list(ORDER),
        "blas_threads": 1,
        "preparation_workers": 3,
        "host_reserved": False,
        "warmups": 0,
        "solve_gate": 2.0,
        "provenance": {
            "created_at": base._provenance(source)["created_at"],
            "free_before_bytes": free,
            "evidence_budget_bytes": cpu.MAX_EVIDENCE,
            "scope": "Already-spent KITTI data, complete CPU calibration; no accuracy claim",
        },
    }
    _validate(protocol, "protocol")
    write(folder / "protocol.json", protocol)
    attempts = {
        "observations": [],
        "provenance": {"protocol_sha256": base.sha(folder / "protocol.json")},
    }
    for index, label in enumerate(ORDER):
        root = folder / label
        worker = f"{index:02d}-numba"
        path = root / worker
        # The original worker creates its own result folder; logs live next to it
        # until completion, then are moved inside for one closed artifact set.
        logs = root / f"{index:02d}-logs"
        logs.mkdir()
        env = os.environ.copy()
        env.update(
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
            CUDA_CACHE_DISABLE="1",
            CUPY_CACHE_DIR=str(folder / "unused-cupy"),
            NUMBA_CACHE_DIR=str(folder / "unused-numba"),
        )
        command = [
            sys.executable,
            str(folder / "toolchain/tools/online_grid_reuse_worker.py"),
            "numba",
            str(args.fixture.resolve()),
            str(root / "source"),
            str(path),
            commits[label],
            str(root / "receipt.json"),
        ]
        print(json.dumps({"starting": label, "worker": worker}), flush=True)
        start = time.perf_counter()
        status = execute_worker(command, env, logs)
        seconds = time.perf_counter() - start
        path.mkdir(exist_ok=True)
        for name in ("stdout.txt", "stderr.txt"):
            (logs / name).replace(path / name)
        logs.rmdir()
        observation = {
            "label": label,
            "worker": worker,
            "seconds": seconds,
            "exit_code": status,
            "artifact_sha256": {
                file.name: base.sha(file) for file in path.iterdir() if file.is_file()
            },
        }
        attempts["observations"].append(observation)
        _validate(attempts, "attempts")
        write(folder / "attempts.json", attempts)
        print(
            json.dumps(
                {key: item for key, item in observation.items() if key != "artifact_sha256"}
            ),
            flush=True,
        )
        if status:
            raise ValueError("worker failed; original attempt and logs retained")
        if (
            sum(file.stat().st_size for file in folder.rglob("*") if file.is_file())
            > cpu.MAX_EVIDENCE
        ):
            raise ValueError("comparison exceeds its 64 MiB evidence budget")
    base.verify_fixture(args.fixture, receipt)
    value = audit(folder, verify_git=True)
    write(folder / "summary.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--verify-git", action="store_true")
    args = parser.parse_args()
    if args.audit:
        value = audit(args.audit, verify_git=args.verify_git)
    else:
        if not all((args.baseline, args.fixture, args.receipt, args.output)):
            parser.error("measurement requires --baseline, --fixture, --receipt and --output")
        value = run(args)
    print(json.dumps(value, indent=2, allow_nan=False))
    return 0 if value["passes_solve_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
