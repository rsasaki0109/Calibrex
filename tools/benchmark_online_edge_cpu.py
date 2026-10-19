"""Small, frozen, cold-worker CPU comparison on the existing 48 KITTI pairs.

Raw images/scans are read in place. No dataset is downloaded or copied. The
evidence folder has a 64 MiB limit and requires 20 GiB free before launching.
"""

from __future__ import annotations

import argparse
import ast
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
from tools import benchmark_online_edge_cuda as base  # noqa: E402

ORDER = ("numpy", "fused", "numba", "numba", "fused", "numpy")
TOOLCHAIN = (*base.TOOLCHAIN, "tools/benchmark_online_edge_cpu.py")
MIN_FREE = 20 * 1024**3
MAX_EVIDENCE = 64 * 1024**2
POLICY = "ordered_fp64_projection/native_transform_dot_sum;fastmath=false;cache=false"
CPU_COUNTS = (
    "objective_evaluations",
    "point_rows",
    "grid_calls",
    "grid_pose_evaluations",
    "numba_frame_pose_evaluations",
    "native_frame_pose_evaluations",
)


def schemas(root: Path) -> dict[str, dict]:
    """Derive closed CPU benchmark schemas from the existing quality contract."""
    values = {
        name: base.read(root / f"tools/schemas/online_edge_cuda_{name}.schema.json")
        for name in ("config", "result", "records")
    }
    for key in ("edge_evaluation_backend", "grid_evaluation_backend"):
        values["config"]["properties"][key] = {"enum": ["numpy", "numba"]}
    result = values["result"]
    result["$defs"].pop("OnlineEdgeExecution")
    result["$defs"]["OnlineCpuExecution"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["backend", "runtime", "kernel_sha256", "policy", *CPU_COUNTS],
        "properties": {
            "backend": {"const": "numba"},
            "runtime": {"type": "string", "minLength": 1},
            "kernel_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "policy": {"const": POLICY},
            **{key: {"type": "integer", "minimum": 0} for key in CPU_COUNTS},
        },
    }
    result["properties"]["result"]["properties"]["execution"] = {
        "$ref": "#/$defs/OnlineCpuExecution"
    }

    def variants(value: object) -> None:
        if isinstance(value, dict):
            if value.get("enum") == ["numpy", "fused", "cuda"]:
                value["enum"] = ["numpy", "fused", "numba"]
            for child in value.values():
                variants(child)
        elif isinstance(value, list):
            for child in value:
                variants(child)

    for value in values.values():
        variants(value)
    for record in values["records"]["oneOf"]:
        props = record["properties"]
        kind = props.get("kind", {}).get("const")
        if kind == "protocol":
            props["order"] = {"const": list(ORDER)}
            props["toolchain_sha256"].update(
                minProperties=len(TOOLCHAIN), maxProperties=len(TOOLCHAIN)
            )
        elif kind == "comparison":
            props["observation_count"] = {"const": len(ORDER)}
            for key, names in (
                ("whole_worker_median_seconds", ["numpy", "fused", "numba"]),
                ("ratios", ["numpy", "fused"]),
            ):
                props[key] = {
                    "type": "object",
                    "required": names,
                    "additionalProperties": False,
                    "properties": {
                        name: {"type": "number", "exclusiveMinimum": 0} for name in names
                    },
                }
    return values


def kernel_digests(source: Path) -> dict[str, str]:
    """Read decorated kernel source without importing or executing the snapshot."""
    text = (source / "src/calibrex/solvers/numba_online_edge_adapter.py").read_text(
        encoding="utf-8"
    )
    lines = text.splitlines(keepends=True)
    values = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in {"_interpolate", "_image_response"}:
            start = min(node.lineno, *(item.lineno for item in node.decorator_list)) - 1
            values[node.name] = hashlib.sha256(
                "".join(lines[start : node.end_lineno]).encode()
            ).hexdigest()
    if set(values) != {"_interpolate", "_image_response"}:
        raise ValueError("frozen CPU kernels missing")
    return values


def _check_lineage(value: dict, protocol: dict, folder: Path, variant: str) -> None:
    expected = {
        "source_commit": protocol["source_commit"],
        "source_kind": protocol["source_kind"],
        "base_git_commit": protocol["base_git_commit"],
        "variant": variant,
        "protocol_sha256": base.sha(folder / "protocol.json"),
        "acquisition_receipt_sha256": protocol["receipt_sha256"],
        "source_manifest_sha256": protocol["source_manifest_sha256"],
        "launcher_sha256": protocol["toolchain_sha256"]["tools/online_edge_cuda_worker.py"],
    }
    if any(value["provenance"][key] != item for key, item in expected.items()):
        raise ValueError("CPU worker lineage differs")


def audit(folder: Path, *, verify_git: bool = False) -> dict:
    """Recheck full work, hashes and results on CPU without optional imports."""
    import jsonschema

    protocol, attempts = base.read(folder / "protocol.json"), base.read(folder / "attempts.json")
    schema = {
        name: base.read(folder / f"schemas/{name}.schema.json")
        for name in ("records", "result", "config")
    }
    for value in (protocol, attempts):
        jsonschema.validate(value, schema["records"])
    commit = protocol["source_commit"]
    if protocol["source_kind"] != "git-commit" or commit != protocol["base_git_commit"]:
        raise ValueError("CPU cohort requires committed source")
    if any(value["provenance"]["source_commit"] != commit for value in (protocol, attempts)):
        raise ValueError("CPU cohort lineage differs")
    if attempts["provenance"]["protocol_sha256"] != base.sha(folder / "protocol.json"):
        raise ValueError("CPU attempts changed")
    manifest = base._verify_sources(folder / "source")
    if base.sha(folder / "source/manifest.json") != protocol["source_manifest_sha256"]:
        raise ValueError("CPU source manifest changed")
    if set(protocol["toolchain_sha256"]) != set(TOOLCHAIN):
        raise ValueError("CPU toolchain file set differs")
    for name, digest in protocol["toolchain_sha256"].items():
        if base.sha(folder / "toolchain" / name) != digest:
            raise ValueError("CPU toolchain changed")
    for name, digest in protocol["provenance"]["schema_sha256"].items():
        if base.sha(folder / "schemas" / name) != digest:
            raise ValueError("CPU schema changed")
    if base.sha(folder / "receipt.json") != protocol["receipt_sha256"]:
        raise ValueError("CPU acquisition receipt changed")
    if verify_git:
        frozen = base._git_files(commit, ("src/calibrex", *TOOLCHAIN))
        expected = {
            name.removeprefix("src/"): hashlib.sha256(data).hexdigest()
            for name, data in frozen.items()
            if name.startswith("src/calibrex/") and name.endswith(".py")
        }
        if expected != manifest or any(
            hashlib.sha256(frozen[name]).hexdigest() != digest
            for name, digest in protocol["toolchain_sha256"].items()
        ):
            raise ValueError("CPU Git content differs")
    if [item["variant"] for item in attempts["observations"]] != list(ORDER):
        raise ValueError("incomplete CPU counterbalance")
    kernels, times = kernel_digests(folder / "source"), {name: [] for name in set(ORDER)}
    results, configs, captures, hashes = [], [], [], {}
    for index, observation in enumerate(attempts["observations"]):
        variant = observation["variant"]
        worker = f"{index:02d}-{variant}"
        if observation["worker"] != worker or observation["exit_code"] != 0:
            raise ValueError("CPU worker failed or was replaced")
        duration = observation["seconds"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("CPU duration invalid")
        times[variant].append(duration)
        root = folder / worker
        output = {
            name: base.read(root / f"{name}.json") for name in ("result", "config", "measurement")
        }
        for name, value in output.items():
            jsonschema.validate(value, schema["records" if name == "measurement" else name])
            _check_lineage(value, protocol, folder, variant)
        measured, config, quality = (
            output["measurement"],
            output["config"],
            output["result"]["result"],
        )
        for name in ("result", "config"):
            digest = base.sha(root / f"{name}.json")
            if digest != measured[f"{name}_sha256"]:
                raise ValueError("CPU output changed")
            hashes[f"{worker}/{name}.json"] = digest
        if any(
            manifest[name] != digest for name, digest in measured["imported_source_sha256"].items()
        ):
            raise ValueError("CPU imported source differs")
        if measured["evaluations"] != 21958 or measured["point_rows"] != 395476000:
            raise ValueError("complete CPU workload differs")
        if measured["native_raw_grid_pose_evaluations"] != (21840 if variant == "fused" else 0):
            raise ValueError("compiled CPU benchmark-control workload differs")
        if len(quality["timeline"]) != 30 or len(quality["holdout_frame_ids"]) != 10:
            raise ValueError("CPU rolling/holdout population differs")
        for key in ("edge_evaluation_backend", "grid_evaluation_backend"):
            if config.pop(key) != ("numba" if variant == "numba" else "numpy"):
                raise ValueError("CPU backend configuration differs")
        config.pop("provenance")
        execution = quality.get("execution")
        if variant == "numba":
            expected = {
                "backend": "numba",
                "kernel_sha256": kernels["_interpolate"],
                "policy": POLICY,
                "objective_evaluations": 21958,
                "point_rows": 395476000,
                "grid_calls": 30,
                "grid_pose_evaluations": 21840,
                "numba_frame_pose_evaluations": 197738,
                "native_frame_pose_evaluations": 0,
            }
            if not execution or any(execution[key] != value for key, value in expected.items()):
                raise ValueError("compiled CPU execution differs")
            if measured["image_kernel_sha256"] != kernels["_image_response"]:
                raise ValueError("compiled CPU image kernel differs")
            software = measured["software"]
            if execution["runtime"] != (
                f"calibrex.numba_online_edge/v0.1;numba={software['numba']};numpy={software['numpy']}"
            ):
                raise ValueError("compiled CPU software lineage differs")
        elif execution is not None or measured["image_kernel_sha256"] is not None:
            raise ValueError("NumPy control used an accelerator")
        results.append(base._comparable(quality))
        configs.append(config)
        captures.append(measured["prepared_capture_sha256"])
    if (
        any(value != results[0] for value in results)
        or any(value != configs[0] for value in configs)
        or len(set(captures)) != 1
    ):
        raise ValueError("complete CPU results, configurations or prepared bits differ")
    medians = {key: statistics.median(items) for key, items in times.items()}
    ratios = {key: medians[key] / medians["numba"] for key in ("numpy", "fused")}
    result = {
        "kind": "comparison",
        "complete_results_exact": True,
        "prepared_snapshots_exact": True,
        "observation_count": len(ORDER),
        "whole_worker_median_seconds": medians,
        "ratios": ratios,
        "passes_gate": ratios["numpy"] >= 2.0,
        "source_binding_git_verified": verify_git,
        "result_sha256": hashes,
        "provenance": {
            **base._provenance(protocol),
            "protocol_sha256": base.sha(folder / "protocol.json"),
            "attempts_sha256": base.sha(folder / "attempts.json"),
        },
    }
    jsonschema.validate(result, schema["records"])
    if (folder / "comparison.json").exists():
        saved = base.read(folder / "comparison.json")
        jsonschema.validate(saved, schema["records"])
        if any(
            saved[key] != value
            for key, value in result.items()
            if key not in {"provenance", "source_binding_git_verified"}
        ):
            raise ValueError("saved CPU comparison differs")
        for key in (
            "source_commit",
            "source_kind",
            "base_git_commit",
            "protocol_sha256",
            "attempts_sha256",
        ):
            if saved["provenance"][key] != result["provenance"][key]:
                raise ValueError("saved CPU comparison lineage differs")
    return result


def run(args: argparse.Namespace) -> dict:
    parent = args.output.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(parent).free
    if free < MIN_FREE:
        raise ValueError("CPU comparison requires 20 GiB free; reuse the existing dataset")
    commit = subprocess.check_output(["git", "rev-parse", args.commit], text=True).strip()
    frozen = base._git_files(commit, TOOLCHAIN)
    if any(
        (ROOT / name).read_text(encoding="utf-8").encode() != frozen[name] for name in TOOLCHAIN
    ):
        raise ValueError("commit CPU benchmark tools before measuring")
    receipt = base.read(args.receipt)
    base.verify_fixture(args.fixture, receipt)
    folder = args.output.resolve()
    folder.mkdir()
    base._freeze(folder / "source", commit)
    for path in (folder / "source/src").rglob("*.py"):
        path.write_bytes(path.read_text(encoding="utf-8").encode())
    (folder / "source/manifest.json").write_text(
        json.dumps(base._manifest(folder / "source/src"), indent=2) + "\n", encoding="utf-8"
    )
    for name, data in frozen.items():
        target = folder / "toolchain" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (folder / "schemas").mkdir()
    for name, value in schemas(folder / "toolchain").items():
        (folder / f"schemas/{name}.schema.json").write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
    (folder / "receipt.json").write_bytes(args.receipt.read_bytes())
    record_schema = folder / "schemas/records.schema.json"
    protocol = {
        "kind": "protocol",
        "source_commit": commit,
        "base_git_commit": commit,
        "source_kind": "git-commit",
        "source_manifest_sha256": base.sha(folder / "source/manifest.json"),
        "toolchain_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in frozen.items()
        },
        "receipt_sha256": base.sha(folder / "receipt.json"),
        "order": list(ORDER),
        "blas_threads": 1,
        "preparation_workers": 3,
        "warmups": 0,
        "host_reserved": False,
        "whole_worker_gate": 2.0,
    }
    protocol["provenance"] = {
        **base._provenance(protocol),
        "disk_free_before_bytes": free,
        "evidence_budget_bytes": MAX_EVIDENCE,
        "schema_sha256": {
            path.name: base.sha(path) for path in (folder / "schemas").glob("*.json")
        },
    }
    base._write(folder / "protocol.json", protocol, record_schema)
    observations = []
    for index, variant in enumerate(ORDER):
        worker = f"{index:02d}-{variant}"
        env = os.environ.copy()
        env.update(
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            NUMBA_CACHE_DIR=str(folder / "cache" / worker),
            CUPY_CACHE_DIR=str(folder / "unused-cupy"),
            CUDA_CACHE_DISABLE="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        print(json.dumps({"starting": worker}), flush=True)
        with (
            (folder / (worker + "-stdout.txt")).open("wb") as stdout,
            (folder / (worker + "-stderr.txt")).open("wb") as stderr,
        ):
            start = time.perf_counter()
            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        str(folder / "toolchain/tools/online_edge_cuda_worker.py"),
                        variant,
                        str(args.fixture.resolve()),
                        str(folder / "source"),
                        str(folder / worker),
                        commit,
                        str(folder / "receipt.json"),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=600,
                    check=False,
                )
                status = process.returncode
            except subprocess.TimeoutExpired:
                status = -1
            observation = {
                "variant": variant,
                "worker": worker,
                "seconds": time.perf_counter() - start,
                "exit_code": status,
            }
        observations.append(observation)
        base._write(
            folder / "attempts.json",
            {
                "kind": "attempts",
                "observations": observations,
                "provenance": {
                    **base._provenance(protocol),
                    "protocol_sha256": base.sha(folder / "protocol.json"),
                },
            },
            record_schema,
        )
        print(json.dumps(observation), flush=True)
        if status:
            raise ValueError("CPU worker failed; original attempt retained")
        if sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) > MAX_EVIDENCE:
            raise ValueError("CPU evidence exceeds its 64 MiB budget")
    base.verify_fixture(args.fixture, receipt)
    value = audit(folder, verify_git=True)
    base._write(folder / "comparison.json", value, record_schema)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--verify-git", action="store_true")
    args = parser.parse_args()
    if args.audit:
        value = audit(args.audit, verify_git=args.verify_git)
    else:
        if not all((args.fixture, args.receipt, args.output)):
            parser.error("measurement requires --fixture, --receipt and --output")
        value = run(args)
    print(json.dumps(value, indent=2, allow_nan=False))
    return 0 if value["passes_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
