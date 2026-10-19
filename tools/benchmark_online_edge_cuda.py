"""Frozen-source, complete-worker online edge comparison and CPU-only audit.

Use an already-downloaded, checksum-pinned KITTI subset. No raw data is copied
to the evidence directory. This is performance development on spent data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_bspline_lidar_imu import _freeze, _manifest, _verify_sources  # noqa: E402

ORDER = ("numpy", "fused", "cuda", "cuda", "fused", "numpy")
TOOLCHAIN = (
    "tools/benchmark_online_edge_cuda.py",
    "tools/online_edge_cuda_worker.py",
    "tools/online_edge_fused_control.py",
    "tools/benchmark_bspline_lidar_imu.py",
    "tools/schemas/online_edge_cuda_result.schema.json",
    "tools/schemas/online_edge_cuda_config.schema.json",
    "tools/schemas/online_edge_cuda_records.schema.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _provenance(protocol: dict) -> dict:
    return {
        "source_commit": protocol["source_commit"],
        "base_git_commit": protocol["base_git_commit"],
        "source_kind": protocol["source_kind"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _write(path: Path, value: dict, schema: Path) -> None:
    import jsonschema

    jsonschema.validate(value, read(schema))
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _git_files(commit: str, paths: tuple[str, ...]) -> dict[str, bytes]:
    raw = subprocess.check_output(["git", "archive", commit, *paths])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        files = {}
        for member in archive.getmembers():
            if member.isfile():
                content = archive.extractfile(member)
                assert content is not None
                files[member.name] = content.read()
        return files


def verify_fixture(folder: Path, receipt: dict) -> None:
    expected_files = {
        f"{stream}/data/{index:010d}.{extension}"
        for stream, extension in (("image_02", "png"), ("velodyne_points", "bin"))
        for index in range(15, 486, 10)
    } | {
        "calib_cam_to_cam.txt",
        "calib_velo_to_cam.txt",
        "calib_imu_to_velo.txt",
        "image_02/timestamps.txt",
        "velodyne_points/timestamps.txt",
    }
    if (
        receipt["complete"] is not True
        or receipt["frame_indices"] != list(range(15, 486, 10))
        or receipt["size"] != 11320688364
        or receipt["etag"] != "e933e363d73c5e057e18e6890463b47a-1350"
        or set(receipt["files"]) != expected_files
        or receipt["source_url"]
        != (
            "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
            "2011_09_30_drive_0018/2011_09_30_drive_0018_sync.zip"
        )
    ):
        raise ValueError("requires the pinned 48-pair KITTI drive0018 subset")
    folder = folder.resolve()
    for name, info in receipt["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("fixture member escapes its root")
        if path.stat().st_size != info["size"] or sha(path) != info["sha256"]:
            raise ValueError(f"fixture member changed: {name}")


def _kernel_digests(source: Path) -> dict[str, str]:
    module = source / "src/calibrex/solvers/cuda_online_edge_adapter.py"
    values = {}
    for node in ast.parse(module.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"_KERNEL", "_IMAGE_KERNEL"}
        ):
            values[node.targets[0].id] = hashlib.sha256(
                ast.literal_eval(node.value).encode()
            ).hexdigest()
    if set(values) != {"_KERNEL", "_IMAGE_KERNEL"}:
        raise ValueError("frozen kernel definitions missing")
    return values


def _comparable(result: dict) -> dict:
    value = json.loads(json.dumps(result))
    value.pop("execution", None)
    return value


def audit(folder: Path, *, verify_git: bool = False) -> dict:
    """Revalidate all full outputs and timings without importing CUDA or Calibrex."""
    import jsonschema

    protocol = read(folder / "protocol.json")
    attempts = read(folder / "attempts.json")
    meta_schema = folder / "schemas/records.schema.json"
    result_schema = read(folder / "schemas/result.schema.json")
    config_schema = read(folder / "schemas/config.schema.json")
    jsonschema.validate(protocol, read(meta_schema))
    jsonschema.validate(attempts, read(meta_schema))
    commit = protocol["source_commit"]
    if (protocol["source_kind"] == "git-commit" and commit != protocol["base_git_commit"]) or (
        protocol["source_kind"] == "working-copy" and commit is not None
    ):
        raise ValueError("source kind and Git lineage differ")
    if (
        protocol["provenance"]["source_commit"] != commit
        or attempts["provenance"]["source_commit"] != commit
        or attempts["provenance"]["protocol_sha256"] != sha(folder / "protocol.json")
    ):
        raise ValueError("cohort lineage differs")
    if set(protocol["toolchain_sha256"]) != set(TOOLCHAIN):
        raise ValueError("toolchain file set differs")
    manifest = _verify_sources(folder / "source")
    if sha(folder / "source/manifest.json") != protocol["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    for name, digest in protocol["toolchain_sha256"].items():
        if sha(folder / "toolchain" / name) != digest:
            raise ValueError("frozen toolchain changed")
    for name in ("result", "config", "records"):
        path = folder / "schemas" / (name + ".schema.json")
        if (
            sha(path)
            != protocol["toolchain_sha256"][f"tools/schemas/online_edge_cuda_{name}.schema.json"]
        ):
            raise ValueError("artifact schema changed")
    if sha(folder / "receipt.json") != protocol["receipt_sha256"]:
        raise ValueError("acquisition receipt changed")
    if verify_git:
        if commit is None:
            raise ValueError("working-copy evidence has no committed source binding")
        expected = _git_files(commit, ("src/calibrex", *TOOLCHAIN))
        git_manifest = {
            name.removeprefix("src/"): hashlib.sha256(data).hexdigest()
            for name, data in expected.items()
            if name.startswith("src/calibrex/") and name.endswith(".py")
        }
        if git_manifest != manifest:
            raise ValueError("Git source binding differs")
        for name, digest in protocol["toolchain_sha256"].items():
            if hashlib.sha256(expected[name]).hexdigest() != digest:
                raise ValueError("Git toolchain binding differs")
    observations = attempts["observations"]
    if [item["variant"] for item in observations] != list(ORDER):
        raise ValueError("incomplete counterbalanced cohort")
    kernels = _kernel_digests(folder / "source")
    values, snapshots, configurations = [], [], []
    times: dict[str, list[float]] = {variant: [] for variant in set(ORDER)}
    artifact_hashes = {}
    for index, observation in enumerate(observations):
        variant = observation["variant"]
        worker = f"{index:02d}-{variant}"
        if observation["worker"] != worker or observation["exit_code"] != 0:
            raise ValueError("worker failed or was substituted")
        if not math.isfinite(observation["seconds"]) or observation["seconds"] <= 0:
            raise ValueError("invalid worker duration")
        times[variant].append(observation["seconds"])
        root = folder / worker
        result = read(root / "result.json")
        config = read(root / "config.json")
        measured = read(root / "measurement.json")
        jsonschema.validate(result, result_schema)
        jsonschema.validate(config, config_schema)
        jsonschema.validate(measured, read(meta_schema))
        if measured["native_raw_grid_pose_evaluations"] != (21840 if variant == "fused" else 0):
            raise ValueError("raw-only CPU control workload differs")
        configuration = dict(config)
        configuration.pop("provenance")
        for key in ("edge_evaluation_backend", "grid_evaluation_backend"):
            if configuration.pop(key) != ("cuda" if variant == "cuda" else "numpy"):
                raise ValueError("configuration backend differs")
        configurations.append(configuration)
        for value in (result, config, measured):
            provenance = value["provenance"]
            if (
                provenance["source_commit"] != commit
                or provenance["source_kind"] != protocol["source_kind"]
                or provenance["base_git_commit"] != protocol["base_git_commit"]
                or provenance["protocol_sha256"] != sha(folder / "protocol.json")
                or provenance["acquisition_receipt_sha256"] != protocol["receipt_sha256"]
                or provenance["source_manifest_sha256"] != protocol["source_manifest_sha256"]
                or provenance["launcher_sha256"]
                != protocol["toolchain_sha256"]["tools/online_edge_cuda_worker.py"]
                or provenance["variant"] != variant
            ):
                raise ValueError("worker lineage differs")
        for artifact in ("result", "config"):
            digest = sha(root / (artifact + ".json"))
            if digest != measured[artifact + "_sha256"]:
                raise ValueError("worker artifact changed")
            artifact_hashes[f"{worker}/{artifact}.json"] = digest
        if any(
            manifest[name] != digest for name, digest in measured["imported_source_sha256"].items()
        ):
            raise ValueError("executed source receipt differs")
        quality = result["result"]
        if len(quality["timeline"]) != 30 or len(quality["holdout_frame_ids"]) != 10:
            raise ValueError("full rolling or holdout workload missing")
        snapshots.append(measured["prepared_capture_sha256"])
        execution = quality.get("execution")
        if variant == "cuda":
            if (
                not execution
                or execution["kernel_sha256"] != kernels["_KERNEL"]
                or measured["image_kernel_sha256"] != kernels["_IMAGE_KERNEL"]
                or execution["capture_sha256"] != snapshots[-1]
                or execution["policy"]
                != "fp64_raw_score_enclosure/v0.1;native_ambiguous_and_finalists"
                or execution["grid_calls"] != 30
                or execution["grid_pose_evaluations"] != 21840
                or execution["cuda_frame_pose_evaluations"] != 196560
                or execution["max_device_snapshot_bytes"] > execution["device_snapshot_limit_bytes"]
                or execution["max_scratch_bytes"] >= execution["owned_device_limit_bytes"]
                or measured["cuda_cache_empty"] is not True
                or measured["cuda_driver_cache_disabled"] is not True
            ):
                raise ValueError("CUDA execution contract differs")
        elif execution is not None or measured["image_kernel_sha256"] is not None:
            raise ValueError("CPU control used CUDA")
        elif measured["evaluations"] != 21958 or measured["point_rows"] != 395476000:
            raise ValueError("CPU control evaluation workload differs")
        values.append(_comparable(quality))
    if (
        any(value != values[0] for value in values)
        or any(value != configurations[0] for value in configurations)
        or len(set(snapshots)) != 1
    ):
        raise ValueError("complete results or prepared frame bits differ")
    medians = {variant: statistics.median(items) for variant, items in times.items()}
    ratios = {variant: medians[variant] / medians["cuda"] for variant in ("numpy", "fused")}
    comparison = {
        "kind": "comparison",
        "complete_results_exact": True,
        "prepared_snapshots_exact": True,
        "observation_count": len(observations),
        "whole_worker_median_seconds": medians,
        "ratios": ratios,
        "passes_gate": all(value >= 2.0 for value in ratios.values()),
        "source_binding_git_verified": verify_git,
        "result_sha256": artifact_hashes,
        "provenance": {
            **_provenance(protocol),
            "protocol_sha256": sha(folder / "protocol.json"),
            "attempts_sha256": sha(folder / "attempts.json"),
        },
    }
    if (folder / "comparison.json").exists():
        saved = read(folder / "comparison.json")
        jsonschema.validate(saved, read(meta_schema))
        for key, value in comparison.items():
            if key not in {"provenance", "source_binding_git_verified"} and saved[key] != value:
                raise ValueError("saved comparison differs from complete observations")
        for key in (
            "source_commit",
            "base_git_commit",
            "source_kind",
            "protocol_sha256",
            "attempts_sha256",
        ):
            if saved["provenance"][key] != comparison["provenance"][key]:
                raise ValueError("saved comparison lineage differs")
    return comparison


def run(args: argparse.Namespace) -> dict:
    folder = args.output.resolve()
    folder.mkdir()
    if args.working_copy and args.commit != "HEAD":
        raise ValueError("working-copy mode uses the current HEAD as its base")
    base = subprocess.check_output(["git", "rev-parse", args.commit], text=True).strip()
    commit = None if args.working_copy else base
    if args.working_copy:
        frozen = {name: (ROOT / name).read_text(encoding="utf-8").encode() for name in TOOLCHAIN}
    else:
        frozen = _git_files(commit, TOOLCHAIN)
        if any(
            (ROOT / name).read_text(encoding="utf-8").encode() != data
            for name, data in frozen.items()
        ):
            raise ValueError("commit benchmark tools before measuring")
    _freeze(folder / "source", commit)
    if args.working_copy:
        for path in (folder / "source/src").rglob("*.py"):
            path.write_bytes(path.read_text(encoding="utf-8").encode())
        (folder / "source/manifest.json").write_text(
            json.dumps(_manifest(folder / "source/src"), indent=2) + "\n", encoding="utf-8"
        )
    for name, data in frozen.items():
        target = folder / "toolchain" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    schema_dir = folder / "schemas"
    schema_dir.mkdir()
    for name in ("config", "result", "records"):
        (schema_dir / (name + ".schema.json")).write_bytes(
            frozen[f"tools/schemas/online_edge_cuda_{name}.schema.json"]
        )
    schema = schema_dir / "records.schema.json"
    receipt = read(args.receipt)
    verify_fixture(args.fixture, receipt)
    (folder / "receipt.json").write_bytes(args.receipt.read_bytes())
    protocol = {
        "kind": "protocol",
        "source_commit": commit,
        "base_git_commit": base,
        "source_kind": "working-copy" if args.working_copy else "git-commit",
        "source_manifest_sha256": sha(folder / "source/manifest.json"),
        "toolchain_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in frozen.items()
        },
        "receipt_sha256": sha(folder / "receipt.json"),
        "order": list(ORDER),
        "blas_threads": 1,
        "preparation_workers": 3,
        "warmups": 0,
        "host_reserved": False,
        "whole_worker_gate": 2.0,
    }
    protocol["provenance"] = _provenance(protocol)
    _write(folder / "protocol.json", protocol, schema)
    observations = []
    for index, variant in enumerate(ORDER):
        worker = f"{index:02d}-{variant}"
        env = os.environ.copy()
        env.update(
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            CUPY_CACHE_DIR=str(folder / "cache" / (worker + "-cupy")),
            NUMBA_CACHE_DIR=str(folder / "cache" / (worker + "-numba")),
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
                        commit or "working-copy",
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
            seconds = time.perf_counter() - start
        observation = {
            "variant": variant,
            "worker": worker,
            "seconds": seconds,
            "exit_code": status,
        }
        observations.append(observation)
        _write(
            folder / "attempts.json",
            {
                "kind": "attempts",
                "observations": observations,
                "provenance": {
                    **_provenance(protocol),
                    "protocol_sha256": sha(folder / "protocol.json"),
                },
            },
            schema,
        )
        print(json.dumps(observation), flush=True)
        if status:
            raise ValueError("worker failed; all attempted artifacts retained")
    verify_fixture(args.fixture, receipt)
    comparison = audit(folder, verify_git=not args.working_copy)
    _write(folder / "comparison.json", comparison, schema)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument(
        "--working-copy",
        action="store_true",
        help="freeze development bytes without a committed-source claim",
    )
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--verify-git", action="store_true")
    args = parser.parse_args()
    if args.audit:
        value = audit(args.audit.resolve(), verify_git=args.verify_git)
    else:
        if not all((args.fixture, args.receipt, args.output)):
            parser.error("--fixture, --receipt, and --output are required")
        value = run(args)
    print(json.dumps(value, indent=2), flush=True)
    return 0 if value["passes_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
