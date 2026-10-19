"""Compare complete offline estimates on 48 already-spent, pinned KITTI pairs.

Commit source and this tool before measuring. Images/scans are read in place.
The gate is 2x on estimate_from_frames, including preparation and every fit and
control. Fresh-worker time, including decoding and imports, is separate.
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
from tools import benchmark_online_edge_cuda as base  # noqa: E402

ORDER = ("serial", "parallel", "parallel", "serial", "serial", "parallel")
WORKERS = {"serial": 1, "parallel": 8}
TOOLCHAIN = (
    "tools/benchmark_offline_edge_preparation.py",
    "tools/benchmark_online_edge_cuda.py",
    "tools/benchmark_bspline_lidar_imu.py",
)
RESULT_SCHEMA = "schemas/camera_lidar_edge.schema.json"


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def quality(value: dict) -> bytes:
    value = json.loads(json.dumps(value))
    value["options"].pop("preparation_workers")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def worker(args: argparse.Namespace) -> None:
    folder = args.output.resolve()
    sys.path.insert(0, str(folder / "source/src"))
    import numpy as np

    from calibrex.core.camera_lidar_edge import CameraLidarEdgeProvenance
    from calibrex.data.kitti import read_calibration_file, read_png_luminance_array
    from calibrex.data.ros2_camera_lidar import CameraLidarFrame
    from calibrex.evaluation import camera_lidar_edge as edge

    manifest = base._verify_sources(folder / "source")
    protocol = base.read(folder / "protocol.json")
    if base.sha(Path(__file__)) != protocol["toolchain_sha256"][TOOLCHAIN[0]]:
        raise ValueError("worker differs from frozen toolchain")
    fixture = args.fixture.resolve()
    base.verify_fixture(fixture, base.read(folder / "receipt.json"))
    frames = []
    for index in range(15, 486, 10):
        gray = read_png_luminance_array(fixture / f"image_02/data/{index:010d}.png")
        if gray is None:
            raise ValueError("unsupported PNG")
        image = np.rint(gray).clip(0, 255).astype(np.uint8)
        points = np.fromfile(
            fixture / f"velodyne_points/data/{index:010d}.bin", dtype="<f4"
        ).reshape(-1, 4)[:, :3]
        frames.append(CameraLidarFrame(index / 10, index / 10, image, points, "lidar", "camera"))
    calibration = read_calibration_file(fixture / "calib_cam_to_cam.txt")
    extrinsic = read_calibration_file(fixture / "calib_velo_to_cam.txt")
    projection = np.asarray(calibration["P_rect_02"]).reshape(3, 4)
    intrinsic = projection[:, :3]
    rect, velo, shift = np.eye(4), np.eye(4), np.eye(4)
    rect[:3, :3] = np.asarray(calibration["R_rect_00"]).reshape(3, 3)
    velo[:3, :3] = np.asarray(extrinsic["R"]).reshape(3, 3)
    velo[:3, 3] = extrinsic["T"]
    shift[:3, 3] = np.linalg.solve(intrinsic, projection[:, 3])
    candidate = shift @ rect @ velo
    camera = edge.EdgeCamera(
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
    )
    provenance = CameraLidarEdgeProvenance(
        generator="offline-edge-preparation-benchmark",
        generator_version="0.1",
        dataset_family="KITTI raw, spent performance development",
        sequence_ids=["2011_09_30_drive_0018_sync"],
        image_topic="image_02",
        lidar_topic="velodyne_points",
        input_sha256=protocol["receipt_sha256"],
        input_digest_scope=(
            "Pinned 48-pair receipt; luminance rounded to uint8; index/10 timestamps; "
            "reference initialized performance control, no accuracy evaluation"
        ),
        dataset_license="KITTI non-commercial share-alike",
        created_at=protocol["provenance"]["created_at"],
    )
    original_prepare, original_score = edge.prepare_frames, edge.EdgeObjective.score
    prep, calls = {}, 0

    def prepare(*arguments, **kwargs):
        start = time.perf_counter()
        prepared, dropped = original_prepare(*arguments, **kwargs)
        prep["seconds"] = time.perf_counter() - start
        digest = hashlib.sha256()
        for frame in prepared:
            digest.update(repr((frame.time_s, frame.gap_s, frame.scale, frame.rings)).encode())
            for array in (frame.edge, frame.points, frame.weights):
                digest.update(repr((array.shape, array.dtype.str)).encode())
                digest.update(array.tobytes(order="C"))
        prep.update(
            frames=len(prepared),
            points=sum(len(frame.points) for frame in prepared),
            capture_sha256=digest.hexdigest(),
            dropped=dropped,
        )
        return prepared, dropped

    def score(*arguments, **kwargs):
        nonlocal calls
        calls += 1
        return original_score(*arguments, **kwargs)

    label = args.worker.split("-", 1)[1]
    edge.prepare_frames, edge.EdgeObjective.score = prepare, score
    try:
        start = time.perf_counter()
        artifact = edge.estimate_from_frames(
            frames,
            camera,
            candidate,
            options=edge.EdgeAlignmentOptions(preparation_workers=WORKERS[label]),
            provenance=provenance,
        )
        seconds = time.perf_counter() - start
    finally:
        edge.prepare_frames, edge.EdgeObjective.score = original_prepare, original_score
    destination = folder / args.worker
    write(destination / "result.json", artifact.model_dump(mode="json"))
    imported = {}
    for name, module in list(sys.modules.items()):
        if name.startswith("calibrex") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(folder / "source/src"):
                raise ValueError("import escaped frozen source")
            relative = path.relative_to(folder / "source/src").as_posix()
            if base.sha(path) != manifest[relative]:
                raise ValueError("import differs from manifest")
            imported[relative] = base.sha(path)
    base._verify_sources(folder / "source")
    write(
        destination / "measurement.json",
        {
            "estimate_seconds": seconds,
            "preparation": prep,
            "score_calls": calls,
            "imported_source_sha256": imported,
            "result_sha256": base.sha(destination / "result.json"),
            "provenance": {"protocol_sha256": base.sha(folder / "protocol.json")},
        },
    )


def audit(folder: Path, *, verify_git: bool = False) -> dict:
    """Replay complete records without importing Calibrex or executing snapshots."""
    import jsonschema

    protocol = base.read(folder / "protocol.json")
    attempts = base.read(folder / "attempts.json")
    if protocol["order"] != list(ORDER) or protocol["workers"] != WORKERS:
        raise ValueError("comparison order or worker budgets differ")
    if protocol["estimate_gate"] != 2.0 or protocol["blas_threads"] != 1:
        raise ValueError("gate or BLAS budget differs")
    if (
        protocol["source_kind"] != "git-commit"
        or protocol["source_commit"] != protocol["base_git_commit"]
    ):
        raise ValueError("committed source lineage differs")
    manifest = base._verify_sources(folder / "source")
    if base.sha(folder / "source/manifest.json") != protocol["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    if set(protocol["toolchain_sha256"]) != set(TOOLCHAIN):
        raise ValueError("toolchain set differs")
    for name, digest in protocol["toolchain_sha256"].items():
        if base.sha(folder / "toolchain" / name) != digest:
            raise ValueError("toolchain changed")
    if verify_git:
        frozen = base._git_files(
            protocol["source_commit"], ("src/calibrex", *TOOLCHAIN, RESULT_SCHEMA)
        )
        expected = {
            name.removeprefix("src/"): hashlib.sha256(data).hexdigest()
            for name, data in frozen.items()
            if name.startswith("src/") and name.endswith(".py")
        }
        normalized = {
            name: hashlib.sha256(
                (folder / "source/src" / name).read_text(encoding="utf-8").encode()
            ).hexdigest()
            for name in manifest
        }
        if expected != normalized:
            raise ValueError("Git source differs")
        for name in (*TOOLCHAIN, RESULT_SCHEMA):
            path = folder / ("result.schema.json" if name == RESULT_SCHEMA else "toolchain/" + name)
            if path.read_bytes() != frozen[name]:
                raise ValueError("Git toolchain/schema differs")
    if base.sha(folder / "receipt.json") != protocol["receipt_sha256"]:
        raise ValueError("acquisition receipt changed")
    if base.sha(folder / "result.schema.json") != protocol["result_schema_sha256"]:
        raise ValueError("result schema changed")
    schema = base.read(folder / "result.schema.json")
    if len(attempts["observations"]) != len(ORDER):
        raise ValueError("missing attempt")
    protocol_sha = base.sha(folder / "protocol.json")
    if attempts["provenance"]["protocol_sha256"] != protocol_sha:
        raise ValueError("attempts lineage differs")
    reference, preparation, work = None, None, None
    samples = {
        label: {key: [] for key in ("estimate", "preparation", "fresh_worker")} for label in WORKERS
    }
    for index, (label, observation) in enumerate(zip(ORDER, attempts["observations"], strict=True)):
        name = f"{index:02d}-{label}"
        if observation["worker"] != name or observation["exit_code"] != 0:
            raise ValueError("worker order or status differs")
        files = {path.name: base.sha(path) for path in (folder / name).iterdir() if path.is_file()}
        if files != observation["artifact_sha256"] or set(files) != {
            "result.json",
            "measurement.json",
            "stdout.txt",
            "stderr.txt",
        }:
            raise ValueError("worker artifacts differ")
        result = base.read(folder / name / "result.json")
        jsonschema.validate(result, schema)
        if result["options"]["preparation_workers"] != WORKERS[label]:
            raise ValueError("result worker budget differs")
        result_quality = quality(result)
        reference = result_quality if reference is None else reference
        if result_quality != reference:
            raise ValueError("complete quality bits differ")
        measurement = base.read(folder / name / "measurement.json")
        if (
            measurement["provenance"]["protocol_sha256"] != protocol_sha
            or measurement["result_sha256"] != files["result.json"]
        ):
            raise ValueError("measurement lineage differs")
        imports = measurement["imported_source_sha256"]
        if not imports or any(manifest.get(name) != digest for name, digest in imports.items()):
            raise ValueError("actual imports differ")
        prepared = {
            key: value for key, value in measurement["preparation"].items() if key != "seconds"
        }
        preparation = prepared if preparation is None else preparation
        work = measurement["score_calls"] if work is None else work
        if prepared != preparation or measurement["score_calls"] != work or work < 1:
            raise ValueError("prepared data or objective work differs")
        times = {
            "estimate": measurement["estimate_seconds"],
            "preparation": measurement["preparation"]["seconds"],
            "fresh_worker": observation["seconds"],
        }
        if (
            any(not math.isfinite(value) or value <= 0 for value in times.values())
            or not times["preparation"] < times["estimate"] < times["fresh_worker"]
        ):
            raise ValueError("invalid timing boundaries")
        for key, value in times.items():
            samples[label][key].append(value)
    medians = {
        label: {key: statistics.median(values) for key, values in sample.items()}
        for label, sample in samples.items()
    }
    ratios = {key: medians["serial"][key] / medians["parallel"][key] for key in medians["serial"]}
    value = {
        "medians_seconds": medians,
        "ratios": ratios,
        "passes_estimate_gate": ratios["estimate"] >= protocol["estimate_gate"],
        "complete_quality_bits_equal": True,
        "prepared_bits_equal": True,
        "quality_sha256": hashlib.sha256(reference).hexdigest(),
        "score_calls": work,
        "prepared": preparation,
        "provenance": {
            "source_commit": protocol["source_commit"],
            "protocol_sha256": protocol_sha,
            "attempts_sha256": base.sha(folder / "attempts.json"),
        },
    }
    if (folder / "summary.json").exists() and value != base.read(folder / "summary.json"):
        raise ValueError("saved summary differs")
    return value


def run(args: argparse.Namespace) -> dict:
    folder = args.output.resolve()
    if shutil.disk_usage(folder.parent).free < 20 * 1024**3:
        raise ValueError("requires 20 GiB free")
    commit = subprocess.check_output(["git", "rev-parse", args.commit], text=True).strip()
    frozen = base._git_files(commit, (*TOOLCHAIN, RESULT_SCHEMA))
    for name in TOOLCHAIN:
        if (ROOT / name).read_text(encoding="utf-8").encode() != frozen[name]:
            raise ValueError("commit toolchain before measuring")
    base.verify_fixture(args.fixture, base.read(args.receipt))
    folder.mkdir()
    base._freeze(folder / "source", commit)
    # Git archive honors Windows checkout line endings. Bind the canonical
    # Python text, as the existing online benchmark drivers do.
    for path in (folder / "source/src").rglob("*.py"):
        path.write_bytes(path.read_text(encoding="utf-8").encode())
    write(folder / "source/manifest.json", base._manifest(folder / "source/src"))
    for name in TOOLCHAIN:
        path = folder / "toolchain" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(frozen[name])
    (folder / "result.schema.json").write_bytes(frozen[RESULT_SCHEMA])
    (folder / "receipt.json").write_bytes(args.receipt.read_bytes())
    protocol = {
        "source_commit": commit,
        "base_git_commit": commit,
        "source_kind": "git-commit",
        "source_manifest_sha256": base.sha(folder / "source/manifest.json"),
        "toolchain_sha256": {name: hashlib.sha256(frozen[name]).hexdigest() for name in TOOLCHAIN},
        "result_schema_sha256": base.sha(folder / "result.schema.json"),
        "receipt_sha256": base.sha(args.receipt),
        "order": list(ORDER),
        "workers": WORKERS,
        "estimate_gate": 2.0,
        "blas_threads": 1,
        "host_reserved": False,
        "warmups": 0,
        "provenance": {
            **base._provenance(
                {"source_commit": commit, "base_git_commit": commit, "source_kind": "git-commit"}
            ),
            "scope": (
                "Spent-data performance control; complete estimate API and fresh-worker "
                "time reported separately; no accuracy or SOTA claim"
            ),
        },
    }
    write(folder / "protocol.json", protocol)
    attempts = {
        "observations": [],
        "provenance": {"protocol_sha256": base.sha(folder / "protocol.json")},
    }
    for index, label in enumerate(ORDER):
        name = f"{index:02d}-{label}"
        destination = folder / name
        destination.mkdir()
        env = os.environ.copy()
        env.update(
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        command = [
            sys.executable,
            str(folder / "toolchain" / TOOLCHAIN[0]),
            "--worker",
            name,
            "--output",
            str(folder),
            "--fixture",
            str(args.fixture.resolve()),
        ]
        print(json.dumps({"starting": name}), flush=True)
        start = time.perf_counter()
        with (
            (destination / "stdout.txt").open("wb") as stdout,
            (destination / "stderr.txt").open("wb") as stderr,
        ):
            try:
                status = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=600,
                    check=False,
                ).returncode
            except subprocess.TimeoutExpired:
                status = -1
            except OSError:
                status = -2
        observation = {
            "worker": name,
            "exit_code": status,
            "seconds": time.perf_counter() - start,
            "artifact_sha256": {
                path.name: base.sha(path) for path in destination.iterdir() if path.is_file()
            },
        }
        attempts["observations"].append(observation)
        write(folder / "attempts.json", attempts)
        print(
            json.dumps(
                {key: value for key, value in observation.items() if key != "artifact_sha256"}
            ),
            flush=True,
        )
        if status:
            raise ValueError("worker failed; original attempt and logs retained")
        if sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) > 64 * 1024**2:
            raise ValueError("evidence exceeds 64 MiB budget")
    base.verify_fixture(args.fixture, base.read(args.receipt))
    value = audit(folder, verify_git=True)
    write(folder / "summary.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--verify-git", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    if args.audit:
        value = audit(args.audit, verify_git=args.verify_git)
    else:
        if not all((args.fixture, args.receipt, args.output)):
            parser.error("measurement requires --fixture, --receipt and --output")
        value = run(args)
    print(json.dumps(value, indent=2, allow_nan=False))
    return 0 if value["passes_estimate_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
