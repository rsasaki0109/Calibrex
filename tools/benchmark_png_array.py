"""Compare complete PNG-to-array reads, keeping every fresh-worker observation.

The rows control includes its required NumPy conversion. Decode timings include
file reads, validation, lazy Pillow loading and ordered parallel decoding. Parent
times additionally include startup, source/input checks, pixel hashing and saving.
Neither timing is a complete calibration benchmark. Inputs are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def execute_worker(command: list[str], env: dict[str, str], folder: Path) -> int:
    """Retain logs and return a failed observation even when launch or timeout fails."""
    with (folder / "stdout.txt").open("wb") as stdout, (folder / "stderr.txt").open("wb") as stderr:
        try:
            return subprocess.run(
                command, stdout=stdout, stderr=stderr, env=env, timeout=120, check=False
            ).returncode
        except subprocess.TimeoutExpired:
            return -1
        except OSError:
            return -2


def summary_schema() -> dict:
    positive = {"type": "number", "exclusiveMinimum": 0}
    timings = {
        "type": "object",
        "additionalProperties": False,
        "required": ["rows", "array"],
        "properties": dict.fromkeys(("rows", "array"), positive),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "scope",
            "observations",
            "images_per_worker",
            "pixels_per_worker",
            "decode_medians_seconds",
            "whole_worker_medians_seconds",
            "decode_ratio",
            "whole_worker_ratio",
            "all_float64_pixels_equal",
            "provenance",
        ],
        "properties": {
            "scope": {"type": "string"},
            "observations": {"type": "integer", "minimum": 4},
            "images_per_worker": {"type": "integer", "minimum": 1},
            "pixels_per_worker": {"type": "integer", "minimum": 1},
            "decode_medians_seconds": timings,
            "whole_worker_medians_seconds": timings,
            "decode_ratio": positive,
            "whole_worker_ratio": positive,
            "all_float64_pixels_equal": {"const": True},
            "provenance": {"type": "object"},
        },
    }


def worker(protocol_path: Path, variant: str, folder: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    protocol = read(protocol_path)
    output = protocol_path.parent
    if sha(Path(__file__)) != protocol["launcher_sha256"]:
        raise ValueError("worker launcher changed")
    source = output / "source"
    manifest = read(output / "source-manifest.json")
    for name, digest in manifest.items():
        if sha(source / "src" / name) != digest:
            raise ValueError("frozen source changed")
    fixture = Path(protocol["fixture"])
    for name, digest in protocol["inputs_sha256"].items():
        if sha(fixture / name) != digest:
            raise ValueError("input changed")
    sys.path.insert(0, str(source / "src"))
    import numpy as np

    from calibrex.data.kitti import read_png_luminance, read_png_luminance_array

    def decode(name: str):
        if variant == "array":
            value = read_png_luminance_array(fixture / name)
        else:
            image = read_png_luminance(fixture / name)
            value = None if image is None else np.asarray(image.rows, dtype=np.float64)
        if value is None:
            raise ValueError(f"unsupported PNG: {name}")
        return value

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=protocol["workers"]) as executor:
        images = list(executor.map(decode, protocol["inputs_sha256"]))
    elapsed = time.perf_counter() - started
    pixels = {
        name: {
            "shape": list(value.shape),
            "dtype": "float64-le",
            "sha256": hashlib.sha256(value.astype("<f8", copy=False).tobytes()).hexdigest(),
        }
        for name, value in zip(protocol["inputs_sha256"], images, strict=True)
    }
    imports = {}
    for name, module in sys.modules.copy().items():
        if name == "calibrex" or name.startswith("calibrex."):
            path = Path(module.__file__).resolve()
            relative = path.relative_to(source / "src").as_posix()
            if sha(path) != manifest[relative]:
                raise ValueError("import outside frozen source")
            imports[name] = relative
    write(
        folder / "measurement.json",
        {
            "variant": variant,
            "decode_seconds": elapsed,
            "pixels": pixels,
            "images": len(images),
            "pixel_count": sum(value.size for value in images),
            "array_bytes": sum(value.nbytes for value in images),
            "imports": imports,
            "provenance": {
                "protocol_sha256": sha(protocol_path),
                "source_manifest_sha256": sha(output / "source-manifest.json"),
                "launcher_sha256": sha(Path(__file__)),
            },
        },
    )


def audit(output: Path) -> dict:
    """Replay all observations with CPU dependencies; never execute the capsule."""
    import jsonschema

    protocol = read(output / "protocol.json")
    manifest = read(output / "source-manifest.json")
    if sha(output / "source-manifest.json") != protocol["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    if sha(output / "source.tar.gz") != protocol["source_capsule_sha256"]:
        raise ValueError("source capsule changed")
    with tarfile.open(output / "source.tar.gz", "r:gz") as capsule:
        entries = {}
        for member in capsule.getmembers():
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or not member.name.startswith("calibrex/")
                or not member.name.endswith(".py")
                or member.name in entries
            ):
                raise ValueError("unexpected source capsule member")
            stream = capsule.extractfile(member)
            entries[member.name] = hashlib.sha256(stream.read()).hexdigest()
        if entries != manifest:
            raise ValueError("source capsule inventory differs")
    if sha(output / "launcher.py") != protocol["launcher_sha256"]:
        raise ValueError("launcher changed")
    attempts = read(output / "attempts.json")
    if attempts["provenance"]["protocol_sha256"] != sha(output / "protocol.json"):
        raise ValueError("attempt lineage differs")
    rows = attempts["observations"]
    repeats = protocol.get("repetitions", len(protocol["order"]) // 2)
    expected_order = [
        v
        for index in range(repeats)
        for v in (("rows", "array") if index % 2 == 0 else ("array", "rows"))
    ]
    if (
        repeats < 2
        or protocol["order"] != expected_order
        or [row["variant"] for row in rows] != expected_order
    ):
        raise ValueError("incomplete or reordered observations")
    decode_times, whole_times = (
        {v: [] for v in ("rows", "array")},
        {v: [] for v in ("rows", "array")},
    )
    reference = None
    for index, row in enumerate(rows):
        variant = row["variant"]
        if row["folder"] != f"{index:02d}-{variant}" or row["exit_code"] != 0:
            raise ValueError("failed or replaced worker")
        path = output / row["folder"] / "measurement.json"
        if sha(path) != row["measurement_sha256"]:
            raise ValueError("worker measurement changed")
        value = read(path)
        if (
            value["provenance"]
            != {
                "protocol_sha256": sha(output / "protocol.json"),
                "source_manifest_sha256": protocol["source_manifest_sha256"],
                "launcher_sha256": protocol["launcher_sha256"],
            }
            or value["variant"] != variant
        ):
            raise ValueError("worker lineage differs")
        pixels = value["pixels"]
        if (
            len(protocol["inputs_sha256"]) != 48
            or set(pixels) != set(protocol["inputs_sha256"])
            or value["images"] != len(pixels)
        ):
            raise ValueError("decoded image set differs")
        if any(
            len(item["shape"]) != 2 or any(type(n) is not int or n < 1 for n in item["shape"])
            for item in pixels.values()
        ):
            raise ValueError("invalid pixel dimensions")
        count = sum(item["shape"][0] * item["shape"][1] for item in pixels.values())
        if value["pixel_count"] != count or value["array_bytes"] != count * 8:
            raise ValueError("pixel work differs")
        if any(
            item["dtype"] != "float64-le"
            or len(item["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in item["sha256"])
            for item in pixels.values()
        ):
            raise ValueError("invalid pixel identity")
        if reference is None:
            reference = pixels
        if pixels != reference:
            raise ValueError("decoded float64 pixels differ")
        if not value["imports"] or any(path not in manifest for path in value["imports"].values()):
            raise ValueError("unbound source import")
        durations = (value["decode_seconds"], row["seconds"])
        if (
            any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0.0 for t in durations)
            or durations[0] > durations[1]
        ):
            raise ValueError("invalid or rebound timing")
        decode_times[variant].append(value["decode_seconds"])
        whole_times[variant].append(row["seconds"])
    decode = {key: statistics.median(times) for key, times in decode_times.items()}
    whole = {key: statistics.median(times) for key, times in whole_times.items()}
    result = {
        "scope": "48-image PNG-to-array read stage; complete calibration excluded",
        "observations": len(rows),
        "images_per_worker": len(reference),
        "pixels_per_worker": count,
        "decode_medians_seconds": decode,
        "whole_worker_medians_seconds": whole,
        "decode_ratio": decode["rows"] / decode["array"],
        "whole_worker_ratio": whole["rows"] / whole["array"],
        "all_float64_pixels_equal": True,
        "provenance": {
            "protocol_sha256": sha(output / "protocol.json"),
            "attempts_sha256": sha(output / "attempts.json"),
        },
    }
    jsonschema.validate(result, summary_schema())
    if (output / "summary.json").exists() and read(output / "summary.json") != result:
        raise ValueError("saved summary differs from complete observations")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--variant", choices=("rows", "array"))
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.worker:
        if not args.variant or not args.folder:
            parser.error("worker mode requires variant and folder")
        worker(args.worker.resolve(), args.variant, args.folder.resolve())
        return
    if args.audit:
        print(json.dumps(audit(args.audit.resolve()), indent=2))
        return
    if not args.fixture or not args.output_dir or args.repetitions < 2 or args.workers < 1:
        parser.error("requires fixture, new output-dir, repetitions >= 2 and workers >= 1")
    root = Path.cwd()
    sys.path.insert(0, str(root))
    from tools.benchmark_bspline_lidar_imu import _freeze

    output, fixture = args.output_dir.resolve(), args.fixture.resolve()
    disk_free = shutil.disk_usage(root).free
    if disk_free < 20 * 2**30:
        raise ValueError("requires a 20 GiB free-space reserve")
    files = sorted(fixture.glob("image_02/data/*.png"))
    if len(files) != 48:
        raise ValueError("requires the existing fixed 48-image KITTI fixture")
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    _freeze(output / "source", commit)
    shutil.copyfile(output / "source/manifest.json", output / "source-manifest.json")
    shutil.copyfile(Path(__file__), output / "launcher.py")
    with tarfile.open(output / "source.tar.gz", "w:gz") as capsule:
        for name in read(output / "source-manifest.json"):
            data = (output / "source/src" / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), 0
            capsule.addfile(info, io.BytesIO(data))
    order = [
        variant
        for repeat in range(args.repetitions)
        for variant in (("rows", "array") if repeat % 2 == 0 else ("array", "rows"))
    ]
    protocol = {
        "scope": __doc__,
        "source_commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "source_manifest_sha256": sha(output / "source-manifest.json"),
        "source_capsule_sha256": sha(output / "source.tar.gz"),
        "launcher_sha256": sha(output / "launcher.py"),
        "fixture": fixture.relative_to(root).as_posix(),
        "inputs_sha256": {p.relative_to(fixture).as_posix(): sha(p) for p in files},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pillow")},
        "workers": args.workers,
        "repetitions": args.repetitions,
        "order": order,
        "warmups_discarded": 0,
        "blas_threads": 1,
        "host_reserved": False,
        "decode_stage_gate": 2.0,
        "disk_free_before_bytes": disk_free,
        "evidence_budget_bytes": 64 * 2**20,
        "limitations": [
            "Already-spent public inputs; no accuracy/SOTA claim.",
            "Input hash checks precede timing and warm the OS file cache.",
            "Source/input checks and pixel hashes are excluded from decode time; "
            "parent time includes them.",
            "Neither measurement includes calibration.",
        ],
    }
    write(output / "protocol.json", protocol)
    attempts = {
        "observations": [],
        "provenance": {"protocol_sha256": sha(output / "protocol.json")},
    }
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for index, variant in enumerate(order):
        folder = output / f"{index:02d}-{variant}"
        folder.mkdir()
        start = time.perf_counter()
        exit_code = execute_worker(
            [
                sys.executable,
                str(output / "launcher.py"),
                "--worker",
                str(output / "protocol.json"),
                "--variant",
                variant,
                "--folder",
                str(folder),
            ],
            env,
            folder,
        )
        observation = {
            "variant": variant,
            "folder": folder.name,
            "seconds": time.perf_counter() - start,
            "exit_code": exit_code,
            "measurement_sha256": sha(folder / "measurement.json") if exit_code == 0 else None,
        }
        attempts["observations"].append(observation)
        write(output / "attempts.json", attempts)
        print(json.dumps(observation), flush=True)
        if exit_code:
            raise ValueError("worker failed; original attempt retained")
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > 64 * 2**20:
            raise ValueError("evidence exceeded 64 MiB budget")
    result = audit(output)
    write(output / "summary.schema.json", summary_schema())
    write(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
