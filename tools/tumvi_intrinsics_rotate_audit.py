#!/usr/bin/env python3
"""Audit targetless intrinsics against the published TUM-VI Kalibr reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import yaml
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from calibrex.core.exceptions import DatasetError
from calibrex.data.rosbag1 import iter_messages

if __package__:
    from .tumvi_camera_focal_audit import BAGS, CAMCHAIN_SHA256, CAMERAS, ROOT, SESSIONS, sha256
else:
    from tumvi_camera_focal_audit import BAGS, CAMCHAIN_SHA256, CAMERAS, ROOT, SESSIONS, sha256

DEVELOPMENT = {"room1": (2971525053, "b11ddfbd-56a1196d83910")}
CAMCHAIN = ROOT / "docs/assets/tumvi_camera_focal/camchain.yaml"
PLAN = ROOT / "docs/benchmarks/tumvi_intrinsics_rotate_preregistration.yaml"
DURATION_S = 10
STEP_S = 2
MIN_AXIS_DEG = 20
POSE_TOPIC = "/vrpn_client/raw_transform"
COEFFICIENTS = ("k1", "k2", "k3", "k4")


def bag_url(session: str) -> str:
    return f"{BAGS}/dataset-{session}_512_16.bag"


def response_header(headers: str, name: str) -> str | None:
    for line in reversed(headers.splitlines()):
        key, _, value = line.partition(":")
        if key.lower() == name:
            return value.strip()
    return None


def download(session: str, bag: Path) -> None:
    """Fetch contiguous pinned ranges, then assemble and delete them in order."""
    size, etag = (DEVELOPMENT | SESSIONS)[session]
    parts = min(6, size)
    bounds = [(size * i // parts, size * (i + 1) // parts - 1) for i in range(parts)]
    with TemporaryDirectory(dir=bag.parent, prefix=f"{session}-") as directory:

        def fetch(bounds: tuple[int, int]) -> Path:
            start, end = bounds
            part = Path(directory) / str(start)
            completed = subprocess.run(
                [
                    "curl",
                    "-fsS",
                    "--max-time",
                    "900",
                    "--dump-header",
                    "-",
                    "-H",
                    f'If-Match: "{etag}"',
                    "--range",
                    f"{start}-{end}",
                    "--max-filesize",
                    str(end - start + 1),
                    "-o",
                    str(part),
                    bag_url(session),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if (
                response_header(completed.stdout, "etag") != f'"{etag}"'
                or response_header(completed.stdout, "content-range")
                != f"bytes {start}-{end}/{size}"
                or part.stat().st_size != end - start + 1
            ):
                raise ValueError(f"{session}: range {start}-{end} does not match pinned bag")
            return part

        with ThreadPoolExecutor(max_workers=parts) as executor, bag.open("xb") as target:
            for part in executor.map(fetch, bounds):
                with part.open("rb") as source:
                    shutil.copyfileobj(source, target)
                part.unlink()
    if bag.stat().st_size != size:
        raise ValueError(f"{session}: assembled bag does not match pinned size")


def verify_download(session: str, bag: Path, headers: str) -> None:
    size, etag = (DEVELOPMENT | SESSIONS)[session]
    if response_header(headers, "etag") != f'"{etag}"' or bag.stat().st_size != size:
        raise ValueError(f"{session}: downloaded bag does not match pinned size and ETag")


def read_motion(bag: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    """Read OptiTrack rig poses and the synchronized camera time origin."""
    camera_topic = "/cam0/image_raw"
    first_image_ns = None
    last_image_ns = None
    stamps, poses = [], []
    for connection, bag_ns, payload in iter_messages(bag, topics={POSE_TOPIC, camera_topic}):
        _, sec, ns = struct.unpack_from("<III", payload)
        stamp_ns = sec * 10**9 + ns or bag_ns
        if connection.topic == camera_topic:
            if first_image_ns is None:
                first_image_ns = stamp_ns
            last_image_ns = stamp_ns
            continue
        if connection.message_type != "geometry_msgs/TransformStamped":
            raise ValueError("unexpected OptiTrack message type")
        parent_len = struct.unpack_from("<I", payload, 12)[0]
        child_len = struct.unpack_from("<I", payload, 16 + parent_len)[0]
        poses.append(struct.unpack_from("<7d", payload, 20 + parent_len + child_len))
        stamps.append(stamp_ns)
    if first_image_ns is None or last_image_ns is None or not poses:
        raise ValueError("missing image timestamps or OptiTrack poses")
    order = np.argsort(stamps, kind="stable")
    ordered_stamps = np.asarray(stamps)[order]
    keep = (ordered_stamps >= first_image_ns) & (ordered_stamps <= last_image_ns)
    if not np.any(keep):
        raise ValueError("OptiTrack poses do not overlap the images")
    samples = np.asarray(poses)[order][keep]
    if not np.all(np.isfinite(samples)):
        raise ValueError("nonfinite OptiTrack pose")
    times = (ordered_stamps[keep] - first_image_ns) * 1e-9
    return times, samples[:, :3], Rotation.from_quat(samples[:, 3:])


def select_window(
    times: np.ndarray, positions: np.ndarray, rotations: Rotation
) -> dict[str, Any] | None:
    """Choose the qualifying window of smallest measured rig displacement."""
    best = None
    for start in range(0, math.floor(times[-1]) - DURATION_S + 1, STEP_S):
        grid = start + np.arange(DURATION_S * 10 + 1) / 10
        indices = np.searchsorted(times, grid)
        if indices[-1] == len(times) or np.any(times[indices] - grid > 0.1):
            continue
        orientation = rotations[indices]
        angle_span = np.ptp((orientation[0].inv() * orientation).as_rotvec(degrees=True), axis=0)
        if np.count_nonzero(angle_span >= MIN_AXIS_DEG) < 2:
            continue
        diameter = float(pdist(positions[indices]).max())
        if best is None or diameter < best["rig_position_diameter_m"]:
            best = {
                "start_seconds": start,
                "duration_seconds": DURATION_S,
                "rig_position_diameter_m": diameter,
                "rig_rotation_vector_span_deg": angle_span.tolist(),
            }
    return best


def rotate_command(bag: Path, camera: str, output: Path, start: int) -> list[str]:
    command = [
        sys.executable,
        "-c",
        "import sys; from calibrex.cli.main import main; sys.exit(main())",
        "intrinsics",
        "rotate",
        "--bag",
        str(bag),
        "--topic",
        f"/{camera}/image_raw",
        "--distortion",
        "equidistant",
        "--seconds",
        str(DURATION_S),
        "--camera-name",
        camera,
        "--output",
        str(output),
        "--json",
    ]
    if start:
        command.extend(("--start-seconds", str(start)))
    return command


def errors(estimate: dict[str, Any], kalibr: dict[str, Any]) -> dict[str, float]:
    fx, fy, cx, cy = estimate["intrinsics"]
    rfx, rfy, rcx, rcy = kalibr["intrinsics"]
    return {
        "fx_pct": 100 * (fx / rfx - 1),
        "fy_pct": 100 * (fy / rfy - 1),
        "cx_px": cx - rcx,
        "cy_px": cy - rcy,
        **{
            name: value - ref
            for name, value, ref in zip(
                COEFFICIENTS,
                estimate["distortion_coeffs"],
                kalibr["distortion_coeffs"],
                strict=True,
            )
        },
    }


def verdict(delta: dict[str, float], gates: dict[str, float]) -> str:
    if not all(math.isfinite(x) for x in delta.values()):
        return "inconclusive"
    passed = (
        max(abs(delta["fx_pct"]), abs(delta["fy_pct"])) <= gates["max_focal_error_pct"]
        and max(abs(delta["cx_px"]), abs(delta["cy_px"])) <= gates["max_principal_error_px"]
    )
    return "pass" if passed else "fail"


def run_session(
    session: str,
    bag: Path,
    out: Path,
    reference: dict[str, Any],
    gates: dict[str, float] | None,
) -> dict[str, Any]:
    try:
        window = select_window(*read_motion(bag))
        reason = "no qualifying OptiTrack window"
    except (DatasetError, ValueError, struct.error) as exc:
        window = None
        reason = str(exc)
    size, etag = (DEVELOPMENT | SESSIONS)[session]
    record: dict[str, Any] = {
        "bag": {"url": bag_url(session), "size_bytes": size, "etag": etag},
        "window": window,
        "cameras": {},
    }
    for camera in CAMERAS:
        if window is None:
            record["cameras"][camera] = {
                "status": "inconclusive",
                "reason": reason,
            }
            continue
        output = out / f"{session}_{camera}_camchain.yaml"
        command = rotate_command(bag, camera, output, window["start_seconds"])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
        camera_record: dict[str, Any] = {
            "command": command,
            "returncode": completed.returncode,
            "stderr": completed.stderr,
        }
        if completed.returncode == 0:
            estimate = json.loads(completed.stdout)[camera]
            delta = errors(estimate, reference[camera])
            camera_record.update(
                estimate=estimate,
                errors=delta,
                status=verdict(delta, gates) if gates is not None else "development",
            )
        else:
            camera_record.update(status="inconclusive", reason="rotate exited unsuccessfully")
        record["cameras"][camera] = camera_record
    return record


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_digest() -> str:
    """Pin the estimator, adapters and audit code without committing source changes."""
    paths = set(git("ls-files", "--", "src", "tools").splitlines())
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    digest = hashlib.sha256()
    for name in sorted(paths):
        digest.update(name.encode() + b"\0" + sha256(ROOT / name).encode() + b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", "evaluation"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bag", type=Path, help="disposable fetched room1 bag (development only)")
    parser.add_argument("--response-headers", type=Path, help="curl response headers for --bag")
    args = parser.parse_args()
    if bool(args.bag) != bool(args.response_headers):
        parser.error("--bag and --response-headers must be specified together")
    gates = None
    if args.split == "evaluation":
        if args.bag:
            parser.error("--bag is only available for development")
        plan = yaml.safe_load(PLAN.read_text())
        if git("status", "--porcelain", "--", str(PLAN)):
            parser.error("evaluation requires a committed protocol")
        if source_digest() != plan["frozen_source_sha256"]:
            parser.error("src and tools must match the preregistered source digest")
        gates = plan["claims"]["tumvi-rotate-agreement"]["gates"]
    if sha256(CAMCHAIN) != CAMCHAIN_SHA256:
        parser.error("Kalibr reference does not match pinned SHA256")
    reference = yaml.safe_load(CAMCHAIN.read_text())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    import scipy

    provenance = {
        "split": args.split,
        "commit": git("rev-parse", "HEAD"),
        "source_sha256": source_digest(),
        "reference_sha256": CAMCHAIN_SHA256,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
    }
    sessions = DEVELOPMENT if args.split == "development" else SESSIONS
    for session in sessions:
        artifact = args.output_dir / f"{session}.json"
        if artifact.exists():
            parser.error(f"{artifact} already exists")
        bag = args.bag or args.work_dir / f"dataset-{session}_512_16.bag"
        if args.bag is None and bag.exists():
            parser.error(f"download target already exists: {bag}")
        print(f"{session}: verifying bag", file=sys.stderr, flush=True)
        if args.bag is not None:
            verify_download(session, bag, args.response_headers.read_text())
        try:
            if args.bag is None:
                try:
                    download(session, bag)
                except (subprocess.CalledProcessError, ValueError, OSError) as exc:
                    message = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else exc
                    print(f"{session}: bag download failed: {message}", file=sys.stderr)
                    return 1
            print(f"{session}: selecting window and running rotate", file=sys.stderr, flush=True)
            record = run_session(session, bag, args.output_dir, reference, gates)
            artifact.write_text(json.dumps({"provenance": provenance, **record}, indent=2) + "\n")
        finally:
            bag.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
