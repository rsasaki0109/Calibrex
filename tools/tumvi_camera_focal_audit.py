#!/usr/bin/env python3
"""The pre-registered camera-focal check audit on unseen TUM-VI room sequences.

An evaluation harness, not part of the Calibrex package. TUM-VI (Schubert et al., IROS
2018; CC BY 4.0) records a hand-held rig with two 190 deg fisheye cameras and a BMI160
IMU, calibrated with Kalibr. For each sequence in ``SESSIONS`` it downloads the 512x512
calibrated bag (pinned by size and ETag), runs ``calibrex check --pairs
camera-imu,camera-focal`` three times with three candidates written from Kalibr's
camchain (as calibrated, focal lengths 3 % long, 3 % short), records each camera's
``camera-focal`` verdict, and deletes the bag::

    python tools/tumvi_camera_focal_audit.py --camchain CAMCHAIN --work-dir WORK --output-dir RUNS

See docs/benchmarks/tumvi_camera_focal_preregistration.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BAGS = "https://cdn2.vision.in.tum.de/tumvi/calibrated/512_16"
SESSIONS: dict[str, tuple[int, str]] = {
    # sequence: (bag size in bytes, ETag)
    "room2": (3035570964, "b4ef2314-56a11992ff3a9"),
    "room3": (2971328666, "b11ae09a-56a119e3a1b83"),
    "room4": (2346890121, "8be2b389-56a11a2682bd9"),
    "room5": (2998956689, "b2c07291-56a11a591bcf8"),
    "room6": (2776625052, "a57fef9c-56a11a948d850"),
}
CAMCHAIN_SHA256 = "3a0c0285f4d0a3a1678053b2e5912e3e250e78ac09bd009f6d7d8bba669d5703"
"""Kalibr's camchain (dso/camchain.yaml of TUM-VI's dataset-calib-cam1_512_16 export)."""
CAMERAS = ("cam0", "cam1")
VARIANTS: dict[str, float] = {"kalibr": 1.0, "plus3": 1.03, "minus3": 0.97}
PAIRS = "camera-imu,camera-focal"

MAX_KALIBR_FAILS = 0
MIN_KALIBR_JUDGED_FRACTION = 0.25
MAX_PERTURBED_PASSES = 0
MIN_PERTURBED_FAIL_FRACTION = 0.25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_candidates(camchain: Path, out: Path) -> dict[str, Path]:
    """Kalibr's camchain, and copies with fx and fy scaled by each variant's factor."""

    if sha256(camchain) != CAMCHAIN_SHA256:
        raise SystemExit(f"{camchain}: not the pinned Kalibr camchain")
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, scale in VARIANTS.items():
        chain = yaml.safe_load(camchain.read_text(encoding="utf-8"))
        for camera in CAMERAS:
            fx, fy, cx, cy = chain[camera]["intrinsics"]
            chain[camera]["intrinsics"] = [fx * scale, fy * scale, cx, cy]
        path = out / f"camchain_{name}.yaml"
        path.write_text(yaml.safe_dump(chain, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths


def check_command(bag: Path, candidate: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        "import sys; from calibrex.cli.main import main; sys.exit(main())",
        "check",
        str(bag),
        "--tf",
        str(candidate),
        "--pairs",
        PAIRS,
        "--no-cache",
        "--fail-on",
        "never",
        "--output",
        str(output),
    ]


def download(session: str, target: Path) -> None:
    size, etag = SESSIONS[session]
    url = f"{BAGS}/dataset-{session}_512_16.bag"
    subprocess.run(
        ["curl", "-sSf", "--retry", "8", "--retry-all-errors", "-C", "-",
         "-H", f'If-Match: "{etag}"', "-o", str(target), url],
        check=True,
    )  # fmt: skip
    if target.stat().st_size != size:
        raise SystemExit(f"{url}: {target.stat().st_size} bytes, pinned {size}")


def focal_records(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each camera's camera-focal pair record, keyed cam0/cam1."""

    records = {}
    for pair in artifact["pairs"]:
        if pair["pair"] != "camera-focal":
            continue
        camera = pair["sensors"][0].split("/")[0]
        components = (pair.get("focal_scale") or {}).get("components", [])
        # an axis counts as judged only when check gave the pair a verdict: an inconclusive
        # pair (its estimator failed its own controls) judged nothing, whatever its numbers
        verdict_given = pair["status"] in ("pass", "warn", "fail")
        judged = [
            c for c in components if verdict_given and c.get("status") in ("pass", "warn", "fail")
        ]
        records[camera] = {
            "status": pair["status"],
            "reason_code": pair.get("reason_code"),
            "judged_axes": [c["name"] for c in judged],
            "axes": {
                c["name"]: {
                    "scale": c.get("scale"),
                    "scale_error": c.get("scale_error"),
                    "scale_std": c.get("scale_std"),
                    "tolerance": c.get("tolerance"),
                    "status": c.get("status"),
                }
                for c in components
            },
        }
    return records


def verdicts(scores: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """The two pre-registered claims over every (sequence, camera) case."""

    kalibr = [scores[s]["kalibr"].get(c, {}) for s in scores for c in CAMERAS]
    perturbed = [
        scores[s][v].get(c, {}) for s in scores for v in ("plus3", "minus3") for c in CAMERAS
    ]
    expected = len(SESSIONS) * len(CAMERAS)
    kalibr_fails = sum(r.get("status") == "fail" for r in kalibr)
    kalibr_judged = sum(bool(r.get("judged_axes")) for r in kalibr) / expected
    perturbed_passes = sum(r.get("status") == "pass" for r in perturbed)
    perturbed_fails = sum(r.get("status") == "fail" for r in perturbed) / (2 * expected)
    complete = len(kalibr) == expected and all(kalibr) and all(perturbed)
    honest = complete and kalibr_fails <= MAX_KALIBR_FAILS
    honest = honest and kalibr_judged >= MIN_KALIBR_JUDGED_FRACTION
    detects = complete and perturbed_passes <= MAX_PERTURBED_PASSES
    detects = detects and perturbed_fails >= MIN_PERTURBED_FAIL_FRACTION
    return {
        "honesty": {
            "verdict": "supported" if honest else "refuted",
            "kalibr_fails": kalibr_fails,
            "kalibr_judged_fraction": kalibr_judged,
        },
        "detection": {
            "verdict": "supported" if detects else "refuted",
            "perturbed_passes": perturbed_passes,
            "perturbed_fail_fraction": perturbed_fails,
        },
        "complete": complete,
    }


def run_session(session: str, candidates: dict[str, Path], work: Path, runs: Path) -> dict:
    bag = work / f"dataset-{session}_512_16.bag"
    download(session, bag)
    result: dict[str, Any] = {}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for name, candidate in candidates.items():
        output = runs / f"{session}_{name}.json"
        completed = subprocess.run(
            check_command(bag, candidate, output), capture_output=True, text=True, env=env
        )
        (runs / f"{session}_{name}.log").write_text(completed.stdout + completed.stderr)
        artifact = json.loads(output.read_text()) if output.exists() else {"pairs": []}
        result[name] = focal_records(artifact)
    bag.unlink()
    (runs / f"{session}_score.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camchain", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = write_candidates(args.camchain, args.output_dir / "candidates")
    scores = {}
    for session in SESSIONS:
        try:
            scores[session] = run_session(session, candidates, args.work_dir, args.output_dir)
        except Exception as error:  # a sequence that cannot be run leaves its cases empty
            scores[session] = {"error": repr(error), **{name: {} for name in VARIANTS}}
        partial = args.work_dir / f"dataset-{session}_512_16.bag"
        partial.unlink(missing_ok=True)
    summary = {"scores": scores, "verdicts": verdicts(scores)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    shutil.rmtree(args.work_dir, ignore_errors=True)
    print(json.dumps(summary["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
