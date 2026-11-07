#!/usr/bin/env python3
"""The pre-registered camera-imu check v2 audit on unseen TUM-VI sequences.

An evaluation harness, not part of the Calibrex package. v1
(``tools/tumvi_camera_imu_audit.py``, corridor1-corridor5) was refuted: a T_cam_imu 2 deg
wrong in pitch passed as ``pass (partial)`` where pitch was unchecked. v2 judges what a
user running ``calibrex check --fail-on partial`` gets: a case is *flagged* when its
camera-imu pair would make that run exit 1 (``check_exit_status`` of the pair).

For each sequence in ``SESSIONS`` it downloads the 512x512 calibrated bag (pinned by size
and ETag, resumed from the bytes on disk), runs ``calibrex check --pairs camera-imu
--fail-on partial`` with four candidates written from Kalibr's camchain (as calibrated,
and with T_cam_imu turned 2 deg about the camera's x, y and z axes) sharing one estimator
cache, records each camera's camera-imu record and each run's exit status, and deletes
the bag and the cache::

    python tools/tumvi_camera_imu_v2_audit.py --camchain CAMCHAIN --work-dir WORK --output-dir RUNS

See docs/benchmarks/tumvi_camera_imu_v2_preregistration.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.tumvi_camera_imu_audit import (  # noqa: E402
    AXIS_VECTORS,
    CAMERAS,
    VARIANTS,
    write_candidates,
)

from calibrex.check.verdict import check_exit_status  # noqa: E402

BAGS = "https://cdn2.vision.in.tum.de/tumvi/calibrated/512_16"
SESSIONS: dict[str, tuple[int, str]] = {
    # sequence: (bag size in bytes, ETag); the five smallest unseen calibrated bags
    "slides2": (5167095556, "133fb9f04-56a11b425b78a"),
    "slides1": (5876912463, "15e4a914f-56a11af7075fa"),
    "slides3": (7614358916, "1c5d9e584-56a11be871d86"),
    "magistrale5": (9392089416, "22fcfe948-56a10633639d7"),
    "magistrale3": (10214128670, "260cf3c1e-56a103e156263"),
}
PAIRS = "camera-imu"
FAIL_ON = "partial"
DOWNLOAD_ATTEMPTS = 20

MAX_KALIBR_FAILS = 0
MIN_KALIBR_FULL_PASS_FRACTION = 0.5
MAX_TURNED_UNFLAGGED = 0
MIN_TURNED_DETECTED_FRACTION = 0.5


def check_command(bag: Path, candidate: Path, output: Path, cache: Path) -> list[str]:
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
        "--cache-dir",
        str(cache),
        "--fail-on",
        FAIL_ON,
        "--output",
        str(output),
    ]


def download(sessions: dict[str, tuple[int, str]], session: str, target: Path) -> None:
    """Fetch the pinned bag, resuming a cut-off transfer from the bytes already on disk."""

    size, etag = sessions[session]
    url = f"{BAGS}/dataset-{session}_512_16.bag"
    for attempt in range(DOWNLOAD_ATTEMPTS):
        if target.exists() and target.stat().st_size >= size:
            break
        if attempt:
            time.sleep(min(60, 5 * attempt))
        subprocess.run(
            ["curl", "-sSf", "-C", "-", "-H", f'If-Match: "{etag}"', "-o", str(target), url],
            check=False,
        )  # fmt: skip
    got = target.stat().st_size if target.exists() else 0
    if got != size:
        raise SystemExit(f"{url}: {got} bytes, pinned {size}")


def imu_records(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each camera's camera-imu pair record, keyed cam0/cam1, with its exit status."""

    records = {}
    for pair in artifact["pairs"]:
        if pair["pair"] != "camera-imu":
            continue
        camera = pair["sensors"][0].split("/")[0]
        verdict_given = pair["status"] in ("pass", "warn", "fail")
        axes = pair.get("axes", [])
        partial = int(pair.get("coverage") == "partial")
        records[camera] = {
            "status": pair["status"],
            "coverage": pair.get("coverage"),
            "reason_code": pair.get("reason_code"),
            "exit_status": check_exit_status(pair["status"], partial, FAIL_ON),
            "judged_axes": [a["name"] for a in axes if verdict_given and a["name"] in AXIS_VECTORS],
            "unchecked_axes": [u["name"] for u in pair.get("unchecked_axes", [])],
            "axes": {
                a["name"]: {
                    "candidate_error": a.get("candidate_error"),
                    "estimate_std": a.get("estimate_std"),
                    "tolerance": a.get("tolerance"),
                    "status": a.get("status"),
                }
                for a in axes
            },
        }
    return records


def full_pass(record: dict[str, Any]) -> bool:
    return record.get("status") == "pass" and record.get("coverage") == "full"


def detected(record: dict[str, Any], axis: str) -> bool:
    """A turned case is detected when the pair fails on the axis that carries the error."""

    return record.get("status") == "fail" and record["axes"].get(axis, {}).get("status") == "fail"


def verdicts(
    scores: dict[str, dict[str, dict[str, Any]]], sessions: dict[str, tuple[int, str]] = SESSIONS
) -> dict[str, Any]:
    """The two pre-registered claims over every (sequence, camera, candidate) case."""

    kalibr = [scores[s]["kalibr"].get(c, {}) for s in scores for c in CAMERAS]
    turned = [
        (scores[s][v].get(c, {}), axis)
        for s in scores
        for v, axis in VARIANTS.items()
        if axis is not None
        for c in CAMERAS
    ]
    expected = len(sessions) * len(CAMERAS)
    expected_turned = expected * (len(VARIANTS) - 1)
    kalibr_fails = sum(r.get("status") == "fail" for r in kalibr)
    kalibr_full = sum(full_pass(r) for r in kalibr) / expected
    unflagged = sum(r.get("exit_status") == 0 for r, _axis in turned if r)
    turned_detected = sum(detected(r, axis) for r, axis in turned if r) / expected_turned
    complete = len(kalibr) == expected and all(kalibr) and all(r for r, _axis in turned)
    honest = complete and kalibr_fails <= MAX_KALIBR_FAILS
    honest = honest and kalibr_full >= MIN_KALIBR_FULL_PASS_FRACTION
    detects = complete and unflagged <= MAX_TURNED_UNFLAGGED
    detects = detects and turned_detected >= MIN_TURNED_DETECTED_FRACTION
    return {
        "honesty": {
            "verdict": "supported" if honest else "refuted",
            "kalibr_fails": kalibr_fails,
            "kalibr_full_pass_fraction": kalibr_full,
        },
        "detection": {
            "verdict": "supported" if detects else "refuted",
            "turned_unflagged": unflagged,
            "turned_detected_fraction": turned_detected,
        },
        "complete": complete,
    }


def run_session(
    session: str,
    candidates: dict[str, Path],
    work: Path,
    runs: Path,
    sessions: dict[str, tuple[int, str]] = SESSIONS,
) -> dict:
    bag = work / f"dataset-{session}_512_16.bag"
    cache = work / f"cache-{session}"
    download(sessions, session, bag)
    result: dict[str, Any] = {"exit_status": {}}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for name, candidate in candidates.items():
        output = runs / f"{session}_{name}.json"
        completed = subprocess.run(
            check_command(bag, candidate, output, cache), capture_output=True, text=True, env=env
        )
        (runs / f"{session}_{name}.log").write_text(completed.stdout + completed.stderr)
        result["exit_status"][name] = completed.returncode
        artifact = json.loads(output.read_text()) if output.exists() else {"pairs": []}
        result[name] = imu_records(artifact)
    bag.unlink()
    shutil.rmtree(cache, ignore_errors=True)
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
        (args.work_dir / f"dataset-{session}_512_16.bag").unlink(missing_ok=True)
        shutil.rmtree(args.work_dir / f"cache-{session}", ignore_errors=True)
    summary = {"scores": scores, "verdicts": verdicts(scores)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    shutil.rmtree(args.work_dir, ignore_errors=True)
    print(json.dumps(summary["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
