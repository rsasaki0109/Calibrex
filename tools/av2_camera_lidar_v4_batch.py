#!/usr/bin/env python3
"""Sequential Argoverse 2 development batch for the camera-LiDAR v4 check.

Runs the 30 development logs fixed in ``docs/assets/camera_lidar_v4/av2_split.json``;
it never names an evaluation or reserve log. For each log the batch:

1. re-lists the log's conversion members and requires their key/size/ETag digest to
   equal the committed pin;
2. downloads exactly those members, each under ``If-Match`` on its listed ETag with an
   exact length check;
3. converts with ``calibrex convert av2`` (``ring_front_center``, ``up_lidar``);
4. runs the seven real checks through ``kitti_camera_lidar_v3_replay.capture_drive``
   and replays the default rule against the opt-in 0.5 deg cross-start agreement rule.

Raw members and the bag are then deleted; NPZ inputs stay only while free disk remains
above a reserve. Every failure is retained as its own attempt. Development only.

    python tools/av2_camera_lidar_v4_batch.py run WORK_ROOT [--log ID] [--retry-failed]
    python tools/av2_camera_lidar_v4_batch.py publish WORK_ROOT DEST
    python tools/av2_camera_lidar_v4_batch.py gates DEST OUT.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "calibrex.av2_camera_lidar_v4_batch/v0.1"
SPLIT = ROOT / "docs/assets/camera_lidar_v4/av2_split.json"
RULE_DOCUMENT = ROOT / "docs/benchmarks/camera_lidar_v4_candidates.md"
AGREEMENT = {"start_agreement_deg": 0.5}
EVALUATION_LOGS = 150
RETRIES = 3
START_FREE_BYTES = 6 * 1024**3
KEEP_ARRAYS_FREE_BYTES = 12 * 1024**3
FROZEN = (
    "tools/av2_camera_lidar_v4_batch.py",
    "tools/av2_camera_lidar_v4_split.py",
    "tools/kitti_camera_lidar_v3_replay.py",
    "tools/kitti_camera_lidar_v3_development.py",
    "tools/kitti_camera_lidar_sota_run.py",
    "tools/kitti360_camera_lidar_v3_gates.py",
    "tools/score_kitti_camera_lidar_v2.py",
    "src/calibrex/evaluation/camera_lidar_edge.py",
    "src/calibrex/data/av2_to_rosbag2.py",
)
PUBLISHED_SUFFIXES = {".json", ".log", ".yaml", ".txt"}
RETRYABLE = {"FAIL", "RUNNING"}
TRANSIENT = (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError)


class LogReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA
    log: str
    attempt: int
    status: str
    stage: str
    error: str | None = None
    details: dict[str, Any] = {}
    provenance: dict[str, Any] = {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool(name: str) -> ModuleType:
    key = f"_av2_v4_batch_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "tools" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def _commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()


def development_logs() -> dict[str, dict[str, Any]]:
    """The fixed development logs; evaluation and reserve logs are never returned."""

    split = json.loads(SPLIT.read_text())
    logs = {item["log"]: item for item in split["development"]}
    evaluation = {item["log"] for item in split["evaluation"]}
    if set(logs) & evaluation or any(item["split"] != "val" for item in logs.values()):
        raise ValueError("development logs must be validation logs outside the evaluation set")
    return logs


def members(pin: dict[str, Any], lister: Callable[..., list[Any]] | None = None) -> list[Any]:
    """The log's conversion members, re-listed; their digest must equal the pin."""

    split_tool = _tool("av2_camera_lidar_v4_split")
    lister = lister or split_tool.s3_list
    base = f"{split_tool.PREFIX}{pin['split']}/{pin['log']}/"
    objects = [o for o in lister(base, "/") if not isinstance(o, str)]
    objects += [o for o in lister(base + "calibration/", "/") if not isinstance(o, str)]
    tables = {
        base + "city_SE3_egovehicle.feather",
        base + "calibration/egovehicle_SE3_sensor.feather",
        base + "calibration/intrinsics.feather",
    }
    found = [o for o in objects if o[0] in tables]
    found += [
        o
        for o in lister(base + "sensors/lidar/", "")
        if not isinstance(o, str) and o[0].endswith(".feather")
    ]
    found += [
        o
        for o in lister(base + f"sensors/cameras/{split_tool.CAMERA}/", "")
        if not isinstance(o, str) and o[0].endswith(".jpg")
    ]
    listed = sorted(found)
    digest = hashlib.sha256()
    for key, size, etag in listed:
        digest.update(f"{key}\0{size}\0{etag}\n".encode())
    if digest.hexdigest() != pin["members_sha256"]:
        raise ValueError(f"{pin['log']}: listed members differ from the committed pin")
    return listed


def _download(key: str, size: int, etag: str, target: Path) -> int:
    url = _tool("av2_camera_lidar_v4_split").BUCKET + urllib.parse.quote(key)
    for attempt in range(RETRIES + 1):
        request = Request(url, headers={"If-Match": etag, "Accept-Encoding": "identity"})
        try:
            with urlopen(request, timeout=120) as response:
                if response.status != 200 or response.headers.get("ETag") != etag:
                    raise ValueError(f"{key}: status or ETag differs from the pin")
                data = response.read(size + 1)
        except urllib.error.HTTPError as exc:
            raise ValueError(f"{key}: HTTP {exc.code}") from None
        except TRANSIENT:
            if attempt == RETRIES:
                raise
            time.sleep(2.0 * 2**attempt)
            continue
        if len(data) == size:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return size
        if attempt == RETRIES:
            raise ValueError(f"{key}: body {len(data)} bytes, expected {size}")
        time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def acquire(pin: dict[str, Any], raw: Path, listed: list[Any]) -> int:
    """Download every pinned member into ``raw/<split>/<log>/``; returns bytes."""

    split_tool = _tool("av2_camera_lidar_v4_split")
    base = f"{split_tool.PREFIX}{pin['split']}/{pin['log']}/"
    log_dir = raw / pin["split"] / pin["log"]
    jobs = []
    for key, size, etag in listed:
        relative = key[len(base) :]
        target = log_dir / relative
        if not target.resolve().is_relative_to(log_dir.resolve()):
            raise ValueError(f"unsafe member path {key}")
        jobs.append((key, size, etag, target))
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        return sum(pool.map(lambda job: _download(*job), jobs))


def _latest(work: Path, log: str) -> tuple[int, dict[str, Any] | None]:
    attempts = sorted(
        int(p.name.removeprefix("attempt_")) for p in (work / log).glob("attempt_*") if p.is_dir()
    )
    if not attempts:
        return 0, None
    report = work / log / f"attempt_{attempts[-1]}" / "report.json"
    return attempts[-1], json.loads(report.read_text()) if report.is_file() else None


def run_log(
    log: str,
    work: Path,
    *,
    keep_arrays_free: int = KEEP_ARRAYS_FREE_BYTES,
    free: Callable[[], int] | None = None,
    lister: Callable[..., list[Any]] | None = None,
) -> LogReport:
    """One new attempt for one development log; the attempt directory is never reused."""

    pin = development_logs()[log]
    free = free or (lambda: shutil.disk_usage(work).free)
    attempt = _latest(work, log)[0] + 1
    out = work / log / f"attempt_{attempt}"
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = LogReport(
        log=log,
        attempt=attempt,
        status="RUNNING",
        stage="preflight",
        details={"pin": pin},
        provenance={
            "source_commit": _commit(),
            "frozen_modules_sha256": {name: _sha(ROOT / name) for name in FROZEN},
            "split_sha256": _sha(SPLIT),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "development_only": True,
            "evaluation_bytes_read": 0,
            "new_accuracy_claim": False,
        },
    )

    def save() -> None:
        report.details["wall_s"] = round(time.monotonic() - started, 1)
        (out / "report.json").write_text(report.model_dump_json(indent=2) + "\n", "utf-8")

    replay = _tool("kitti_camera_lidar_v3_replay")
    try:
        report.details["free_bytes_at_start"] = free()
        if report.details["free_bytes_at_start"] < START_FREE_BYTES:
            raise RuntimeError("insufficient free disk before acquisition")
        report.stage = "fetch"
        save()
        listed = members(pin, lister)
        report.details["fetched_bytes"] = acquire(pin, out / "raw", listed)
        report.stage = "convert"
        save()
        from calibrex.data.av2_to_rosbag2 import convert_av2_to_rosbag2

        converted = convert_av2_to_rosbag2(
            out / "raw" / pin["split"] / log,
            out / "bag",
            camera="ring_front_center",
            lidar="up_lidar",
            command=[sys.executable, str(Path(__file__)), "run", "DEVELOPMENT_ONLY"],
        )
        shutil.copyfile(converted.provenance_path, out / "conversion.json")
        report.details["conversion_sha256"] = _sha(out / "conversion.json")
        shutil.rmtree(out / "raw")
        report.details["raw_deleted"] = True
        report.stage = "capture"
        save()
        code = replay.capture_drive(out / "bag", log, out / "snapshots" / log)
        report.details["capture_exit_code"] = code
        shutil.rmtree(out / "bag")
        report.details["bag_deleted"] = True
        if code != 0:
            raise RuntimeError("native capture failed; see retained per-run logs")
        report.stage = "compare"
        save()
        result = replay.compare_drives(
            out / "snapshots", out / "comparison", drives=[log], proposed=AGREEMENT
        )
        report.details["comparison_errors"] = result["errors"]
        if result["errors"]:
            raise RuntimeError(f"comparison errors: {result['errors']}")
        report.stage = "complete"
        report.status = "PASS_DEVELOPMENT"
    except Exception as exc:
        report.status = "FAIL"
        report.error = f"{type(exc).__name__}: {exc}"
        (out / "failure.txt").write_text(traceback.format_exc(), "utf-8")
    finally:
        for leftover in ("raw", "bag"):
            if (out / leftover).exists():
                shutil.rmtree(out / leftover)
                report.details[f"{leftover}_deleted"] = True
        arrays = sorted((out / "snapshots").rglob("arrays.npz"))
        if arrays and free() < keep_arrays_free:
            report.details["arrays_deleted"] = {
                p.relative_to(out).as_posix(): _sha(p) for p in arrays
            }
            for path in arrays:
                path.unlink()
        report.details["arrays_kept"] = len(sorted((out / "snapshots").rglob("arrays.npz")))
        save()
    return report


def run(work: Path, chosen: Sequence[str] | None = None, retry_failed: bool = False) -> int:
    """Process logs in order under a single-runner lock; earlier attempts stay untouched."""

    import fcntl

    known = development_logs()
    order = list(known) if not chosen else list(chosen)
    unknown = [item for item in order if item not in known]
    if unknown or len(set(order)) != len(order):
        raise ValueError(f"choose distinct development logs: {unknown}")
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".batch.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("another batch runner holds this work root") from None
        failures = 0
        for log in order:
            attempt, previous = _latest(work, log)
            retry = retry_failed and previous is not None and previous["status"] in RETRYABLE
            if previous is not None and not retry:
                print(log, "skip", previous["status"], f"attempt_{attempt}", flush=True)
                continue
            report = run_log(log, work)
            failures += report.status != "PASS_DEVELOPMENT"
            print(log, report.status, report.stage, report.details.get("wall_s"), flush=True)
    return 1 if failures else 0


def publish(work: Path, dest: Path) -> int:
    """Copy reports and non-sensor receipts; never images, clouds, bags or NPZ arrays."""

    count = 0
    for report_path in sorted(work.glob("*/attempt_*/report.json")):
        source = report_path.parent
        target = dest / source.parent.name / source.name
        if target.exists():
            continue
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix in PUBLISHED_SUFFIXES:
                relative = path.relative_to(source)
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                (target / relative).write_bytes(path.read_bytes())
                count += 1
    files = [
        {"path": p.relative_to(dest).as_posix(), "bytes": p.stat().st_size, "sha256": _sha(p)}
        for p in sorted(dest.rglob("*"))
        if p.is_file() and p.name not in ("manifest.json", "gates.json")
    ]
    manifest = {
        "schema_version": SCHEMA,
        "policy": "unmodified reports and non-sensor receipts; no images, clouds, bags, "
        "feather tables or NPZ inputs",
        "files": files,
        "provenance": _provenance(),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    return count


def _provenance() -> dict[str, Any]:
    return {
        "generator": "tools/av2_camera_lidar_v4_batch.py",
        "generator_sha256": _sha(Path(__file__)),
        "git_commit": _commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def scores_from(dest: Path) -> dict[str, dict[str, Any]]:
    """Each development log's latest published scores under both rules (failed if absent)."""

    from dataclasses import asdict

    scorer = _tool("score_kitti_camera_lidar_v2")
    scores: dict[str, dict[str, Any]] = {"current": {}, "proposed": {}}
    for log in development_logs():
        attempt, report = _latest(dest, log)
        path = dest / log / f"attempt_{attempt}" / "comparison" / "comparison.json"
        if report is None or report["status"] != "PASS_DEVELOPMENT" or not path.is_file():
            reason = "missing" if report is None else f"{report['status']}: {report['error']}"
            for rule in scores:
                scores[rule][log] = asdict(scorer.DriveScore("failed", reason))
            continue
        comparison = json.loads(path.read_text())["scores"]
        for rule in scores:
            scores[rule][log] = comparison[rule][log]
    return scores


def gates(dest: Path, output: Path) -> dict[str, Any]:
    """The committed v4 rule applied to the development scores (no verdict)."""

    if output.exists():
        raise ValueError("refuse overwrite")
    rule = _tool("kitti360_camera_lidar_v3_gates")
    scores = scores_from(dest)
    primary = rule.primary(scores["current"], evaluation_units=EVALUATION_LOGS)
    metrics = primary["metrics"]
    scored = metrics["scored_drives"]
    primary["gates"]["vendor-calibration-flagged"] = rule.allowance(
        len(metrics["flagged_baselines"]), scored
    ) | {"flagged": metrics["flagged_baselines"]}
    primary["gates"].pop("vendor-calibration-not-flagged")
    result = {
        "schema_version": SCHEMA,
        "analysis": "gate derivation from the 30 Argoverse 2 development logs; no verdict",
        "primary": primary,
        "secondary": rule.secondary(scores["proposed"]),
        "provenance": _provenance()
        | {
            "rule_document_sha256": _sha(RULE_DOCUMENT),
            "manifest_sha256": _sha(dest / "manifest.json"),
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("work", type=Path)
    r.add_argument("--log", action="append")
    r.add_argument("--retry-failed", action="store_true")
    u = sub.add_parser("publish")
    u.add_argument("work", type=Path)
    u.add_argument("dest", type=Path)
    g = sub.add_parser("gates")
    g.add_argument("dest", type=Path)
    g.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.work, args.log, args.retry_failed)
    if args.command == "publish":
        print(publish(args.work, args.dest), "files published")
        return 0
    result = gates(args.dest, args.output)
    print(json.dumps({k: result[k]["gates"] for k in ("primary", "secondary")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
