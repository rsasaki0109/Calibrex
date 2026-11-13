#!/usr/bin/env python3
"""Run the frozen INS-LiDAR rotation audit v2, or its development run on spent recordings.

One case at a time: acquire the declared streams (pinned by size, ETag or SHA-256), run
the native default fit once in a subprocess, record the result, then delete the raw
inputs. The run resumes by the same command:

* a case whose method attempt finished (PASS or FAIL) is never run again;
* a method attempt left RUNNING by a killed process produced no result anyone saw, so it
  is recorded as interrupted and run again;
* a case whose inputs could not be acquired is retried within the transport budget of
  ``ins_lidar_sota_acquire`` (two attempts per object or archive fetch). A case still
  missing leaves the audit incomplete.

::

    python tools/ins_lidar_sota_v2_evaluate.py PREREGISTRATION WORK RUNS
    python tools/ins_lidar_sota_v2_evaluate.py --development WORK RUNS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.ins_lidar_sota_acquire import (  # noqa: E402
    _json,
    acquire_case,
    file_digest,
    verify_input_receipt,
)
from tools.ins_lidar_sota_evaluate import exclusive_run  # noqa: E402
from tools.ins_lidar_sota_protocol import (  # noqa: E402
    AcquisitionReceipt,
    EvaluationCase,
    MetadataPin,
    verify_frozen_source,
)
from tools.ins_lidar_sota_v2_protocol import (  # noqa: E402
    DEVELOPMENT_CASES,
    AcquisitionPolicy,
    RunManifest,
    load_preregistration,
)
from tools.score_ins_lidar_sota import CaseSpec  # noqa: E402

V1_PREREGISTRATION = "docs/benchmarks/ins_lidar_rotation_v1_preregistration.yaml"


@dataclass(frozen=True)
class Plan:
    """What the run reads and from where; the pre-registration, or the development plan."""

    evaluation: list[EvaluationCase]
    metadata: dict[str, MetadataPin]
    calibration_archive_sha256: dict[str, str]
    acquisition: AcquisitionPolicy
    frozen_code_commit: str
    identity_sha256: str
    development: bool


def _head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def development_plan(root: Path = ROOT) -> Plan:
    """v1's four evaluation recordings with v1's pins: spent data, for development only."""

    v1 = yaml.safe_load((root / V1_PREREGISTRATION).read_text("utf-8"))
    metadata = {family: MetadataPin(**pin) for family, pin in v1["metadata"].items()}
    for pin in metadata.values():
        if file_digest(root / pin.path) != pin.sha256:
            raise ValueError(f"metadata pin mismatch: {pin.path}")
    cases = [
        EvaluationCase(case_id=case, dataset_family=family, rig_id=rig)  # type: ignore[arg-type]
        for case, family, rig in DEVELOPMENT_CASES
    ]
    return Plan(
        evaluation=cases,
        metadata=metadata,
        calibration_archive_sha256=dict(v1["calibration_archive_sha256"]),
        acquisition=AcquisitionPolicy(),
        frozen_code_commit=_head(root),
        identity_sha256=hashlib.sha256(
            json.dumps([DEVELOPMENT_CASES, v1["metadata"]], sort_keys=True).encode()
        ).hexdigest(),
        development=True,
    )


def registered_plan(path: Path, root: Path = ROOT) -> Plan:
    """The pre-registration, committed on the merged main whose src/tools are frozen."""

    prereg = load_preregistration(path, root)
    head = verify_frozen_source(prereg, root, require_main=True)  # type: ignore[arg-type]
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=root)
    if yaml.safe_load(committed) != yaml.safe_load(path.read_bytes()):
        raise ValueError("pre-registration differs from merged main")
    return Plan(
        evaluation=list(prereg.evaluation),
        metadata=dict(prereg.metadata),
        calibration_archive_sha256=dict(prereg.calibration_archive_sha256),
        acquisition=prereg.acquisition,
        frozen_code_commit=prereg.frozen_code_commit,
        identity_sha256=file_digest(path),
        development=False,
    )


def load_plan(prereg_path: Path | None) -> Plan:
    return development_plan() if prereg_path is None else registered_plan(prereg_path)


def worker(plan: Plan, case_index: int, work: Path, output: Path) -> None:
    """Run one native default fit; reference poses never seed LiDAR odometry."""

    if not 0 <= case_index < len(plan.evaluation):
        raise ValueError("case index outside the declared evaluation")
    case = plan.evaluation[case_index]
    receipt = AcquisitionReceipt.model_validate_json(
        (work / "acquisition.json").read_text()
    ).model_dump(mode="json")
    if (
        receipt["case_id"] != case.case_id
        or receipt["provenance"]["frozen_code_commit"] != plan.frozen_code_commit
        or receipt["provenance"]["source_metadata_sha256"]
        != plan.metadata[case.dataset_family].sha256
    ):
        raise ValueError("case input receipt mismatch")
    verify_input_receipt(receipt, work)
    sequence = (work / receipt["sequence_dir"]).resolve()
    if not sequence.is_relative_to(work.resolve()):
        raise ValueError("sequence escapes acquisition directory")
    import calibrex
    from calibrex.core.validation import validate_file
    from calibrex.evaluation.boreas_ins_lidar import run_boreas_ins_lidar_hand_eye
    from calibrex.evaluation.ins_lidar_hand_eye import (
        InsLidarRunOptions,
        run_kitti_ins_lidar_hand_eye,
    )

    if not Path(calibrex.__file__).resolve().is_relative_to((ROOT / "src").resolve()):
        raise ValueError(f"calibrex is imported from another checkout: {calibrex.__file__}")
    if output.exists():
        raise ValueError("method artifacts must never be overwritten")
    method = (
        run_kitti_ins_lidar_hand_eye
        if case.dataset_family == "kitti_raw"
        else run_boreas_ins_lidar_hand_eye
    )
    artifact = method(
        sequence,
        InsLidarRunOptions(),
        max_frames=plan.acquisition.max_frames,
        command=[sys.executable, *sys.argv],
    )
    verify_input_receipt(receipt, work)  # the fit did not change its inputs
    artifact.save(output)
    validation = validate_file(output, kind="ins-lidar-hand-eye")
    if not validation.valid:
        raise ValueError(str(validation))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_aside(path: Path, number: int) -> str | None:
    """Keep what an interrupted attempt left behind under another name."""

    if not path.exists():
        return None
    kept = path.with_name(f"{path.stem}.interrupted-{number}{path.suffix}")
    path.rename(kept)
    return kept.name


def delete_raw_inputs(receipt: dict[str, Any], work: Path) -> int:
    """Delete the method inputs of a finished case; the receipt keeps their digests."""

    root = work.resolve()
    freed = 0
    for item in receipt["files"]:
        path = (root / item["path"]).resolve()
        if path.is_relative_to(root) and path.is_file():
            freed += path.stat().st_size
            path.unlink()
    return freed


def evaluate(prereg_path: Path | None, work: Path, output: Path) -> dict[str, Any]:
    """Run every declared case once, keeping every failure and interruption on record."""

    plan = load_plan(prereg_path)
    work, output = work.resolve(), output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(work / "run.lock"):
        manifest_path = output / "run.json"
        identity = {
            "preregistration_sha256": plan.identity_sha256,
            "frozen_code_commit": plan.frozen_code_commit,
            "development": plan.development,
        }
        if manifest_path.exists():
            manifest = RunManifest.model_validate_json(manifest_path.read_text()).model_dump(
                mode="json"
            )
            if any(manifest.get(k) != v for k, v in identity.items()):
                raise ValueError("run directory belongs to another registration")
        else:
            manifest = {
                "schema_version": "calibrex.ins_lidar_sota_run/v0.2",
                **identity,
                "cases": [
                    CaseSpec(**c.model_dump(), drive_ids=c.drive_ids).model_dump()
                    for c in plan.evaluation
                ],
                "input_receipts": {},
                "attempts": {},
                "interrupted_attempts": [],
                "acquisition_errors": [],
                "provenance": {
                    "generator": "tools.ins_lidar_sota_v2_evaluate",
                    "source_head": _head(ROOT),
                    "python": platform.python_version(),
                    "created_at": _now(),
                    "command": [sys.executable, *sys.argv],
                },
            }
            _json(manifest_path, manifest)
        for index, case in enumerate(plan.evaluation):
            prior = manifest["attempts"].get(case.case_id)
            case_work = work / case.case_id
            artifact_path = output / f"{case.case_id}.yaml"
            log_path = output / f"{case.case_id}.log"
            if prior is not None and prior["status"] != "RUNNING":
                continue  # a finished attempt, pass or fail, is the case's one attempt
            if prior is not None:
                number = 1 + sum(
                    item["case_id"] == case.case_id for item in manifest["interrupted_attempts"]
                )
                manifest["interrupted_attempts"].append(
                    {
                        **prior,
                        "case_id": case.case_id,
                        "noted_at": _now(),
                        "kept_artifact": _set_aside(artifact_path, number),
                        "kept_log": _set_aside(log_path, number),
                    }
                )
                del manifest["attempts"][case.case_id]
                _json(manifest_path, manifest)
            try:
                receipt = acquire_case(case, plan, ROOT, case_work)  # type: ignore[arg-type]
            except Exception as error:
                manifest["acquisition_errors"].append(
                    {
                        "case_id": case.case_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "at": _now(),
                    }
                )
                _json(manifest_path, manifest)
                continue
            receipt_path = case_work / "acquisition.json"
            manifest["input_receipts"][case.case_id] = {
                "path": str(receipt_path),
                "sha256": file_digest(receipt_path),
                "work_dir": str(case_work),
                "raw_deleted": False,
            }
            if artifact_path.exists():
                raise ValueError("unrecorded result exists; preserve it for investigation")
            command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
            command += ["--development"] if prereg_path is None else [str(prereg_path.resolve())]
            command += [str(case_work), str(artifact_path), "--worker-index", str(index)]
            attempt: dict[str, Any] = {
                "status": "RUNNING",
                "command": command,
                "started_at": _now(),
            }
            manifest["attempts"][case.case_id] = attempt
            _json(manifest_path, manifest)
            with log_path.open("xb") as log:
                try:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=plan.acquisition.method_timeout_seconds,
                        check=False,
                    )
                    attempt.update(
                        status="PASS" if result.returncode == 0 else "FAIL",
                        returncode=result.returncode,
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    attempt.update(status="FAIL", error_type=type(error).__name__, error=str(error))
            if attempt["status"] == "PASS":
                manifest["cases"][index].update(
                    artifact_path=artifact_path.name, artifact_sha256=file_digest(artifact_path)
                )
            attempt["finished_at"] = _now()
            _json(manifest_path, manifest)
            if plan.acquisition.delete_raw_after_method:
                freed = delete_raw_inputs(receipt, case_work)
                manifest["input_receipts"][case.case_id].update(
                    raw_deleted=True, raw_deleted_bytes=freed
                )
                _json(manifest_path, manifest)
        RunManifest.model_validate(manifest)
        _json(manifest_path, manifest)
        return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="[PREREGISTRATION] WORK RUNS")
    parser.add_argument("--development", action="store_true", help="the spent v1 recordings")
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    expected = 2 if args.development else 3
    if len(args.paths) != expected:
        parser.error("give PREREGISTRATION WORK RUNS, or --development WORK RUNS")
    prereg_path = None if args.development else args.paths[0]
    first, second = args.paths[-2], args.paths[-1]
    if args.worker_index is not None:
        worker(load_plan(prereg_path), args.worker_index, first, second)
        return 0
    manifest = evaluate(prereg_path, first, second)
    finished = sum(a["status"] != "RUNNING" for a in manifest["attempts"].values())
    print(
        json.dumps(
            {
                "cases": len(manifest["cases"]),
                "finished": finished,
                "passed": sum(a["status"] == "PASS" for a in manifest["attempts"].values()),
                "interrupted": len(manifest["interrupted_attempts"]),
                "acquisition_errors": len(manifest["acquisition_errors"]),
            }
        )
    )
    return 0 if finished == len(manifest["cases"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
