"""Run the four pre-registered INS--LiDAR cases once from merged, frozen main.

Acquisition may resume within its recorded transport budget. Method execution
never automatically repeats: failure, interruption and timeout remain evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ins_lidar_sota_acquire import _json, acquire_case, file_digest, verify_input_receipt
from tools.ins_lidar_sota_protocol import (
    AcquisitionReceipt,
    InsLidarPreregistration,
    RunManifest,
    load_preregistration,
    verify_frozen_source,
)
from tools.score_ins_lidar_sota import CaseSpec

ROOT = Path(__file__).resolve().parents[1]


def verify_raw_budget(prereg: InsLidarPreregistration, root: Path) -> int:
    """Bound two complete acquisition attempts before downloading any payload."""
    inventories = {
        f: json.loads((root / pin.path).read_text()) for f, pin in prereg.metadata.items()
    }
    required = 0
    for case in prereg.evaluation:
        if case.dataset_family == "kitti_raw":
            record = next(
                r for r in inventories["kitti_raw"]["selected"] if r["case_id"] == case.case_id
            )
            required += record["selected_output_bytes_upper_bound"] + 16384
        else:
            record = next(
                r for r in inventories["boreas"]["recordings"] if r["recording_id"] == case.case_id
            )
            required += sum(r["bytes"] for r in record["lidar"])
            required += record["reference_trajectory"]["bytes"]
            required += record["reference_calibration"]["bytes"]
    worst_case = required * prereg.acquisition.object_attempts
    if worst_case > prereg.acquisition.maximum_additional_raw_bytes:
        raise ValueError("two acquisition attempts exceed the frozen raw disk budget")
    return worst_case


def verify_registered_main(path: Path, root: Path) -> tuple[InsLidarPreregistration, str]:
    """Check registration is committed on frozen main before any payload is read."""
    prereg = load_preregistration(path, root)
    head = verify_frozen_source(prereg, root, require_main=True)
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=root)
    # Git may check out CRLF on Windows; compare the parsed document as well as
    # require a clean tracked registration, rather than compare checkout bytes.
    import yaml

    if yaml.safe_load(committed) != yaml.safe_load(path.read_bytes()):
        raise ValueError("pre-registration differs from merged main")
    return prereg, head


@contextmanager
def exclusive_run(path: Path) -> Iterator[None]:
    """Use an OS lock released on process exit, including abnormal termination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def worker(prereg_path: Path, case_index: int, work: Path, output: Path) -> None:
    """Run one native default fit; reference poses never seed LiDAR odometry."""
    prereg, _ = verify_registered_main(prereg_path, ROOT)
    if not 0 <= case_index < len(prereg.evaluation):
        raise ValueError("case index outside the declared evaluation")
    case = prereg.evaluation[case_index]
    receipt = AcquisitionReceipt.model_validate_json(
        (work / "acquisition.json").read_text()
    ).model_dump(mode="json")
    if (
        receipt["case_id"] != case.case_id
        or receipt["provenance"]["frozen_code_commit"] != prereg.frozen_code_commit
        or receipt["provenance"]["source_metadata_sha256"]
        != prereg.metadata[case.dataset_family].sha256
    ):
        raise ValueError("case input receipt mismatch")
    verify_input_receipt(receipt, work)
    sequence = (work / receipt["sequence_dir"]).resolve()
    if not sequence.is_relative_to(work.resolve()):
        raise ValueError("sequence escapes acquisition directory")
    from calibrex.core.validation import validate_file
    from calibrex.evaluation.boreas_ins_lidar import run_boreas_ins_lidar_hand_eye
    from calibrex.evaluation.ins_lidar_hand_eye import (
        InsLidarRunOptions,
        run_kitti_ins_lidar_hand_eye,
    )

    if output.exists():
        raise ValueError("method artifacts must never be overwritten")
    method = (
        run_kitti_ins_lidar_hand_eye
        if case.dataset_family == "kitti_raw"
        else run_boreas_ins_lidar_hand_eye
    )
    artifact = method(
        sequence, InsLidarRunOptions(), max_frames=600, command=[sys.executable, *sys.argv]
    )
    verify_input_receipt(receipt, work)
    artifact.save(output)
    validation = validate_file(output, kind="ins-lidar-hand-eye")
    if not validation.valid:
        raise ValueError(str(validation))


def evaluate(prereg_path: Path, work: Path, output: Path) -> dict[str, Any]:
    """Retain all four cases, raw-input custody, and every method failure."""
    prereg, head = verify_registered_main(prereg_path, ROOT)
    verify_raw_budget(prereg, ROOT)
    work, output = work.resolve(), output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(work / "run.lock"):
        manifest_path = output / "run.json"
        identity = {
            "preregistration_sha256": file_digest(prereg_path),
            "frozen_code_commit": prereg.frozen_code_commit,
        }
        if manifest_path.exists():
            manifest = RunManifest.model_validate_json(manifest_path.read_text()).model_dump(
                mode="json"
            )
            if any(manifest.get(k) != v for k, v in identity.items()):
                raise ValueError("run directory belongs to another registration")
        else:
            manifest = {
                "schema_version": "calibrex.ins_lidar_sota_run/v0.1",
                **identity,
                "cases": [
                    CaseSpec(**c.model_dump(), drive_ids=c.drive_ids).model_dump()
                    for c in prereg.evaluation
                ],
                "input_data_verified": False,
                "input_receipts": {},
                "attempts": {},
                "provenance": {
                    "generator": "tools.ins_lidar_sota_evaluate",
                    "source_head": head,
                    "python": platform.python_version(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "command": [sys.executable, *sys.argv],
                },
            }
            _json(manifest_path, manifest)
        for index, case in enumerate(prereg.evaluation):
            prior = manifest["attempts"].get(case.case_id)
            case_work = work / case.case_id
            if prior is not None:
                # A previous RUNNING state also consumes the method attempt.
                # Never guess that an interrupted fit did not read its data.
                continue
            try:
                acquire_case(case, prereg, ROOT, case_work)
            except Exception as error:
                manifest.setdefault("acquisition_errors", []).append(
                    {
                        "case_id": case.case_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                _json(manifest_path, manifest)
                continue
            receipt_path = case_work / "acquisition.json"
            manifest["input_receipts"][case.case_id] = {
                "path": str(receipt_path),
                "sha256": file_digest(receipt_path),
                "work_dir": str(case_work),
            }
            artifact_path = output / f"{case.case_id}.yaml"
            if artifact_path.exists():
                raise ValueError("unrecorded result exists; preserve it for investigation")
            command = [
                sys.executable,
                "-X",
                "utf8",
                str(Path(__file__).resolve()),
                str(prereg_path.resolve()),
                str(case_work),
                str(artifact_path),
                "--worker-index",
                str(index),
            ]
            attempt = {
                "status": "RUNNING",
                "command": command,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest["attempts"][case.case_id] = attempt
            _json(manifest_path, manifest)
            with (output / f"{case.case_id}.log").open("xb") as log:
                try:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=prereg.acquisition.method_timeout_seconds,
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
            attempt["finished_at"] = datetime.now(timezone.utc).isoformat()
            _json(manifest_path, manifest)
        for info in manifest["input_receipts"].values():
            receipt_path = Path(info["path"])
            if file_digest(receipt_path) != info["sha256"]:
                raise ValueError("input receipt changed on resume")
            verify_input_receipt(json.loads(receipt_path.read_text()), Path(info["work_dir"]))
        manifest["input_data_verified"] = len(manifest["input_receipts"]) == len(prereg.evaluation)
        RunManifest.model_validate(manifest)
        _json(manifest_path, manifest)
        return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_index is not None:
        worker(args.preregistration, args.worker_index, args.work_dir, args.output_dir)
        return 0
    manifest = evaluate(args.preregistration, args.work_dir, args.output_dir)
    return 0 if all(c["artifact_path"] for c in manifest["cases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
