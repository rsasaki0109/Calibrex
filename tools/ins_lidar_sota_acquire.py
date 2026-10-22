"""Bounded, digest-receipted acquisition for the frozen INS--LiDAR audit."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import kitti_drive_fetch
from tools.boreas_ins_lidar_metadata import BoreasInsLidarMetadata
from tools.ins_lidar_sota_inventory import KittiInsLidarInventory
from tools.ins_lidar_sota_protocol import (
    AcquisitionReceipt,
    EvaluationCase,
    InsLidarPreregistration,
)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".writing-{time.time_ns()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _download(
    url: str,
    path: Path,
    *,
    size: int | None,
    etag: str | None,
    expected_sha256: str | None = None,
    cap: int,
) -> dict[str, Any]:
    """Reuse verified successes, retry transport failures twice, keep every failed attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_name(path.name + ".receipt.json")
    history_path = path.with_name(path.name + ".attempts.json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["url"] != url or receipt["etag"] != etag:
            raise ValueError("download identity changed on resume")
        if (
            receipt["bytes"] > cap
            or (size is not None and receipt["bytes"] != size)
            or (expected_sha256 is not None and receipt["sha256"] != expected_sha256)
        ):
            raise ValueError("download pin/budget changed on resume")
        if receipt["temporary"] not in (path.name + ".part-1", path.name + ".part-2"):
            raise ValueError("invalid temporary download path")
        candidate = path if path.exists() else path.with_name(receipt["temporary"])
        if (
            candidate.stat().st_size != receipt["bytes"]
            or file_digest(candidate) != receipt["sha256"]
        ):
            raise ValueError("download changed on resume")
        if not path.exists():
            candidate.rename(path)
        return receipt
    if path.exists():
        raise ValueError("unreceipted existing input must not be overwritten")
    if size is not None and size > cap:
        raise ValueError("object exceeds acquisition byte budget")
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    while len(history) < 2:
        number = len(history) + 1
        temporary = path.with_name(path.name + f".part-{number}")
        attempt = {
            "number": number,
            "url": url,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "RUNNING",
        }
        history.append(attempt)
        _json(history_path, history)
        try:
            headers = {} if etag is None else {"If-Match": '"' + etag + '"'}
            request = urllib.request.Request(url, headers=headers)
            digest, count = hashlib.sha256(), 0
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("xb") as stream,
            ):
                if response.status != 200 or (
                    etag is not None and response.headers["ETag"].strip('"') != etag
                ):
                    raise ValueError("object status or ETag differs from its pin")
                for block in iter(lambda: response.read(1 << 20), b""):
                    count += len(block)
                    if count > cap or (size is not None and count > size):
                        raise ValueError("object exceeds declared size/budget")
                    digest.update(block)
                    stream.write(block)
            if size is not None and count != size:
                raise OSError("truncated object")
            actual = digest.hexdigest()
            if expected_sha256 is not None and actual != expected_sha256:
                raise ValueError("object SHA-256 differs from its pin")
            receipt = {
                "url": url,
                "etag": etag,
                "bytes": count,
                "sha256": actual,
                "temporary": temporary.name,
            }
            _json(receipt_path, receipt)
            temporary.rename(path)
            attempt["status"] = "PASS"
            _json(history_path, history)
            return receipt
        except Exception as error:
            attempt.update(status="FAIL", error_type=type(error).__name__, error=str(error))
            _json(history_path, history)
            transient = isinstance(error, (OSError, urllib.error.URLError, ConnectionError))
            if isinstance(error, urllib.error.HTTPError):
                transient = error.code >= 500 or error.code == 429
            if not transient or number == 2:
                raise
            time.sleep(1)
    raise ValueError("download attempt budget exhausted; original attempts retained")


def verify_input_receipt(receipt: dict[str, Any], work: Path) -> None:
    """Re-hash every acquired method input; reject escapes and missing/replaced files."""
    root = work.resolve()
    for item in receipt["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("input path escapes its acquisition directory")
        if path.stat().st_size != item["bytes"] or file_digest(path) != item["sha256"]:
            raise ValueError(f"input digest mismatch: {item['path']}")


def acquire_case(
    case: EvaluationCase,
    prereg: InsLidarPreregistration,
    source_root: Path,
    work: Path,
) -> dict[str, Any]:
    """Acquire only the declared streams of one case, retaining all transport records."""
    work.mkdir(parents=True, exist_ok=True)
    final = work / "acquisition.json"
    pin = prereg.metadata[case.dataset_family]
    if file_digest(source_root / pin.path) != pin.sha256:
        raise ValueError("source inventory differs from pre-registration")
    if final.exists():
        receipt = AcquisitionReceipt.model_validate_json(final.read_text()).model_dump(mode="json")
        if (
            receipt["case_id"] != case.case_id
            or receipt["provenance"]["source_metadata_sha256"] != pin.sha256
            or receipt["provenance"]["frozen_code_commit"] != prereg.frozen_code_commit
        ):
            raise ValueError("acquisition directory belongs to another case")
        verify_input_receipt(receipt, work)
        return receipt
    if shutil.disk_usage(work).free < prereg.acquisition.minimum_free_disk_bytes:
        raise OSError("less than the frozen 20 GiB free-disk floor")
    source_records = []
    if case.dataset_family == "boreas":
        metadata = BoreasInsLidarMetadata.model_validate_json((source_root / pin.path).read_text())
        candidates = [r for r in metadata.recordings if r.recording_id == case.case_id]
        if len(candidates) != 1 or candidates[0].missing_required:
            raise ValueError("Boreas case lacks required metadata")
        candidate = candidates[0]
        sequence = work / "data" / case.case_id
        objects = [
            candidate.reference_trajectory,
            candidate.reference_calibration,
            *candidate.lidar,
        ]
        if any(obj is None for obj in objects) or len(objects) != 602:
            raise ValueError("Boreas acquisition must contain 600 scans and two reference files")

        def get(obj: Any) -> tuple[Path, dict[str, Any]]:
            if not obj.key.startswith(case.case_id + "/"):
                raise ValueError("Boreas object belongs to a different recording")
            relative = Path(obj.key).relative_to(case.case_id)
            path = sequence / relative
            if not path.resolve().is_relative_to(sequence.resolve()):
                raise ValueError("Boreas object escapes its acquisition directory")
            limit = (
                prereg.acquisition.max_reference_object_bytes
                if str(relative).replace("\\", "/") == "applanix/gps_post_process.csv"
                else prereg.acquisition.max_point_object_bytes
            )
            receipt = _download(
                "https://boreas.s3.us-west-2.amazonaws.com/"
                + urllib.parse.quote(obj.key, safe="/"),
                path,
                size=obj.bytes,
                etag=obj.etag,
                cap=limit,
            )
            return path, receipt

        with ThreadPoolExecutor(max_workers=4) as executor:
            obtained = list(executor.map(get, objects))
        paths = [p for p, _ in obtained]
        source_records = [r for _, r in obtained]
    else:
        metadata = KittiInsLidarInventory.model_validate_json((source_root / pin.path).read_text())
        matches = [r for r in metadata.selected if r.case_id == case.case_id]
        if len(matches) != 1:
            raise ValueError("KITTI case missing from its frozen inventory")
        candidate = matches[0]
        date = candidate.date
        calibration = work / f"{date}_calib.zip"
        source_records.append(
            _download(
                kitti_drive_fetch.S3 + f"/{date}_calib.zip",
                calibration,
                size=None,
                etag=None,
                expected_sha256=prereg.calibration_archive_sha256[date],
                cap=16384,
            )
        )
        failure_path = work / "fetch_attempts.json"
        attempts = json.loads(failure_path.read_text()) if failure_path.exists() else []
        record = None
        successful = [a for a in attempts if a["status"] == "PASS"]
        if successful:
            destination = work / f"fetch-{successful[-1]['number']}"
            record = json.loads((destination / "fetch-record.json").read_text())
            cached = json.loads((destination / "output-digests.json").read_text())
            verify_input_receipt({"files": cached}, destination)
        for number in range(len(attempts) + 1, 3):
            if record is not None:
                break
            destination = work / f"fetch-{number}"
            destination.mkdir()
            attempt = {"number": number, "status": "RUNNING", "source_url": candidate.url}
            attempts.append(attempt)
            _json(failure_path, attempts)
            try:
                record = kitti_drive_fetch.fetch_drive(
                    date,
                    candidate.drive,
                    600,
                    destination,
                    (candidate.size_bytes, candidate.etag),
                    streams=("velodyne_points", "oxts"),
                    max_fetch_bytes=min(2 * 1024**3, candidate.selected_range_bytes),
                )
                if (
                    kitti_drive_fetch.members_sha256(record["members"])
                    != candidate.selected_members_sha256
                ):
                    raise ValueError("KITTI selected members differ from frozen metadata")
                if record["frames"] != {"velodyne_points": 600, "oxts": 600}:
                    raise ValueError("incomplete KITTI streams")
                _json(destination / "fetch-record.json", record)
                _json(
                    destination / "output-digests.json",
                    [
                        {
                            "path": m["name"],
                            "bytes": (destination / m["name"]).stat().st_size,
                            "sha256": file_digest(destination / m["name"]),
                        }
                        for m in record["members"]
                    ],
                )
                attempt["status"] = "PASS"
                _json(failure_path, attempts)
                break
            except Exception as error:
                record = None
                attempt.update(status="FAIL", error_type=type(error).__name__, error=str(error))
                _json(failure_path, attempts)
                if not isinstance(error, OSError) or number == 2:
                    raise
        if record is None:
            raise ValueError("KITTI fetch attempt budget exhausted")
        if (record["size_bytes"], record["etag"]) != (candidate.size_bytes, candidate.etag):
            raise ValueError("cached KITTI archive differs from its frozen pin")
        if kitti_drive_fetch.members_sha256(record["members"]) != candidate.selected_members_sha256:
            raise ValueError("cached KITTI members differ from frozen metadata")
        with zipfile.ZipFile(io.BytesIO(calibration.read_bytes())) as archive:
            content = archive.read(f"{date}/calib_imu_to_velo.txt")
        cal_path = destination / date / "calib_imu_to_velo.txt"
        cal_path.write_bytes(content)
        sequence = destination / date / case.drive_ids[0]
        paths = [p for p in (destination / date).rglob("*") if p.is_file()]
        source_records.append(record)
    files = [
        {
            "path": p.relative_to(work).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": file_digest(p),
        }
        for p in sorted(paths)
    ]
    receipt = {
        "schema_version": "calibrex.ins_lidar_audit_acquisition/v0.1",
        "case_id": case.case_id,
        "dataset_family": case.dataset_family,
        "verified": True,
        "sequence_dir": sequence.relative_to(work).as_posix(),
        "files": files,
        "source_records": source_records,
        "provenance": {
            "generator": "tools.ins_lidar_sota_acquire",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_metadata_sha256": pin.sha256,
            "frozen_code_commit": prereg.frozen_code_commit,
            "source_sha256": file_digest(Path(__file__)),
            "timestamp_policy": (
                "KITTI text timestamps cut to first 600 indices; Boreas full independent INS CSV"
            ),
        },
    }
    verify_input_receipt(receipt, work)
    AcquisitionReceipt.model_validate(receipt)
    _json(final, receipt)
    return receipt
