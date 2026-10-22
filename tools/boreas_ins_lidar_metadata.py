"""Inventory Boreas INS--LiDAR candidates using S3 listings only, never sensor objects.

This is readiness evidence, not a pre-registration or an evaluation. Candidate
roles follow recording names, never motion, reference poses, or method scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BUCKET_URL = "https://boreas.s3.us-west-2.amazonaws.com/"
MAX_LISTING_BYTES = 2 * 1024 * 1024
MAX_PAGES = 100
ORIGINAL_RECORDING = re.compile(r"boreas-202[01]-\d{2}-\d{2}-\d{2}-\d{2}/\Z")
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
SELECTION_RULE = (
    "First three lexicographically ordered original Boreas (2020/2021) recording "
    "prefixes: first development candidate, next two evaluation candidates. "
    "First 600 lexicographically ordered numeric-stem LiDAR .bin keys per recording; "
    "all selected recordings retained even if required files are missing. "
    "Names, sizes and ETags only; no sensor or calibration object bodies read."
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectMetadata(_Strict):
    key: str
    bytes: int = Field(ge=0)
    etag: str
    last_modified: str


class ListingReceipt(_Strict):
    url: str
    bytes: int = Field(ge=0, le=MAX_LISTING_BYTES)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RecordingMetadata(_Strict):
    recording_id: str
    role: Literal["development_candidate", "evaluation_candidate"]
    lidar: list[ObjectMetadata]
    reference_trajectory: ObjectMetadata | None
    reference_calibration: ObjectMetadata | None
    selected_lidar_bytes: int = Field(ge=0)
    missing_required: list[str]


class MetadataProvenance(_Strict):
    generator: str = "tools.boreas_ins_lidar_metadata"
    created_at: str
    git_commit: str | None
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    command: list[str]
    bucket_url: str = BUCKET_URL
    dataset_license: str = "CC BY 4.0 (AWS Open Data Registry)"
    accuracy_claim: Literal[False] = False
    exposure_audit_complete: Literal[False] = False
    notice: str = (
        "Metadata readiness only. ETags pin object identity but are not SHA-256. "
        "Recording freshness requires a separate repository/history/local exposure audit. "
        "Do not run evaluation before frozen code and pre-registration are merged."
    )


class BoreasInsLidarMetadata(_Strict):
    schema_version: Literal["calibrex.boreas_ins_lidar_metadata/v0.1"] = (
        "calibrex.boreas_ins_lidar_metadata/v0.1"
    )
    provenance: MetadataProvenance
    selection_rule: str = SELECTION_RULE
    selected_frame_limit: int = Field(gt=0, le=600)
    recordings: list[RecordingMetadata]
    requests: list[ListingReceipt]
    sensor_object_download_bytes: Literal[0] = 0
    calibration_object_download_bytes: Literal[0] = 0


def fetch_listing(url: str) -> bytes:
    """GET only a bucket listing with a bounded response; never GET an object."""
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(parsed.query)
    if not url.startswith(BUCKET_URL + "?") or query.get("list-type") != ["2"]:
        raise ValueError("only S3 ListObjectsV2 requests are allowed")
    with urllib.request.urlopen(url, timeout=60) as response:
        body = response.read(MAX_LISTING_BYTES + 1)
    if len(body) > MAX_LISTING_BYTES:
        raise ValueError("listing exceeds metadata byte bound")
    return body


class _Inventory:
    def __init__(self, fetch: Callable[[str], bytes]) -> None:
        self.fetch = fetch
        self.receipts: list[ListingReceipt] = []

    def pages(self, prefix: str, *, delimiter: str | None = None) -> Iterator[ET.Element]:
        token: str | None = None
        seen: set[str] = set()
        for _ in range(MAX_PAGES):
            query = {"list-type": "2", "max-keys": "1000", "prefix": prefix}
            if delimiter is not None:
                query["delimiter"] = delimiter
            if token is not None:
                query["continuation-token"] = token
            url = BUCKET_URL + "?" + urllib.parse.urlencode(query)
            body = self.fetch(url)
            if len(body) > MAX_LISTING_BYTES:
                raise ValueError("listing exceeds metadata byte bound")
            self.receipts.append(
                ListingReceipt(url=url, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
            )
            tree = ET.fromstring(body)
            if tree.findtext("s3:Name", namespaces=NS) != "boreas":
                raise ValueError("unexpected bucket")
            if tree.findtext("s3:Prefix", default="", namespaces=NS) != prefix:
                raise ValueError("unexpected listing prefix")
            yield tree
            if tree.findtext("s3:IsTruncated", namespaces=NS) == "false":
                return
            token = tree.findtext("s3:NextContinuationToken", namespaces=NS)
            if not token or token in seen:
                raise ValueError("missing or repeated pagination token")
            seen.add(token)
        raise ValueError("too many metadata pages")

    def objects(self, prefix: str, *, limit: int | None = None) -> list[ObjectMetadata]:
        result: list[ObjectMetadata] = []
        previous: str | None = None
        for page in self.pages(prefix):
            for node in page.findall("s3:Contents", NS):
                key = node.findtext("s3:Key", default="", namespaces=NS)
                if not key.startswith(prefix) or (previous is not None and key <= previous):
                    raise ValueError("object keys must be unique, sorted and inside the prefix")
                previous = key
                if key.endswith("/"):
                    continue
                if limit is not None and not re.fullmatch(r"\d+\.bin", key[len(prefix) :]):
                    continue
                result.append(
                    ObjectMetadata(
                        key=key,
                        bytes=int(node.findtext("s3:Size", default="-1", namespaces=NS)),
                        etag=node.findtext("s3:ETag", default="", namespaces=NS).strip('"'),
                        last_modified=node.findtext("s3:LastModified", default="", namespaces=NS),
                    )
                )
                if limit is not None and len(result) == limit:
                    return result
        return result


def inventory(
    provenance: MetadataProvenance,
    *,
    fetch: Callable[[str], bytes] = fetch_listing,
    frame_limit: int = 600,
) -> BoreasInsLidarMetadata:
    """Build a schema-valid metadata-only readiness record, without claiming freshness."""
    if not 1 <= frame_limit <= 600:
        raise ValueError("frame_limit must be between 1 and 600")
    reader = _Inventory(fetch)
    prefixes = []
    for page in reader.pages("", delimiter="/"):
        prefixes.extend(
            node.findtext("s3:Prefix", default="", namespaces=NS)
            for node in page.findall("s3:CommonPrefixes", NS)
        )
    names = sorted({p for p in prefixes if ORIGINAL_RECORDING.fullmatch(p)})[:3]
    if len(names) != 3:
        raise ValueError("three original Boreas recording prefixes required")
    recordings = []
    for index, prefix in enumerate(names):
        lidar = reader.objects(prefix + "lidar/", limit=frame_limit)
        aux = reader.objects(prefix + "applanix/") + reader.objects(prefix + "calib/")
        by_key = {obj.key: obj for obj in aux}
        trajectory_key = prefix + "applanix/gps_post_process.csv"
        calibration_key = prefix + "calib/T_applanix_lidar.txt"
        missing = [key for key in (trajectory_key, calibration_key) if key not in by_key]
        if len(lidar) < frame_limit:
            missing.append(f"{frame_limit} LiDAR scans (found {len(lidar)})")
        recordings.append(
            RecordingMetadata(
                recording_id=prefix.rstrip("/"),
                role="development_candidate" if index == 0 else "evaluation_candidate",
                lidar=lidar,
                reference_trajectory=by_key.get(trajectory_key),
                reference_calibration=by_key.get(calibration_key),
                selected_lidar_bytes=sum(obj.bytes for obj in lidar),
                missing_required=missing,
            )
        )
    return BoreasInsLidarMetadata(
        provenance=provenance,
        selected_frame_limit=frame_limit,
        recordings=recordings,
        requests=reader.receipts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    schema_path = args.output.with_suffix(".schema.json")
    raw_dir = args.output.with_suffix(".responses")
    if args.output.exists() or schema_path.exists() or raw_dir.exists():
        parser.error("output exists; retain it and choose a fresh path")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    provenance = MetadataProvenance(
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command=[sys.executable, *sys.argv],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    raw_dir.mkdir()
    response_count = 0

    def capture_listing(url: str) -> bytes:
        nonlocal response_count
        body = fetch_listing(url)
        with (raw_dir / f"{response_count:03d}.xml").open("xb") as stream:
            stream.write(body)
        response_count += 1
        return body

    result = inventory(provenance, fetch=capture_listing)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(result.model_dump_json(indent=2) + "\n")
    with schema_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result.model_json_schema(), indent=2) + "\n")
    print(
        json.dumps(
            {
                "recordings": [
                    {
                        "id": r.recording_id,
                        "role": r.role,
                        "scans": len(r.lidar),
                        "selected_lidar_bytes": r.selected_lidar_bytes,
                        "missing": r.missing_required,
                    }
                    for r in result.recordings
                ],
                "listing_bytes": sum(r.bytes for r in result.requests),
                "sensor_object_download_bytes": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
