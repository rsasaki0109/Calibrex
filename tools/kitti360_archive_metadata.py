#!/usr/bin/env python3
"""List an official KITTI-360 ZIP using exact metadata-only HTTP ranges.

No local headers or member contents are requested. ZIP comments, split archives
and extended ZIP64 end records are deliberately unsupported: never widen a tail
request to discover them. Pass a URL obtained from the official distribution;
this tool does not guess perspective archive URLs or select evaluation windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
MAX_DIRECTORY_BYTES = 64 * 1024 * 1024
OFFICIAL_HOST = "s3.eu-central-1.amazonaws.com"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def urlopen(request: Request, *, timeout: float) -> Any:
    """Keep default proxy and TLS handling, but never follow archive redirects."""

    return build_opener(_NoRedirect()).open(request, timeout=timeout)


class MetadataReader:
    """Pin one official archive and retain only exact ZIP metadata ranges."""

    def __init__(self, url: str, *, timeout: float = 30.0) -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != OFFICIAL_HOST
            or not parsed.path.startswith("/avg-projects/KITTI-360/")
            or not parsed.path.endswith(".zip")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "use an official KITTI-360 HTTPS ZIP URL, without credentials or query"
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.url, self.timeout = url, timeout
        self.parts: list[tuple[str, int, bytes]] = []
        request = Request(url, method="HEAD", headers={"Accept-Encoding": "identity"})
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError("HEAD must return 200 without a redirect")
            headers = {key.lower(): value for key, value in response.headers.items()}
        self.size = int(headers.get("content-length", "0"))
        self.etag = headers.get("etag", "")
        if self.size < 22 or self.size > (1 << 63) - 1:
            raise ValueError("missing or invalid ZIP Content-Length")
        if re.fullmatch(r'"[^"\r\n]+"', self.etag) is None:
            raise ValueError("HEAD must supply a strong quoted ETag")
        if headers.get("content-encoding", "identity").lower() != "identity":
            raise ValueError("encoded archive responses are unsupported")
        self.advertised_ranges = headers.get("accept-ranges")
        self.last_modified = headers.get("last-modified")

    def read(self, start: int, end: int, role: str) -> bytes:
        """Require a pinned exact 206 before reading any response body."""

        if not 0 <= start <= end < self.size:
            raise ValueError("metadata range outside archive")
        length = end - start + 1
        request = Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "If-Match": self.etag,
                "Accept-Encoding": "identity",
            },
        )
        with urlopen(request, timeout=self.timeout) as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            if response.status != 206 or response.geturl() != self.url:
                raise ValueError("range request must return 206 without a redirect")
            if headers.get("content-range") != f"bytes {start}-{end}/{self.size}":
                raise ValueError("Content-Range does not match the exact metadata request")
            if int(headers.get("content-length", "-1")) != length:
                raise ValueError("range Content-Length mismatch")
            if headers.get("etag") != self.etag:
                raise ValueError("range ETag differs from HEAD")
            if headers.get("content-encoding", "identity").lower() != "identity":
                raise ValueError("encoded range responses are unsupported")
            payload = response.read(length)
        if len(payload) != length:
            raise ValueError("truncated metadata range")
        self.parts.append((role, start, payload))
        return payload


def list_members(reader: MetadataReader) -> list[dict[str, Any]]:
    """Read only the fixed end records and central directory of an uncommented ZIP."""

    eocd_offset = reader.size - 22
    eocd = reader.read(eocd_offset, reader.size - 1, "eocd")
    signature, disk, cd_disk, disk_count, count, cd_size, cd_offset, comment = struct.unpack(
        "<4s4H2IH", eocd
    )
    if signature != b"PK\x05\x06" or comment:
        raise ValueError("uncommented ZIP required; refusing a broader tail read")
    if disk != 0 or cd_disk != 0 or disk_count != count:
        raise ValueError("split ZIP archives are unsupported")
    metadata_start = eocd_offset
    if count == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        locator_offset = eocd_offset - 20
        locator = reader.read(locator_offset, eocd_offset - 1, "zip64_locator")
        marker, record_disk, record_offset, disks = struct.unpack("<4sIQI", locator)
        if marker != b"PK\x06\x07" or record_disk != 0 or disks != 1:
            raise ValueError("invalid or split ZIP64 locator")
        if record_offset < 0 or record_offset + 56 != locator_offset:
            raise ValueError("extended or displaced ZIP64 end records are unsupported")
        record = reader.read(record_offset, locator_offset - 1, "zip64_eocd")
        values = struct.unpack("<4sQ2H2I4Q", record)
        marker, record_size, _, _, disk64, cd_disk64, disk_count64, count64, size64, offset64 = (
            values
        )
        if marker != b"PK\x06\x06" or record_size != 44:
            raise ValueError("invalid or extended ZIP64 end record")
        if disk64 != 0 or cd_disk64 != 0 or disk_count64 != count64:
            raise ValueError("split ZIP64 archives are unsupported")
        for short, full, sentinel in (
            (count, count64, 0xFFFF),
            (cd_size, size64, 0xFFFFFFFF),
            (cd_offset, offset64, 0xFFFFFFFF),
        ):
            if short != sentinel and short != full:
                raise ValueError("ZIP and ZIP64 end records disagree")
        count, cd_size, cd_offset = count64, size64, offset64
        metadata_start = record_offset
    if cd_offset + cd_size != metadata_start or cd_size > MAX_DIRECTORY_BYTES:
        raise ValueError("central directory bounds or metadata size limit violated")
    if bool(count) != bool(cd_size):
        raise ValueError("central directory entry count and size disagree")
    if cd_size:
        reader.read(cd_offset, cd_offset + cd_size - 1, "central_directory")
    # Sparse holes stand in for all local headers and member payloads. ZipFile
    # lists the directory only; no member is opened or extracted.
    with tempfile.TemporaryDirectory() as scratch:
        skeleton = Path(scratch) / "metadata.zip"
        with skeleton.open("wb") as stream:
            stream.truncate(reader.size)
            for _, offset, payload in reader.parts:
                stream.seek(offset)
                stream.write(payload)
        with zipfile.ZipFile(skeleton) as archive:
            members = archive.infolist()
    if len(members) != count:
        raise ValueError("central directory entry count mismatch")
    names = [item.filename for item in members]
    if len(set(names)) != len(names):
        raise ValueError("duplicate member names in central directory")
    if any(not 0 <= item.header_offset < cd_offset for item in members):
        raise ValueError("member local-header offsets lie outside the payload region")
    return [
        {
            "name": item.filename,
            "size_bytes": item.file_size,
            "compressed_bytes": item.compress_size,
            "compression": item.compress_type,
            "crc32": f"{item.CRC:08x}",
            "local_header_offset": item.header_offset,
            "is_directory": item.is_dir(),
        }
        for item in members
    ]


def inventory(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Produce provenance for metadata only, without selecting or reading sensor records."""

    reader = MetadataReader(url, timeout=timeout)
    members = list_members(reader)
    sequence_ids = sorted(
        {
            match.group(1)
            for member in members
            if (match := re.search(r"(?:^|/)2013_05_28_drive_(\d{4})_sync/", member["name"]))
        }
    )
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "format": "calibrex.kitti360_archive_metadata/v1",
        "metadata_only": True,
        "url": url,
        "size_bytes": reader.size,
        "etag": reader.etag,
        "last_modified": reader.last_modified,
        "advertised_accept_ranges": reader.advertised_ranges,
        "verified_exact_http_ranges": True,
        "sequence_ids": sequence_ids,
        "members": members,
        "provenance": {
            "generator": "tools/kitti360_archive_metadata.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ranges": [
                {
                    "role": role,
                    "start": offset,
                    "end": offset + len(payload) - 1,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for role, offset, payload in reader.parts
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="verified official archive URL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; retain the previous metadata inventory")
    report = inventory(args.url, timeout=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
