"""Remote listings must never fetch neighboring local headers or sensor payloads."""

import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import pytest
from tests.unit.test_score_hilti_camera_camera import _load

metadata = _load("kitti360_archive_metadata")
URL = (
    "https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_3d_raw/"
    "2013_05_28_drive_0000_sync_velodyne.zip"
)
MEMBER = "data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000000.bin"
ETAG = '"synthetic-metadata-test"'


def archive(*, zip64: bool = False, comment: bytes = b"", empty: bool = False) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_STORED) as output:
        if not empty:
            output.writestr(MEMBER, b"synthetic sensor payload: must never be requested" * 5000)
        output.comment = comment
    blob = data.getvalue()
    if not zip64:
        return blob
    end = len(blob) - 22
    fields = list(struct.unpack("<4s4H2IH", blob[end:]))
    record = struct.pack("<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, fields[4], *fields[4:7])
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, end, 1)
    fields[3:7] = [0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF]
    return blob[:end] + record + locator + struct.pack("<4s4H2IH", *fields)


class Response:
    def __init__(self, status: int, headers: dict[str, str], payload: bytes) -> None:
        self.status, self.headers, self.payload = status, headers, payload
        self.reads: list[int] = []
        self.url = URL

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        self.reads.append(size)
        return self.payload[:size]


class SparseArchive:
    """A large synthetic ZIP whose non-metadata region is never served."""

    def __init__(self, data: bytes, base: int, central_start: int) -> None:
        self.data, self.base, self.central_start = data, base, central_start

    def __len__(self) -> int:
        return self.base + len(self.data)

    def __getitem__(self, region: slice) -> bytes:
        assert self.central_start <= region.start <= region.stop <= len(self)
        return self.data[region.start - self.base : region.stop - self.base]


class OffsetBytesIO(io.BytesIO):
    """Let zipfile write genuine large offsets without a disk or memory hole."""

    base = (1 << 32) + 17

    def tell(self) -> int:
        return self.base + super().tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        local_offset = offset - self.base if whence == io.SEEK_SET else offset
        return self.base + super().seek(local_offset, whence)


class HTTP:
    def __init__(self, blob: bytes | SparseArchive) -> None:
        self.blob = blob
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.responses: list[Response] = []
        self.bad: str | None = None

    def __call__(self, request: Any, **_: Any) -> Response:
        assert request.full_url == URL
        method = request.get_method()
        headers = {key.lower(): value for key, value in request.header_items()}
        self.requests.append((method, headers))
        assert headers["accept-encoding"] == "identity"
        if method == "HEAD":
            response = Response(200, {"Content-Length": str(len(self.blob)), "ETag": ETAG}, b"")
            if self.bad == "weak_head_etag":
                response.headers["ETag"] = "W/" + ETAG
            if self.bad == "missing_head_length":
                del response.headers["Content-Length"]
        else:
            assert headers["if-match"] == ETAG
            start, end = (int(item) for item in headers["range"].removeprefix("bytes=").split("-"))
            response = Response(
                206,
                {
                    "Content-Range": f"bytes {start}-{end}/{len(self.blob)}",
                    "Content-Length": str(end - start + 1),
                    "ETag": ETAG,
                },
                self.blob[start : end + 1],
            )
            if self.bad == "ignored_range":
                response.status = 200
                response.payload = self.blob
            elif self.bad == "wrong_range":
                response.headers["Content-Range"] = f"bytes 0-{end}/{len(self.blob)}"
            elif self.bad == "wrong_length":
                response.headers["Content-Length"] = "999999"
            elif self.bad == "changed_etag":
                response.headers["ETag"] = '"changed"'
            elif self.bad == "missing_etag":
                del response.headers["ETag"]
            elif self.bad == "encoded":
                response.headers["Content-Encoding"] = "gzip"
            elif self.bad == "redirect":
                response.url += "?redirected"
            elif self.bad == "truncated":
                response.payload = response.payload[:-1]
        self.responses.append(response)
        return response


@pytest.mark.parametrize("zip64", [False, True])
def test_exact_ranges_list_a_zip_without_touching_payload(
    zip64: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blob = archive(zip64=zip64)
    http = HTTP(blob)
    monkeypatch.setattr(metadata, "urlopen", http)
    output = tmp_path / "metadata.json"
    assert metadata.main(["--url", URL, "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["metadata_only"] is True
    assert report["sequence_ids"] == ["0000"]
    assert all(isinstance(item, str) for item in report["sequence_ids"])
    assert report["etag"] == ETAG and report["size_bytes"] == len(blob)
    assert report["advertised_accept_ranges"] is None  # Actual 206 proves support.
    assert report["verified_exact_http_ranges"] is True
    with zipfile.ZipFile(io.BytesIO(blob)) as expected:
        (info,) = expected.infolist()
        central_start = expected.start_dir
    assert report["members"] == [
        {
            "name": MEMBER,
            "size_bytes": info.file_size,
            "compressed_bytes": info.compress_size,
            "compression": info.compress_type,
            "crc32": f"{info.CRC:08x}",
            "local_header_offset": info.header_offset,
            "is_directory": False,
        }
    ]
    ranges = report["provenance"]["ranges"]
    assert [r["role"] for r in ranges] == (
        ["eocd", "zip64_locator", "zip64_eocd", "central_directory"]
        if zip64
        else ["eocd", "central_directory"]
    )
    assert ranges[0]["end"] - ranges[0]["start"] + 1 == 22
    for part, response in zip(ranges, http.responses[1:], strict=True):
        assert part["start"] >= central_start  # All local headers and sensor bytes precede it.
        assert response.reads == [part["end"] - part["start"] + 1]
        assert part["sha256"] == hashlib.sha256(blob[part["start"] : part["end"] + 1]).hexdigest()
    assert http.responses[0].reads == []
    assert (
        report["provenance"]["generator_sha256"]
        == hashlib.sha256(Path(metadata.__file__).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "bad",
    [
        "ignored_range",
        "wrong_range",
        "wrong_length",
        "changed_etag",
        "missing_etag",
        "encoded",
        "redirect",
    ],
)
def test_invalid_http_response_is_rejected_before_reading_any_body(
    bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = HTTP(archive())
    http.bad = bad
    monkeypatch.setattr(metadata, "urlopen", http)
    with pytest.raises(ValueError):
        metadata.inventory(URL)
    assert len(http.requests) == 2
    assert all(response.reads == [] for response in http.responses)


@pytest.mark.parametrize("bad", ["weak_head_etag", "missing_head_length"])
def test_unpinnable_head_stops_without_a_get(bad: str, monkeypatch: pytest.MonkeyPatch) -> None:
    http = HTTP(archive())
    http.bad = bad
    monkeypatch.setattr(metadata, "urlopen", http)
    with pytest.raises(ValueError):
        metadata.inventory(URL)
    assert [request[0] for request in http.requests] == ["HEAD"]
    assert http.responses[0].reads == []


def test_truncated_range_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    http = HTTP(archive())
    http.bad = "truncated"
    monkeypatch.setattr(metadata, "urlopen", http)
    with pytest.raises(ValueError, match="truncated"):
        metadata.inventory(URL)
    assert len(http.requests) == 2


def test_commented_zip_does_not_trigger_a_tail_search(monkeypatch: pytest.MonkeyPatch) -> None:
    http = HTTP(archive(comment=b"synthetic comment"))
    monkeypatch.setattr(metadata, "urlopen", http)
    with pytest.raises(ValueError, match="broader tail"):
        metadata.inventory(URL)
    assert len(http.requests) == 2
    assert http.responses[-1].reads == [22]


def test_bad_directory_bounds_stop_before_directory_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = archive()
    fields = list(struct.unpack("<4s4H2IH", blob[-22:]))
    fields[6] -= 1
    http = HTTP(blob[:-22] + struct.pack("<4s4H2IH", *fields))
    monkeypatch.setattr(metadata, "urlopen", http)
    with pytest.raises(ValueError, match="bounds"):
        metadata.inventory(URL)
    assert len(http.requests) == 2


def test_empty_zip_needs_only_the_end_record(monkeypatch: pytest.MonkeyPatch) -> None:
    http = HTTP(archive(empty=True))
    monkeypatch.setattr(metadata, "urlopen", http)
    report = metadata.inventory(URL)
    assert report["members"] == report["sequence_ids"] == []
    assert len(http.requests) == 2


def test_real_zip64_offsets_above_four_gibibytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with OffsetBytesIO() as stream:
        with zipfile.ZipFile(stream, "w") as output:
            output.writestr(MEMBER, b"synthetic payload")
        data = stream.getvalue()
        with zipfile.ZipFile(stream) as reference:
            central_start = reference.start_dir
            (info,) = reference.infolist()
    assert len(data) < 4096
    http = HTTP(SparseArchive(data, OffsetBytesIO.base, central_start))
    monkeypatch.setattr(metadata, "urlopen", http)
    report = metadata.inventory(URL)
    assert report["size_bytes"] == OffsetBytesIO.base + len(data) > (1 << 32)
    assert report["members"][0]["local_header_offset"] == info.header_offset > (1 << 32)
    assert len(http.requests) == 5


def test_displaced_zip64_locator_is_rejected_before_its_target_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = bytearray(archive(zip64=True))
    struct.pack_into("<Q", blob, len(blob) - 22 - 20 + 8, 0)  # Points at the local header.
    http = HTTP(bytes(blob))
    monkeypatch.setattr(metadata, "urlopen", http)
    with pytest.raises(ValueError, match="displaced"):
        metadata.inventory(URL)
    assert [response.reads for response in http.responses] == [[], [22], [20]]


def test_oversized_directory_stops_before_its_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    http = HTTP(archive())
    monkeypatch.setattr(metadata, "urlopen", http)
    monkeypatch.setattr(metadata, "MAX_DIRECTORY_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        metadata.inventory(URL)
    assert len(http.requests) == 2


def test_directory_entry_count_mismatch_is_not_reported_as_an_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = archive()
    fields = list(struct.unpack("<4s4H2IH", blob[-22:]))
    fields[3:5] = [2, 2]
    http = HTTP(blob[:-22] + struct.pack("<4s4H2IH", *fields))
    monkeypatch.setattr(metadata, "urlopen", http)
    with pytest.raises(ValueError, match="entry count mismatch"):
        metadata.inventory(URL)
    assert len(http.requests) == 3


def test_output_and_url_guards_precede_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = HTTP(archive())
    monkeypatch.setattr(metadata, "urlopen", http)
    output = tmp_path / "previous.json"
    output.write_text("previous inventory")
    with pytest.raises(SystemExit):
        metadata.main(["--url", URL, "--output", str(output)])
    assert output.read_text() == "previous inventory"
    with pytest.raises(ValueError, match="official"):
        metadata.inventory("https://example.com/unverified.zip")
    assert http.requests == []
