"""Remote ZIP64 metadata must not allocate the archive's multi-gigabyte holes."""

from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from tools import kitti_drive_fetch, kitti_drive_metadata
from tools.kitti_zip_metadata import read_central_directory


def _large_archive(comment: bytes) -> tuple[int, dict[int, bytes], list[str]]:
    names = [
        "2011_09_30_drive_0018_sync/image_02/data/0000000015.png",
        "2011_09_30_drive_0018_sync/velodyne_points/data/0000000015.bin",
    ]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name in names:
            archive.writestr(name, b"fixture")
    original = stream.getvalue()
    end = original.rfind(b"PK\x05\x06")
    length, offset = struct.unpack_from("<II", original, end + 12)
    directory = original[offset : offset + length]
    entries, cursor = [], 0
    for index in range(2):
        name_size, extra_size, comment_size = struct.unpack_from("<3H", directory, cursor + 28)
        entry_size = 46 + name_size + extra_size + comment_size
        header = bytearray(directory[cursor : cursor + 46])
        struct.pack_into("<I", header, 42, 0xFFFFFFFF)
        struct.pack_into("<H", header, 30, extra_size + 12)
        name = directory[cursor + 46 : cursor + 46 + name_size]
        rest = directory[cursor + 46 + name_size : cursor + entry_size]
        extra = struct.pack("<HHQ", 1, 8, (8 << 30) + 100 * index)
        entries.append(bytes(header) + name + extra + rest)
        cursor += entry_size
    directory = b"".join(entries)
    directory_offset = 12 << 30
    record_offset = directory_offset + len(directory)
    record = struct.pack(
        "<4sQ2H2I4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 2, 2,
        len(directory), directory_offset,
    )
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, record_offset, 1)
    end_record = struct.pack(
        "<4s4H2IH", b"PK\x05\x06", 0, 0, 0xFFFF, 0xFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, len(comment),
    )
    footer = record + locator + end_record + comment
    return record_offset + len(footer), {directory_offset: directory, record_offset: footer}, names


@pytest.mark.parametrize(
    "comment", [b"", b"x" * 65535, b"comment with PK\x05\x06 inside"],
    ids=["empty", "maximum-comment", "embedded-signature"],
)
@pytest.mark.parametrize("consumer", ["helper", "fetcher", "metadata"])
def test_large_zip64_metadata_preserves_offsets_without_file_writes(
    monkeypatch: pytest.MonkeyPatch, comment: bytes, consumer: str,
) -> None:
    size, sections, names = _large_archive(comment)
    requests = []

    def read_range(start: int, end: int) -> bytes:
        assert 0 <= start <= end < size
        assert end - start + 1 <= (1 << 16) + 42
        requests.append((start, end))
        data = bytearray(end - start + 1)
        for offset, content in sections.items():
            low, high = max(start, offset), min(end + 1, offset + len(content))
            if low < high:
                data[low - start : high - start] = content[low - offset : high - offset]
        return bytes(data)

    def no_file(*args, **kwargs):
        pytest.fail("metadata parsing must not create or open filesystem payloads")

    monkeypatch.setattr(Path, "open", no_file)
    if consumer == "helper":
        infos = read_central_directory(size, read_range)
    elif consumer == "fetcher":
        monkeypatch.setattr(
            kitti_drive_fetch, "fetch_range", lambda url, lo, hi, etag: read_range(lo, hi)
        )
        infos = kitti_drive_fetch.central_directory("public-url", size, "pinned-etag")
    else:
        monkeypatch.setattr(kitti_drive_metadata, "head", lambda url: (size, "pinned-etag"))
        monkeypatch.setattr(kitti_drive_metadata, "fetch", lambda url, lo, hi: read_range(lo, hi))
        result = kitti_drive_metadata.metadata("0018", "2011_09_30")
        assert result == {
            "size_bytes": size, "etag": "pinned-etag", "entries": 2,
            "frames": {"image_00": 0, "image_01": 0, "image_02": 1,
                       "image_03": 0, "velodyne_points": 1, "oxts": 0},
        }
        infos = read_central_directory(size, read_range)
    assert [info.filename for info in infos] == names
    assert [info.header_offset for info in infos] == [(8 << 30), (8 << 30) + 100]
    assert all(info.file_size == 7 for info in infos)
    assert sum(high - low + 1 for low, high in requests) < 140000


def test_empty_zip_and_invalid_directory() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w"):
        pass
    raw = stream.getvalue()
    assert read_central_directory(len(raw), lambda lo, hi: raw[lo : hi + 1]) == []
    with pytest.raises(ValueError, match="no end"):
        read_central_directory(80, lambda lo, hi: b"x" * (hi - lo + 1))


def test_zip64_record_is_used_even_when_directory_size_and_offset_fit_uint32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.BytesIO()
    with monkeypatch.context() as patch:
        patch.setattr(zipfile, "ZIP64_LIMIT", 1024)
        with zipfile.ZipFile(stream, "w") as archive:
            archive.comment = b"x" * 65535
            archive.writestr("large-member.bin", b"x" * 2048)
    raw = stream.getvalue()
    end = len(raw) - 65535 - 22
    assert struct.unpack_from("<II", raw, end + 12)[1] < 0xFFFFFFFF
    infos = read_central_directory(len(raw), lambda lo, hi: raw[lo : hi + 1])
    assert [(info.filename, info.header_offset, info.file_size) for info in infos] == [
        ("large-member.bin", 0, 2048)
    ]


@pytest.mark.parametrize("tool", ["kitti_drive_fetch.py", "kitti_drive_metadata.py"])
def test_tools_still_import_when_invoked_outside_the_checkout(tool: str, tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "tools" / tool
    subprocess.run(
        [sys.executable, "-c", "import runpy,sys; runpy.run_path(sys.argv[1])", str(script)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
