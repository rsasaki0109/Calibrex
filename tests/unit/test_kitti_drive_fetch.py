"""The KITTI drive-head fetcher: member selection, the pinned ETag, CRC checks and zip64."""

import struct
import zipfile
from pathlib import Path

import pytest
from tests.unit.test_score_hilti_camera_camera import _load

fetcher = _load("kitti_drive_fetch")
DATE, DRIVE = "2011_09_30", "0042"
PREFIX = f"{DATE}/{DATE}_drive_{DRIVE}_sync/"
STREAMS = {"image_00": "png", "image_02": "png", "oxts": "txt", "velodyne_points": "bin"}


def _archive(path: Path, frames: int) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for stream, extension in STREAMS.items():
            for index in range(frames):
                # alternate stored and deflated members, as both occur in KITTI's archives
                archive.writestr(
                    f"{PREFIX}{stream}/data/{index:010d}.{extension}",
                    f"{stream} frame {index};".encode() * 40,
                    compress_type=zipfile.ZIP_DEFLATED if index % 2 else zipfile.ZIP_STORED,
                )
        lines = "".join(f"2011-09-30 12:00:{index:02d}.000\n" for index in range(frames))
        for name in fetcher.TIMESTAMP_FILES:
            archive.writestr(PREFIX + name, lines)
        archive.writestr(PREFIX + "image_00/timestamps.txt", lines)
        archive.writestr(PREFIX + "oxts/dataformat.txt", "lat: latitude\n")


class _Bucket:
    """One S3 object served the way curl reads it: HEAD, and ranges under If-Match."""

    def __init__(self, data: bytes, etag: str = "abc123-7") -> None:
        self.data = data
        self.etag = etag
        self.ranges: list[tuple[int, int]] = []
        self.if_match: list[str] = []

    def curl(self, args: list[str]) -> bytes:
        if args[0] == "-I":
            text = f'HTTP/1.1 200 OK\r\nContent-Length: {len(self.data)}\r\nETag: "{self.etag}"\r\n'
            return text.encode()
        assert args[0] == "-H" and args[2] == "-r"
        self.if_match.append(args[1])
        if args[1] != f'If-Match: "{self.etag}"':
            raise OSError("curl exit 22: The requested URL returned error: 412")
        start, end = (int(value) for value in args[3].split("-"))
        self.ranges.append((start, end))
        return self.data[start : end + 1]


def _serve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frames: int = 8) -> _Bucket:
    _archive(tmp_path / "drive.zip", frames)
    bucket = _Bucket((tmp_path / "drive.zip").read_bytes())
    monkeypatch.setattr(fetcher, "_curl", bucket.curl)
    return bucket


def test_the_head_of_a_drive_is_fetched_member_by_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bucket = _serve(monkeypatch, tmp_path)
    out = tmp_path / "out"
    record = fetcher.fetch_drive(DATE, DRIVE, 5, out, (len(bucket.data), bucket.etag))
    assert record["frames"] == {"image_02": 5, "velodyne_points": 5, "oxts": 5}
    assert record["archive_frames"] == {"image_02": 8, "velodyne_points": 8, "oxts": 8}
    assert record["crc_verified"] is True
    assert record["member_count"] == len(record["members"]) == 15 + 6
    assert record["members_sha256"] == fetcher.members_sha256(record["members"][::-1])
    assert (record["size_bytes"], record["etag"]) == (len(bucket.data), bucket.etag)
    # every range came from the object the HEAD request saw
    assert set(bucket.if_match) == {f'If-Match: "{bucket.etag}"'}
    assert sum(hi - lo + 1 for lo, hi in record["ranges"]) == record["fetched_bytes"]
    with zipfile.ZipFile(tmp_path / "drive.zip") as archive:
        frame = f"{PREFIX}velodyne_points/data/{3:010d}.bin"
        assert (out / frame).read_bytes() == archive.read(frame)
    assert not (out / f"{PREFIX}image_02/data/{5:010d}.png").exists()
    assert not (out / f"{PREFIX}image_00").exists()  # streams camera-lidar does not read
    stamps = (out / PREFIX / "velodyne_points/timestamps_start.txt").read_text().splitlines()
    assert stamps == [f"2011-09-30 12:00:{index:02d}.000" for index in range(5)]
    assert (out / PREFIX / "oxts/dataformat.txt").read_text() == "lat: latitude\n"


def test_a_drive_shorter_than_the_head_is_fetched_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _serve(monkeypatch, tmp_path, frames=3)
    record = fetcher.fetch_drive(DATE, DRIVE, 600, tmp_path / "out")
    assert (
        record["frames"]
        == record["archive_frames"]
        == {
            "image_02": 3,
            "velodyne_points": 3,
            "oxts": 3,
        }
    )


def test_an_archive_that_differs_from_its_pin_is_not_fetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bucket = _serve(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="differ from the pin"):
        fetcher.fetch_drive(DATE, DRIVE, 5, tmp_path / "out", (len(bucket.data), "other-7"))
    with pytest.raises(ValueError, match="differ from the pin"):
        fetcher.fetch_drive(DATE, DRIVE, 5, tmp_path / "out", (len(bucket.data) + 1, bucket.etag))
    assert bucket.ranges == []


def test_an_archive_replaced_after_the_head_request_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bucket = _serve(monkeypatch, tmp_path)
    head = fetcher.head

    def head_then_replace(url: str) -> tuple[int, str]:
        seen = head(url)
        bucket.etag = "new-7"  # the object is replaced right after the HEAD request
        return seen

    monkeypatch.setattr(fetcher, "head", head_then_replace)
    with pytest.raises(OSError, match="412"):
        fetcher.fetch_drive(DATE, DRIVE, 5, tmp_path / "out")


def test_a_corrupt_member_fails_its_crc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bucket = _serve(monkeypatch, tmp_path)
    with zipfile.ZipFile(tmp_path / "drive.zip") as archive:
        info = archive.getinfo(f"{PREFIX}image_02/data/{2:010d}.png")  # a stored member
    data = bytearray(bucket.data)
    data[info.header_offset + 30 + len(info.filename) + 10] ^= 0xFF
    bucket.data = bytes(data)
    with pytest.raises(ValueError, match="CRC"):
        fetcher.fetch_drive(DATE, DRIVE, 5, tmp_path / "out")


def test_a_zip64_archive_is_read_through_its_zip64_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # KITTI archives over 4 GiB keep the central directory's offset in the zip64 records;
    # a small zip64 limit makes zipfile write them, and the 32-bit field is set as such a
    # writer leaves it (0xFFFFFFFF)
    with monkeypatch.context() as patch:
        patch.setattr(zipfile, "ZIP64_LIMIT", 1 << 10)
        _archive(tmp_path / "drive.zip", 8)
    data = bytearray((tmp_path / "drive.zip").read_bytes())
    eocd = data.rfind(b"PK\x05\x06")
    assert data.rfind(b"PK\x06\x07", 0, eocd) > 0
    struct.pack_into("<I", data, eocd + 16, 0xFFFFFFFF)
    bucket = _Bucket(bytes(data))
    monkeypatch.setattr(fetcher, "_curl", bucket.curl)
    infos = fetcher.central_directory("url", len(bucket.data), bucket.etag)
    assert max(info.header_offset for info in infos) > 1 << 10
    record = fetcher.fetch_drive(DATE, DRIVE, 5, tmp_path / "out")
    assert record["frames"] == {"image_02": 5, "velodyne_points": 5, "oxts": 5}
