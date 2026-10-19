"""Parse remote ZIP metadata without materializing the archive's address space."""

from __future__ import annotations

import io
import struct
import zipfile
from collections.abc import Callable, Sequence


class _MetadataView(io.RawIOBase):
    """Seek in original archive coordinates, reading only fetched metadata."""

    def __init__(self, size: int, sections: Sequence[tuple[int, bytes]]) -> None:
        super().__init__()
        self._size = size
        self._sections = sorted(sections)
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError("unknown seek origin")
        if position < 0:
            raise OSError("negative metadata seek")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        available = max(0, self._size - self._position)
        remaining = available if size < 0 else min(size, available)
        cursor = self._position
        chunks: list[bytes] = []
        while remaining:
            for start, data in self._sections:
                if start <= cursor < start + len(data):
                    count = min(remaining, start + len(data) - cursor)
                    chunks.append(data[cursor - start : cursor - start + count])
                    cursor += count
                    remaining -= count
                    break
            else:
                raise OSError("ZIP parser requested archive bytes that were not fetched")
        self._position = cursor
        return b"".join(chunks)


def read_central_directory(
    size: int, read_range: Callable[[int, int], bytes]
) -> list[zipfile.ZipInfo]:
    """Read directory entries and retain original local-header offsets.

    The reader must return the requested inclusive range from one pinned object.
    Only the central directory, tail and optional ZIP64 record are retained.
    No zero-filled or sparse temporary file is created, even on Windows.
    """
    tail_length = min(size, (1 << 16) + 22 + 20)
    tail_start = size - tail_length
    tail = read_range(tail_start, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    while eocd >= 0:
        if eocd + 22 <= len(tail):
            comment_size = struct.unpack_from("<H", tail, eocd + 20)[0]
            if eocd + 22 + comment_size == len(tail):
                break
        eocd = tail.rfind(b"PK\x05\x06", 0, eocd)
    if eocd < 0:
        raise ValueError("no end of central directory")
    directory_size, directory_offset = struct.unpack_from("<II", tail, eocd + 12)
    # A metadata-only view omits the archive comment. Original byte offsets stay
    # valid, and a signature inside a comment cannot confuse zipfile's scan.
    metadata_tail = tail[: eocd + 20] + b"\x00\x00"
    sections = [(tail_start, metadata_tail)]
    locator = eocd - 20
    has_locator = locator >= 0 and tail[locator : locator + 4] == b"PK\x06\x07"
    entries = struct.unpack_from("<2H", tail, eocd + 8)
    if (directory_offset == 0xFFFFFFFF or directory_size == 0xFFFFFFFF
            or 0xFFFF in entries) and not has_locator:
        raise ValueError("no zip64 end of central directory locator")
    if has_locator:
        record_offset = struct.unpack_from("<Q", tail, locator + 8)[0]
        record = read_range(record_offset, record_offset + 55)
        if len(record) != 56 or record[:4] != b"PK\x06\x06":
            raise ValueError("invalid zip64 end of central directory record")
        directory_size, directory_offset = struct.unpack_from("<QQ", record, 40)
        sections.append((record_offset, record))
    if directory_offset + directory_size > tail_start + eocd:
        raise ValueError("central directory extends beyond its end record")
    if directory_size:
        directory = read_range(directory_offset, directory_offset + directory_size - 1)
        sections.append((directory_offset, directory))
    with (
        _MetadataView(tail_start + eocd + 22, sections) as view,
        zipfile.ZipFile(view) as archive,
    ):
        return sorted(archive.infolist(), key=lambda info: info.header_offset)
