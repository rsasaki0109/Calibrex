from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from calibrex.core import provenance


@pytest.mark.parametrize("size", [0, 1, 16, 17, 18, 67, 68, 134, 139, 4097])
def test_buffered_hash_preserves_full_bytes_and_partial_tail(tmp_path, monkeypatch, size):
    monkeypatch.setattr(provenance, "_LARGE_FILE_THRESHOLD_BYTES", 17)
    monkeypatch.setattr(provenance, "_LARGE_FILE_BUFFER_BYTES", 67)
    data = bytes(index % 251 for index in range(size))
    path = tmp_path / "input.bin"
    path.write_bytes(data)
    assert provenance.sha256_path(path) == hashlib.sha256(data).hexdigest()


def test_directory_manifest_order_and_file_content_remain_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_LARGE_FILE_THRESHOLD_BYTES", 17)
    monkeypatch.setattr(provenance, "_LARGE_FILE_BUFFER_BYTES", 67)
    files = {"z.bin": b"large\0data"*23, "nested/a.bin": b"short", "empty.bin": b""}
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
    expected = hashlib.sha256()
    for path in sorted(tmp_path.rglob("*")):
        if path.is_file():
            expected.update(str(path.relative_to(tmp_path)).encode("utf-8"))
            expected.update(path.read_bytes())
    assert provenance.sha256_path(tmp_path) == expected.hexdigest()
    assert provenance.sha256_path(tmp_path / "missing") is None


def test_actual_large_file_crosses_buffer_boundary(tmp_path: Path):
    path = tmp_path / "large.bin"
    block = bytes(range(256))*4096
    expected = hashlib.sha256()
    with path.open("wb") as stream:
        for _ in range(64):
            stream.write(block)
            expected.update(block)
        stream.write(b"partial tail")
        expected.update(b"partial tail")
    assert provenance.sha256_path(path) == expected.hexdigest()
