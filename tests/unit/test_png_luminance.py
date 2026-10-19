from __future__ import annotations

import builtins
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from calibrex.data import kitti
from calibrex.data.pillow_png_adapter import _decode_luminance_array, _decode_luminance_rows


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def _png(
    width: int,
    height: int,
    channels: int,
    compressed: bytes,
    *,
    bit_depth: int = 8,
    interlace: int = 0,
    ancillary: bytes = b"",
) -> bytes:
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    header = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, interlace)
    middle = len(compressed) // 2
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + ancillary
        + _chunk(b"IDAT", compressed[:middle])
        + _chunk(b"IDAT", compressed[middle:])
        + _chunk(b"IEND", b"")
    )


def _encoded_rows(pixels: np.ndarray[Any, Any], filter_type: int) -> bytes:
    height, width, channels = pixels.shape
    previous = [0] * (width * channels)
    raw = bytearray()
    for row in pixels.reshape(height, -1).tolist():
        raw.append(filter_type)
        for index, value in enumerate(row):
            left = row[index - channels] if index >= channels else 0
            up = previous[index]
            corner = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            else:
                estimate = left + up - corner
                distances = (abs(estimate - left), abs(estimate - up), abs(estimate - corner))
                predictor = (left, up, corner)[distances.index(min(distances))]
            raw.append((value - predictor) & 255)
        previous = row
    return bytes(raw)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("filter_type", [0, 1, 2, 3, 4])
def test_compiled_png_matches_native_for_every_color_and_filter(
    channels: int,
    filter_type: int,
) -> None:
    pytest.importorskip("PIL.Image")
    pixels = np.random.default_rng(8100 + channels).integers(
        0,
        256,
        size=(11, 17, channels),
        dtype=np.uint8,
    )
    if channels in (2, 4):
        pixels[0, 0, -1] = 0
    raw = _encoded_rows(pixels, filter_type)
    compressed = zlib.compress(raw)
    native = kitti._decode_png_luminance(_png(17, 11, channels, compressed))
    assert native is not None
    assert _decode_luminance_rows(compressed, raw, 17, 11, channels) == native.rows
    array = _decode_luminance_array(compressed, raw, 17, 11, channels)
    assert array is not None
    assert array.dtype == np.float64
    assert array.flags.c_contiguous
    np.testing.assert_array_equal(array.view(np.uint64), np.asarray(native.rows).view(np.uint64))
    # Include invisible alpha and byte overflow: no integer luminance rounding.
    first = pixels[0, 0].tolist()
    expected = (
        float(first[0]) if channels <= 2 else 0.299 * first[0] + 0.587 * first[1] + 0.114 * first[2]
    )
    assert native.rows[0][0] == expected


@pytest.fixture
def large_png() -> tuple[bytes, kitti.LuminanceImage]:
    width, height, channels = 256, 256, 4
    pixels = np.random.default_rng(991).integers(
        0,
        256,
        size=(height, width, channels),
        dtype=np.uint8,
    )
    raw = _encoded_rows(pixels, 4)
    # Ancillary gamma/ICC metadata and original CRC handling cannot affect pixels.
    ancillary = _chunk(b"gAMA", struct.pack(">I", 45455))
    png = _png(width, height, channels, zlib.compress(raw), ancillary=ancillary)
    native = kitti._unfilter_png_luminance(raw, width, height, channels)
    assert native is not None
    return png, native


def test_large_png_uses_optional_decoder_and_preserves_rows(
    large_png: tuple[bytes, kitti.LuminanceImage],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL.Image")
    from calibrex.data import pillow_png_adapter

    png, expected = large_png
    calls = []
    original = pillow_png_adapter._decode_luminance_rows

    def traced(*args: Any) -> tuple[tuple[float, ...], ...] | None:
        result = original(*args)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(pillow_png_adapter, "_decode_luminance_rows", traced)
    path = tmp_path / "camera.png"
    path.write_bytes(png)
    assert kitti.read_png_luminance(path) == expected
    assert calls == [True]


def test_missing_optional_decoder_preserves_native_fallback(
    large_png: tuple[bytes, kitti.LuminanceImage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builtins.__import__

    def without_pillow(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError("optional image package is absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pillow)
    png, expected = large_png
    assert kitti._decode_png_luminance(png) == expected


def test_decoder_failure_falls_back_without_changing_global_policy(
    large_png: tuple[bytes, kitti.LuminanceImage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = pytest.importorskip("PIL.Image")
    png, expected = large_png
    limit = image.MAX_IMAGE_PIXELS

    def failed(*args: Any, **kwargs: Any) -> Any:
        raise OSError("optional compiled decoder unavailable")

    monkeypatch.setattr(image, "open", failed)
    assert kitti._decode_png_luminance(png) == expected
    assert limit == image.MAX_IMAGE_PIXELS
    monkeypatch.setattr(image, "MAX_IMAGE_PIXELS", 1)
    assert kitti._decode_png_luminance(png) == expected
    assert image.MAX_IMAGE_PIXELS == 1


@pytest.mark.parametrize("bit_depth,interlace", [(16, 0), (8, 1)])
def test_optional_decoder_does_not_expand_supported_encodings(
    bit_depth: int,
    interlace: int,
) -> None:
    raw = b"\0\xff\0\x80\0" * 256 * 256
    png = _png(256, 256, 4, zlib.compress(raw), bit_depth=bit_depth, interlace=interlace)
    assert kitti._decode_png_luminance(png) is None


@pytest.mark.parametrize("invalid", ["filter", "short", "zlib"])
def test_malformed_large_png_stays_unreadable(invalid: str) -> None:
    row = bytes(256 * 3)
    raw = b"\0" + row
    raw *= 256
    if invalid == "filter":
        raw = b"\5" + raw[1:]
    elif invalid == "short":
        raw = raw[:-1]
    compressed = b"invalid" if invalid == "zlib" else zlib.compress(raw)
    assert kitti._decode_png_luminance(_png(256, 256, 3, compressed)) is None


def test_small_png_and_module_import_do_not_load_pillow() -> None:
    png = _png(1, 1, 1, zlib.compress(b"\0\xff"))
    script = f"""
import sys
from calibrex.data import kitti
assert not any(n == 'PIL' or n.startswith('PIL.') for n in sys.modules)
image = kitti._decode_png_luminance(bytes.fromhex('{png.hex()}'))
assert image.rows == ((255.0,),)
assert 'calibrex.data.pillow_png_adapter' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_original_crc_and_trailing_stream_acceptance_stay_native() -> None:
    pytest.importorskip("PIL.Image")
    raw = b"\0\xff\x10\x80" * 4 + b"ignored"
    # Extra decompressed data and bytes after the zlib stream are ignored by
    # the native decoder. Original CRCs are also deliberately not validated.
    compressed = zlib.compress(raw) + b"trailing"
    png = bytearray(_png(1, 4, 3, compressed))
    png[29:33] = b"\0\0\0\0"
    native = kitti._unfilter_png_luminance(raw, 1, 4, 3)
    assert native is not None
    assert _decode_luminance_rows(compressed, raw, 1, 4, 3) == native.rows


@pytest.mark.parametrize("compiled", [False, True])
def test_array_reader_preserves_pixels_and_caller_ownership(
    tmp_path: Path,
    large_png: tuple[bytes, kitti.LuminanceImage],
    compiled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrex.data import read_png_luminance_array

    png, expected = large_png
    if not compiled:
        monkeypatch.setattr(
            "calibrex.data.pillow_png_adapter._decode_luminance_array", lambda *_args: None
        )
    path = tmp_path / "camera.png"
    path.write_bytes(png)
    actual = read_png_luminance_array(path)
    assert actual is not None
    assert actual.shape == (expected.height, expected.width)
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous and actual.flags.writeable
    native = np.asarray(expected.rows)
    np.testing.assert_array_equal(actual.view(np.uint64), native.view(np.uint64))
    actual[:] = -1.0
    second = read_png_luminance_array(path)
    assert second is not None
    np.testing.assert_array_equal(second.view(np.uint64), native.view(np.uint64))
    assert kitti.read_png_luminance(path) == expected


@pytest.mark.parametrize("invalid", ["missing", "header", "filter", "short", "zlib", "depth"])
def test_array_reader_has_the_same_rejection_boundary(tmp_path: Path, invalid: str) -> None:
    path = tmp_path / "invalid.png"
    if invalid != "missing":
        raw = (b"\0" + bytes(256 * 3)) * 256
        if invalid == "filter":
            raw = b"\5" + raw[1:]
        elif invalid == "short":
            raw = raw[:-1]
        compressed = b"invalid" if invalid == "zlib" else zlib.compress(raw)
        payload = _png(256, 256, 3, compressed, bit_depth=16 if invalid == "depth" else 8)
        path.write_bytes(b"invalid" if invalid == "header" else payload)
    assert kitti.read_png_luminance(path) is None
    assert kitti.read_png_luminance_array(path) is None
