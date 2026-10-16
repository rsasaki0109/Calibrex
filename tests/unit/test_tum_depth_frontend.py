"""Exact integer PNG reconstruction and unchanged depth samples."""

import builtins
import struct
import zlib

import numpy as np
import pytest

from calibrex.data import tum_rgbd as tum


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode(pixels, filters):
    rows = pixels.astype(">u2").view(np.uint8).reshape(len(pixels), -1).tolist()
    previous = [0] * len(rows[0])
    raw = bytearray()
    for row, kind in zip(rows, filters, strict=True):
        raw.append(kind)
        for index, value in enumerate(row):
            left = row[index - 2] if index >= 2 else 0
            above = previous[index]
            corner = previous[index - 2] if index >= 2 else 0
            if kind == 0:
                predictor = 0
            elif kind == 1:
                predictor = left
            elif kind == 2:
                predictor = above
            elif kind == 3:
                predictor = (left + above) // 2
            else:
                estimate = left + above - corner
                distances = (abs(estimate - left), abs(estimate - above), abs(estimate - corner))
                predictor = (left, above, corner)[distances.index(min(distances))]
            raw.append((value - predictor) & 255)
        previous = row
    return bytes(raw)


def png(raw, width, height, *, header=None):
    compressed = zlib.compress(raw)
    middle = len(compressed) // 2
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header or struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0))
        + chunk(b"gAMA", struct.pack(">I", 45455))
        + chunk(b"IDAT", compressed[:middle])
        + chunk(b"IDAT", compressed[middle:])
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize("shape", [(1, 1), (7, 3), (256, 129), (480, 640)])
@pytest.mark.parametrize(
    "pattern", [(0,), (1,), (2,), (3,), (4,), (0, 1, 2, 2, 1, 0, 2), (0, 4, 1, 3, 2)]
)
def test_all_png_filters_preserve_every_16bit_pixel_and_sample(shape, pattern, tmp_path):
    height, width = shape
    pixels = np.random.default_rng(711).integers(0, 65536, size=shape, dtype=np.uint16)
    pixels.flat[: min(pixels.size, 6)] = [0, 65535, 255, 65280, 256, 65534][: min(pixels.size, 6)]
    filters = [pattern[row % len(pattern)] for row in range(height)]
    raw = encode(pixels, filters)
    path = tmp_path / "depth.png"
    path.write_bytes(png(raw, width, height))
    decoded = tum.read_tum_depth_png(path)
    expected = tum.TUMDepthImage(width, height, tuple(pixels.reshape(-1).tolist()))
    assert decoded == expected
    assert all(type(value) is int for value in decoded.raw_values)
    assert tum.sample_tum_depth_points(decoded, max_points=73) == tum.sample_tum_depth_points(
        expected, max_points=73
    )


def test_replaced_scanline_function_retains_one_call_per_row(tmp_path, monkeypatch):
    pixels = np.full((256, 129), 65535, dtype=np.uint16)
    path = tmp_path / "depth.png"
    path.write_bytes(png(encode(pixels, [1] * 256), 129, 256))
    calls = []
    original = tum._unfilter_png_scanline

    def unfilter(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(tum, "_unfilter_png_scanline", unfilter)
    assert tum.read_tum_depth_png(path).raw_values == (65535,) * pixels.size
    assert calls == [1] * 256


def test_small_depth_decode_keeps_numpy_lazy(tmp_path, monkeypatch):
    pixels = np.asarray([[0, 5000], [10000, 65535]], dtype=np.uint16)
    path = tmp_path / "small.png"
    path.write_bytes(png(encode(pixels, [1, 2]), 2, 2))
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name != "numpy", "small depth images do not need an array import"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert tum.read_tum_depth_png(path).raw_values == (0, 5000, 10000, 65535)


@pytest.mark.parametrize("position,value", [(2, 8), (3, 2), (4, 1), (5, 1), (6, 1)])
def test_unsupported_depth_png_headers_keep_original_error(position, value, tmp_path):
    fields = [129, 256, 16, 0, 0, 0, 0]
    fields[position] = value
    path = tmp_path / "bad.png"
    path.write_bytes(png(bytes(256 * 259), 129, 256, header=struct.pack(">IIBBBBB", *fields)))
    with pytest.raises(ValueError, match="non-interlaced 16-bit grayscale"):
        tum.read_tum_depth_png(path)


@pytest.mark.parametrize("kind", ["signature", "missing", "size", "filter"])
def test_invalid_depth_streams_are_still_rejected(kind, tmp_path):
    path = tmp_path / "bad.png"
    if kind == "signature":
        data = b"not a PNG"
        message = "not a PNG"
    elif kind == "missing":
        data = b"\x89PNG\r\n\x1a\n" + chunk(b"IEND", b"")
        message = "missing IHDR or IDAT"
    elif kind == "size":
        data = png(bytes(256 * 259 - 1), 129, 256)
        message = "decompressed size"
    else:
        raw = bytearray(256 * 259)
        raw[-259] = 5
        data = png(raw, 129, 256)
        message = "unsupported PNG filter type 5"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        tum.read_tum_depth_png(path)
