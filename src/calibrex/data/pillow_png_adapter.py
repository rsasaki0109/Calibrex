"""Optional compiled PNG unfiltering with native floating-point luminance.

The caller retains its existing header and zlib validation. A canonical PNG
containing only that header and the selected IDAT stream avoids differences in
ancillary-chunk handling, CRC acceptance, or animation between image readers.
Pillow is imported only when a sufficiently large PNG is actually decoded.
"""

from __future__ import annotations

import struct
import zlib
from io import BytesIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def _decode_luminance_rows(
    compressed: bytes,
    raw: bytes,
    width: int,
    height: int,
    channels: int,
) -> tuple[tuple[float, ...], ...] | None:
    """Decode a validated 8-bit stream, or request the native fallback."""
    luminance = _decode_luminance_array(compressed, raw, width, height, channels)
    if luminance is None:
        return None
    return tuple(tuple(row) for row in luminance.tolist())


def _decode_luminance_array(
    compressed: bytes,
    raw: bytes,
    width: int,
    height: int,
    channels: int,
) -> NDArray[np.float64] | None:
    """Keep decoded luminance in a contiguous array without Python pixel objects."""
    row_length = width * channels
    expected = height * (row_length + 1)
    if (
        len(raw) < expected
        or len(compressed) >= 2**32
        or any(raw[index] > 4 for index in range(0, expected, row_length + 1))
    ):
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    # Leave Pillow's process-wide limits and warning policies untouched.
    if Image.MAX_IMAGE_PIXELS is not None and width * height > Image.MAX_IMAGE_PIXELS:
        return None
    import numpy as np

    modes = {1: ("L", 0), 2: ("LA", 4), 3: ("RGB", 2), 4: ("RGBA", 6)}
    mode, color_type = modes[channels]
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    canonical = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    try:
        with Image.open(BytesIO(canonical), formats=("PNG",)) as image:
            if image.mode != mode or image.size != (width, height):
                return None
            pixels = np.asarray(image, dtype=np.float64)
    except (OSError, ValueError, OverflowError):
        return None
    if channels == 1:
        luminance = pixels
    elif channels == 2:
        luminance = pixels[:, :, 0]
    else:
        # Match the native three multiplications and two additions, without an
        # 8-bit conversion, alpha compositing, gamma adjustment, or fused FMA.
        luminance = (
            (0.299 * pixels[:, :, 0]) + (0.587 * pixels[:, :, 1]) + (0.114 * pixels[:, :, 2])
        )
    return np.ascontiguousarray(luminance)
