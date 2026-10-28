"""Shared compression primitives for bag readers; optional codecs load lazily."""


def decompress_lz4_frames(data: bytes) -> bytes:
    """Decode every LZ4 frame, leaving codec error context to the caller."""
    import lz4.frame

    parts: list[bytes] = []
    remaining = memoryview(data)
    while True:
        decoded, consumed = lz4.frame.decompress(remaining, return_bytes_read=True)
        if consumed <= 0 or consumed > len(remaining):
            raise ValueError("invalid LZ4 frame byte count")
        parts.append(decoded)
        remaining = remaining[consumed:]
        if not remaining:
            return b"".join(parts)
