"""Generate the deterministic KITTI-shaped Camera-LiDAR demo fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from collections.abc import Sequence
from pathlib import Path

_SEQUENCE = Path("2011_09_26/2011_09_26_drive_0005_sync")
_TIMESTAMPS = "2011-09-26 13:00:00.000000000\n2011-09-26 13:00:01.000000000\n"


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _edge_png(
    *,
    width: int,
    height: int,
    feature_pixels: set[tuple[int, int]],
) -> bytes:
    bright_pixels = feature_pixels
    raw_rows = b"".join(
        b"\x00"
        + b"".join(
            b"\xff\xff\xff" if (x, y) in bright_pixels else b"\x00\x00\x00" for x in range(width)
        )
        for y in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(raw_rows))
        + _png_chunk(b"IEND", b"")
    )


def _velodyne_payload() -> bytes:
    points = [
        *((float(x), float(y), 5.0, 1.0) for x in range(3) for y in range(3)),
        *((0.0, float(y), 8.0, 1.0) for y in range(3)),
    ]
    return b"".join(struct.pack("<ffff", *point) for point in points)


def _oxts_packet(*, yaw_rad: float, vn: float, ve: float) -> str:
    fields = [
        49.0,
        8.0,
        110.0,
        0.0,
        0.0,
        yaw_rad,
        vn,
        ve,
        0.0,
        0.0,
        0.0,
        0.2,
        0.0,
        0.0,
        0.0,
    ]
    return " ".join(str(value) for value in fields)


def _fixture_files(root: Path) -> dict[Path, bytes]:
    sequence = root / _SEQUENCE
    focal_px = 150.0
    principal_px = 150.0
    feature_pixels = {
        (
            round(principal_px + focal_px * x / z),
            round(principal_px + focal_px * y / z),
        )
        for x, y, z in [
            *((float(x), float(y), 5.0) for x in range(3) for y in range(3)),
            *((0.0, float(y), 8.0) for y in range(3)),
        ]
    }
    image = _edge_png(
        width=300,
        height=300,
        feature_pixels=feature_pixels,
    )
    points = _velodyne_payload()
    return {
        sequence / "image_02/data/0000000000.png": image,
        sequence / "image_02/data/0000000001.png": image,
        sequence / "velodyne_points/data/0000000000.bin": points,
        sequence / "velodyne_points/data/0000000001.bin": points,
        sequence / "oxts/data/0000000000.txt": _oxts_packet(
            yaw_rad=0.0,
            vn=2.0,
            ve=0.0,
        ).encode("utf-8"),
        sequence / "oxts/data/0000000001.txt": _oxts_packet(
            yaw_rad=0.2,
            vn=4.0,
            ve=1.0,
        ).encode("utf-8"),
        sequence / "image_02/timestamps.txt": _TIMESTAMPS.encode("utf-8"),
        sequence / "velodyne_points/timestamps.txt": (
            b"2011-09-26 13:00:00.005000000\n2011-09-26 13:00:01.005000000\n"
        ),
        sequence / "oxts/timestamps.txt": (
            b"2011-09-26 13:00:00.000000000\n2011-09-26 13:00:02.000000000\n"
        ),
        root / "2011_09_26/calib_velo_to_cam.txt": (b"R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n"),
        root / "2011_09_26/calib_cam_to_cam.txt": (
            b"S_rect_02: 300 300\n"
            b"R_rect_00: 1 0 0 0 1 0 0 0 1\n"
            b"P_rect_02: 150 0 150 0 0 150 150 0 0 0 1 0\n"
        ),
    }


def generate_fixture(root: Path, *, force: bool = False) -> dict[str, object]:
    """Write the deterministic fixture and return file-level provenance."""

    files = _fixture_files(root)
    existing = [path for path in files if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "fixture files already exist; pass --force to replace only the declared files"
        )

    records: list[dict[str, object]] = []
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return {
        "schema_version": "calibrex.synthetic_fixture_generation/v0.1",
        "generator": "tools/generate_kitti_lidar_camera_demo_fixture.py",
        "output_root": str(root),
        "files": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the deterministic files declared by this generator",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a fixture from command-line arguments."""

    args = _parser().parse_args(argv)
    try:
        result = generate_fixture(args.output_root, force=args.force)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
