"""Compare decoded Velodyne packets with a reference cloud decoded by someone else.

Two references are supported.

Autoware sample bags hold the raw packets of each Velodyne next to a concatenated cloud
in ``base_link``. This decodes one scan per sensor, transforms it with the bag's
``/tf_static`` and reports nearest-neighbour distances to the concatenated cloud
(median, 90th percentile, fraction within 2/5/10 cm) overall and per laser ring::

    python tools/validate_velodyne_decode.py BAG

The concatenated cloud may be cropped (points inside the vehicle or beyond a range are
removed), so the statistics are over decoded points that have a reference point within
``--match-m`` (default 0.5 m); the matched fraction is printed too.

LidarView's test recordings (``tools/download_public_dataset.py lidarview_velodyne``)
pair a pcap with the frames LidarView decoded from it, as VTK ``.vtp`` files. For each
frame this decodes the packets whose timestamps fall in it and reports, for every
reference point, the distance to the nearest decoded point (sensor frame, manual axes),
plus the decoded per-point time against LidarView's::

    python tools/validate_velodyne_decode.py --pcap X.pcap --reference X_1.vtp \
        [--reference X_2.vtp ...] [--calibration db.xml]
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from calibrex.check.frame_tree import StaticFrameTree
from calibrex.check.tf_sources import load_bag_tf_static
from calibrex.data import velodyne_packets as vp
from calibrex.data.rosbag2 import decode_pointcloud2, iter_topic_messages, list_rosbag2_connections

CONCAT_TOPIC = "/sensing/lidar/concatenated/pointcloud"


def _stats(dist: np.ndarray) -> str:
    return (
        f"median={np.median(dist) * 100:.2f} cm p90={np.percentile(dist, 90) * 100:.2f} cm "
        f"<2cm={(dist < 0.02).mean():.2f} <5cm={(dist < 0.05).mean():.2f} "
        f"<10cm={(dist < 0.10).mean():.2f}"
    )


def _decode_with_rings(scan: vp.VelodynePacketScan) -> tuple[np.ndarray, np.ndarray]:
    """Points (Autoware axes, x forward) and laser ring of every non-empty return."""

    points: list[np.ndarray] = []
    rings: list[np.ndarray] = []
    for packet in scan.packets:
        decoded = vp.decode_velodyne_packet(packet)
        keep = (decoded.range_m >= 0.4) & (decoded.range_m <= 200.0)
        points.append(decoded.xyz[keep])
        rings.append(decoded.laser[keep])
    xyz_all = np.concatenate(points)
    return np.stack([xyz_all[:, 1], -xyz_all[:, 0], xyz_all[:, 2]], axis=1), np.concatenate(rings)


def read_pcap_payloads(path: Path, size: int = vp.PACKET_SIZE) -> list[bytes]:
    """UDP payloads of ``size`` bytes from a classic pcap (Ethernet or raw IPv4)."""

    data = path.read_bytes()
    if len(data) < 24:
        raise ValueError(f"{path} is not a pcap file")
    magic = data[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    else:
        raise ValueError(f"{path} is not a classic pcap file (pcapng is not read)")
    (linktype,) = struct.unpack(endian + "I", data[20:24])
    payloads: list[bytes] = []
    position = 24
    while position + 16 <= len(data):
        (included,) = struct.unpack(endian + "I", data[position + 8 : position + 12])
        frame = data[position + 16 : position + 16 + included]
        position += 16 + included
        if linktype == 1:  # Ethernet, optionally one 802.1Q tag
            header = 18 if frame[12:14] == b"\x81\x00" else 14
            frame = frame[header:]
        if len(frame) < 28 or frame[0] >> 4 != 4 or frame[9] != 17:
            continue
        payload = frame[(frame[0] & 0x0F) * 4 + 8 :]
        if len(payload) == size:
            payloads.append(payload)
    return payloads


_VTK_TYPES = {
    "Float64": "<f8",
    "Float32": "<f4",
    "Int64": "<i8",
    "UInt64": "<u8",
    "Int32": "<i4",
    "UInt32": "<u4",
    "UInt16": "<u2",
    "UInt8": "u1",
}


def read_vtp_point_data(path: Path) -> dict[str, np.ndarray]:
    """Point arrays of a VTK XML PolyData file with raw, uncompressed appended data."""

    data = path.read_bytes()
    marker = data.find(b'<AppendedData encoding="raw">')
    if marker < 0:
        raise ValueError(f"{path}: only raw appended VTK data is read")
    start = data.index(b"_", marker) + 1
    header = data[:marker].decode("utf-8", errors="replace")
    if 'header_type="UInt64"' not in header or 'compressor="' in header:
        raise ValueError(f"{path}: expected uncompressed data with UInt64 headers")
    arrays: dict[str, np.ndarray] = {}
    for match in re.finditer(r"<DataArray ([^>]*)>", header):
        attributes = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
        name, kind = attributes.get("Name"), attributes.get("type")
        if name is None or name in arrays or kind not in _VTK_TYPES:
            continue
        offset = start + int(attributes["offset"])
        (length,) = struct.unpack("<Q", data[offset : offset + 8])
        values = np.frombuffer(data[offset + 8 : offset + 8 + length], dtype=_VTK_TYPES[kind])
        components = int(attributes.get("NumberOfComponents", "1"))
        arrays[name] = values.reshape(-1, components) if components > 1 else values
    return arrays


@dataclass(frozen=True)
class FrameComparison:
    """One LidarView frame against the packets decoded here."""

    reference: str
    packet_format: str
    reference_points: int
    decoded_points: int
    median_m: float
    p90_m: float
    within_1mm: float
    time_offset_us: float
    time_spread_us: float


def compare_lidarview_frame(
    payloads: list[bytes],
    reference: Path,
    calibration: vp.VelodyneCalibration | None = None,
) -> FrameComparison:
    """Decode the packets of one LidarView frame and compare point positions and times.

    Packets are selected by their top-of-hour timestamp (bytes 1200-1203), so a frame
    that starts or ends mid-packet adds a few decoded points outside it; the
    statistics are over the reference points. The decoded time of each matched point
    is compared with LidarView's per-point timestamp: the median difference is a
    stamp convention (the packet stamp marks its first or its last firing), the
    spread around it is the timing error.
    """

    arrays = read_vtp_point_data(reference)
    reference_xyz = np.stack([arrays["X"], arrays["Y"], arrays["Z"]], axis=1)
    reference_time = arrays["timestamp"].astype(np.float64)
    low, high = reference_time.min(), reference_time.max()
    xyz: list[np.ndarray] = []
    times: list[np.ndarray] = []
    packet_format = ""
    for payload in payloads:
        stamp_us = int.from_bytes(payload[1200:1204], "little")
        if not low <= stamp_us <= high:
            continue
        packet_format = packet_format or vp.velodyne_packet_format(payload).description
        decoded = vp.decode_velodyne_packet(payload, calibration)
        xyz.append(decoded.xyz)
        times.append(stamp_us + decoded.time_s * 1.0e6)
    if not xyz:
        raise ValueError(f"no packet of the pcap falls in {reference.name}")
    decoded_xyz = np.concatenate(xyz)
    decoded_time = np.concatenate(times)
    distance, index = cKDTree(decoded_xyz).query(reference_xyz)
    matched = distance < 1.0e-3
    delta = reference_time[matched] - decoded_time[index[matched]]
    offset = float(np.median(delta)) if delta.size else float("nan")
    spread = float(np.median(np.abs(delta - offset))) if delta.size else float("nan")
    return FrameComparison(
        reference=reference.name,
        packet_format=packet_format,
        reference_points=int(reference_xyz.shape[0]),
        decoded_points=int(decoded_xyz.shape[0]),
        median_m=float(np.median(distance)),
        p90_m=float(np.percentile(distance, 90)),
        within_1mm=float(matched.mean()),
        time_offset_us=offset,
        time_spread_us=spread,
    )


def _main_lidarview(args: argparse.Namespace) -> int:
    calibration = (
        vp.load_velodyne_calibration(args.calibration) if args.calibration is not None else None
    )
    payloads = read_pcap_payloads(args.pcap)
    print(f"{args.pcap.name}: {len(payloads)} data packets")
    medians: list[float] = []
    for reference in args.reference:
        result = compare_lidarview_frame(payloads, reference, calibration)
        medians.append(result.median_m)
        print(
            f"{result.reference} ({result.packet_format}): {result.reference_points} reference / "
            f"{result.decoded_points} decoded points; median {result.median_m * 1e3:.3f} mm, "
            f"p90 {result.p90_m * 1e3:.3f} mm, {result.within_1mm:.1%} within 1 mm; time "
            f"{result.time_offset_us:+.1f} us stamp convention, "
            f"{result.time_spread_us:.2f} us median deviation"
        )
    print(f"median point distance over {len(medians)} frame(s): {np.median(medians) * 1e3:.3f} mm")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path, nargs="?")
    parser.add_argument("--scan-index", type=int, default=30)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--concat-topic", default=CONCAT_TOPIC)
    parser.add_argument("--match-m", type=float, default=0.5)
    parser.add_argument("--pcap", type=Path, help="LidarView test pcap")
    parser.add_argument(
        "--reference", type=Path, action="append", default=[], help="LidarView frame (.vtp)"
    )
    parser.add_argument("--calibration", type=Path, help="db.xml or ROS YAML (HDL-64E, VLS-128)")
    args = parser.parse_args(argv)
    if args.pcap is not None:
        if not args.reference:
            parser.error("--pcap needs at least one --reference frame")
        return _main_lidarview(args)
    if args.bag is None:
        parser.error("give a BAG, or --pcap with --reference frames")
    source = load_bag_tf_static(args.bag)
    if source is None:
        parser.error("the bag has no /tf_static")
    tree = StaticFrameTree(source.edges)
    concat = None
    for index, (_c, ts, data) in enumerate(iter_topic_messages(args.bag, args.concat_topic)):
        if index == args.scan_index:
            concat = (ts, decode_pointcloud2("c", ts, data))
            break
    if concat is None:
        parser.error(f"{args.concat_topic} has fewer than {args.scan_index + 1} messages")
    stamp, cloud = concat
    reference = cKDTree(cloud.xyz)
    print(f"concatenated cloud: {cloud.point_count} points at {stamp}")
    topics = sorted(
        c.topic
        for c, _ in list_rosbag2_connections(args.bag)
        if c.message_type == vp.VELODYNE_SCAN_TYPE
    )
    for topic in topics:
        best = None
        for _c, ts, data in iter_topic_messages(args.bag, topic):
            if best is None or abs(ts - stamp) < abs(best[0] - stamp):
                best = (ts, data)
            if ts > stamp + 2 * 10**8:
                break
        if best is None:
            continue
        scan = vp.decode_velodyne_scan_packets(best[1])
        model = vp.velodyne_packet_format(scan.packets[0]).description
        points, rings = _decode_with_rings(scan)
        transform = tree.lookup(args.base_frame, scan.frame_id)
        if transform is None:
            print(f"{topic}: no transform {args.base_frame} <- {scan.frame_id}")
            continue
        rotation = Rotation.from_quat(transform.rotation_quat_xyzw).as_matrix()
        in_base = points @ rotation.T + np.asarray(transform.translation_m)
        dist, _ = reference.query(in_base)
        matched = dist < args.match_m
        print(
            f"{topic} ({model}, scan {(best[0] - stamp) * 1e-9:+.3f} s from the cloud stamp): "
            f"{matched.sum()}/{dist.size} points within {args.match_m} m; "
            f"{_stats(dist[matched])}"
        )
        for ring in range(int(rings.max()) + 1):
            selected = dist[matched & (rings == ring)]
            if selected.size:
                print(f"  ring {ring:2d}: n={selected.size:5d} {_stats(selected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
