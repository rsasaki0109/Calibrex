"""Compare decoded Velodyne packets with a concatenated base_link cloud of the same bag.

Autoware sample bags hold the raw packets of each Velodyne next to a concatenated cloud
in ``base_link``. This decodes one scan per sensor, transforms it with the bag's
``/tf_static`` and reports nearest-neighbour distances to the concatenated cloud
(median, 90th percentile, fraction within 2/5/10 cm) overall and per laser ring::

    python tools/validate_velodyne_decode.py BAG

The concatenated cloud may be cropped (points inside the vehicle or beyond a range are
removed), so the statistics are over decoded points that have a reference point within
``--match-m`` (default 0.5 m); the matched fraction is printed too.
"""

from __future__ import annotations

import argparse
import sys
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
        model = vp.velodyne_model_of_packet(packet)
        decode = vp._decode_vlp16_packet if model == "vlp16" else vp._decode_vlp32c_packet
        channels = 16 if model == "vlp16" else 32
        xyz, info, _ = decode(packet)
        keep = (info[:, 1] >= 0.4) & (info[:, 1] <= 200.0)
        points.append(xyz[keep])
        rings.append((np.arange(xyz.shape[0]) % channels)[keep])
    xyz_all = np.concatenate(points)
    return np.stack([xyz_all[:, 1], -xyz_all[:, 0], xyz_all[:, 2]], axis=1), np.concatenate(rings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("--scan-index", type=int, default=30)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--concat-topic", default=CONCAT_TOPIC)
    parser.add_argument("--match-m", type=float, default=0.5)
    args = parser.parse_args(argv)
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
        model = vp.velodyne_model_of_packet(scan.packets[0])
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
