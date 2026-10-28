"""Adapt the KITTI converter's xyz/intensity/time clouds to Koide scan-start time.

This changes the time origin, not the azimuth timing model or point geometry.
Only the converter's explicit five-float layout is accepted; other fields must
not be silently discarded. Image messages remain byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from calibrex.data.ros_cdr import decode_ros2_pointcloud2
from calibrex.data.ros_cdr_writer import encode_pointcloud2
from calibrex.data.rosbag2 import iter_messages, require_cdr_connection
from calibrex.data.rosbag2_writer import Rosbag2Writer


class TimingRecord(BaseModel):
    """Acquisition-time preservation evidence, in seconds and integer nanoseconds."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source_header_ns: int
    output_header_ns: int
    header_shift_ns: int
    point_count: int
    source_offset_min_s: float
    source_offset_max_s: float
    output_offset_min_s: float
    output_offset_max_s: float
    max_absolute_time_change_s: float


class MessageRecord(BaseModel):
    """Source/output message identity, including unchanged camera messages."""

    model_config = ConfigDict(extra="forbid")
    topic: str
    source_bag_timestamp_ns: int
    source_sha256: str
    output_bag_timestamp_ns: int
    output_sha256: str
    timing: TimingRecord | None = None


class AdaptationReport(BaseModel):
    """Schema for an explicitly derived bag; no timing measurement claim."""

    model_config = ConfigDict(extra="forbid")
    scope: str
    source: str
    destination: str
    messages: list[MessageRecord]


def rebase_kitti_cloud(topic: str, timestamp_ns: int, data: bytes) -> tuple[bytes, dict]:
    """Preserve acquisition times while moving the header to the earliest point."""
    cloud = decode_ros2_pointcloud2(topic, timestamp_ns, data, point_time_field="time")
    expected = [(name, 4 * i, 7, 1) for i, name in enumerate(("x", "y", "z", "intensity", "time"))]
    actual = [(f.name, f.offset, f.datatype, f.count) for f in cloud.fields]
    if actual != expected or cloud.point_step != 20 or cloud.height != 1:
        raise ValueError("expected unorganized KITTI xyz/intensity/time float32 layout")
    times = cloud.point_time_offsets_s
    if (
        times is None
        or len(times) == 0
        or cloud.nonfinite_xyz_count
        or cloud.intensity is None
        or not np.isfinite(times).all()
        or not np.isfinite(cloud.intensity).all()
    ):
        raise ValueError("cloud must contain finite points, intensity and explicit times")
    # Floor, not round: the first offset remains nonnegative at nanosecond precision.
    shift_ns = math.floor(float(np.min(times)) * 1_000_000_000)
    rebased = np.asarray(times, dtype=np.float64) - shift_ns * 1e-9
    if rebased.max() >= 1 or rebased.max() == rebased.min():
        raise ValueError("expected a nonzero scan duration shorter than one second")
    order = np.argsort(rebased, kind="stable")
    packed = np.empty(
        len(times), dtype=[("xyz", "<f4", (3,)), ("intensity", "<f4"), ("time", "<f8")]
    )
    packed["xyz"] = cloud.xyz[order]
    packed["intensity"] = cloud.intensity[order]
    packed["time"] = rebased[order]
    new_stamp = cloud.timestamp_ns + shift_ns
    encoded = encode_pointcloud2(
        frame_id=cloud.frame_id,
        timestamp_ns=new_stamp,
        fields=(*expected[:4], ("time", 16, 8, 1)),
        point_step=24,
        data=packed.tobytes(),
    )
    residual = shift_ns * 1e-9 + rebased - times
    return encoded, {
        "source_header_ns": cloud.timestamp_ns,
        "output_header_ns": new_stamp,
        "header_shift_ns": shift_ns,
        "point_count": len(times),
        "source_offset_min_s": float(np.min(times)),
        "source_offset_max_s": float(np.max(times)),
        "output_offset_min_s": float(rebased.min()),
        "output_offset_max_s": float(rebased.max()),
        "max_absolute_time_change_s": float(np.max(np.abs(residual))),
    }


def adapt_bag(source: Path, destination: Path) -> dict:
    """Write a derived bag, preserving non-cloud payloads and recording hashes."""
    report = {
        "scope": "KITTI azimuth-model time-origin adaptation, not measured timing",
        "source": str(source),
        "destination": str(destination),
        "messages": [],
    }
    rows = []
    with Rosbag2Writer(destination) as writer:
        topics = set()
        for connection, timestamp, payload in iter_messages(source):
            require_cdr_connection(connection)
            if connection.topic not in topics:
                writer.add_topic(connection.topic, connection.message_type)
                topics.add(connection.topic)
            original = payload
            record = {
                "topic": connection.topic,
                "source_bag_timestamp_ns": timestamp,
                "source_sha256": hashlib.sha256(original).hexdigest(),
            }
            if connection.message_type == "sensor_msgs/msg/PointCloud2":
                payload, timing = rebase_kitti_cloud(connection.topic, timestamp, payload)
                record["timing"] = timing
                timestamp = timing["output_header_ns"]
            record["output_bag_timestamp_ns"] = timestamp
            record["output_sha256"] = hashlib.sha256(payload).hexdigest()
            report["messages"].append(record)
            rows.append((timestamp, connection.topic, payload))
        for timestamp, topic, payload in sorted(rows):
            writer.write(topic, timestamp, payload)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = adapt_bag(args.source, args.destination)
    validated = AdaptationReport.model_validate(report)
    args.report.write_text(validated.model_dump_json(indent=2) + "\n")
    args.report.with_suffix(".schema.json").write_text(
        json.dumps(AdaptationReport.model_json_schema(), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
