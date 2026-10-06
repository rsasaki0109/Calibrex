"""Shift the header stamps of chosen topics of a rosbag2 (clock-offset known-bad control).

A sensor whose clock offset changed between two sessions (a driver update, a PTP
or hardware-timestamping change) stamps the same physical event ``dt`` later or
earlier than before. :func:`shift_stamps_in_bag` copies a bag message by message
and adds ``shift_s`` to the ``std_msgs/Header`` stamp of every message on the
chosen topics; every other byte of every message is copied unchanged.

Which stamp the estimators use matters: ``calibrex`` decodes the **header stamp**
of an Imu, Image or PointCloud2 message (falling back to the bag's log time only
when the header stamp is zero), so the header stamp is what is shifted. The log
(receive) time is the time the recorder saw the message; a clock-offset change
does not move it, so it is left alone by default (``shift_log_time=True`` shifts
it too, which keeps the bag's time order consistent when the shift is large). A
message whose header stamp is zero is left unchanged and counted. Per-point time
fields of a point cloud (absolute times in some drivers) are *not* shifted.

The point of the control is a *known* change: a drift detector that sees the
original and the shifted bag must flag the pair's time offset with a magnitude of
about ``shift_s``, and only that.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from calibrex.core.exceptions import DatasetError
from calibrex.data.rosbag2 import iter_messages, list_rosbag2_connections
from calibrex.data.rosbag2_writer import Rosbag2Writer

_NS = 1_000_000_000


def shift_header_stamp(data: bytes, shift_ns: int) -> bytes | None:
    """The CDR payload with its leading ``Header.stamp`` moved by ``shift_ns``.

    Returns ``None`` when the stamp is zero (an unstamped message) or the payload is
    too short; the stamp is the first field after the 4-byte CDR encapsulation header.
    """

    if len(data) < 12 or data[0] != 0 or data[1] not in (0, 1, 2, 3):
        return None
    endian = "<" if data[1] in (1, 3) else ">"
    sec, nsec = struct.unpack_from(f"{endian}iI", data, 4)
    total = int(sec) * _NS + int(nsec)
    if total == 0:
        return None
    total += shift_ns
    if total < 0:
        raise DatasetError("the shift moves a header stamp before the epoch")
    new_sec, new_nsec = divmod(total, _NS)
    buffer = bytearray(data)
    struct.pack_into(f"{endian}iI", buffer, 4, new_sec, new_nsec)
    return bytes(buffer)


@dataclass(frozen=True)
class ShiftStampsSummary:
    """What :func:`shift_stamps_in_bag` wrote."""

    source: str
    destination: str
    topics: tuple[str, ...]
    shift_s: float
    shift_log_time: bool
    messages_shifted: int
    messages_unstamped: int
    messages_copied: int
    max_duration_s: float | None


def shift_stamps_in_bag(
    source: str | Path,
    destination: str | Path,
    shift_s: float,
    *,
    topics: Sequence[str],
    shift_log_time: bool = False,
    max_duration_s: float | None = None,
    overwrite: bool = False,
) -> ShiftStampsSummary:
    """Copy ``source`` to ``destination`` with ``shift_s`` added to the header stamps of ``topics``.

    ``topics`` must name topics whose messages start with a ``std_msgs/Header``. With
    ``max_duration_s`` only messages within that many seconds of the first message are
    copied (an estimator run with the same ``--max-duration-s`` sees the same data).
    """

    if not topics:
        raise ValueError("name at least one topic to shift")
    connections = [c for c, _ in list_rosbag2_connections(source)]
    unknown = sorted(set(topics) - {c.topic for c in connections})
    if unknown:
        raise DatasetError(f"not a topic of {source}: {', '.join(unknown)}")
    chosen = set(topics)
    shift_ns = round(shift_s * _NS)
    shifted = unstamped = copied = 0
    first_ns: int | None = None
    with Rosbag2Writer(destination, overwrite=overwrite) as writer:
        for connection in connections:
            latched = "durability: 1" in (connection.offered_qos_profiles or "")
            writer.add_topic(connection.topic, connection.message_type, latched=latched)
        for connection, timestamp_ns, payload in iter_messages(source):
            if first_ns is None:
                first_ns = timestamp_ns
            if max_duration_s is not None and timestamp_ns - first_ns > max_duration_s * 1e9:
                break  # messages arrive in timestamp order: nothing later is in range
            if connection.topic in chosen:
                moved = shift_header_stamp(payload, shift_ns)
                if moved is None:
                    unstamped += 1
                else:
                    payload = moved
                    shifted += 1
                    if shift_log_time:
                        timestamp_ns = max(0, timestamp_ns + shift_ns)
            writer.write(connection.topic, timestamp_ns, payload)
            copied += 1
    return ShiftStampsSummary(
        source=str(source),
        destination=str(destination),
        topics=tuple(sorted(chosen)),
        shift_s=shift_s,
        shift_log_time=shift_log_time,
        messages_shifted=shifted,
        messages_unstamped=unstamped,
        messages_copied=copied,
        max_duration_s=max_duration_s,
    )
