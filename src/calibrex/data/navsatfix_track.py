"""GNSS tracks from a bag's ``sensor_msgs/NavSatFix`` topic.

:func:`read_navsatfix_track` builds the :class:`~calibrex.data.rtk_slam.GnssTrack`
the GNSS-LiDAR estimator consumes, so a bag needs no separate ``rtk.txt``.

Mapping from NavSatFix to the ``rtk.txt`` columns (verified on RTK-SLAM, whose
``/gnss/fix`` is the same receiver stream as ``rtk.txt``):

* **time**: the header stamp, the clock the LiDAR header stamps use as well.
  On RTK-SLAM the stamps equal ``rtk.txt`` to the microsecond.
* **position**: ``latitude``/``longitude`` in degrees and ``altitude`` in
  metres above the WGS84 ellipsoid (the ROS convention, and what
  ``geodetic_to_enu`` expects).
* **quality**: ``NavSatStatus.status`` is -1 (no fix), 0 (fix), 1 (SBAS) or
  2 (GBAS, i.e. RTK). It cannot tell RTK-fixed from RTK-float; RTK-SLAM's
  driver publishes RTK-fixed as 0 and everything else (``rtk.txt`` status 1 and
  2: single and float) as -1. A fix is therefore *usable* when its status is at
  least ``min_status`` (default 0) and its standard deviation is at most
  ``max_sigma_m`` (default 0.15 m), so a bag of ordinary single-point GPS
  (status 0, metres of covariance) is rejected rather than treated as RTK.
* **standard deviation**: ``sqrt(trace(position_covariance))``, the 3-D RMS
  position error. It matches ``rtk.txt``'s ``blt_std`` (median ratio 1.0000
  on construction_seq1). When ``position_covariance_type`` is 0 (unknown) the
  covariance is not trusted: the fix is kept only if its status is GBAS (2),
  with ``fallback_sigma_m`` as its std, and the summary records how many fixes
  used the fallback.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibrex.data import ros_cdr
from calibrex.data.ros_messages import NavSatFixMessage
from calibrex.data.rosbag2 import iter_topic_messages
from calibrex.data.rtk_slam import GnssTrack, geodetic_to_enu

GBAS_STATUS = 2


@dataclass(frozen=True)
class NavSatFixTrackOptions:
    """Which fixes are usable and how their std is derived."""

    min_status: int = 0
    max_sigma_m: float = 0.15
    fallback_sigma_m: float = 0.05
    sigma_floor_m: float = 0.01

    def __post_init__(self) -> None:
        if min(self.max_sigma_m, self.fallback_sigma_m, self.sigma_floor_m) <= 0.0:
            raise ValueError("max_sigma_m, fallback_sigma_m and sigma_floor_m must be positive")


@dataclass(frozen=True)
class NavSatFixTrackSummary:
    """What was read from the topic and why fixes were dropped."""

    topic: str
    messages: int
    kept: int
    rejected_no_fix: int
    rejected_sigma: int
    rejected_unknown_quality: int
    rejected_invalid: int
    covariance_type_counts: dict[int, int]
    fallback_sigma_epochs: int
    frame_id: str
    first_stamp_s: float | None
    last_stamp_s: float | None


def track_from_navsatfix(
    messages: Iterable[NavSatFixMessage],
    options: NavSatFixTrackOptions | None = None,
    *,
    topic: str = "",
    origin: tuple[float, float, float] | None = None,
) -> tuple[GnssTrack, NavSatFixTrackSummary]:
    """Filter NavSatFix messages into a time-sorted ENU track (see the module docstring)."""

    opts = options or NavSatFixTrackOptions()
    rows: list[tuple[float, float, float, float, float]] = []
    counts: dict[int, int] = {}
    messages_seen = no_fix = too_noisy = unknown_quality = invalid = fallback = 0
    frame_id = ""
    first: float | None = None
    last: float | None = None
    for message in messages:
        messages_seen += 1
        frame_id = frame_id or message.frame_id
        stamp = message.timestamp_ns * 1.0e-9
        first = stamp if first is None else min(first, stamp)
        last = stamp if last is None else max(last, stamp)
        counts[message.position_covariance_type] = (
            counts.get(message.position_covariance_type, 0) + 1
        )
        position = (message.latitude_deg, message.longitude_deg, message.altitude_m)
        if not all(math.isfinite(value) for value in position):
            invalid += 1
            continue
        if message.status < opts.min_status:
            no_fix += 1
            continue
        trace = (
            message.position_covariance[0]
            + message.position_covariance[4]
            + message.position_covariance[8]
        )
        known = message.position_covariance_type in (1, 2, 3) and math.isfinite(trace) and trace > 0
        if known:
            sigma = math.sqrt(trace)
            if sigma > opts.max_sigma_m:
                too_noisy += 1
                continue
        elif message.status >= GBAS_STATUS:
            sigma = opts.fallback_sigma_m
            fallback += 1
        else:
            unknown_quality += 1
            continue
        rows.append((stamp, *position, max(sigma, opts.sigma_floor_m)))
    summary = NavSatFixTrackSummary(
        topic=topic,
        messages=messages_seen,
        kept=len(rows),
        rejected_no_fix=no_fix,
        rejected_sigma=too_noisy,
        rejected_unknown_quality=unknown_quality,
        rejected_invalid=invalid,
        covariance_type_counts=counts,
        fallback_sigma_epochs=fallback,
        frame_id=frame_id,
        first_stamp_s=first,
        last_stamp_s=last,
    )
    if len(rows) < 2:
        raise ValueError(
            f"{topic or 'NavSatFix'}: fewer than two usable GNSS fixes of {messages_seen} "
            f"(no fix {no_fix}, std over {opts.max_sigma_m:g} m {too_noisy}, "
            f"unknown covariance {unknown_quality}, invalid {invalid})"
        )
    table = np.array(rows, dtype=np.float64)
    table = table[np.argsort(table[:, 0], kind="stable")]
    chosen = origin or (float(table[0, 1]), float(table[0, 2]), float(table[0, 3]))
    track = GnssTrack(
        times_s=table[:, 0],
        enu_m=geodetic_to_enu(table[:, 1], table[:, 2], table[:, 3], chosen),
        sigma_m=table[:, 4],
        origin_lat_lon_height=chosen,
        rejected_epochs=messages_seen - len(rows),
    )
    return track, summary


def read_navsatfix_track(
    bag: str | Path,
    topic: str,
    options: NavSatFixTrackOptions | None = None,
    *,
    origin: tuple[float, float, float] | None = None,
) -> tuple[GnssTrack, NavSatFixTrackSummary]:
    """Read a bag's NavSatFix topic into a GNSS track."""

    def messages() -> Iterable[NavSatFixMessage]:
        for _conn, timestamp_ns, payload in iter_topic_messages(bag, topic):
            yield ros_cdr.decode_ros2_navsatfix(topic, timestamp_ns, payload)

    return track_from_navsatfix(messages(), options, topic=topic, origin=origin)
