"""Explain an uninformative result by what the recording's own motion offered.

An IMU-LiDAR or camera-IMU rotation is only observable when the rig rotates about
every axis. When a pair ends ``no_judgeable_axes``, :func:`recording_motion_note`
reads the IMU gyro of the same recording and reports how long it was and how far
it rotated about each IMU axis (integrated angular rate), so the user learns
whether the recording is too short or too static, and what to record instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from calibrex.core.exceptions import DatasetError
from calibrex.data.rosbag2 import decode_imu, iter_topic_messages

# Heuristic: a rotation axis counts as excited by an excursion of this many degrees.
EXCITED_EXCURSION_DEG = 10.0
RECOMMENDED_DURATION_S = 60.0
_AXES = ("x", "y", "z")


def rotation_excursions_deg(
    times_s: Sequence[float], angular_velocity_rps: NDArray[np.float64]
) -> tuple[float, tuple[float, float, float]]:
    """Duration (s) and per-axis peak-to-peak integrated angular rate (degrees)."""

    t = np.asarray(times_s, dtype=np.float64)
    w = np.asarray(angular_velocity_rps, dtype=np.float64)
    if t.size < 2:
        return 0.0, (0.0, 0.0, 0.0)
    dt = np.diff(t, prepend=t[0])
    angle = np.degrees(np.cumsum(w * dt[:, None], axis=0))
    span = angle.max(axis=0) - angle.min(axis=0)
    return float(t[-1] - t[0]), (float(span[0]), float(span[1]), float(span[2]))


def motion_note(duration_s: float, excursions_deg: Sequence[float]) -> str:
    """Sentence describing the recording's rotation excitation and what to record."""

    parts = ", ".join(
        f"{axis} {value:.1f} deg" for axis, value in zip(_AXES, excursions_deg, strict=True)
    )
    weak = [
        axis
        for axis, value in zip(_AXES, excursions_deg, strict=True)
        if value < EXCITED_EXCURSION_DEG
    ]
    if not weak and duration_s >= RECOMMENDED_DURATION_S:
        return (
            f"recording motion: {duration_s:.0f} s, integrated IMU rotation {parts}; the "
            "excitation looks sufficient, so the estimator's own noise (not the length) "
            "limits it"
        )
    needs = []
    if duration_s < RECOMMENDED_DURATION_S:
        needs.append(f"about {RECOMMENDED_DURATION_S:.0f} s or more")
    if weak:
        needs.append(
            f"turns or tilts about the IMU {'/'.join(weak)} "
            f"{'axes' if len(weak) > 1 else 'axis'} (at least ~{EXCITED_EXCURSION_DEG:.0f} deg)"
        )
    return (
        f"recording too short or too static: {duration_s:.0f} s with integrated IMU rotation "
        f"{parts}; need {' and '.join(needs)}"
    )


def recording_motion_note(
    bag: Path, imu_topic: str, *, max_duration_s: float | None = None
) -> str | None:
    """Read the IMU of ``bag`` and return a :func:`motion_note`, or ``None`` if unreadable."""

    times: list[float] = []
    rates: list[list[float]] = []
    try:
        for _conn, timestamp_ns, payload in iter_topic_messages(bag, imu_topic):
            message = decode_imu(imu_topic, timestamp_ns, payload)
            time_s = message.timestamp_ns * 1.0e-9
            if max_duration_s is not None and times and time_s - times[0] > max_duration_s:
                break
            times.append(time_s)
            rates.append(list(message.angular_velocity))
    except DatasetError:
        return None
    if len(times) < 2:
        return None
    duration, excursions = rotation_excursions_deg(times, np.asarray(rates))
    return motion_note(duration, excursions)
