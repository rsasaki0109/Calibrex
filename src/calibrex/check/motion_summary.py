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
from calibrex.data.rosbag2 import decode_imu, iter_topic_messages, require_cdr_connection

# Heuristic: a rotation axis counts as excited by an excursion of this many degrees.
EXCITED_EXCURSION_DEG = 10.0
RECOMMENDED_DURATION_S = 60.0
# Heuristic: an axis counts as rotated about at this RMS angular rate (deg/s).
EXCITED_RATE_DPS = 5.0
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


ADVICE_MARK = "; to judge "
"""Where an unchecked rotation axis's reason turns into advice (:func:`rotation_axis_advice`)."""
_ROTATION_AXES = ("roll", "pitch", "yaw")
_CAMERA_MOTION = {"x": "tilt it up and down", "y": "pan it left and right", "z": "roll it"}


def rotation_rates_dps(
    angular_velocity_rps: NDArray[np.float64], rotation_parent_imu: NDArray[np.float64]
) -> tuple[float, float, float]:
    """RMS angular rate (deg/s) about the x, y, z axes of the frame the errors are given in."""

    rates = np.asarray(angular_velocity_rps, dtype=np.float64) @ np.asarray(rotation_parent_imu).T
    rms = np.degrees(np.sqrt(np.mean(rates**2, axis=0))) if rates.size else np.zeros(3)
    return float(rms[0]), float(rms[1]), float(rms[2])


def rotation_axis_advice(
    axis: str,
    frame: str,
    rates_dps: Sequence[float],
    *,
    camera: bool = False,
    duration_s: float | None = None,
) -> str:
    """Whether recording other motion would let the rotation error about one axis be judged.

    A mounting error about an axis does not change the rotation rate measured about that
    axis; it shows only in rotation about the other two. So ``roll`` (x) needs rotation
    about y or z, ``pitch`` (y) about x or z and ``yaw`` (z) about x or y.

    When the rig hardly rotated about either of those (both RMS rates under
    ``EXCITED_RATE_DPS``, a heuristic) the text says which rotation to add. When it did,
    the text says so: the motion is not what is missing, and re-recording the same way is
    not expected to help. TUM-VI's corridor4 leaves pitch unchecked at 34 and 27 deg/s
    about x and z, about what room1 has (42 and 26) with every axis judged. A recording
    shorter than ``RECOMMENDED_DURATION_S`` is the exception: the same motion for longer
    narrows every axis (the LiU GoPro sequences are 30 to 45 s).
    """

    index = _ROTATION_AXES.index(axis)
    others = [name for k, name in enumerate(_AXES) if k != index]
    offered = ", ".join(f"{name} {rate:.0f}" for name, rate in zip(_AXES, rates_dps, strict=True))
    measured = f"this recording's RMS rotation rate about {frame} was {offered} deg/s"
    excited = max(rate for k, rate in enumerate(rates_dps) if k != index) >= EXCITED_RATE_DPS
    if excited and duration_s is not None and duration_s < RECOMMENDED_DURATION_S:
        return (
            f"{axis}, record the same motion for longer: the rig did rotate about {others[0]} "
            f"and {others[1]} ({measured}), but only for {duration_s:.0f} s; record about "
            f"{RECOMMENDED_DURATION_S:.0f} s or more"
        )
    if excited:
        return (
            f"{axis}, other motion is not what is missing: the rig did rotate about "
            f"{others[0]} and {others[1]}, the axes that show an error about {_AXES[index]} "
            f"({measured}); the estimator's noise on this axis limits it, so recording the "
            "same way again is not expected to help"
        )
    motions = (
        " (" + ", or ".join(f"{_CAMERA_MOTION[name]} for {name}" for name in others) + ")"
        if camera
        else ""
    )
    return (
        f"{axis}, rotate the rig about the {others[0]} or {others[1]} axis of {frame}{motions}: "
        f"an error about {_AXES[index]} does not show in rotation about {_AXES[index]} itself, "
        f"and {measured}"
    )


def read_gyro(
    bag: Path, imu_topic: str, *, max_duration_s: float | None = None
) -> tuple[list[float], NDArray[np.float64]] | None:
    """Times (s) and angular velocities (rad/s) of the IMU of ``bag``, or ``None`` if unreadable."""

    times: list[float] = []
    rates: list[list[float]] = []
    try:
        for connection, timestamp_ns, payload in iter_topic_messages(bag, imu_topic):
            require_cdr_connection(connection)
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
    return times, np.asarray(rates, dtype=np.float64)


def recording_motion_note(
    bag: Path, imu_topic: str, *, max_duration_s: float | None = None
) -> str | None:
    """Read the IMU of ``bag`` and return a :func:`motion_note`, or ``None`` if unreadable."""

    gyro = read_gyro(bag, imu_topic, max_duration_s=max_duration_s)
    if gyro is None:
        return None
    duration, excursions = rotation_excursions_deg(*gyro)
    return motion_note(duration, excursions)
