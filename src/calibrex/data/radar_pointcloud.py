"""Radar point clouds: a ``sensor_msgs/PointCloud2`` that carries each return's Doppler.

A 4D radar driver publishes its detections as a point cloud with an extra float field for
the radial (Doppler) velocity. Its name differs between drivers; the names below are
tried in order. ``calibrex check`` gives such a topic the ``radar`` role instead of
``lidar``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DOPPLER_FIELD_NAMES: tuple[str, ...] = (
    "v_doppler_mps",
    "doppler",
    "Doppler",
    "radial_velocity",
    "v_r",
    "vr",
    "velocity",
)
"""PointCloud2 fields read as a return's Doppler velocity (m/s), first match wins."""

_FLOAT_DATATYPES = frozenset({7, 8})  # FLOAT32, FLOAT64


def doppler_field(fields: Sequence[Any]) -> str | None:
    """The first float field of ``fields`` named as a Doppler velocity, or ``None``."""

    by_name = {item.name: item for item in fields}
    for name in DOPPLER_FIELD_NAMES:
        item = by_name.get(name)
        if item is not None and int(item.datatype) in _FLOAT_DATATYPES:
            return name
    return None
