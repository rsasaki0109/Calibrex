"""Stream LiDAR scans into reliable odometry windows.

Calibration evaluations that compare LiDAR odometry with another sensor (GNSS
positions, IMU rates) share one pipeline: read a bag once, run odometry only
while the other sensor covers the scans, restart it after every gap, cut it
at unreliable registrations, and split it into fixed-length windows that each
get their own nuisance alignment and serve as holdout and jackknife units.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.scan_to_scan_odometry import (
    IncrementalScanOdometry,
    RotationModel,
    ScanOdometryOptions,
    ScanRegistration,
)

FloatArray: TypeAlias = NDArray[np.float64]
ScanItem: TypeAlias = tuple[float, FloatArray] | tuple[float, FloatArray, FloatArray | None]


@dataclass(frozen=True)
class WindowingOptions:
    """Window length, registration reliability, and odometry settings."""

    window_duration_s: float = 10.0
    min_window_scans: int = 20
    min_correspondences: int = 200
    max_registration_rmse_m: float = 0.3
    odometry: ScanOdometryOptions = field(
        default_factory=lambda: ScanOdometryOptions(
            voxel_size_m=0.3, min_range_m=1.0, max_range_m=60.0, local_map_scans=5
        )
    )


@dataclass
class OdometrySegmenter:
    """Accumulates windows while scans stream in; see :func:`collect_odometry_windows`."""

    options: WindowingOptions
    covers: Callable[[float], bool] | None = None
    prefix: str = ""
    rotation_model: RotationModel | None = None
    windows: list[OdometryWindow] = field(default_factory=list)
    scans_read: int = 0
    scans_covered: int = 0
    segments: int = 0
    unreliable: int = 0
    _odometry: IncrementalScanOdometry | None = None
    _times: list[float] = field(default_factory=list)

    def add(self, time_s: float, scan: FloatArray, offsets_s: FloatArray | None = None) -> None:
        """Feed one scan; an uncovered scan ends the current odometry segment."""

        self.scans_read += 1
        if self.covers is not None and not self.covers(time_s):
            self.flush()
            return
        self.scans_covered += 1
        if self._odometry is None:
            self._odometry = IncrementalScanOdometry(
                self.options.odometry, rotation_model=self.rotation_model
            )
            self._times = []
        self._odometry.add(scan, time_s, offsets_s)
        self._times.append(time_s)

    def flush(self) -> None:
        """Close the current segment and cut it into windows."""

        if self._odometry is None:
            return
        result = self._odometry.result()
        self._odometry = None
        if len(result.poses) < 2:
            return
        self.segments += 1
        poses = np.stack(result.poses)
        times = np.array(self._times)
        start = 0
        for index, registration in enumerate(result.registrations, start=1):
            if not self._reliable(registration):
                self.unreliable += 1
                self._cut(times[start:index], poses[start:index])
                start = index
        self._cut(times[start:], poses[start:])

    def _reliable(self, registration: ScanRegistration) -> bool:
        return (
            registration.correspondences >= self.options.min_correspondences
            and math.isfinite(registration.rmse_m)
            and registration.rmse_m <= self.options.max_registration_rmse_m
        )

    def _cut(self, times: FloatArray, poses: FloatArray) -> None:
        if len(times) < 2:
            return
        duration = self.options.window_duration_s
        for begin in np.arange(times[0], times[-1] + 1e-9, duration):
            mask = (times >= begin) & (times < begin + duration)
            if np.count_nonzero(mask) >= self.options.min_window_scans:
                block = len(self.windows)
                self.windows.append(
                    OdometryWindow(
                        window_id=f"{self.prefix}w{block}",
                        block=block,
                        times_s=times[mask],
                        poses=poses[mask],
                    )
                )


def collect_odometry_windows(
    scans: Iterable[ScanItem],
    options: WindowingOptions,
    *,
    covers: Callable[[float], bool] | None = None,
    prefix: str = "",
    max_scans: int | None = None,
    into: OdometrySegmenter | None = None,
    rotation_model: RotationModel | None = None,
) -> OdometrySegmenter:
    """Stream scans once and return the segmenter holding every window."""

    segmenter = into or OdometrySegmenter(options, covers, rotation_model=rotation_model)
    segmenter.prefix = prefix
    if covers is not None:
        segmenter.covers = covers
    for count, item in enumerate(scans):
        if max_scans is not None and count >= max_scans:
            break
        segmenter.add(*item)
    segmenter.flush()
    return segmenter
