"""Find the clock offset between a camera recorded apart from the IMU and the IMU.

The speed of rotation does not depend on the axes it is measured in: the camera's
angular speed ``|omega_cam(t)|`` equals the gyro's ``|omega_imu(t + offset)|`` whatever
the mounting rotation is. So the offset is found before the mounting is known, by
correlating the two speed signals over every lag where they overlap by enough.

The camera speed comes from the tracked frame-to-frame rotations
(:func:`~calibrex.evaluation.visual_rotation.track_camera_rotations`); both signals are
averaged onto one time grid and compared with a Pearson correlation per lag. The best
lag is accepted only if it is clearly better than any other lag more than
``exclusion_s`` away (a motion that repeats would otherwise give a confident wrong
answer); the fine offset is left to the camera-IMU estimator, which fits its own
time offset afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ClockAlignmentOptions:
    """Grid, search and acceptance settings (the thresholds are heuristics)."""

    grid_s: float = 0.02
    min_overlap_fraction: float = 0.5
    """Lags where less than this share of the camera recording overlaps the IMU are skipped."""
    exclusion_s: float = 1.0
    min_correlation: float = 0.5
    min_margin: float = 0.15
    """The best lag must beat the best lag outside ``exclusion_s`` by this much."""


@dataclass(frozen=True)
class ClockAlignment:
    """``camera time + offset_s = IMU time``, and how sure the search is about it."""

    offset_s: float | None
    correlation: float
    runner_up: float
    overlap_s: float
    accepted: bool
    reason: str


def camera_angular_speed(
    times_s: FloatArray, orientations: FloatArray, segment: NDArray[np.int64]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Interval starts, ends and angular speed (rad/s) between consecutive tracked frames."""

    t = np.asarray(times_s, dtype=np.float64)
    if len(t) < 2:
        empty = np.zeros(0)
        return empty, empty, empty
    same = segment[1:] == segment[:-1]
    relative = Rotation.from_matrix(np.einsum("nji,njk->nik", orientations[:-1], orientations[1:]))
    angle = np.linalg.norm(relative.as_rotvec(), axis=1)
    dt = np.diff(t)
    keep = same & (dt > 0.0)
    return t[:-1][keep], t[1:][keep], angle[keep] / dt[keep]


def _bin_mean(times: FloatArray, values: FloatArray, start: float, n: int, step: float):  # type: ignore[no-untyped-def]
    index = np.floor((times - start) / step).astype(np.int64)
    inside = (index >= 0) & (index < n)
    total = np.bincount(index[inside], weights=values[inside], minlength=n)
    count = np.bincount(index[inside], minlength=n).astype(np.float64)
    mean = np.divide(total, count, out=np.zeros(n), where=count > 0)
    return mean, count > 0


def _masked_pearson(x: FloatArray, mx: FloatArray, y: FloatArray, my: FloatArray):  # type: ignore[no-untyped-def]
    """Pearson correlation of ``x`` shifted by every lag against ``y`` over shared valid bins.

    Returns ``(lags, r, overlap_bins)``; lag ``k`` pairs ``x[i]`` with ``y[i + k]``.
    """

    n = len(x) + len(y) - 1
    size = 1 << int(np.ceil(np.log2(max(n, 2))))

    def corr(a: FloatArray, b: FloatArray) -> FloatArray:
        fa = np.fft.rfft(a[::-1], size)
        fb = np.fft.rfft(b, size)
        return np.asarray(np.fft.irfft(fa * fb, size)[:n], dtype=np.float64)

    xm, ym = x * mx, y * my
    count = corr(mx, my)
    sx, sy = corr(xm, my), corr(mx, ym)
    sxx, syy = corr(xm * x, my), corr(mx, ym * y)
    sxy = corr(xm, ym)
    lags = np.arange(n) - (len(x) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / count
        vx = sxx - sx**2 / count
        vy = syy - sy**2 / count
        r = cov / np.sqrt(vx * vy)
    r[~np.isfinite(r) | (count < 3)] = -np.inf
    return lags, r, np.rint(count)


def align_camera_clock(
    camera_times_s: FloatArray,
    camera_orientations: FloatArray,
    camera_segment: NDArray[np.int64],
    gyro_times_s: FloatArray,
    gyro_rps: FloatArray,
    options: ClockAlignmentOptions | None = None,
) -> ClockAlignment:
    """The offset that best lines the camera's angular speed up with the gyro's."""

    opts = options or ClockAlignmentOptions()
    starts, ends, speed = camera_angular_speed(camera_times_s, camera_orientations, camera_segment)
    if len(speed) < 10:
        return ClockAlignment(None, 0.0, 0.0, 0.0, False, "fewer than 10 tracked frame pairs")
    step = opts.grid_s
    c0 = float(starts[0])
    nc = int(np.ceil((float(ends[-1]) - c0) / step))
    middle = 0.5 * (starts + ends)
    cam, cam_ok = _bin_mean(middle, speed, c0, nc, step)
    g_t = np.asarray(gyro_times_s, dtype=np.float64)
    g0 = float(g_t[0])
    ng = int(np.ceil((float(g_t[-1]) - g0) / step))
    gyro, gyro_ok = _bin_mean(g_t, np.linalg.norm(gyro_rps, axis=1), g0, ng, step)
    lags, r, overlap = _masked_pearson(cam, cam_ok.astype(float), gyro, gyro_ok.astype(float))
    need = opts.min_overlap_fraction * float(cam_ok.sum())
    r = np.where(overlap >= need, r, -np.inf)
    if not np.isfinite(r).any():
        return ClockAlignment(
            None, 0.0, 0.0, 0.0, False,
            f"the camera ({nc * step:.0f} s) and IMU ({ng * step:.0f} s) recordings never "
            f"overlap by {opts.min_overlap_fraction:.0%} of the camera recording",
        )  # fmt: skip
    best = int(np.argmax(r))
    shift = 0.0
    if 0 < best < len(r) - 1 and np.isfinite(r[best - 1]) and np.isfinite(r[best + 1]):
        a, b, c = r[best - 1], r[best], r[best + 1]
        denominator = a - 2.0 * b + c
        shift = 0.5 * (a - c) / denominator if denominator < 0.0 else 0.0
    offset = g0 - c0 + (lags[best] + shift) * step
    far = np.abs(lags - lags[best]) * step > opts.exclusion_s
    runner_up = float(np.max(r[far])) if np.isfinite(r[far]).any() else -1.0
    peak = float(r[best])
    overlap_s = max(0.0, min(float(ends[-1]) + offset, float(g_t[-1])) - max(c0 + offset, g0))
    if peak < opts.min_correlation:
        return ClockAlignment(
            None, peak, runner_up, overlap_s, False,
            f"the camera and gyro rotation speeds correlate at most {peak:.2f} "
            f"(needs {opts.min_correlation:g}): too little rotation, or not the same rig",
        )  # fmt: skip
    if peak - runner_up < opts.min_margin:
        return ClockAlignment(
            None, peak, runner_up, overlap_s, False,
            f"two clock offsets fit almost equally well ({peak:.2f} and {runner_up:.2f} more "
            f"than {opts.exclusion_s:g} s apart): the motion repeats; give the offset as "
            "NAME=PATH@SECONDS",
        )  # fmt: skip
    return ClockAlignment(
        float(offset), peak, runner_up, overlap_s, True,
        f"camera clock + {offset:.3f} s = IMU clock (rotation-speed correlation {peak:.2f}, "
        f"next best {runner_up:.2f}, {overlap_s:.0f} s overlap)",
    )  # fmt: skip
