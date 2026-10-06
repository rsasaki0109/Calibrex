"""Excitation analysis of the translation (lever-arm) estimators.

Both lever-arm estimators see the lever arm ``l`` through the rotation
between two poses of the same rigid body::

    (R_j - R_i) l = R_i (Exp(theta) - I) l,       theta = Log(R_i^T R_j)

(the accelerometer's double-integrated specific force for the IMU, the antenna
position for the GNSS antenna).  To first order ``(Exp(theta) - I) e_a =
theta x e_a``, whose norm is the component of ``theta`` *perpendicular* to the
axis ``a``.  So the information on the axis ``a`` of ``l`` is, before nuisance
parameters::

    I_a = sum_k |(R_j - R_i) e_a|^2 / sigma_k^2  ~  sum_k |theta_k,perp(a)|^2 / sigma_k^2

Rotation about ``a`` itself gives nothing on ``a``: yaw alone leaves the
vertical lever-arm component unobservable, and pure translation leaves all
three unobservable.  The data-only std of the axis (from the information matrix
the solver already builds and the noise level it estimates from its residuals)
is the actual information; its ratio to ``I_a`` is the efficiency lost to
nuisance parameters (gravity, velocity and accelerometer bias per segment for the
IMU; the map alignment of each window for the GNSS antenna).

For the same motion the information grows linearly with recording time, so the
std shrinks as 1/sqrt(time) and the recording needed to reach a bound is
``T * (std / bound)^2``.  When the motion does not excite the axis at all, that
extrapolation is meaningless; the module then says which rotation is missing and
how much of a reference motion (a ramp at a stated rate over the estimator's
pairing interval) would supply the missing information.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.excitation import AxisExcitation, ExcitationAxisName, ExcitationCause

FloatArray: TypeAlias = NDArray[np.float64]
AXES: tuple[ExcitationAxisName, ExcitationAxisName, ExcitationAxisName] = ("x", "y", "z")
Kind: TypeAlias = Literal["imu_lidar", "gnss_lidar"]

MAX_NEEDED_S = 3600.0
"""Repeating the recorded motion for longer than this is not a plan (cause ``no_excitation``)."""
REFERENCE_RATE_DPS = 30.0
"""Rotation rate of the reference motion that the recommendation is sized for."""
WEAK_ROTATION_DEG = 3.0
"""RMS rotation below which an axis counts as not rotated."""
WEAK_FRACTION = 0.25
"""An axis rotating less than this fraction of the strongest axis counts as not rotated."""
FALLBACK_EFFICIENCY = 0.25
MIN_RELIABLE_WINDOWS = 12
"""Fewer fitted windows than this make the jackknife, and so the extrapolated duration, indicative
only (on the validation recordings, predictions from fewer windows were off by up to 27x)."""


@dataclass(frozen=True)
class ExcitationInputs:
    """What an estimator hands over: its std, information and the recorded rotation.

    ``ideal_information`` is the 3x3 ``sum (R_j - R_i)^T (R_j - R_i) / sigma^2``
    (1/m^2) before nuisance parameters; ``rotation_moment`` the 3x3
    ``sum theta theta^T`` (rad^2) in the sensor frame of the lever arm, over the
    same ``samples`` paired poses spanning ``covered_s`` seconds of the fit.
    ``recording_s`` is the time the estimate used in total (fit and held-out),
    ``interval_s`` the mean time between the two poses of a pair.
    ``std_scaling_m`` is the std that shrinks with more data (the larger of the
    analytic and the jackknife std); ``sensitivity_m`` the part that does not.
    """

    kind: Kind
    std_analytic_m: FloatArray
    std_scaling_m: FloatArray
    std_reported_m: FloatArray
    bound_m: float
    ideal_information: FloatArray
    rotation_moment: FloatArray
    samples: int
    covered_s: float
    recording_s: float
    interval_s: float
    sensitivity_m: FloatArray | None = None
    windows: int = 0
    """Odometry windows the fit used (0: unknown)."""


def rotation_moment(rotation_vectors: FloatArray) -> FloatArray:
    """``sum theta theta^T`` of ``(n, 3)`` rotation vectors."""

    vectors = np.asarray(rotation_vectors, dtype=np.float64).reshape(-1, 3)
    return np.asarray(vectors.T @ vectors, dtype=np.float64)


def relative_rotation_vectors(first: FloatArray, second: FloatArray) -> FloatArray:
    """Body-frame ``Log(R_first^T R_second)`` of two ``(n, 3, 3)`` rotation stacks."""

    from scipy.spatial.transform import Rotation

    relative = np.transpose(first, (0, 2, 1)) @ second
    return np.asarray(Rotation.from_matrix(relative).as_rotvec(), dtype=np.float64)


def ideal_information(
    first: FloatArray, second: FloatArray, weights: FloatArray | float = 1.0
) -> FloatArray:
    """``sum w (R_j - R_i)^T (R_j - R_i)`` for ``(n, 3, 3)`` rotation stacks."""

    difference = np.asarray(second - first, dtype=np.float64)
    w = np.broadcast_to(np.asarray(weights, dtype=np.float64), (len(difference),))
    return np.asarray(np.einsum("k,kia,kib->ab", w, difference, difference), dtype=np.float64)


def format_duration(seconds: float) -> str:
    """``45 s``, ``12 min`` or ``3.5 h``."""

    if not math.isfinite(seconds):
        return "an unbounded time"
    if seconds < 90.0:
        return f"{max(round(seconds), 1)} s"
    if seconds < 7200.0:
        return f"{round(seconds / 60.0)} min"
    if seconds < 360000.0:
        return f"{seconds / 3600.0:.1f} h"
    return "more than 100 h"


def _cm(value: float) -> str:
    return f"{100.0 * value:.2g} cm" if math.isfinite(value) else "inf"


def diagnose_translation_axes(inputs: ExcitationInputs) -> list[AxisExcitation]:
    """One :class:`AxisExcitation` per lever-arm axis."""

    samples = max(inputs.samples, 1)
    rms_deg = [
        math.degrees(math.sqrt(max(float(inputs.rotation_moment[index, index]), 0.0) / samples))
        for index in range(3)
    ]
    efficiency = _efficiencies(inputs)
    reference = [value for value in efficiency if value is not None and value > 0.0]
    fallback = float(np.median(reference)) if reference else FALLBACK_EFFICIENCY
    trace = float(np.trace(inputs.rotation_moment))
    weight = float(np.trace(inputs.ideal_information)) / (2.0 * trace) if trace > 0.0 else 0.0
    weak = [value < max(WEAK_ROTATION_DEG, WEAK_FRACTION * max(rms_deg)) for value in rms_deg]
    dominant = int(np.argmax(rms_deg))
    results = []
    for index, name in enumerate(AXES):
        perpendicular = [other for other in range(3) if other != index]
        weak_axes = [AXES[other] for other in perpendicular if weak[other]]
        std = float(inputs.std_analytic_m[index])
        scaling = float(inputs.std_scaling_m[index])
        ratio = scaling / inputs.bound_m if math.isfinite(scaling) else math.inf
        needed = inputs.recording_s * ratio**2 if math.isfinite(ratio) else None
        sensitivity = None if inputs.sensitivity_m is None else float(inputs.sensitivity_m[index])
        cause = _cause(inputs, scaling, float(inputs.std_reported_m[index]), needed)
        missing = weak_axes or [AXES[other] for other in perpendicular]
        perp_rms = math.sqrt(sum(rms_deg[other] ** 2 for other in perpendicular) / 2.0)
        text = _recommendation(
            inputs,
            name,
            cause,
            std=scaling,
            reported=float(inputs.std_reported_m[index]),
            sensitivity=sensitivity,
            needed=needed,
            rms_deg=rms_deg,
            perp_rms=perp_rms,
            missing=missing,
            planar=index == dominant and len(weak_axes) == 2,
            added_s=_added_duration_s(inputs, index, weight, efficiency[index] or fallback),
        )
        results.append(
            AxisExcitation(
                name=name,
                cause=cause,
                predicted_std_m=std if math.isfinite(std) else None,
                scaling_std_m=scaling if math.isfinite(scaling) else None,
                bound_std_m=inputs.bound_m,
                information_efficiency=efficiency[index],
                rotation_rms_deg=rms_deg,
                perpendicular_rotation_rms_deg=perp_rms,
                recording_s=inputs.recording_s,
                fit_windows=inputs.windows or None,
                needed_duration_s=needed,
                missing_rotation_axes=list(missing)
                if cause in {"no_excitation", "insufficient_duration"}
                else [],
                recommendation=text,
            )
        )
    return results


def _efficiencies(inputs: ExcitationInputs) -> list[float | None]:
    result: list[float | None] = []
    for index in range(3):
        std = float(inputs.std_analytic_m[index])
        ideal = float(inputs.ideal_information[index, index])
        if not math.isfinite(std) or std <= 0.0 or ideal <= 0.0:
            result.append(None)
            continue
        result.append(float(min(1.0, (1.0 / std**2) / ideal)))
    return result


def _cause(
    inputs: ExcitationInputs, scaling: float, reported: float, needed: float | None
) -> ExcitationCause:
    if not math.isfinite(scaling) or needed is None:
        return "unsolved"
    if reported <= inputs.bound_m:
        return "observable"
    if scaling <= inputs.bound_m:
        return "model_limited"
    if needed <= MAX_NEEDED_S:
        return "insufficient_duration"
    return "no_excitation"


def _added_duration_s(
    inputs: ExcitationInputs, index: int, weight: float, efficiency: float
) -> float | None:
    """Seconds of the reference motion that would supply the missing information."""

    std = float(inputs.std_scaling_m[index])
    have = 0.0 if not math.isfinite(std) or std <= 0.0 else 1.0 / std**2
    missing = max(1.0 / inputs.bound_m**2 - have, 0.0)
    if inputs.covered_s <= 0.0 or weight <= 0.0 or inputs.interval_s <= 0.0:
        return None
    rate = math.radians(REFERENCE_RATE_DPS)
    # A ramp of rate w over the pairing interval t has mean theta^2 = (w t)^2 / 3.
    per_second = (
        efficiency * weight * (inputs.samples / inputs.covered_s) * (rate * inputs.interval_s) ** 2
    ) / 3.0
    return missing / per_second if per_second > 0.0 else None


def _recommendation(
    inputs: ExcitationInputs,
    name: str,
    cause: ExcitationCause,
    *,
    std: float,
    reported: float,
    sensitivity: float | None,
    needed: float | None,
    rms_deg: Sequence[float],
    perp_rms: float,
    missing: Sequence[str],
    planar: bool,
    added_s: float | None,
) -> str:
    bound = _cm(inputs.bound_m)
    if cause == "observable":
        return f"{name} is observable (std {_cm(std)}, bound {bound})."
    if cause == "unsolved":
        need = (
            "IMU-covered LiDAR odometry windows"
            if inputs.kind == "imu_lidar"
            else "RTK-fixed LiDAR odometry windows"
        )
        return (
            f"{name}: no solution; the recording has too few windows for the lever arm "
            f"(it needs {need})."
        )
    if cause == "model_limited":
        reason = "segment-duration sensitivity" if sensitivity else "jackknife spread"
        return (
            f"{name}: the data alone would reach the {bound} bound (std {_cm(std)}), but the "
            f"reported std {_cm(reported)} also covers modelling error ({reason}), which more "
            "of the same motion does not remove."
        )
    thin = (
        f" This is extrapolated from only {inputs.windows} windows, so indicative only."
        if 0 < inputs.windows < MIN_RELIABLE_WINDOWS
        else ""
    )
    caution = (
        f" Caution: the segment-duration sensitivity {_cm(sensitivity)} (modelling error) is "
        "also over the bound and may not shrink with more of the same motion."
        if sensitivity is not None and sensitivity > inputs.bound_m
        else ""
    )
    if cause == "insufficient_duration":
        assert needed is not None
        return (
            f"{name}: the motion excites it ({perp_rms:.0f} deg RMS rotation about the other "
            f"axes) but too briefly: std {_cm(std)} against the {bound} bound. About "
            f"{format_duration(needed)} of the same motion would reach it (this recording: "
            f"{format_duration(inputs.recording_s)}).{thin}{caution}"
        )
    axes = " or ".join(missing)
    spread = "/".join(f"{value:.0f}" for value in rms_deg)
    have = f"The recording rotates only {spread} deg RMS about x/y/z."
    hint = (
        "a level sensor turns for z and tilts for x or y"
        if inputs.kind == "imu_lidar"
        else "for a level rig: turns (figure-eights, S-curves) for z, tilting for x or y"
    )
    motion = f"rotation about the sensor {axes} axis ({hint})"
    need = f"{name}: needs rotation about the sensor {axes} axis."
    reach = (
        f" About {format_duration(added_s)} of {motion} at >= {REFERENCE_RATE_DPS:.0f} deg/s "
        f"would reach the {bound} bound."
        if added_s is not None and added_s <= MAX_NEEDED_S
        else f" Record {motion}."
    )
    text = f"{need} {have}{reach}{thin}{caution}"
    if planar:
        text += (
            f" The motion is rotation about {name} alone (planar), which cannot excite {name}: "
            "measure it and give it with --tf."
        )
    return text
