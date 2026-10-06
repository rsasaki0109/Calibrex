"""Excitation diagnosis of one translation (lever-arm) axis.

A translation axis is observable only when the platform rotates about the
other two sensor axes: both lever-arm estimators (the accelerometer's and the
GNSS antenna's) see the lever arm through ``(R_j - R_i) l``, which is zero for
pure translation and for rotation about the axis of ``l`` itself.  This record
says, for one axis of one recording, how much of that rotation there was, what
the data-only std is predicted to be, how long the same motion would have to be
recorded to reach the observability bound, and what to do about it.

It is an optional addition to estimator evidence and to the ``bag_estimate`` and
``calibration_check`` axis records.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from calibrex.core.result import StrictModel

ExcitationCause = Literal[
    "observable",
    "insufficient_duration",
    "no_excitation",
    "model_limited",
    "unsolved",
]
ExcitationAxisName = Literal["x", "y", "z"]


class AxisExcitation(StrictModel):
    """Why one translation axis is (un)observable on this recording and what would change it.

    ``cause`` is ``observable`` (the reported std is within the bound),
    ``insufficient_duration`` (the motion excites the axis, but the recording is
    too short: ``needed_duration_s`` of the same motion would reach the bound),
    ``no_excitation`` (repeating this motion would take more than an hour, so
    different motion is needed: ``missing_rotation_axes``), ``model_limited``
    (the data-only std is within the bound but the reported std, which also
    covers modelling error, is not) or ``unsolved``.

    ``predicted_std_m`` is the data-only (analytic) std, from the information
    matrix and the residual noise level.  The std scales as 1/sqrt(time) for the
    same motion, so ``needed_duration_s = recording_s * (std / bound)^2`` with
    ``scaling_std_m``, the larger of the analytic and jackknife std.  Rotation is in the
    sensor frame of the lever arm (the LiDAR frame).
    """

    name: ExcitationAxisName
    cause: ExcitationCause
    predicted_std_m: float | None = Field(default=None, ge=0.0)
    scaling_std_m: float | None = Field(
        default=None,
        ge=0.0,
        description="the larger of the analytic and jackknife std: the one that shrinks with "
        "more data and that needed_duration_s extrapolates (the analytic std alone is optimistic "
        "on real data because the odometry residuals are correlated)",
    )
    bound_std_m: float = Field(gt=0.0)
    information_efficiency: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="share of the rotation-implied information that survives the nuisance "
        "parameters (gravity, velocity, accelerometer bias; the per-window map alignment)",
    )
    rotation_rms_deg: list[float] = Field(
        min_length=3,
        max_length=3,
        description="RMS rotation between the paired poses about the sensor x, y, z axes",
    )
    perpendicular_rotation_rms_deg: float = Field(
        ge=0.0, description="RMS rotation about the two axes perpendicular to this one"
    )
    recording_s: float = Field(ge=0.0)
    fit_windows: int | None = Field(
        default=None,
        ge=0,
        description="odometry windows the fit used; the duration extrapolation is indicative "
        "only below about a dozen",
    )
    needed_duration_s: float | None = Field(
        default=None,
        ge=0.0,
        description="recording time of the same motion that would reach the bound; absent when "
        "the std is not finite",
    )
    missing_rotation_axes: list[ExcitationAxisName] = Field(default_factory=list)
    recommendation: str
