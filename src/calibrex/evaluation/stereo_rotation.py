"""A stereo pair's relative rotation, judged from its images alone.

Rectifying a pair with the candidate ``T_left_right`` puts every scene point on the same
row of both images when the candidate is right. When it is wrong by a small rotation of
the right camera, ``delta`` (rectified axes: x along the baseline, z the optical axis),
the right image's rows move by::

    dy = -f delta_x (1 + y^2 / f^2) + delta_z x + delta_y x y / f

at rectified pixel ``(x, y)`` from the centre. A rotation about the baseline shifts every
row, one about the optical axis tilts them across the image, and one about the vertical
axis bends them towards the corners, which wide (fisheye) images reach. A vertical
component of the baseline adds a term in disparity, fitted as a nuisance.

:func:`estimate_stereo_rotation` matches textured patches between the rectified images
(:func:`calibrex.evaluation.fisheye_stereo.vertical_offset_samples`), fits the pattern
by trimmed least squares, and reports the candidate's error about the left camera's
axes with the ``calibrex check`` convention (``rotvec(R_candidate R_estimate^T)``). Its
uncertainty is a block jackknife over frames; a control turns the candidate by
``control_deg`` about each axis and re-measures, so an axis the images do not constrain
is reported as such. The translation is not judged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.core.stereo_rotation import (
    StereoRotationArtifact,
    StereoRotationAxis,
    StereoRotationProvenance,
)
from calibrex.data import ros_cdr
from calibrex.data.ros2_camera_imu import to_gray
from calibrex.data.rosbag2 import iter_messages
from calibrex.evaluation.fisheye_stereo import (
    VERTICAL_OFFSET_COLUMNS,
    EquidistantCamera,
    FisheyeStereoRectification,
    RadtanCamera,
    StereoCamera,
    rectify_fisheye_pair,
    vertical_offset_samples,
)
from calibrex.evaluation.visual_rotation import CameraModel

FloatArray: TypeAlias = NDArray[np.float64]
AXES: tuple[Literal["roll", "pitch", "yaw"], ...] = ("roll", "pitch", "yaw")


@dataclass(frozen=True)
class StereoRotationOptions:
    """Frame selection, matching and evidence settings."""

    max_pairs: int = 48
    spacing_s: float = 1.0
    max_seconds: float | None = None
    max_gap_s: float = 0.002
    """Images further apart than this are not a pair (2 ms: hardware-synchronised)."""
    max_unsynchronised_gap_s: float = 0.03
    """The pairing tolerance when the rig's rotation is known (:class:`StereoTiming`)."""
    max_time_std_s: float = 0.002
    patches_per_pair: int = 300
    max_disparity_px: int = 60
    blocks: int = 6
    min_patches: int = 500
    control_deg: float = 0.2
    min_response: float = 0.5
    max_response: float = 1.5
    max_std_deg: float = 0.1

    def __post_init__(self) -> None:
        if self.blocks < 3 or self.max_pairs < self.blocks:
            raise ValueError("need three jackknife blocks and at least as many pairs")
        if (
            self.control_deg <= 0.0
            or self.spacing_s < 0.0
            or self.max_gap_s <= 0.0
            or self.max_unsynchronised_gap_s < self.max_gap_s
        ):
            raise ValueError("control_deg and max_gap_s must be positive, spacing_s not negative")


@dataclass(frozen=True)
class StereoTiming:
    """When each pair's images were stamped and how the rig turned at the time.

    ``gaps_s`` is the right image's stamp minus the left one's, per pair, and
    ``rates_rad_s`` the left camera's angular rate about its own axes at the pair (from a
    gyro). With them, a pair taken ``gap + time_offset`` apart while the rig turns at
    ``omega`` reads the candidate's error as ``error - omega (gap + time_offset)``, and the
    fit recovers both the error and the cameras' clock offset.
    """

    gaps_s: FloatArray
    rates_rad_s: FloatArray
    source: str


@dataclass(frozen=True)
class StereoFrames:
    """Image pairs with the stamps of both images (seconds)."""

    pairs: list[tuple[FloatArray, FloatArray]]
    left_times_s: FloatArray
    right_times_s: FloatArray


@dataclass(frozen=True)
class _Measurement:
    error_deg: FloatArray
    std_deg: FloatArray
    time_offset_s: float | None
    time_offset_std_s: float | None
    patches: int
    rows_known: int
    vertical_baseline_ratio: float
    residual_std_px: float
    rectification: FisheyeStereoRectification


def stereo_camera(model: CameraModel, width: int, height: int) -> StereoCamera:
    """The rectification camera of a check camera model and its image size."""

    if model.distortion_model == "equidistant":
        k = (*model.distortion, 0.0, 0.0, 0.0, 0.0)
        return EquidistantCamera(
            model.fx, model.fy, model.cx, model.cy, (k[0], k[1], k[2], k[3]), width, height
        )
    return RadtanCamera(
        model.fx, model.fy, model.cx, model.cy, tuple(model.distortion), width, height
    )


def read_stereo_pairs(
    bag: str | Path,
    left_topic: str,
    right_topic: str,
    options: StereoRotationOptions | None = None,
) -> list[tuple[FloatArray, FloatArray]]:
    """Grayscale image pairs taken within ``max_gap_s`` of each other, ``spacing_s`` apart."""

    return read_stereo_frames(bag, left_topic, right_topic, options).pairs


def read_stereo_frames(
    bag: str | Path,
    left_topic: str,
    right_topic: str,
    options: StereoRotationOptions | None = None,
    *,
    max_gap_s: float | None = None,
) -> StereoFrames:
    """Image pairs, each image with its nearest other within ``max_gap_s``, and their stamps.

    ``max_gap_s`` defaults to the options' synchronised tolerance. Pairs are at least
    ``spacing_s`` apart, at most ``max_pairs``, from the first ``max_seconds`` of the bag
    when given.
    """

    opts = options or StereoRotationOptions()
    gap_ns = round((opts.max_gap_s if max_gap_s is None else max_gap_s) * 1.0e9)
    pending: dict[str, list[tuple[int, int, bytes]]] = {left_topic: [], right_topic: []}
    pairs: list[tuple[FloatArray, FloatArray]] = []
    times: list[tuple[float, float]] = []
    first_ns: int | None = None
    last_ns: int | None = None
    for connection, stamp, payload in iter_messages(bag, topics={left_topic, right_topic}):
        topic = connection.topic
        header = ros_cdr.decode_ros2_image(topic, stamp, payload, include_data=False)
        time_ns = header.timestamp_ns
        first_ns = time_ns if first_ns is None else first_ns
        if opts.max_seconds is not None and time_ns - first_ns > opts.max_seconds * 1.0e9:
            break
        other = right_topic if topic == left_topic else left_topic
        near = [item for item in pending[other] if abs(item[0] - time_ns) <= gap_ns]
        if not near:
            pending[topic] = [*pending[topic][-4:], (time_ns, stamp, payload)]
            continue
        match = min(near, key=lambda item: abs(item[0] - time_ns))
        pending[other] = [item for item in pending[other] if item[0] > match[0]]
        if last_ns is not None and time_ns - last_ns < opts.spacing_s * 1.0e9:
            continue
        last_ns = time_ns
        images = {topic: (stamp, payload), other: (match[1], match[2])}
        stamps = {topic: time_ns, other: match[0]}
        pairs.append(
            (_gray(left_topic, *images[left_topic]), _gray(right_topic, *images[right_topic]))
        )
        times.append((stamps[left_topic] * 1.0e-9, stamps[right_topic] * 1.0e-9))
        if len(pairs) >= opts.max_pairs:
            break
    stamps_s = np.asarray(times, dtype=np.float64).reshape(-1, 2)
    return StereoFrames(pairs, stamps_s[:, 0], stamps_s[:, 1])


def estimate_stereo_rotation(
    pairs: Sequence[tuple[FloatArray, FloatArray]],
    left: StereoCamera,
    right: StereoCamera,
    candidate: FloatArray,
    *,
    provenance: StereoRotationProvenance,
    options: StereoRotationOptions | None = None,
    timing: StereoTiming | None = None,
) -> StereoRotationArtifact:
    """The candidate ``T_left_right``'s rotation error about the left camera's axes.

    With ``timing``, the cameras' clock offset is fitted with it (see :class:`StereoTiming`).
    """

    opts = options or StereoRotationOptions()
    candidate = np.asarray(candidate, dtype=np.float64)

    def artifact(
        status: Literal["pass", "warn", "fail", "inconclusive"],
        reason: str,
        measured: _Measurement | None,
        axes: Sequence[StereoRotationAxis] = (),
    ) -> StereoRotationArtifact:
        return StereoRotationArtifact(
            policy_status=status,
            policy_reasons=[reason],
            frames=len(pairs),
            patches=measured.patches if measured else 0,
            patches_constraining_rows=measured.rows_known if measured else 0,
            baseline_m=float(np.linalg.norm(candidate[:3, 3])),
            rectified_focal_px=measured.rectification.fx if measured else float(left.fx),
            rectified_from_left=(
                measured.rectification.rectified_from_left if measured else np.eye(3)
            ).tolist(),
            axes=list(axes),
            vertical_baseline_ratio=measured.vertical_baseline_ratio if axes and measured else None,
            residual_std_px=measured.residual_std_px if axes and measured else None,
            time_offset_s=measured.time_offset_s if axes and measured else None,
            time_offset_std_s=measured.time_offset_std_s if axes and measured else None,
            time_offset_status=time_status(measured) if axes and measured and timing else None,
            timing_source=timing.source if timing else None,
            max_stamp_gap_s=float(np.abs(timing.gaps_s).max()) if timing and len(pairs) else None,
            options=asdict(opts),
            limitations=[
                "rotation only: the translation between the cameras is not judged",
                "the images must share texture that varies across rows (blank walls do not)",
            ],
            provenance=provenance,
        )

    def time_status(measured: _Measurement) -> Literal["estimated", "unobservable"]:
        std = measured.time_offset_std_s
        return "estimated" if std is not None and std <= opts.max_time_std_s else "unobservable"

    if timing is not None and (
        len(timing.gaps_s) != len(pairs) or np.shape(timing.rates_rad_s) != (len(pairs), 3)
    ):
        raise ValueError("timing needs one gap and one 3-axis rate per pair")
    if len(pairs) < opts.blocks:
        return artifact(
            "inconclusive",
            f"{len(pairs)} synchronised image pairs, fewer than {opts.blocks}",
            None,
        )
    base = _measure(pairs, left, right, candidate, opts, timing)
    if base.rows_known < opts.min_patches:
        return artifact(
            "inconclusive",
            f"{base.rows_known} matched patches constrain rows, fewer than {opts.min_patches}",
            base,
        )
    axes: list[StereoRotationAxis] = []
    for index, name in enumerate(AXES):
        turned = candidate.copy()
        turn = Rotation.from_rotvec(np.radians(opts.control_deg) * np.eye(3)[index]).as_matrix()
        turned[:3, :3] = turn @ candidate[:3, :3]
        moved = _measure(pairs, left, right, turned, opts, timing)
        response = float((moved.error_deg[index] - base.error_deg[index]) / opts.control_deg)
        std = float(base.std_deg[index])
        problems = []
        if std > opts.max_std_deg:
            problems.append(f"jackknife std {std:.3f} deg exceeds {opts.max_std_deg:g} deg")
        if not opts.min_response <= response <= opts.max_response:
            problems.append(
                f"turning the candidate {opts.control_deg:g} deg moved the estimate by "
                f"{response * opts.control_deg:+.3f} deg (response {response:.2f}, outside "
                f"{opts.min_response:g} to {opts.max_response:g})"
            )
        axes.append(
            StereoRotationAxis(
                name=name,
                candidate_error_deg=float(base.error_deg[index]),
                std_deg=std,
                control_response=response,
                observable=not problems,
                reason="; ".join(problems)
                or f"the {opts.control_deg:g} deg control moved the estimate by "
                f"{response * opts.control_deg:+.3f} deg",
            )
        )
    observable = [axis.name for axis in axes if axis.observable]
    return artifact(
        "pass" if len(observable) == 3 else "warn" if observable else "inconclusive",
        f"{base.rows_known} of {base.patches} matched patches constrain rows; observable: "
        + (", ".join(observable) or "none"),
        base,
        axes,
    )


def _measure(
    pairs: Sequence[tuple[FloatArray, FloatArray]],
    left: StereoCamera,
    right: StereoCamera,
    candidate: FloatArray,
    options: StereoRotationOptions,
    timing: StereoTiming | None = None,
) -> _Measurement:
    rectification = rectify_fisheye_pair(left, right, _inverse(candidate))
    rectified = [(rectification.rectify_left(a), rectification.rectify_right(b)) for a, b in pairs]
    samples = vertical_offset_samples(
        rectified,
        patches_per_pair=options.patches_per_pair,
        max_disparity_px=options.max_disparity_px,
    )
    column = {name: samples[:, i] for i, name in enumerate(VERTICAL_OFFSET_COLUMNS)}
    known = column["vertical_texture"] > 0.5
    f = rectification.fx
    x = column["column_px"][known] - rectification.cx
    y = column["row_px"][known] - rectification.cy
    pair = column["pair"][known].astype(int)
    # row offset per radian of the right camera's turn about the rectified x, y, z axes,
    # mapped to the candidate's error about the left camera's axes (rectified = Rr left)
    rectified_rows = f * np.c_[-(1.0 + (y / f) ** 2), x * y / f**2, x / f]
    rotation = rectified_rows @ rectification.rectified_from_left
    values = column["offset_px"][known]
    columns = [rotation, column["disparity_px"][known][:, None]]
    if timing is not None:
        rates = np.asarray(timing.rates_rad_s, dtype=np.float64)[pair]
        gaps = np.asarray(timing.gaps_s, dtype=np.float64)[pair]
        turned = (rotation * rates).sum(axis=1)  # row offset per second of time offset
        values = values + turned * gaps  # the stamped gap's turn, known
        columns.append(-turned[:, None])
    design = np.hstack(columns)

    def fit(rows: NDArray[np.bool_]) -> tuple[FloatArray, float]:
        return _trimmed_fit(design[rows], values[rows])

    if len(values) < design.shape[1] + 1:
        nan = np.full(3, np.nan)
        return _Measurement(
            nan, nan, None, None, len(samples), int(known.sum()), np.nan, np.nan, rectification
        )
    coefficients, residual = fit(np.ones(len(values), dtype=bool))
    blocks = np.array_split(np.arange(len(pairs)), options.blocks)
    jackknife = np.array([fit(~np.isin(pair, block))[0] for block in blocks])
    count = len(blocks)
    std = np.sqrt((count - 1) / count * ((jackknife - jackknife.mean(axis=0)) ** 2).sum(axis=0))
    return _Measurement(
        error_deg=np.degrees(coefficients[:3]),
        std_deg=np.degrees(std[:3]),
        time_offset_s=float(coefficients[4]) if timing is not None else None,
        time_offset_std_s=float(std[4]) if timing is not None else None,
        patches=len(samples),
        rows_known=int(known.sum()),
        vertical_baseline_ratio=float(coefficients[3]),
        residual_std_px=residual,
        rectification=rectification,
    )


def _trimmed_fit(design: FloatArray, values: FloatArray) -> tuple[FloatArray, float]:
    keep = np.ones(len(values), dtype=bool)
    coefficients = np.zeros(design.shape[1])
    spread = 0.0
    for _ in range(4):
        coefficients, *_ = np.linalg.lstsq(design[keep], values[keep], rcond=None)
        error = values - design @ coefficients
        spread = 1.4826 * float(np.median(np.abs(error[keep])))
        keep = np.abs(error) <= 3.0 * max(spread, 1.0e-6)
    return coefficients, spread


def _inverse(transform: FloatArray) -> FloatArray:
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def _gray(topic: str, stamp: int, payload: bytes) -> FloatArray:
    image = ros_cdr.decode_ros2_image(topic, stamp, payload)
    if image.data is None:
        raise ValueError(f"{topic}: image without data")
    gray = to_gray(
        bytes(image.data),
        height=image.height,
        width=image.width,
        step=image.step,
        encoding=image.encoding,
    )
    return np.asarray(gray, dtype=np.float64)
