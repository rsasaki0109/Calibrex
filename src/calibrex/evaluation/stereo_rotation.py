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
    min_shared_view: float = 0.05
    """Pairs whose cameras share less of the left camera's view are not checked."""
    patches_per_pair: int = 300
    max_disparity_px: int = 60
    blocks: int = 6
    min_patches: int = 500
    control_deg: float = 0.2
    min_response: float = 0.5
    max_response: float = 1.5
    max_std_deg: float = 0.1
    rows_px: int = 4
    """Rows searched each way when matching a patch on the final measurement."""
    coarse_rows_px: int = 12
    """Rows searched each way on a first, coarse measurement (about 2 deg at 350 px)."""
    coarse_patches_per_pair: int = 100
    coarse_min_disparity_px: int = -30
    """The coarse search also looks right: a turn about the rectified y axis moves far
    points to negative disparity (about 6 px per degree at 350 px)."""
    recentre_deg: float = 0.2
    """An error beyond this re-centres the rectification on the estimate and measures again."""
    max_recentres: int = 3
    fit_row_scale: bool = True
    """Fit a row offset proportional to the row: the two rectified images' vertical scales
    differ when one camera's focal length or distortion is slightly off. Where the matched
    patches sit on one side of the image (cameras far apart), it can hardly be told from a
    turn: the images then leave it to ``row_scale_prior``."""
    row_scale_prior: float = 0.003
    """Std of that scale mismatch before the images are seen (0.3%: a pixel at 350 px)."""

    def __post_init__(self) -> None:
        if self.rows_px < 1 or self.coarse_rows_px < self.rows_px or self.max_recentres < 0:
            raise ValueError("need rows_px >= 1, coarse_rows_px >= rows_px, max_recentres >= 0")
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
    row_scale_ratio: float | None = None
    row_scale_ratio_std: float | None = None
    noise_std_deg: FloatArray | None = None
    """The jackknife part of ``std_deg`` (the images' noise), without the scale's."""
    scale_std_deg: FloatArray | None = None


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


def shared_view_fraction(
    left: StereoCamera,
    right: StereoCamera,
    candidate: FloatArray,
    *,
    samples: int = 20000,
    range_m: float = 10.0,
) -> float:
    """The share of the left camera's view the right camera also sees, under the candidate.

    Directions spread evenly over the sphere, ``range_m`` away from the left camera, are
    projected into both images; the fraction is those inside both over those inside the
    left one (0 when the left camera sees none).
    """

    index = np.arange(samples) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / samples)
    azimuth = np.pi * (1.0 + 5.0**0.5) * index
    points = (
        range_m
        * np.c_[np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)]
    )
    to_right = _inverse(np.asarray(candidate, dtype=np.float64))
    in_right = points @ to_right[:3, :3].T + to_right[:3, 3]
    seen_left = _inside(left, points)
    seen_both = seen_left & _inside(right, in_right)
    return float(seen_both.sum() / seen_left.sum()) if seen_left.any() else 0.0


def _inside(camera: StereoCamera, points: FloatArray) -> NDArray[np.bool_]:
    u, v, front = camera.project(points)
    return np.asarray(front & (u >= 0.0) & (v >= 0.0) & (u < camera.width) & (v < camera.height))


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

    recentred_by: list[float] | None = None

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
            row_scale_ratio=measured.row_scale_ratio if axes and measured else None,
            row_scale_ratio_std=measured.row_scale_ratio_std if axes and measured else None,
            residual_std_px=measured.residual_std_px if axes and measured else None,
            time_offset_s=measured.time_offset_s if axes and measured else None,
            time_offset_std_s=measured.time_offset_std_s if axes and measured else None,
            time_offset_status=time_status(measured) if axes and measured and timing else None,
            timing_source=timing.source if timing else None,
            max_stamp_gap_s=float(np.abs(timing.gaps_s).max()) if timing and len(pairs) else None,
            recentred_by_deg=recentred_by,
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
    # a candidate far off moves rows beyond the row search: a coarse measurement with a
    # wider search finds the error, the rectification is re-centred on it, and the final
    # measurement (with the usual search) is made there; the error is their composition
    working = candidate
    first = _measure(pairs, left, right, candidate, opts, timing, coarse=True)
    if _usable(first, opts, coarse=True) and np.linalg.norm(first.error_deg) > opts.recentre_deg:
        working = _recentred(candidate, first.error_deg)
    base = _measure(pairs, left, right, working, opts, timing)
    for _ in range(opts.max_recentres):
        if not _usable(base, opts) or np.linalg.norm(base.error_deg) <= opts.recentre_deg:
            break
        working = _recentred(working, base.error_deg)
        base = _measure(pairs, left, right, working, opts, timing)
    recentred = _error_deg(candidate, working)
    if np.linalg.norm(recentred) > 0.0:
        recentred_by = [float(value) for value in recentred]
    total_error = (
        _error_deg(candidate, _recentred(working, base.error_deg))
        if np.all(np.isfinite(base.error_deg))
        else base.error_deg
    )
    if base.rows_known < opts.min_patches:
        return artifact(
            "inconclusive",
            f"{base.rows_known} matched patches constrain rows, fewer than {opts.min_patches}",
            base,
        )
    axes: list[StereoRotationAxis] = []
    for index, name in enumerate(AXES):
        turned = working.copy()
        turn = Rotation.from_rotvec(np.radians(opts.control_deg) * np.eye(3)[index]).as_matrix()
        turned[:3, :3] = turn @ working[:3, :3]
        moved = _measure(pairs, left, right, turned, opts, timing)
        response = float((moved.error_deg[index] - base.error_deg[index]) / opts.control_deg)
        std = float(base.std_deg[index])
        noise = float(base.noise_std_deg[index]) if base.noise_std_deg is not None else std
        problems = []
        if noise > opts.max_std_deg:
            problems.append(f"jackknife std {noise:.3f} deg exceeds {opts.max_std_deg:g} deg")
        if not opts.min_response <= response <= opts.max_response:
            problems.append(
                f"turning the candidate {opts.control_deg:g} deg moved the estimate by "
                f"{response * opts.control_deg:+.3f} deg (response {response:.2f}, outside "
                f"{opts.min_response:g} to {opts.max_response:g})"
            )
        axes.append(
            StereoRotationAxis(
                name=name,
                candidate_error_deg=float(total_error[index]),
                std_deg=std,
                scale_std_deg=(
                    float(base.scale_std_deg[index]) if base.scale_std_deg is not None else None
                ),
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
    *,
    coarse: bool = False,
) -> _Measurement:
    rectification = rectify_fisheye_pair(left, right, _inverse(candidate))
    # with timing, each pair is rectified for the rig's turn over its stamped gap, so its
    # rows are left with the (small) error and clock offset to measure: the right image
    # was taken with the relative rotation exp(omega gap) R
    per_pair = [rectification] * len(pairs)
    if timing is not None:
        turns = (
            np.asarray(timing.rates_rad_s, dtype=np.float64)
            * np.asarray(timing.gaps_s, dtype=np.float64)[:, None]
        )
        for index, turn in enumerate(turns):
            if np.linalg.norm(turn) > 1.0e-9:
                moved = candidate.copy()
                moved[:3, :3] = Rotation.from_rotvec(turn).as_matrix() @ candidate[:3, :3]
                per_pair[index] = rectify_fisheye_pair(left, right, _inverse(moved))
    rectified = [
        (rect.rectify_left(a), rect.rectify_right(b))
        for rect, (a, b) in zip(per_pair, pairs, strict=True)
    ]
    samples = vertical_offset_samples(
        rectified,
        patches_per_pair=options.coarse_patches_per_pair if coarse else options.patches_per_pair,
        rows_px=options.coarse_rows_px if coarse else options.rows_px,
        max_disparity_px=options.max_disparity_px,
        min_disparity_px=options.coarse_min_disparity_px if coarse else 0,
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
    to_rectified = np.stack([rect.rectified_from_left for rect in per_pair])[pair]
    rotation = np.einsum("nj,njk->nk", rectified_rows, to_rectified)
    values = column["offset_px"][known]
    columns = [rotation, column["disparity_px"][known][:, None]]
    if timing is not None:
        rates = np.asarray(timing.rates_rad_s, dtype=np.float64)[pair]
        columns.append(-(rotation * rates).sum(axis=1)[:, None])  # per second of offset
    design = np.hstack(columns)
    blocks = np.array_split(np.arange(len(pairs)), options.blocks)

    def jackknife_std(design: FloatArray, values: FloatArray) -> FloatArray:
        jackknife = np.array(
            [
                _trimmed_fit(design[~np.isin(pair, block)], values[~np.isin(pair, block)])[0]
                for block in blocks
            ]
        )
        count = len(blocks)
        return np.sqrt(
            (count - 1) / count * ((jackknife - jackknife.mean(axis=0)) ** 2).sum(axis=0)
        )

    if len(values) < design.shape[1] + 2:
        nan = np.full(3, np.nan)
        return _Measurement(
            nan, nan, None, None, len(samples), int(known.sum()), np.nan, np.nan, rectification
        )
    # the vertical scale mismatch: what the images say about it (free fit), combined with
    # its prior; the rotation is then fitted with it removed, and its remaining uncertainty
    # reaches each axis through how strongly the axis follows it
    row_scale: float | None = None
    row_scale_std: float | None = None
    scale_std = np.zeros(3)
    if options.fit_row_scale:
        scale_column = y / f  # px of offset per pixel of scale at one focal length
        free = np.hstack([design, scale_column[:, None]])
        data_value = float(_trimmed_fit(free, values)[0][-1])
        data_std = float(jackknife_std(free, values)[-1])
        prior = options.row_scale_prior * f
        weight = prior**2 / (prior**2 + data_std**2) if prior > 0.0 else 0.0
        value = data_value * weight
        spread = (
            float(np.sqrt(data_std**2 * prior**2 / (data_std**2 + prior**2)))
            if prior > 0.0
            else 0.0
        )
        values = values - value * scale_column
        follows = -np.linalg.lstsq(design, scale_column, rcond=None)[0][:3]
        scale_std = np.abs(follows) * spread
        row_scale, row_scale_std = value / f, spread / f
    coefficients, residual = _trimmed_fit(design, values)
    std = jackknife_std(design, values)
    noise = std[:3]
    return _Measurement(
        error_deg=np.degrees(coefficients[:3]),
        std_deg=np.degrees(np.hypot(noise, scale_std)),
        time_offset_s=float(coefficients[4]) if timing is not None else None,
        time_offset_std_s=float(std[4]) if timing is not None else None,
        patches=len(samples),
        rows_known=int(known.sum()),
        vertical_baseline_ratio=float(coefficients[3]),
        residual_std_px=residual,
        rectification=rectification,
        row_scale_ratio=row_scale,
        row_scale_ratio_std=row_scale_std,
        noise_std_deg=np.degrees(noise),
        scale_std_deg=np.degrees(scale_std),
    )


def _usable(
    measured: _Measurement, options: StereoRotationOptions, *, coarse: bool = False
) -> bool:
    """Enough matched patches (in proportion, for the sparser coarse measurement)."""

    needed = options.min_patches
    if coarse:
        needed = needed * options.coarse_patches_per_pair // options.patches_per_pair
    return measured.rows_known >= needed and bool(np.all(np.isfinite(measured.error_deg)))


def _recentred(candidate: FloatArray, error_deg: FloatArray) -> FloatArray:
    """The rig the error says: ``R_estimate = exp(-error) R_candidate`` (left camera axes)."""

    moved = candidate.copy()
    moved[:3, :3] = Rotation.from_rotvec(-np.radians(error_deg)).as_matrix() @ candidate[:3, :3]
    return moved


def _error_deg(candidate: FloatArray, estimate: FloatArray) -> FloatArray:
    """``rotvec(R_candidate R_estimate^T)`` in degrees."""

    return np.degrees(Rotation.from_matrix(candidate[:3, :3] @ estimate[:3, :3].T).as_rotvec())


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
