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
        if self.control_deg <= 0.0 or self.spacing_s < 0.0 or self.max_gap_s <= 0.0:
            raise ValueError("control_deg and max_gap_s must be positive, spacing_s not negative")


@dataclass(frozen=True)
class _Measurement:
    error_deg: FloatArray
    std_deg: FloatArray
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
    """Grayscale image pairs taken within ``max_gap_s`` of each other, ``spacing_s`` apart.

    At most ``max_pairs``, from the first ``max_seconds`` of the bag when given.
    """

    opts = options or StereoRotationOptions()
    gap_ns = round(opts.max_gap_s * 1.0e9)
    pending: dict[str, list[tuple[int, int, bytes]]] = {left_topic: [], right_topic: []}
    pairs: list[tuple[FloatArray, FloatArray]] = []
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
        match = next((item for item in pending[other] if abs(item[0] - time_ns) <= gap_ns), None)
        if match is None:
            pending[topic] = [*pending[topic][-4:], (time_ns, stamp, payload)]
            continue
        pending[other].remove(match)
        if last_ns is not None and time_ns - last_ns < opts.spacing_s * 1.0e9:
            continue
        last_ns = time_ns
        images = {topic: (stamp, payload), other: (match[1], match[2])}
        pairs.append(
            (_gray(left_topic, *images[left_topic]), _gray(right_topic, *images[right_topic]))
        )
        if len(pairs) >= opts.max_pairs:
            break
    return pairs


def estimate_stereo_rotation(
    pairs: Sequence[tuple[FloatArray, FloatArray]],
    left: StereoCamera,
    right: StereoCamera,
    candidate: FloatArray,
    *,
    provenance: StereoRotationProvenance,
    options: StereoRotationOptions | None = None,
) -> StereoRotationArtifact:
    """The candidate ``T_left_right``'s rotation error about the left camera's axes."""

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
            options=asdict(opts),
            limitations=[
                "rotation only: the translation between the cameras is not judged",
                "the images must share texture that varies across rows (blank walls do not)",
            ],
            provenance=provenance,
        )

    if len(pairs) < opts.blocks:
        return artifact(
            "inconclusive",
            f"{len(pairs)} synchronised image pairs, fewer than {opts.blocks}",
            None,
        )
    base = _measure(pairs, left, right, candidate, opts)
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
        moved = _measure(pairs, left, right, turned, opts)
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
    design = np.c_[1.0 + (y / f) ** 2, x / f, x * y / f**2, column["disparity_px"][known]]
    values = column["offset_px"][known]
    pair = column["pair"][known]
    to_left = rectification.rectified_from_left.T

    def error(rows: NDArray[np.bool_]) -> tuple[FloatArray, FloatArray]:
        coefficients, residual = _trimmed_fit(design[rows], values[rows])
        # the constant is -f delta_x, the column term f delta_z, the column-row term f delta_y
        measured = np.array([-coefficients[0], coefficients[2], coefficients[1]]) / f
        return np.degrees(to_left @ measured), np.array([coefficients[3], residual])

    if len(values) < 4:
        nan = np.full(3, np.nan)
        return _Measurement(nan, nan, len(samples), int(known.sum()), np.nan, np.nan, rectification)
    everything = np.ones(len(values), dtype=bool)
    estimate, extra = error(everything)
    blocks = np.array_split(np.arange(len(pairs)), options.blocks)
    jackknife = np.array([error(~np.isin(pair, block))[0] for block in blocks])
    count = len(blocks)
    std = np.sqrt((count - 1) / count * ((jackknife - jackknife.mean(axis=0)) ** 2).sum(axis=0))
    return _Measurement(
        estimate,
        std,
        len(samples),
        int(known.sum()),
        float(extra[0]),
        float(extra[1]),
        rectification,
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
