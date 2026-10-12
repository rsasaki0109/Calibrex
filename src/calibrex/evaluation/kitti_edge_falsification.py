"""Data-only falsification of KITTI camera-LiDAR candidates by edge alignment.

The KITTI falsification protocol rejects its declared known-bad candidate only through
a comparison with the dataset's own calibration (see the benchmark page). This module
adds a decision that uses the images and scans alone: the edge alignment of
:mod:`calibrex.evaluation.camera_lidar_edge` (block jackknife and held-out known-bad
controls) is started at each candidate, and the candidate is judged with the per-axis
rule of ``calibrex check`` (:func:`calibrex.check.verdict.judge_pair`). Edges judge
rotation only. Translation along the optical axis is judged from KITTI's stereo pair
(:mod:`calibrex.evaluation.stereo_lidar_depth`) at the rotation the edges estimate, with
the same rule; the other two translation axes are not judged.

KITTI's ``calib_velo_to_cam.txt`` maps Velodyne points into the unrectified camera 0
frame; the images are rectified camera 2. A candidate is carried into that image's
pinhole frame with ``R_rect_00`` and the offset of ``P_rect_02``, exactly as the
protocol's own projection (``R_rect`` then ``P_rect``) does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from calibrex.core.camera_lidar_edge import CameraLidarEdgeProvenance
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel
from calibrex.data.kitti import (
    KITTICameraLidarPair,
    find_camera_lidar_pairs,
    read_camera_projection_model,
    read_png_luminance,
)
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.evaluation.camera_lidar_edge import (
    EdgeAlignmentOptions,
    EdgeCamera,
    estimate_from_frames,
)
from calibrex.evaluation.stereo_lidar_depth import (
    RectifiedStereo,
    StereoDepthEstimate,
    StereoDepthOptions,
    StereoFrame,
    estimate_depth_offset,
)

if TYPE_CHECKING:
    from calibrex.check.verdict import AxisEstimate

FloatArray: TypeAlias = NDArray[np.float64]
FrameSelection = Literal["spread", "first"]
EdgeVerdict = Literal["pass", "warn", "fail", "inconclusive"]
CandidateId = Literal["dataset_reference", "known_bad"]


class KITTIEdgeAxis(StrictModel):
    """One judged rotation axis: candidate error against the check's tolerance."""

    name: str
    candidate_error_deg: float
    tolerance_deg: float
    estimate_std_deg: float
    status: Literal["pass", "warn", "fail"]


class KITTIDepthAxis(StrictModel):
    """Translation along the optical axis: candidate error against the check's tolerance."""

    name: Literal["z"] = "z"
    candidate_error_m: float
    tolerance_m: float
    estimate_std_m: float
    status: Literal["pass", "warn", "fail"]


class KITTIEdgeTrial(StrictModel):
    """The data-only verdict for one candidate: edge rotation and stereo depth."""

    candidate_id: CandidateId
    verdict: EdgeVerdict
    axes: list[KITTIEdgeAxis] = Field(default_factory=list)
    unchecked_rotation_axes: list[str] = Field(default_factory=list)
    depth_axis: KITTIDepthAxis | None = None
    depth: StereoDepthEstimate | None = Field(
        default=None,
        description="stereo-depth estimate at the edge-estimated rotation; None when the "
        "dataset has no right stereo images",
    )
    unchecked_translation_axes: list[str] = Field(default_factory=list)
    frames: int = Field(ge=0)
    edge_points: int = Field(ge=0)


class KITTIEdgeDecision(StrictModel):
    """Data-only decision over one frame selection."""

    method: Literal["edge_alignment/v0.1 with the calibrex check per-axis rule"] = (
        "edge_alignment/v0.1 with the calibrex check per-axis rule"
    )
    frame_selection: FrameSelection = Field(
        description="spread: frames evenly over the drive (the check's policy); first: the "
        "protocol's first frames"
    )
    frame_ids: list[str] = Field(min_length=1)
    status: Literal["pass", "fail", "inconclusive"]
    reason: str
    trials: list[KITTIEdgeTrial] = Field(min_length=2, max_length=2)
    git_commit: str | None = None


def kitti_edge_candidates(
    dataset: str | Path, candidates: Mapping[CandidateId, SE3]
) -> tuple[EdgeCamera, dict[CandidateId, FloatArray]]:
    """The rectified camera 2 and each ``T_camera0_lidar`` candidate in its pinhole frame."""

    camera, rectify, _ = _rectified_camera(dataset)
    return camera, {name: rectify @ _matrix(transform) for name, transform in candidates.items()}


def kitti_stereo(dataset: str | Path) -> RectifiedStereo | None:
    """Cameras 2 and 3 projected from camera 2's pinhole frame (the candidates' frame)."""

    _, rectify, left = _rectified_camera(dataset)
    model = read_camera_projection_model(dataset, camera="image_03")
    if model is None:
        return None
    right = np.asarray(model.projection_matrix, dtype=np.float64).reshape(3, 4)
    # a point X in the pinhole frame sits at X - b in rectified camera 0
    to_rectified = np.eye(4)
    to_rectified[:3, 3] = -rectify[:3, 3]
    return RectifiedStereo(left=left @ to_rectified, right=right @ to_rectified)


def _rectified_camera(dataset: str | Path) -> tuple[EdgeCamera, FloatArray, FloatArray]:
    model = read_camera_projection_model(dataset)
    if model is None:
        msg = f"{dataset}: KITTI camera projection (calib_cam_to_cam.txt) is missing"
        raise ValueError(msg)
    projection = np.asarray(model.projection_matrix, dtype=np.float64).reshape(3, 4)
    fx, fy = projection[0, 0], projection[1, 1]
    cx, cy = projection[0, 2], projection[1, 2]
    # P_rect = K [I | b]: the rectified camera-2 origin sits at -b in rectified camera 0.
    offset_z = projection[2, 3]
    offset_y = (projection[1, 3] - cy * offset_z) / fy
    offset_x = (projection[0, 3] - cx * offset_z) / fx
    rectify = np.eye(4)
    rectify[:3, :3] = np.asarray(model.rectification_matrix, dtype=np.float64).reshape(3, 3)
    rectify[:3, 3] = (offset_x, offset_y, offset_z)
    return EdgeCamera(fx, fy, cx, cy), rectify, projection


def kitti_edge_frames(
    pairs: Sequence[KITTICameraLidarPair],
) -> list[CameraLidarFrame]:
    """Grayscale images and Velodyne points of KITTI frame pairs."""

    frames: list[CameraLidarFrame] = []
    for pair in pairs:
        image = read_png_luminance(pair.camera_path)
        if image is None:
            msg = f"{pair.camera_path}: unreadable PNG"
            raise ValueError(msg)
        gray = np.clip(np.asarray(image.rows, dtype=np.float64), 0.0, 255.0).astype(np.uint8)
        points = np.fromfile(pair.lidar_path, dtype="<f4").reshape(-1, 4)[:, :3]
        frames.append(
            CameraLidarFrame(
                time_s=(pair.lidar_timestamp_ns or 0) * 1.0e-9,
                image_time_s=(pair.camera_timestamp_ns or 0) * 1.0e-9,
                image=gray,
                points=np.ascontiguousarray(points, dtype=np.float32),
                cloud_frame_id="velo_link",
                image_frame_id="camera_02_rectified",
            )
        )
    return frames


def kitti_stereo_frames(
    pairs: Sequence[KITTICameraLidarPair], frames: Sequence[CameraLidarFrame]
) -> list[StereoFrame] | None:
    """The edge frames with their camera-3 images; ``None`` when any is missing."""

    stereo: list[StereoFrame] = []
    for pair, frame in zip(pairs, frames, strict=True):
        right_path = _right_image_path(Path(pair.camera_path))
        image = read_png_luminance(right_path) if right_path is not None else None
        if image is None:
            return None
        stereo.append(
            StereoFrame(
                left=np.asarray(frame.image, dtype=np.float64),
                right=np.clip(np.asarray(image.rows, dtype=np.float64), 0.0, 255.0),
                points=np.asarray(frame.points, dtype=np.float64),
            )
        )
    return stereo


def _right_image_path(left: Path) -> Path | None:
    parts = list(left.parts)
    if "image_02" not in parts:
        return None
    parts[len(parts) - 1 - parts[::-1].index("image_02")] = "image_03"
    path = Path(*parts)
    return path if path.is_file() else None


def select_kitti_pairs(
    dataset: str | Path,
    frame_count: int,
    selection: FrameSelection,
    *,
    max_gap_s: float = EdgeAlignmentOptions.max_gap_s,
) -> list[KITTICameraLidarPair]:
    """``frame_count`` pairs: the first ones, or spread evenly over the drive.

    Pairs whose image and scan are more than ``max_gap_s`` apart (an image whose own scan
    is missing, as on drive 0009) are not used, as in the bag-level check.
    """

    pairs = [
        pair
        for pair in find_camera_lidar_pairs(dataset, max_pairs=1_000_000)
        if pair.delta_ms is None or abs(pair.delta_ms) <= 1.0e3 * max_gap_s
    ]
    if selection == "first" or len(pairs) <= frame_count:
        return pairs[:frame_count]
    indices = np.unique(np.round(np.linspace(0, len(pairs) - 1, frame_count)).astype(int))
    return [pairs[int(index)] for index in indices]


def run_kitti_edge_decision(
    dataset: str | Path,
    candidates: Mapping[CandidateId, SE3],
    *,
    frame_count: int = 50,
    selection: FrameSelection = "spread",
    options: EdgeAlignmentOptions | None = None,
    depth_options: StereoDepthOptions | None = None,
) -> KITTIEdgeDecision:
    """Judge each candidate from the images and scans alone.

    Rotation from edges; translation along the optical axis from the stereo pair, at the
    rotation the edges estimate.
    """

    from calibrex.check.estimators import camera_lidar_axis_estimates, transform_matrix
    from calibrex.check.verdict import VerdictOptions, judge_pair

    commit = git_commit()
    pairs = select_kitti_pairs(dataset, frame_count, selection)
    frames = kitti_edge_frames(pairs)
    camera, transforms = kitti_edge_candidates(dataset, candidates)
    stereo = kitti_stereo(dataset)
    stereo_frames = kitti_stereo_frames(pairs, frames) if stereo is not None else None
    provenance = CameraLidarEdgeProvenance(
        generator="calibrex.evaluation.kitti_edge_falsification",
        generator_version="0.1",
        dataset_family="kitti_raw",
        sequence_ids=[Path(dataset).name],
        image_topic="image_02",
        lidar_topic="velodyne_points",
        input_sha256=(sha256_path(Path(pairs[0].camera_path)) if pairs else None) or "0" * 64,
        input_digest_scope="first selected camera image (the benchmark binds every input)",
        dataset_license="CC BY-NC-SA 3.0",
    )
    trials: list[KITTIEdgeTrial] = []
    for candidate_id, transform in transforms.items():
        artifact = estimate_from_frames(
            frames,
            camera,
            transform,
            options=options or EdgeAlignmentOptions(),
            provenance=provenance,
            parent_frame="camera_02_rectified",
            child_frame="velo_link",
        )
        estimates = camera_lidar_axis_estimates(artifact, transform)
        depth: StereoDepthEstimate | None = None
        if stereo is not None and stereo_frames is not None and artifact.transform is not None:
            # the candidate's translation at the rotation the edges estimate
            at_rotation = transform_matrix(transform[:3, 3], artifact.transform.rotation_quat_xyzw)
            depth = estimate_depth_offset(stereo_frames, stereo, at_rotation, depth_options)
            estimates = [_depth_estimate(depth) if e.name == "z" else e for e in estimates]
        judgement = judge_pair(estimates, VerdictOptions())
        depth_judgement = next((axis for axis in judgement.axes if axis.name == "z"), None)
        trials.append(
            KITTIEdgeTrial(
                candidate_id=candidate_id,
                verdict=judgement.verdict,
                axes=[
                    KITTIEdgeAxis(
                        name=axis.name,
                        candidate_error_deg=float(axis.candidate_error),
                        tolerance_deg=float(axis.tolerance),
                        estimate_std_deg=float(axis.estimate_std),
                        status=axis.status,
                    )
                    for axis in judgement.axes
                    if axis.unit == "deg"
                ],
                unchecked_rotation_axes=[
                    item.name
                    for item in judgement.unchecked
                    if item.name in {"roll", "pitch", "yaw"}
                ],
                depth_axis=(
                    KITTIDepthAxis(
                        candidate_error_m=float(depth_judgement.candidate_error),
                        tolerance_m=float(depth_judgement.tolerance),
                        estimate_std_m=float(depth_judgement.estimate_std),
                        status=depth_judgement.status,
                    )
                    if depth_judgement is not None
                    else None
                ),
                depth=depth,
                unchecked_translation_axes=[
                    item.name for item in judgement.unchecked if item.name in {"x", "y", "z"}
                ],
                frames=artifact.samples.frames,
                edge_points=artifact.samples.edge_points,
            )
        )
    status, reason = _decision(trials)
    return KITTIEdgeDecision(
        frame_selection=selection,
        frame_ids=[f"camera:{pair.camera_index}:lidar:{pair.lidar_index}" for pair in pairs],
        status=status,
        reason=reason,
        trials=trials,
        git_commit=commit,
    )


def _depth_estimate(depth: StereoDepthEstimate) -> AxisEstimate:
    from calibrex.check.verdict import AxisEstimate

    if depth.observable and depth.offset_m is not None and depth.std_m is not None:
        return AxisEstimate("z", "m", depth.offset_m, depth.std_m, True)
    code: Literal["unobservable", "control_not_detected"] = (
        "control_not_detected" if depth.control_response is not None else "unobservable"
    )
    return AxisEstimate(
        "z",
        "m",
        depth.offset_m or 0.0,
        depth.std_m or 0.0,
        False,
        f"stereo depth: {depth.reason}",
        code,
    )


def _decision(
    trials: Sequence[KITTIEdgeTrial],
) -> tuple[Literal["pass", "fail", "inconclusive"], str]:
    by_id = {trial.candidate_id: trial for trial in trials}
    reference, known_bad = by_id["dataset_reference"], by_id["known_bad"]
    if reference.verdict == "pass" and known_bad.verdict == "fail":
        return "pass", "the images and scans pass the reference and fail the known-bad candidate"
    if reference.verdict == "fail" or known_bad.verdict == "pass":
        return (
            "fail",
            f"the images and scans {reference.verdict} the reference and "
            f"{known_bad.verdict} the known-bad candidate",
        )
    return (
        "inconclusive",
        f"the reference reads {reference.verdict} and the known-bad candidate "
        f"{known_bad.verdict}: neither is a clear decision",
    )


def _matrix(transform: SE3) -> FloatArray:
    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(transform.rotation_quat_xyzw)).as_matrix()
    matrix[:3, 3] = transform.translation_m
    return matrix
