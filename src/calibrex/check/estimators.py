"""Adapters from the native pair estimators to the verdict rule of ``calibrex check``.

Each adapter runs one existing estimator on a bag, with the *candidate*
(deployed) transform as its reference, and returns an :class:`EstimatorRun`:
the estimator's artifact(s), per-axis :class:`~calibrex.check.verdict.AxisEstimate`
values and the clock offset. The adapters are looked up in :data:`ESTIMATORS`
so tests can replace them.

Conventions
-----------
Every estimator reports ``T_parent_child`` of its own convention: ``imu-lidar``
reports ``T_lidar_imu``, ``camera-imu`` ``T_cam_imu`` and ``lidar-lidar``
``T_reference_target``. The planner's ``T_first_second`` is converted to that
convention (``imu-lidar`` is inverted) before comparing. Rotation errors are
``rotvec(R_candidate R_estimate^T)`` in degrees, a small rotation about the
parent frame's axes, which is the convention of the estimators' standard
deviations; translation errors are component differences in metres.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.check.cache import EstimatorCache, cached_artifact
from calibrex.check.tf_sources import LoadedSource, camchain_entries
from calibrex.check.verdict import AxisEstimate
from calibrex.core.calibration_check import (
    CheckAxisName,
    CheckPairRecord,
    CheckReasonCode,
    CheckTimeOffset,
    CheckTopicRecord,
)

if TYPE_CHECKING:
    from calibrex.data.livox_ros2 import ScanStore
    from calibrex.evaluation.visual_rotation import CameraModel

FloatArray: TypeAlias = NDArray[np.float64]
PointTimeEncoding = Literal["offset_s", "absolute_ns", "absolute_s"]
POINT_TIME_FIELD_CANDIDATES: tuple[str, ...] = (
    "offset_time",
    "t",
    "time",
    "timestamp",
    "time_stamp",
    "timestamps",
    "point_time",
    "time_offset",
    "relative_time",
)
POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"
IMAGE_TYPE = "sensor_msgs/msg/Image"
CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"
DATASET_FAMILY = "calibrex-check"
DATASET_LICENSE = "user-provided"


class CheckSkipError(Exception):
    """The pair cannot be run: a machine-readable reason and a message."""

    def __init__(self, code: CheckReasonCode, reason: str) -> None:
        super().__init__(reason)
        self.code: CheckReasonCode = code
        self.reason = reason


class SavableArtifact(Protocol):
    """A schema artifact that can be written to YAML or JSON."""

    @property
    def schema_version(self) -> str: ...

    def save(self, path: str | Path) -> None: ...


@dataclass(frozen=True)
class EvidenceArtifact:
    """One estimator artifact to be written to the evidence directory."""

    role: str
    artifact: SavableArtifact
    policy_status: str | None = None
    from_cache: bool | None = None


@dataclass(frozen=True)
class EstimatorRun:
    """What an adapter returns for one pair."""

    estimator: str
    artifacts: tuple[EvidenceArtifact, ...]
    solved: bool
    policy_status: str
    policy_reasons: tuple[str, ...]
    estimates: tuple[AxisEstimate, ...]
    compared: FloatArray
    time_offset: CheckTimeOffset | None = None
    notes: tuple[str, ...] = ()

    @property
    def evidence_from_cache(self) -> bool | None:
        """Whether every estimator artifact came from the cache; ``None`` when none is cacheable."""

        flags = [item.from_cache for item in self.artifacts if item.from_cache is not None]
        return all(flags) if flags else None


@dataclass(frozen=True)
class RunControls:
    """Runtime controls shared by the adapters."""

    max_duration_s: float | None = None
    camera: str | None = None
    imu_lidar_translation: bool = True
    acceleration_unit: Literal["mps2", "g"] = "mps2"
    progress: Callable[[str], None] = field(default=lambda message: None)
    cache: EstimatorCache | None = None
    bag_sha256: str = ""
    scan_store: ScanStore | None = None
    memo: dict[tuple[object, ...], Any] = field(default_factory=dict)
    """Per-run results shared by pairs (the LiDAR odometry of the vehicle pairs)."""


@dataclass(frozen=True)
class PairContext:
    """Everything an adapter needs for one planned pair."""

    bag: Path
    pair: CheckPairRecord
    topic_types: Mapping[str, str]
    sensor_topics: Mapping[str, tuple[str, ...]]
    candidate: FloatArray
    sources: Sequence[LoadedSource]
    controls: RunControls
    topics: Sequence[CheckTopicRecord] = ()


Adapter = Callable[[PairContext], EstimatorRun]


# ------------------------------------------------------------------ conversions


def transform_matrix(
    translation_m: Sequence[float], quaternion_xyzw: Sequence[float]
) -> FloatArray:
    """Build a 4x4 transform from a translation and an xyzw quaternion."""

    from scipy.spatial.transform import Rotation

    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64)).as_matrix()
    matrix[:3, 3] = np.asarray(translation_m, dtype=np.float64)
    return matrix


def invert_transform(matrix: FloatArray) -> FloatArray:
    """Invert a rigid transform."""

    inverse = np.eye(4)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def rotation_errors_deg(candidate: FloatArray, estimate: FloatArray) -> tuple[float, float, float]:
    """Return ``rotvec(R_candidate R_estimate^T)`` in degrees (about parent-frame axes)."""

    from scipy.spatial.transform import Rotation

    local = Rotation.from_matrix(candidate @ estimate.T).as_rotvec()
    return (
        math.degrees(float(local[0])),
        math.degrees(float(local[1])),
        math.degrees(float(local[2])),
    )


def _estimate(
    name: CheckAxisName,
    unit: Literal["deg", "m"],
    error: float,
    record: Any,
) -> AxisEstimate:
    observable = record is not None and record.status == "estimated"
    control = getattr(record, "known_bad_control", None)
    undetected = observable and control is not None and not control.detected
    if undetected and control is not None:
        amount = getattr(control, "amount", None)
        control_unit = getattr(control, "unit", None)
        if amount is None:
            amount, control_unit = control.amount_deg, "deg"
        return AxisEstimate(
            name=name,
            unit=unit,
            candidate_error=error,
            std=float(record.std_reported),
            estimated=False,
            unchecked_reason=(
                f"the estimator's known-bad control on this axis ({amount:g} "
                f"{control_unit}) was not detected on held-out data "
                f"(delta chi-square {control.holdout_delta_chi2:.1f}), so the estimate "
                "is not trusted as a yardstick"
            ),
            unchecked_code="control_not_detected",
        )
    return AxisEstimate(
        name=name,
        unit=unit,
        candidate_error=error,
        std=float(record.std_reported) if record is not None else 0.0,
        estimated=observable,
        unchecked_code=None
        if observable
        else ("unobservable" if record is not None else "no_estimate"),
        unchecked_reason=None
        if observable
        else (
            (
                "constrained only by a declared prior, not by the data"
                if getattr(record, "status", None) == "prior"
                else f"not constrained by the data (reported std {record.std_reported:.4g} {unit})"
            )
            if record is not None
            else "the estimator reported no value"
        ),
    )


def rotation_axis_estimates(
    records: Sequence[Any],
    candidate_rotation: FloatArray,
    estimate_rotation: FloatArray,
) -> list[AxisEstimate]:
    """Roll/pitch/yaw estimates from DoF records and the two rotations."""

    errors = rotation_errors_deg(candidate_rotation, estimate_rotation)
    by_name = {record.name: record for record in records}
    names: tuple[CheckAxisName, ...] = ("roll", "pitch", "yaw")
    return [
        _estimate(name, "deg", errors[index], by_name.get(name)) for index, name in enumerate(names)
    ]


def translation_axis_estimates(
    records: Sequence[Any],
    candidate_translation: Sequence[float],
    estimate_translation: Sequence[float],
) -> list[AxisEstimate]:
    """x/y/z estimates from axis records and the two translations."""

    by_name = {record.name: record for record in records}
    names: tuple[CheckAxisName, ...] = ("x", "y", "z")
    return [
        _estimate(
            name,
            "m",
            float(candidate_translation[index]) - float(estimate_translation[index]),
            by_name.get(name),
        )
        for index, name in enumerate(names)
    ]


def time_offset_of(records: Sequence[Any], estimate_s: float) -> CheckTimeOffset | None:
    """The clock-offset record of a rotation artifact."""

    for record in records:
        if record.name == "time_offset":
            return CheckTimeOffset(
                estimate_s=float(estimate_s),
                std_s=float(record.std_reported),
                status="estimated" if record.status == "estimated" else "unobservable",
            )
    return None


def _pose_rotation(artifact: Any) -> FloatArray:
    from scipy.spatial.transform import Rotation

    return np.asarray(Rotation.from_quat(artifact.rotation_quat_xyzw).as_matrix(), dtype=np.float64)


# ------------------------------------------------------------ point-time field


def detect_point_time(bag: Path, topic: str) -> tuple[str, PointTimeEncoding] | None:
    """Find a per-point time field of a PointCloud2 topic and how to read it.

    Looks at the first message: picks the first of the usual field names and
    classifies its values against the header stamp as an offset in seconds, an
    absolute time in seconds or an absolute time in nanoseconds. Returns
    ``None`` when there is no usable field.
    """

    from calibrex.data import ros_cdr
    from calibrex.data.rosbag2 import iter_topic_messages

    for _conn, timestamp_ns, payload in iter_topic_messages(bag, topic, limit=1):
        cloud = ros_cdr.decode_ros2_pointcloud2(topic, timestamp_ns, payload)
        names = {point_field.name for point_field in cloud.fields}
        for candidate in POINT_TIME_FIELD_CANDIDATES:
            if candidate not in names:
                continue
            decoded = ros_cdr.decode_ros2_pointcloud2(
                topic, timestamp_ns, payload, point_time_field=candidate
            )
            encoding = classify_point_time(decoded.point_time_offsets_s, decoded.timestamp_ns)
            if encoding is not None:
                return candidate, encoding
        return None
    return None


def classify_point_time(values: Any, header_ns: int) -> PointTimeEncoding | None:
    """Classify decoded per-point time values against the header stamp (ns)."""

    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    median = float(np.median(array))
    header_s = header_ns * 1.0e-9
    if header_s > 1.0 and abs(median - header_s) < 1.0e3:
        return "absolute_s"
    if header_ns > 0 and abs(median - header_ns) < 1.0e12:
        return "absolute_ns"
    if float(np.max(np.abs(array))) < 10.0:
        return "offset_s"
    return None


def _point_time_or_skip(bag: Path, topic: str) -> tuple[str, PointTimeEncoding]:
    spec = detect_point_time(bag, topic)
    if spec is None:
        raise CheckSkipError(
            "unsupported_sensor",
            f"{topic} has no usable per-point time field, so its scans cannot be deskewed",
        )
    return spec


def _require_pointcloud2(ctx: PairContext, topics: Sequence[str]) -> str:
    for topic in topics:
        if ctx.topic_types.get(topic) == POINTCLOUD2_TYPE:
            return topic
    raise CheckSkipError(
        "unsupported_sensor",
        f"LiDAR topic(s) {', '.join(topics)} are not sensor_msgs/PointCloud2 "
        "(Livox CustomMsg is not supported by the check yet)",
    )


# ----------------------------------------------------------------- imu-lidar


def run_imu_lidar(ctx: PairContext) -> EstimatorRun:
    """Rotation (and optionally lever arm) of ``T_lidar_imu`` from angular rates."""

    from calibrex.core.imu_lidar_rotation import load_imu_lidar_rotation
    from calibrex.core.imu_lidar_translation import load_imu_lidar_translation
    from calibrex.data.livox_ros2 import LivoxStreamProfile
    from calibrex.evaluation.imu_lidar_rotation import (
        ImuLidarRunOptions,
        rebase_rotation_reference,
        run_livox_imu_lidar_rotation,
    )
    from calibrex.evaluation.imu_lidar_translation import (
        ImuLidarTranslationOptions,
        rebase_translation_reference,
        run_livox_imu_lidar_translation,
    )

    lidar_topic = _require_pointcloud2(ctx, ctx.sensor_topics.get("lidar", ()))
    imu_topics = ctx.sensor_topics.get("imu", ())
    if not imu_topics:
        raise CheckSkipError("missing_topic", "no IMU topic for the pair")
    imu_topic = imu_topics[0]
    field_name, encoding = _point_time_or_skip(ctx.bag, lidar_topic)
    profile = LivoxStreamProfile(
        name=f"ros2-pointcloud2:{lidar_topic}+imu:{imu_topic}",
        point_topic=lidar_topic,
        imu_topic=imu_topic,
        point_time_field=field_name,
        point_time_encoding=encoding,
        acceleration_unit=ctx.controls.acceleration_unit,
    )
    candidate = invert_transform(ctx.candidate)  # T_lidar_imu
    command = ["calibrex", "check", str(ctx.bag), "imu-lidar", lidar_topic, imu_topic]
    reference = "deployed candidate calibration (T_lidar_imu)"
    ctx.controls.progress(f"imu-lidar: rotation on {lidar_topic} + {imu_topic}")
    # The estimate does not depend on the candidate (it is only compared with it
    # afterwards), so it is cached and the reference fields are rebased on a hit.
    rotation_options = {
        "profile": profile,
        "max_seconds": ctx.controls.max_duration_s,
        "options": ImuLidarRunOptions(),
    }
    rotation, rotation_cached = cached_artifact(
        ctx.controls.cache,
        ctx.controls.bag_sha256,
        "imu_lidar_rotation",
        rotation_options,
        loader=load_imu_lidar_rotation,
        compute=lambda: run_livox_imu_lidar_rotation(
            [ctx.bag],
            profile,
            dataset_family=DATASET_FAMILY,
            dataset_license=DATASET_LICENSE,
            reference_rotation=candidate[:3, :3],
            max_seconds=ctx.controls.max_duration_s,
            reference=reference,
            command=command,
            scan_store=ctx.controls.scan_store,
        ),
        rebase=lambda hit: rebase_rotation_reference(hit, candidate[:3, :3]),
    )
    notes = [f"per-point time: field '{field_name}' read as {encoding}"]
    cache_flag = rotation_cached if ctx.controls.cache is not None else None
    artifacts = [
        EvidenceArtifact("rotation", rotation, rotation.policy_status, cache_flag),
    ]
    estimates: list[AxisEstimate] = []
    solved = rotation.rotation_quat_xyzw is not None and rotation.solver_status == "converged"
    if rotation.rotation_quat_xyzw is not None:
        estimates += rotation_axis_estimates(
            rotation.dofs, candidate[:3, :3], _pose_rotation(rotation)
        )
    else:
        estimates += rotation_axis_estimates(rotation.dofs, candidate[:3, :3], candidate[:3, :3])
        estimates = [_unestimated(item) for item in estimates]
    policy = rotation.policy_status
    reasons = list(rotation.policy_reasons)
    if ctx.controls.imu_lidar_translation and solved:
        ctx.controls.progress("imu-lidar: lever arm (translation) pass")
        # The lever arm is solved from the rotation estimate (not the candidate),
        # so its key is the rotation's key plus the translation options.
        translation_options = {
            **rotation_options,
            "rotation_estimator": "imu_lidar_rotation",
            "translation_options": ImuLidarTranslationOptions(),
        }
        translation, translation_cached = cached_artifact(
            ctx.controls.cache,
            ctx.controls.bag_sha256,
            "imu_lidar_translation",
            translation_options,
            loader=load_imu_lidar_translation,
            compute=lambda: run_livox_imu_lidar_translation(
                [ctx.bag],
                profile,
                rotation,
                dataset_family=DATASET_FAMILY,
                dataset_license=DATASET_LICENSE,
                reference_translation=candidate[:3, 3],
                max_seconds=ctx.controls.max_duration_s,
                reference=reference,
                command=[*command, "translation"],
                scan_store=ctx.controls.scan_store,
            ),
            rebase=lambda hit: rebase_translation_reference(hit, candidate[:3, 3]),
        )
        artifacts.append(
            EvidenceArtifact(
                "translation",
                translation,
                translation.policy_status,
                translation_cached if ctx.controls.cache is not None else None,
            )
        )
        notes.append(f"translation estimator policy: {translation.policy_status}")
        if translation.translation_m is not None:
            translation_estimates = translation_axis_estimates(
                translation.axes, candidate[:3, 3], translation.translation_m
            )
            if translation.policy_status == "fail":
                reason = "the lever-arm estimate failed its held-out check: " + "; ".join(
                    translation.policy_reasons
                )
                translation_estimates = [
                    _unestimated(item, reason) for item in translation_estimates
                ]
            estimates += translation_estimates
    elif ctx.controls.imu_lidar_translation:
        notes.append("translation not estimated: the rotation was not solved")
    return EstimatorRun(
        estimator=rotation.method,
        artifacts=tuple(artifacts),
        solved=solved,
        policy_status=policy,
        policy_reasons=tuple(reasons),
        estimates=tuple(estimates),
        compared=candidate,
        time_offset=time_offset_of(rotation.dofs, rotation.time_offset_s),
        notes=tuple(notes),
    )


def _unestimated(
    item: AxisEstimate, reason: str = "the estimator produced no rotation"
) -> AxisEstimate:
    return AxisEstimate(
        item.name, item.unit, item.candidate_error, item.std, False, reason, "no_estimate"
    )


# --------------------------------------------------------------- lidar-lidar


def run_lidar_lidar(ctx: PairContext) -> EstimatorRun:
    """``T_reference_target`` by map registration, started from the candidate."""

    from calibrex.evaluation.lidar_lidar_map import run_ros2_lidar_lidar_map

    reference_topic = _require_pointcloud2(ctx, ctx.sensor_topics.get("first", ()))
    target_topic = _require_pointcloud2(ctx, ctx.sensor_topics.get("second", ()))
    ref_field, ref_encoding = _point_time_or_skip(ctx.bag, reference_topic)
    tgt_field, tgt_encoding = _point_time_or_skip(ctx.bag, target_topic)
    ctx.controls.progress(f"lidar-lidar: {reference_topic} -> {target_topic}")
    artifact = run_ros2_lidar_lidar_map(
        ctx.bag,
        reference_topic=reference_topic,
        target_topic=target_topic,
        initial=ctx.candidate,
        point_time_field=ref_field,
        point_time_encoding=ref_encoding,
        target_point_time_field=tgt_field,
        target_point_time_encoding=tgt_encoding,
        dataset_family=DATASET_FAMILY,
        dataset_license=DATASET_LICENSE,
        reference_transform=ctx.candidate,
        reference="deployed candidate calibration (T_reference_target)",
        max_seconds=ctx.controls.max_duration_s,
        command=["calibrex", "check", str(ctx.bag), "lidar-lidar", reference_topic, target_topic],
    )
    notes = [
        f"per-point time: {reference_topic} field '{ref_field}' ({ref_encoding}), "
        f"{target_topic} field '{tgt_field}' ({tgt_encoding})",
        "the solver starts at the candidate; a candidate far outside the capture range "
        "of the registration is reported through the held-out residual, not a large delta",
    ]
    estimates: list[AxisEstimate] = []
    solved = artifact.transform is not None and artifact.solver_status != "insufficient_samples"
    if artifact.transform is not None:
        estimate = transform_matrix(
            artifact.transform.translation_m, artifact.transform.rotation_quat_xyzw
        )
        estimates += rotation_axis_estimates(artifact.dofs, ctx.candidate[:3, :3], estimate[:3, :3])
        estimates += translation_axis_estimates(
            [r for r in artifact.dofs if r.name in {"x", "y", "z"}],
            ctx.candidate[:3, 3],
            estimate[:3, 3],
        )
    if artifact.reference_holdout_delta_chi2 is not None:
        notes.append(
            "held-out chi-square increase when the candidate replaces the estimate: "
            f"{artifact.reference_holdout_delta_chi2:.1f}"
        )
    return EstimatorRun(
        estimator=artifact.method,
        artifacts=(EvidenceArtifact("map_registration", artifact, artifact.policy_status),),
        solved=solved,
        policy_status=artifact.policy_status,
        policy_reasons=tuple(artifact.policy_reasons),
        estimates=tuple(estimates),
        compared=ctx.candidate,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------- camera-imu


def camera_model_from_camera_info(bag: Path, topic: str) -> CameraModel:
    """Pinhole intrinsics and distortion from the first message of a CameraInfo topic."""

    from calibrex.data import ros_cdr
    from calibrex.data.rosbag2 import iter_topic_messages
    from calibrex.evaluation.visual_rotation import CameraModel

    for _conn, timestamp_ns, payload in iter_topic_messages(bag, topic, limit=1):
        info = ros_cdr.decode_ros2_camera_info(topic, timestamp_ns, payload)
        k = info.k
        if len(k) != 9 or not k[0] > 0.0 or not k[4] > 0.0:
            raise CheckSkipError(
                "missing_intrinsics", f"CameraInfo {topic} is uncalibrated (K is zero)"
            )
        coeffs = tuple(float(value) for value in info.d)
        model = info.distortion_model.lower()
        distortion: Literal["equidistant", "radtan", "none"]
        if not coeffs or not any(coeffs):
            distortion = "none"
        elif model in {"equidistant", "fisheye"}:
            distortion = "equidistant"
            coeffs = coeffs[:4]
        elif model in {"plumb_bob", "rational_polynomial", ""}:
            distortion = "radtan"
        else:
            raise CheckSkipError(
                "missing_intrinsics",
                f"CameraInfo {topic}: unsupported model {info.distortion_model!r}",
            )
        return CameraModel(k[0], k[4], k[2], k[5], distortion, coeffs)
    raise CheckSkipError("missing_intrinsics", f"CameraInfo topic {topic} has no messages")


def _camera_info_topic(ctx: PairContext, image_topic: str) -> str | None:
    folder = image_topic.rsplit("/", 1)[0]
    candidates = sorted(
        topic for topic, message_type in ctx.topic_types.items() if message_type == CAMERA_INFO_TYPE
    )
    for topic in candidates:
        if topic.rsplit("/", 1)[0] == folder:
            return topic
    return None


def resolve_camera_intrinsics(
    ctx: PairContext, image_topic: str, camera_frame: str
) -> tuple[CameraModel, str, str]:
    """Return ``(model, source description, camera key or '')`` or raise a skip."""

    from calibrex.evaluation.visual_rotation import CameraModel

    for source in ctx.sources:
        if source.path is None or source.kind not in {"kalibr_camchain", "rtk_slam_calib"}:
            continue
        entries = camchain_entries(source.path)
        for key, frame, entry in entries:
            if frame == camera_frame and "intrinsics" in entry:
                try:
                    model = CameraModel.from_kalibr(entry)
                except ValueError as exc:
                    raise CheckSkipError(
                        "missing_intrinsics", f"{source.path.name} {key}: {exc}"
                    ) from exc
                return model, f"{source.path.name} {key}", key
    info_topic = _camera_info_topic(ctx, image_topic)
    if info_topic is not None:
        return (
            camera_model_from_camera_info(ctx.bag, info_topic),
            f"CameraInfo {info_topic}",
            "",
        )
    raise CheckSkipError(
        "missing_intrinsics",
        f"no intrinsics for {image_topic}: give a Kalibr camchain with --tf or record a "
        "CameraInfo topic next to the image topic",
    )


def run_camera_imu(ctx: PairContext) -> EstimatorRun:
    """``T_cam_imu`` rotation from tracked image features against the gyro."""

    if importlib.util.find_spec("cv2") is None:
        raise CheckSkipError(
            "missing_dependency", "camera-imu needs OpenCV: pip install 'calibrex[opencv]'"
        )
    from calibrex.core.imu_lidar_rotation import load_imu_lidar_rotation
    from calibrex.evaluation.camera_imu_rotation import (
        CameraImuRunOptions,
        run_ros2_camera_imu_rotation,
    )
    from calibrex.evaluation.imu_lidar_rotation import rebase_rotation_reference

    camera_topics = ctx.sensor_topics.get("camera", ())
    image_topic = next((t for t in camera_topics if ctx.topic_types.get(t) == IMAGE_TYPE), None)
    if image_topic is None:
        raise CheckSkipError(
            "unsupported_sensor",
            f"camera topic(s) {', '.join(camera_topics)} are not sensor_msgs/Image "
            "(compressed images are not supported by the check yet)",
        )
    imu_topics = ctx.sensor_topics.get("imu", ())
    if not imu_topics:
        raise CheckSkipError("missing_topic", "no IMU topic for the pair")
    camera_frame = ctx.pair.frames[0]
    model, intrinsics_source, _key = resolve_camera_intrinsics(ctx, image_topic, camera_frame)
    ctx.controls.progress(
        f"camera-imu: {image_topic} + {imu_topics[0]} (intrinsics: {intrinsics_source})"
    )
    candidate = ctx.candidate  # T_cam_imu
    run_options = CameraImuRunOptions(max_seconds=ctx.controls.max_duration_s)
    artifact, cached = cached_artifact(
        ctx.controls.cache,
        ctx.controls.bag_sha256,
        "camera_imu_rotation",
        {
            "image_topic": image_topic,
            "imu_topic": imu_topics[0],
            "camera": model,
            "acceleration_unit": ctx.controls.acceleration_unit,
            "options": run_options,
        },
        loader=load_imu_lidar_rotation,
        compute=lambda: run_ros2_camera_imu_rotation(
            [ctx.bag],
            image_topic=image_topic,
            imu_topic=imu_topics[0],
            camera=model,
            dataset_family=DATASET_FAMILY,
            dataset_license=DATASET_LICENSE,
            acceleration_unit=ctx.controls.acceleration_unit,
            reference_rotation=candidate[:3, :3],
            reference="deployed candidate calibration (T_cam_imu)",
            options=run_options,
            command=["calibrex", "check", str(ctx.bag), "camera-imu", image_topic, imu_topics[0]],
        ),
        rebase=lambda hit: rebase_rotation_reference(hit, candidate[:3, :3]),
    )
    solved = artifact.rotation_quat_xyzw is not None and artifact.solver_status == "converged"
    if artifact.rotation_quat_xyzw is not None:
        estimates = rotation_axis_estimates(
            artifact.dofs, candidate[:3, :3], _pose_rotation(artifact)
        )
    else:
        estimates = [
            _unestimated(item)
            for item in rotation_axis_estimates(artifact.dofs, candidate[:3, :3], candidate[:3, :3])
        ]
    return EstimatorRun(
        estimator=f"{artifact.method} (camera)",
        artifacts=(
            EvidenceArtifact(
                "rotation",
                artifact,
                artifact.policy_status,
                cached if ctx.controls.cache is not None else None,
            ),
        ),
        solved=solved,
        policy_status=artifact.policy_status,
        policy_reasons=tuple(artifact.policy_reasons),
        estimates=tuple(estimates),
        compared=candidate,
        time_offset=time_offset_of(artifact.dofs, artifact.time_offset_s),
        notes=(f"intrinsics: {intrinsics_source}",),
    )


# ------------------------------------------------------------- vehicle pairs
#
# lidar-vehicle, imu-vehicle, ins-lidar and lidar-wheel_odometry run the estimators
# of the KITTI benchmarks on bag data (see ``vehicle_inputs``). None of them uses
# the candidate: it enters only when the estimate is compared with it, so the
# artifacts are cached. Sweeps are registered as rigid snapshots, like the KITTI
# runners do; a per-point time field is neither required nor used.

ODOMETRY_TYPE = "nav_msgs/msg/Odometry"
TWIST_TYPES = frozenset(
    {"geometry_msgs/msg/TwistStamped", "geometry_msgs/msg/TwistWithCovarianceStamped"}
)
VEHICLE_PROVENANCE_GENERATOR = "calibrex.check.estimators"
_RIGID_SWEEP_NOTE = (
    "LiDAR sweeps are registered as rigid snapshots; motion within a sweep is not compensated"
)
_PLANAR_NOTE = (
    "The vehicle frame is defined by the motion: it assumes no side slip and no vertical "
    "velocity, and roll needs turns."
)


def _digest_or_topic(ctx: PairContext, *names: str) -> str:
    """The bag digest for artifact provenance (a digest of the topic names without one)."""

    if ctx.controls.bag_sha256:
        return ctx.controls.bag_sha256
    return hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()


def _lidar_odometry(ctx: PairContext, topic: str) -> Any:
    """LiDAR odometry of a topic, computed once per run and shared by the vehicle pairs."""

    from calibrex.check import vehicle_inputs

    key = ("lidar_odometry", str(ctx.bag), topic, ctx.controls.max_duration_s)
    track = ctx.controls.memo.get(key)
    if track is None:
        ctx.controls.progress(f"vehicle pairs: LiDAR odometry on {topic}")
        track = vehicle_inputs.read_lidar_odometry(
            ctx.bag, topic, max_duration_s=ctx.controls.max_duration_s
        )
        ctx.controls.memo[key] = track
    return track


SEGMENT_BLOCK_STRIDE = 100_000
"""Block-number offset between stream segments, so no block (and no motion) spans two."""


def _lidar_motions(track: Any, *, step: int, block_duration_s: float) -> list[Any]:
    """Blocked LiDAR motions of every odometry segment, pooled as the KITTI runners pool drives."""

    from calibrex.solvers.vehicle_frame_solver import motions_from_poses

    motions: list[Any] = []
    for position, (times, poses) in enumerate(track.segment_slices()):
        motions += motions_from_poses(
            times,
            poses,
            step=step,
            block_duration_s=block_duration_s,
            block_offset=position * SEGMENT_BLOCK_STRIDE,
        )
    return motions


def _segment_note(track: Any) -> tuple[str, ...]:
    count = len(track.segment_slices())
    if count < 2:
        return ()
    return (
        f"the LiDAR stream has {count} segments (gaps over 1 s); odometry and motions are "
        "kept within a segment",
    )


def _rotation_estimates_of(
    records: Sequence[Any], rotation_quat_xyzw: Sequence[float] | None, candidate: FloatArray
) -> list[AxisEstimate]:
    """Roll/pitch/yaw estimates against a candidate rotation of the same convention."""

    if rotation_quat_xyzw is None:
        estimates = rotation_axis_estimates(records, candidate, candidate)
        return [_unestimated(item) for item in estimates]
    from scipy.spatial.transform import Rotation

    estimate = np.asarray(Rotation.from_quat(rotation_quat_xyzw).as_matrix(), dtype=np.float64)
    return rotation_axis_estimates(records, candidate, estimate)


def _vehicle_run(
    artifact: Any,
    *,
    estimator: str,
    cached: bool,
    ctx: PairContext,
    estimates: Sequence[AxisEstimate],
    compared: FloatArray,
    solved: bool,
    time_offset: CheckTimeOffset | None = None,
    notes: Sequence[str] = (),
    role: str = "estimate",
) -> EstimatorRun:
    return EstimatorRun(
        estimator=estimator,
        artifacts=(
            EvidenceArtifact(
                role,
                artifact,
                artifact.policy_status,
                cached if ctx.controls.cache is not None else None,
            ),
        ),
        solved=solved,
        policy_status=artifact.policy_status,
        policy_reasons=tuple(artifact.policy_reasons),
        estimates=tuple(estimates),
        compared=compared,
        time_offset=time_offset,
        notes=tuple(notes),
    )


def run_lidar_vehicle(ctx: PairContext) -> EstimatorRun:
    """``R_vehicle_lidar`` from the non-holonomic motion of the LiDAR odometry."""

    from calibrex import __version__
    from calibrex.core.provenance import git_commit
    from calibrex.core.vehicle_frame_rotation import (
        VehicleFrameProvenance,
        load_vehicle_frame_rotation,
    )
    from calibrex.evaluation.vehicle_frame import (
        VehicleFrameRunOptions,
        build_vehicle_frame_artifact,
        evaluate_vehicle_frame,
    )

    lidar_topic = _require_pointcloud2(ctx, ctx.sensor_topics.get("lidar", ()))
    options = VehicleFrameRunOptions()
    candidate = invert_transform(ctx.candidate)  # T_vehicle_lidar

    def compute() -> Any:
        track = _lidar_odometry(ctx, lidar_topic)
        motions = _lidar_motions(
            track, step=options.step_frames, block_duration_s=options.block_duration_s
        )
        evaluation = evaluate_vehicle_frame(motions, options)
        return build_vehicle_frame_artifact(
            evaluation,
            modality="lidar",
            motions=len(motions),
            references=[],
            options=options,
            include_step=True,
            limitations=[
                _PLANAR_NOTE,
                _RIGID_SWEEP_NOTE,
                "The translation is not estimated.",
                *_segment_note(track),
            ],
            provenance=VehicleFrameProvenance(
                generator=VEHICLE_PROVENANCE_GENERATOR,
                generator_version=__version__,
                git_commit=git_commit(),
                command=["calibrex", "check", str(ctx.bag), "lidar-vehicle", lidar_topic],
                dataset_family=DATASET_FAMILY,
                sequence_ids=[ctx.bag.name],
                sensor=lidar_topic,
                input_sha256=_digest_or_topic(ctx, lidar_topic),
                dataset_license=DATASET_LICENSE,
            ),
        )

    artifact, cached = cached_artifact(
        ctx.controls.cache,
        ctx.controls.bag_sha256,
        "lidar_vehicle",
        {
            "topic": lidar_topic,
            "max_duration_s": ctx.controls.max_duration_s,
            "options": options,
        },
        loader=load_vehicle_frame_rotation,
        compute=compute,
        rebase=lambda hit: hit,
    )
    return _vehicle_run(
        artifact,
        estimator="vehicle_frame_rotation (lidar)",
        cached=cached,
        ctx=ctx,
        estimates=_rotation_estimates_of(
            artifact.dofs, artifact.rotation_quat_xyzw, candidate[:3, :3]
        ),
        compared=candidate,
        solved=artifact.rotation_quat_xyzw is not None,
        notes=(
            f"LiDAR odometry on {lidar_topic} ({_RIGID_SWEEP_NOTE})",
            "roll needs turns; an axis the drive does not constrain is listed as unchecked",
            "the translation of T_vehicle_lidar is not estimated",
        ),
    )


def _topic_records(ctx: PairContext, slot: str) -> list[CheckTopicRecord]:
    from calibrex.check.planner import slot_of

    return [record for record in ctx.topics if slot_of(record) == slot]


def run_imu_vehicle(ctx: PairContext) -> EstimatorRun:
    """``R_vehicle_imu`` from the body-frame velocity and rates of an INS in the IMU's frame."""

    from calibrex import __version__
    from calibrex.check import vehicle_inputs
    from calibrex.core.provenance import git_commit
    from calibrex.core.vehicle_frame_rotation import (
        VehicleFrameProvenance,
        load_vehicle_frame_rotation,
    )
    from calibrex.evaluation.vehicle_frame import (
        VehicleFrameRunOptions,
        body_motions,
        build_vehicle_frame_artifact,
        evaluate_vehicle_frame,
    )

    imu_frame = ctx.pair.frames[0]
    candidates = [
        record
        for record in _topic_records(ctx, "ins")
        if record.message_type == ODOMETRY_TYPE or record.message_type in TWIST_TYPES
    ]
    if not candidates:
        raise CheckSkipError(
            "missing_topic",
            "imu-vehicle needs the body-frame velocity and angular rate of an INS (a raw IMU "
            "has no velocity): an Odometry or TwistStamped topic of kind 'ins'; name it with "
            "oxts/ins/gnss or pass --topic-kind TOPIC=ins",
        )
    same_frame = [record for record in candidates if record.mapped_frame == imu_frame]
    if not same_frame:
        raise CheckSkipError(
            "unsupported_sensor",
            "the INS velocity topic(s) "
            + ", ".join(f"{r.topic} ({r.mapped_frame})" for r in candidates)
            + f" are not expressed in the IMU frame '{imu_frame}'",
        )
    # Odometry first (its twist is in the child frame by definition), then the busiest.
    same_frame.sort(key=lambda r: (r.message_type != ODOMETRY_TYPE, -(r.message_count or 0)))
    source = same_frame[0]
    options = VehicleFrameRunOptions()
    candidate = invert_transform(ctx.candidate)  # T_vehicle_imu

    def compute() -> Any:
        track = vehicle_inputs.read_twist_track(
            ctx.bag,
            source.topic,
            source.message_type,
            max_duration_s=ctx.controls.max_duration_s,
        )
        if not np.any(np.abs(track.linear_mps) > 1e-9):
            raise CheckSkipError(
                "unsupported_sensor", f"{source.topic} reports zero velocity in every message"
            )
        motions = []
        for position, piece in enumerate(vehicle_inputs.split_at_gaps(track.times_s)):
            motions += body_motions(
                list(track.times_s[piece]),
                list(track.linear_mps[piece]),
                list(track.angular_rps[piece]),
                block_duration_s=options.block_duration_s,
                block_offset=position * SEGMENT_BLOCK_STRIDE,
            )
        evaluation = evaluate_vehicle_frame(motions, options)
        return build_vehicle_frame_artifact(
            evaluation,
            modality="ins",
            motions=len(motions),
            references=[],
            options=options,
            include_step=False,
            limitations=[
                _PLANAR_NOTE,
                f"The velocity and rates are the INS's own navigation solution on {source.topic}, "
                "taken as expressed in the body frame of the IMU.",
                "The translation is not estimated.",
            ],
            provenance=VehicleFrameProvenance(
                generator=VEHICLE_PROVENANCE_GENERATOR,
                generator_version=__version__,
                git_commit=git_commit(),
                command=["calibrex", "check", str(ctx.bag), "imu-vehicle", source.topic],
                dataset_family=DATASET_FAMILY,
                sequence_ids=[ctx.bag.name],
                sensor=source.topic,
                input_sha256=_digest_or_topic(ctx, source.topic),
                dataset_license=DATASET_LICENSE,
            ),
        )

    artifact, cached = cached_artifact(
        ctx.controls.cache,
        ctx.controls.bag_sha256,
        "imu_vehicle",
        {
            "topic": source.topic,
            "max_duration_s": ctx.controls.max_duration_s,
            "options": options,
        },
        loader=load_vehicle_frame_rotation,
        compute=compute,
        rebase=lambda hit: hit,
    )
    return _vehicle_run(
        artifact,
        estimator="vehicle_frame_rotation (ins)",
        cached=cached,
        ctx=ctx,
        estimates=_rotation_estimates_of(
            artifact.dofs, artifact.rotation_quat_xyzw, candidate[:3, :3]
        ),
        compared=candidate,
        solved=artifact.rotation_quat_xyzw is not None,
        notes=(
            f"velocity and angular rate from {source.topic} ({source.message_type}), "
            f"taken as expressed in '{imu_frame}'; the IMU topic itself carries no velocity",
            "the translation of T_vehicle_imu is not estimated",
        ),
    )


def run_ins_lidar(ctx: PairContext) -> EstimatorRun:
    """``T_ins_lidar`` by trajectory hand-eye calibration of LiDAR odometry against an INS."""

    from calibrex import __version__
    from calibrex.check import vehicle_inputs
    from calibrex.core.ins_lidar_hand_eye import (
        InsLidarHandEyeProvenance,
        load_ins_lidar_hand_eye,
    )
    from calibrex.core.provenance import git_commit
    from calibrex.evaluation.ins_lidar_hand_eye import (
        InsLidarRunOptions,
        _odometry_summary,
        build_ins_lidar_artifact,
        build_sensor_motions,
        evaluate_ins_lidar_hand_eye,
    )
    from calibrex.solvers.trajectory_hand_eye_solver import ReferenceTrajectory

    lidar_topic = _require_pointcloud2(ctx, ctx.sensor_topics.get("lidar", ()))
    ins_topics = [
        topic
        for topic in ctx.sensor_topics.get("ins", ())
        if ctx.topic_types.get(topic) == ODOMETRY_TYPE
    ]
    if not ins_topics:
        raise CheckSkipError(
            "unsupported_sensor",
            "ins-lidar needs an INS pose: a nav_msgs/Odometry topic of kind 'ins' "
            f"(found {', '.join(ctx.sensor_topics.get('ins', ())) or 'none'})",
        )
    ins_topic = ins_topics[0]
    options = InsLidarRunOptions()
    candidate = ctx.candidate  # T_ins_lidar

    def compute() -> Any:
        ins = vehicle_inputs.read_ins_track(
            ctx.bag, ins_topic, max_duration_s=ctx.controls.max_duration_s
        )
        track = _lidar_odometry(ctx, lidar_topic)
        times = list(track.times_s)
        motions = []
        for position, (piece_times, piece_poses) in enumerate(track.segment_slices()):
            motions += build_sensor_motions(
                piece_times,
                piece_poses,
                options,
                block_offset=position * SEGMENT_BLOCK_STRIDE,
                id_prefix=f"segment{position}/",
            )
        reference = ReferenceTrajectory(ins.times_s, ins.poses)
        evaluation = evaluate_ins_lidar_hand_eye(reference, motions, options)
        return build_ins_lidar_artifact(
            evaluation,
            options,
            odometry=_odometry_summary(list(track.registrations), len(times), []),
            provenance=InsLidarHandEyeProvenance(
                generator=VEHICLE_PROVENANCE_GENERATOR,
                generator_version=__version__,
                git_commit=git_commit(),
                command=["calibrex", "check", str(ctx.bag), "ins-lidar", ins_topic, lidar_topic],
                dataset_family=DATASET_FAMILY,
                drive_ids=[ctx.bag.name],
                input_sha256=_digest_or_topic(ctx, ins_topic, lidar_topic),
                dataset_license=DATASET_LICENSE,
            ),
            reference_calibration=None,
            limitations=[
                _RIGID_SWEEP_NOTE + ".",
                "A reference scale and a clock offset are estimated as nuisance parameters "
                "because INS tracks are not metrically exact and their stamps are logging times.",
                "Vehicle motion is close to planar, so the translation along the vertical "
                "rotation axis cannot be observed without a prior.",
                *_segment_note(track),
            ],
        )

    artifact, cached = cached_artifact(
        ctx.controls.cache,
        ctx.controls.bag_sha256,
        "ins_lidar",
        {
            "ins_topic": ins_topic,
            "lidar_topic": lidar_topic,
            "max_duration_s": ctx.controls.max_duration_s,
            "options": options,
        },
        loader=load_ins_lidar_hand_eye,
        compute=compute,
        rebase=lambda hit: hit,
    )
    estimates: list[AxisEstimate]
    transform = artifact.transform
    solved = transform is not None and artifact.solver_status == "converged"
    if transform is not None:
        estimates = _rotation_estimates_of(
            artifact.dofs, transform.rotation_quat_xyzw, candidate[:3, :3]
        )
        estimates += translation_axis_estimates(
            [r for r in artifact.dofs if r.name in {"x", "y", "z"}],
            candidate[:3, 3],
            transform.translation_m,
        )
    else:
        estimates = [
            _unestimated(item)
            for item in rotation_axis_estimates(artifact.dofs, candidate[:3, :3], candidate[:3, :3])
        ]
    return _vehicle_run(
        artifact,
        estimator="ins_lidar_hand_eye",
        cached=cached,
        ctx=ctx,
        estimates=estimates,
        compared=candidate,
        solved=solved,
        time_offset=time_offset_of(artifact.dofs, artifact.time_offset_s or 0.0),
        notes=(
            f"INS pose from {ins_topic}, LiDAR odometry on {lidar_topic} ({_RIGID_SWEEP_NOTE})",
            "an axis constrained only by a prior or by nothing is listed as unchecked",
        ),
    )


def run_lidar_wheel(ctx: PairContext) -> EstimatorRun:
    """``R_wheel_lidar`` from LiDAR odometry against wheel speed and yaw rate."""

    from calibrex import __version__
    from calibrex.check import vehicle_inputs
    from calibrex.core.lidar_wheel_odometry import (
        LidarWheelProvenance,
        load_lidar_wheel_odometry,
    )
    from calibrex.core.provenance import git_commit
    from calibrex.evaluation.lidar_wheel import (
        LIMITATIONS,
        LidarWheelRunOptions,
        build_artifact,
        evaluate_lidar_wheel,
    )
    from calibrex.solvers.lidar_wheel_solver import WheelOdometry

    lidar_topic = _require_pointcloud2(ctx, ctx.sensor_topics.get("lidar", ()))
    wheel_topics = [
        topic
        for topic in ctx.sensor_topics.get("wheel", ())
        if ctx.topic_types.get(topic) == ODOMETRY_TYPE or ctx.topic_types.get(topic) in TWIST_TYPES
    ]
    if not wheel_topics:
        raise CheckSkipError(
            "unsupported_sensor",
            "lidar-wheel_odometry needs wheel speed and yaw rate from an Odometry or "
            f"TwistStamped topic (found {', '.join(ctx.sensor_topics.get('wheel', ())) or 'none'})",
        )
    wheel_topic = wheel_topics[0]
    wheel_type = ctx.topic_types[wheel_topic]
    options = LidarWheelRunOptions()
    candidate = invert_transform(ctx.candidate)  # T_wheel_lidar

    def compute() -> Any:
        track = _lidar_odometry(ctx, lidar_topic)
        motions = _lidar_motions(
            track, step=options.step_frames, block_duration_s=options.block_duration_s
        )
        twist = vehicle_inputs.read_twist_track(
            ctx.bag, wheel_topic, wheel_type, max_duration_s=ctx.controls.max_duration_s
        )
        wheel = WheelOdometry(twist.times_s, twist.linear_mps[:, 0], twist.angular_rps[:, 2])
        evaluation = evaluate_lidar_wheel(motions, wheel, options)
        return build_artifact(
            evaluation,
            options,
            motions=len(motions),
            references=[],
            provenance=LidarWheelProvenance(
                generator=VEHICLE_PROVENANCE_GENERATOR,
                generator_version=__version__,
                git_commit=git_commit(),
                command=["calibrex", "check", str(ctx.bag), "lidar-wheel", wheel_topic],
                dataset_family=DATASET_FAMILY,
                sequence_ids=[ctx.bag.name],
                wheel_source=f"{wheel_topic} ({wheel_type}): linear.x speed, angular.z yaw rate",
                input_sha256=_digest_or_topic(ctx, wheel_topic, lidar_topic),
                dataset_license=DATASET_LICENSE,
            ),
            limitations=[*LIMITATIONS, _RIGID_SWEEP_NOTE + ".", *_segment_note(track)],
        )

    artifact, cached = cached_artifact(
        ctx.controls.cache,
        ctx.controls.bag_sha256,
        "lidar_wheel",
        {
            "wheel_topic": wheel_topic,
            "lidar_topic": lidar_topic,
            "max_duration_s": ctx.controls.max_duration_s,
            "options": options,
        },
        loader=load_lidar_wheel_odometry,
        compute=compute,
        rebase=lambda hit: hit,
    )
    return _vehicle_run(
        artifact,
        estimator="lidar_wheel_odometry",
        cached=cached,
        ctx=ctx,
        estimates=_rotation_estimates_of(
            artifact.parameters, artifact.rotation_quat_xyzw, candidate[:3, :3]
        ),
        compared=candidate,
        solved=artifact.rotation_quat_xyzw is not None,
        time_offset=time_offset_of(artifact.parameters, artifact.time_offset_s or 0.0),
        notes=(
            f"wheel speed (linear.x) and yaw rate (angular.z) from {wheel_topic} ({wheel_type}); "
            f"LiDAR odometry on {lidar_topic} ({_RIGID_SWEEP_NOTE})",
            "only the rotation is judged: the lever, speed scale and clock offset are reported "
            "in the evidence artifact, not compared with the candidate",
        ),
    )


ESTIMATORS: dict[str, Adapter] = {
    "imu-lidar": run_imu_lidar,
    "lidar-lidar": run_lidar_lidar,
    "camera-imu": run_camera_imu,
    "lidar-vehicle": run_lidar_vehicle,
    "imu-vehicle": run_imu_vehicle,
    "ins-lidar": run_ins_lidar,
    "lidar-wheel_odometry": run_lidar_wheel,
}
WIRED_CHECK_PAIRS: frozenset[str] = frozenset(ESTIMATORS)
