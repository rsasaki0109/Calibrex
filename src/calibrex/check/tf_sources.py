"""Readers for candidate static extrinsics: bag ``/tf_static`` and calibration files.

Each reader returns a :class:`LoadedSource` with ``T_parent_child`` edges as
:class:`~calibrex.core.geometry.SE3`, optional topic/role frame hints, the
SHA-256 of what was read, and notes about conventions that were assumed.
:func:`merge_sources` layers sources into one :class:`StaticFrameTree`.
"""

from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from calibrex.check.frame_tree import FrameHints, StaticEdge, StaticFrameTree, normalize_frame_id
from calibrex.check.hints import template_refusal
from calibrex.core.calibration_check import CandidateSourceKind, CheckFramesFile
from calibrex.core.exceptions import DatasetError, FrameGraphError
from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.data.ros_cdr import decode_ros2_tf_message
from calibrex.data.rosbag2 import (
    TF_MESSAGE_TYPE,
    iter_topic_messages,
    list_rosbag2_connections,
    require_cdr_connection,
)

IMU_FRAME_NAME = "imu"
_MAX_TF_STATIC_MESSAGES = 100_000
_ROTATION_TOLERANCE = 1e-3
_CAM_KEY = re.compile(r"^cam\d+$")
_LIDAR_KEY = re.compile(r"^lidar\d+$")


@dataclass(frozen=True)
class LoadedSource:
    """Static edges read from one origin."""

    kind: CandidateSourceKind
    path: Path | None
    sha256: str | None
    edges: tuple[StaticEdge, ...]
    hints: FrameHints = field(default_factory=FrameHints)
    notes: tuple[str, ...] = ()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- bag


def load_bag_tf_static(bag: str | Path) -> LoadedSource | None:
    """Read ``tf_static`` from a bag: the latest transform per child frame.

    A child frame that appears under two different parents is rejected: the
    candidate tree must be unambiguous. Returns ``None`` when the bag has no
    ``tf_static`` topic.
    """

    topics = [
        (connection, count)
        for connection, count in list_rosbag2_connections(bag)
        if connection.message_type == TF_MESSAGE_TYPE
        and connection.topic.rstrip("/").endswith("tf_static")
    ]
    if not topics:
        return None
    latest: dict[str, tuple[str, SE3, str]] = {}
    value_updates = 0
    message_count = 0
    digest = hashlib.sha256()
    for connection, count in topics:
        limit = count if count else _MAX_TF_STATIC_MESSAGES
        for _conn, timestamp_ns, data in iter_topic_messages(bag, connection.topic, limit=limit):
            require_cdr_connection(_conn)
            message_count += 1
            digest.update(connection.topic.encode("utf-8"))
            digest.update(data)
            message = decode_ros2_tf_message(connection.topic, timestamp_ns, data)
            for transform in message.transforms:
                parent = normalize_frame_id(transform.frame_id)
                child = normalize_frame_id(transform.child_frame_id)
                pose = SE3(transform.translation_m, transform.rotation_xyzw)
                previous = latest.get(child)
                if previous is not None:
                    if previous[0] != parent:
                        msg = (
                            f"{connection.topic}: child frame '{child}' appears with conflicting "
                            f"parents '{previous[0]}' and '{parent}'"
                        )
                        raise FrameGraphError(msg)
                    if previous[1] != pose:
                        value_updates += 1
                latest[child] = (parent, pose, connection.topic)
    notes = [f"read {message_count} tf_static message(s) from {len(topics)} topic(s)"]
    if value_updates:
        notes.append(
            f"{value_updates} later message(s) changed a transform value; the latest is used"
        )
    edges = tuple(
        StaticEdge(parent=parent, child=child, transform=pose, source="bag_tf_static")
        for child, (parent, pose, _topic) in sorted(latest.items())
    )
    return LoadedSource(
        kind="bag_tf_static",
        path=None,
        sha256=digest.hexdigest() if message_count else None,
        edges=edges,
        notes=tuple(notes),
    )


# -------------------------------------------------------------------- file entry


def load_tf_file(path: str | Path) -> LoadedSource:
    """Read a candidate calibration file, detecting its format.

    Supported: URDF (fixed joints), ``slac.check_frames/v0.1`` YAML, Kalibr
    camchain-imucam YAML, RTK-SLAM ``calib.yaml``, the Hilti
    ``lidar_calibration.yaml`` sensor list, FAST-LIO-family configs
    (FAST-LIO, FAST-LIO2, Point-LIO, FAST-LIVO, FAST-LIVO2), and the vikit
    ``Pinhole`` camera file FAST-LIVO reads its intrinsics from.
    """

    file_path = Path(path)
    if not file_path.is_file():
        msg = f"calibration file does not exist: {file_path}"
        raise DatasetError(msg)
    digest = sha256_file(file_path)
    text = file_path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if file_path.suffix.lower() in {".urdf", ".xml"} or stripped.startswith("<"):
        return _load_urdf(file_path, text, digest)
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"{file_path}: not valid YAML or URDF: {exc}"
        raise DatasetError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"{file_path}: expected a YAML mapping"
        raise DatasetError(msg)
    refusal = template_refusal(file_path, payload)
    if refusal is not None:
        raise DatasetError(refusal)
    if str(payload.get("schema_version", "")).startswith("slac.check_frames/"):
        return _load_frames_yaml(file_path, payload, digest)
    if isinstance(payload.get("sensors"), dict):
        return _load_hilti_sensors(file_path, payload["sensors"], digest)
    keys = [str(key) for key in payload]
    if "reference_offsets" in payload or any(_LIDAR_KEY.match(key) for key in keys):
        return _load_rtk_slam_calib(file_path, payload, digest)
    if any(_CAM_KEY.match(key) for key in keys):
        return _load_kalibr_camchain(file_path, payload, digest)
    if _fast_lio_section(payload) is not None:
        return _load_fast_lio_config(file_path, payload, digest)
    if "cam_model" in payload and "cam_fx" in payload:
        return _load_vikit_camera(file_path, payload, digest)
    msg = (
        f"{file_path}: unrecognized calibration format (expected URDF, slac.check_frames, "
        "Kalibr camchain-imucam, RTK-SLAM calib.yaml, a Hilti sensors list, a "
        "FAST-LIO-family config, or a FAST-LIVO camera file)"
    )
    raise DatasetError(msg)


def merge_sources(
    bag_source: LoadedSource | None,
    file_sources: Sequence[LoadedSource],
) -> tuple[StaticFrameTree, list[str]]:
    """Layer sources into one tree.

    Files take precedence over bag ``tf_static`` for the same child frame and
    each override is reported. Two files that give the same child frame
    different parents or transforms are rejected.
    """

    overrides: list[str] = []
    chosen: dict[str, StaticEdge] = {}
    origin: dict[str, str] = {}
    if bag_source is not None:
        for edge in bag_source.edges:
            chosen[edge.child] = edge
            origin[edge.child] = "bag tf_static"
    file_children: dict[str, tuple[StaticEdge, str]] = {}
    for source in file_sources:
        label = str(source.path) if source.path is not None else source.kind
        for edge in source.edges:
            seen = file_children.get(edge.child)
            if seen is not None and (
                seen[0].parent != edge.parent or seen[0].transform != edge.transform
            ):
                msg = (
                    f"frame '{edge.child}' is defined differently by {seen[1]} and {label}; "
                    "give each frame once or make the definitions identical"
                )
                raise FrameGraphError(msg)
            file_children[edge.child] = (edge, label)
            existing = chosen.get(edge.child)
            if (
                existing is not None
                and origin[edge.child] == "bag tf_static"
                and (existing.parent != edge.parent or existing.transform != edge.transform)
            ):
                overrides.append(
                    f"{label} overrides bag tf_static for frame '{edge.child}' "
                    f"(parent '{existing.parent}' -> '{edge.parent}')"
                )
            chosen[edge.child] = edge
            origin[edge.child] = label
    return StaticFrameTree(chosen.values()), overrides


def merge_hints(sources: Sequence[LoadedSource]) -> FrameHints:
    """Merge topic and role frame hints; later sources win."""

    topic_frames: dict[str, str] = {}
    role_frames: dict[str, str] = {}
    for source in sources:
        topic_frames.update(source.hints.topic_frames)
        role_frames.update(source.hints.role_frames)
    return FrameHints(topic_frames=topic_frames, role_frames=role_frames)


# ------------------------------------------------------------------------- URDF


def _parse_floats(text: str | None, count: int, where: str) -> list[float]:
    if text is None:
        return [0.0] * count
    try:
        values = [float(item) for item in text.split()]
    except ValueError as exc:
        msg = f"{where}: non-numeric value {text!r} (run xacro first)"
        raise DatasetError(msg) from exc
    if len(values) != count or not all(math.isfinite(item) for item in values):
        msg = f"{where}: expected {count} finite numbers, got {text!r}"
        raise DatasetError(msg)
    return values


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (xyzw) of the ROS fixed-axis rpy rotation ``Rz(yaw) Ry(pitch) Rx(roll)``."""

    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _load_urdf(path: Path, text: str, digest: str) -> LoadedSource:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        msg = f"{path}: invalid URDF XML: {exc}"
        raise DatasetError(msg) from exc
    if root.tag != "robot":
        msg = f"{path}: URDF root element must be <robot>, found <{root.tag}>"
        raise DatasetError(msg)
    edges: list[StaticEdge] = []
    skipped = 0
    for joint in root.findall("joint"):
        name = joint.get("name", "<unnamed>")
        if joint.get("type") != "fixed":
            skipped += 1
            continue
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            msg = f"{path}: joint '{name}' needs <parent> and <child>"
            raise DatasetError(msg)
        parent = parent_el.get("link")
        child = child_el.get("link")
        if not parent or not child:
            msg = f"{path}: joint '{name}' has an empty parent or child link"
            raise DatasetError(msg)
        origin = joint.find("origin")
        xyz = _parse_floats(None if origin is None else origin.get("xyz"), 3, f"joint '{name}' xyz")
        rpy = _parse_floats(None if origin is None else origin.get("rpy"), 3, f"joint '{name}' rpy")
        pose = SE3((xyz[0], xyz[1], xyz[2]), quaternion_from_rpy(rpy[0], rpy[1], rpy[2]))
        edges.append(
            StaticEdge(
                parent=normalize_frame_id(parent),
                child=normalize_frame_id(child),
                transform=pose,
                source="urdf",
            )
        )
    if not edges:
        msg = f"{path}: URDF has no fixed joints"
        raise DatasetError(msg)
    notes = [f"{len(edges)} fixed joint(s) read"]
    if skipped:
        notes.append(f"{skipped} non-fixed joint(s) ignored")
    return LoadedSource(
        kind="urdf", path=path, sha256=digest, edges=tuple(edges), notes=tuple(notes)
    )


# ------------------------------------------------------------------ frames YAML


def _load_frames_yaml(path: Path, payload: dict[str, Any], digest: str) -> LoadedSource:
    try:
        model = CheckFramesFile.model_validate(payload)
    except ValidationError as exc:
        msg = f"{path}: invalid slac.check_frames file: {exc}"
        raise DatasetError(msg) from exc
    edges = tuple(
        StaticEdge(
            parent=normalize_frame_id(entry.parent),
            child=normalize_frame_id(entry.name),
            transform=SE3.from_lists(entry.translation_m, entry.rotation_quat_xyzw),
            source="frames_yaml",
        )
        for entry in model.frames
    )
    return LoadedSource(
        kind="frames_yaml",
        path=path,
        sha256=digest,
        edges=edges,
        hints=FrameHints(
            topic_frames={
                topic: normalize_frame_id(frame) for topic, frame in model.topic_frames.items()
            },
            role_frames={
                role: normalize_frame_id(frame) for role, frame in model.role_frames.items()
            },
        ),
        notes=(f"{len(edges)} frame(s) read",),
    )


# ---------------------------------------------------------------- matrix helpers


def _matrix_to_se3(value: Any, where: str) -> SE3:
    try:
        rows = [[float(item) for item in row] for row in value]
    except (TypeError, ValueError) as exc:
        msg = f"{where}: expected a 4x4 numeric matrix"
        raise DatasetError(msg) from exc
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        msg = f"{where}: expected a 4x4 matrix"
        raise DatasetError(msg)
    if any(abs(a - b) > 1e-9 for a, b in zip(rows[3], [0.0, 0.0, 0.0, 1.0], strict=True)):
        msg = f"{where}: last row must be [0, 0, 0, 1]"
        raise DatasetError(msg)
    rotation = [rows[i][:3] for i in range(3)]
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[i][k] * rotation[j][k] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > _ROTATION_TOLERANCE:
                msg = f"{where}: rotation block is not orthonormal"
                raise DatasetError(msg)
    det = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if det < 0.0:
        msg = f"{where}: rotation block is a reflection (determinant {det:.3f})"
        raise DatasetError(msg)
    quaternion = quaternion_xyzw_from_rotation_matrix(
        [rotation[i][j] for i in range(3) for j in range(3)]
    )
    return SE3((rows[0][3], rows[1][3], rows[2][3]), quaternion)


def _camera_frame_name(key: str, entry: dict[str, Any]) -> tuple[str, str | None]:
    topic = entry.get("rostopic")
    if isinstance(topic, str) and topic.strip():
        return normalize_frame_id(topic), topic
    return key, None


def camchain_entries(path: Path) -> list[tuple[str, str, dict[str, Any]]]:
    """Return ``(key, frame name, entry)`` of every camera in a camchain-style file."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    entries: list[tuple[str, str, dict[str, Any]]] = []
    for key, entry in payload.items():
        if _CAM_KEY.match(str(key)) and isinstance(entry, dict):
            entries.append((str(key), _camera_frame_name(str(key), entry)[0], entry))
    return entries


# ----------------------------------------------------------------------- Kalibr


def _camchain_edges(
    path: Path,
    payload: dict[str, Any],
    kind: CandidateSourceKind,
) -> tuple[list[StaticEdge], dict[str, str], list[str]]:
    edges: list[StaticEdge] = []
    topic_frames: dict[str, str] = {}
    notes: list[str] = []
    for key, entry in payload.items():
        if not _CAM_KEY.match(str(key)) or not isinstance(entry, dict):
            continue
        if "T_cam_imu" not in entry:
            if "intrinsics" not in entry:
                notes.append(f"{key}: no T_cam_imu; ignored")
            continue
        t_cam_imu = _matrix_to_se3(entry["T_cam_imu"], f"{path} {key}.T_cam_imu")
        frame, topic = _camera_frame_name(str(key), entry)
        if topic is None:
            notes.append(f"{key}: no rostopic; the frame is named '{frame}'")
        else:
            topic_frames[topic] = frame
        edges.append(
            StaticEdge(
                parent=IMU_FRAME_NAME,
                child=frame,
                transform=t_cam_imu.inverse(),
                source=kind,
            )
        )
    return edges, topic_frames, notes


def _load_kalibr_camchain(path: Path, payload: dict[str, Any], digest: str) -> LoadedSource:
    edges, topic_frames, notes = _camchain_edges(path, payload, "kalibr_camchain")
    intrinsics_only = [
        str(key)
        for key, entry in payload.items()
        if _CAM_KEY.match(str(key))
        and isinstance(entry, dict)
        and "T_cam_imu" not in entry
        and "intrinsics" in entry
    ]
    if not edges and not intrinsics_only:
        msg = f"{path}: no camN.T_cam_imu or camN.intrinsics entries found"
        raise DatasetError(msg)
    for key in intrinsics_only:
        frame, topic = _camera_frame_name(key, payload[key])
        if topic is not None:
            topic_frames[topic] = frame
    notes.insert(
        0,
        f"Kalibr T_cam_imu inverted to T_{IMU_FRAME_NAME}_cam; the IMU frame is named "
        f"'{IMU_FRAME_NAME}' and camera frames take their rostopic as name",
    )
    if intrinsics_only:
        notes.append(
            "intrinsics only (no T_cam_imu), the extrinsics come from another source: "
            + ", ".join(intrinsics_only)
        )
    return LoadedSource(
        kind="kalibr_camchain",
        path=path,
        sha256=digest,
        edges=tuple(edges),
        hints=FrameHints(topic_frames=topic_frames, role_frames={"imu": IMU_FRAME_NAME}),
        notes=tuple(notes),
    )


# -------------------------------------------------------------------- FAST-LIO


FAST_LIO_LIDAR_FRAME = "lidar"


def _fast_lio_section(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The mapping that holds ``extrinsic_T`` and ``extrinsic_R``, if any.

    FAST-LIO, FAST-LIO2, Point-LIO and FAST-LIVO keep them under ``mapping``,
    FAST-LIVO2 under ``extrin_calib``.
    """

    for name in ("mapping", "extrin_calib"):
        section = payload.get(name)
        if isinstance(section, dict) and "extrinsic_T" in section and "extrinsic_R" in section:
            return section
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rotation_translation_se3(rotation: Any, translation: Any, where: str) -> SE3:
    """``[R | t]`` from a row-major list of 9 and a list of 3, checked like a matrix."""

    try:
        values = [float(item) for item in rotation]
        offset = [float(item) for item in translation]
    except (TypeError, ValueError) as exc:
        msg = f"{where}: expected 9 rotation and 3 translation numbers"
        raise DatasetError(msg) from exc
    if len(values) != 9 or len(offset) != 3:
        msg = f"{where}: expected 9 rotation and 3 translation numbers"
        raise DatasetError(msg)
    rows = [[*values[3 * i : 3 * i + 3], offset[i]] for i in range(3)]
    return _matrix_to_se3([*rows, [0.0, 0.0, 0.0, 1.0]], where)


def _load_fast_lio_config(path: Path, payload: dict[str, Any], digest: str) -> LoadedSource:
    """LiDAR-IMU (and FAST-LIVO camera-LiDAR) extrinsics of a FAST-LIO-family config.

    ``extrinsic_R``, ``extrinsic_T`` are the LiDAR's pose in the IMU frame
    (``p_imu = R p_lidar + T``): ``T_imu_lidar``. FAST-LIVO's ``Rcl``, ``Pcl`` map
    LiDAR points into the camera (``p_cam = Rcl p_lidar + Pcl``): ``T_cam_lidar``.
    """

    section = _fast_lio_section(payload)
    assert section is not None
    t_imu_lidar = _rotation_translation_se3(
        section["extrinsic_R"], section["extrinsic_T"], f"{path} extrinsic_R/extrinsic_T"
    )
    edges = [
        StaticEdge(
            parent=IMU_FRAME_NAME,
            child=FAST_LIO_LIDAR_FRAME,
            transform=t_imu_lidar,
            source="fast_lio_config",
        )
    ]
    common = _mapping(payload.get("common"))
    topic_frames: dict[str, str] = {}
    for key, frame in (("lid_topic", FAST_LIO_LIDAR_FRAME), ("imu_topic", IMU_FRAME_NAME)):
        topic = common.get(key)
        if isinstance(topic, str) and topic.strip():
            topic_frames[topic.strip()] = frame
    notes = [
        f"extrinsic_R/extrinsic_T read as T_{IMU_FRAME_NAME}_{FAST_LIO_LIDAR_FRAME} (the LiDAR "
        f"in the IMU frame); the frames are named '{IMU_FRAME_NAME}' and "
        f"'{FAST_LIO_LIDAR_FRAME}' and common.lid_topic/imu_topic map the topics to them"
    ]
    if section.get("extrinsic_est_en") is True:
        notes.append(
            "extrinsic_est_en is true: FAST-LIO refines the extrinsic online, so the file "
            "holds its initial value, not the one it ran with"
        )
    camera = _mapping(payload.get("camera"))
    holder = next((part for part in (section, camera) if "Rcl" in part and "Pcl" in part), None)
    if holder is not None:
        t_cam_lidar = _rotation_translation_se3(holder["Rcl"], holder["Pcl"], f"{path} Rcl/Pcl")
        image_topic = next(
            (
                value.strip()
                for value in (camera.get("img_topic"), common.get("img_topic"))
                if isinstance(value, str) and value.strip()
            ),
            None,
        )
        camera_frame = normalize_frame_id(image_topic) if image_topic else "camera"
        if image_topic:
            topic_frames[image_topic] = camera_frame
        edges.append(
            StaticEdge(
                parent=FAST_LIO_LIDAR_FRAME,
                child=camera_frame,
                transform=t_cam_lidar.inverse(),
                source="fast_lio_config",
            )
        )
        notes.append(
            f"Rcl/Pcl read as T_cam_{FAST_LIO_LIDAR_FRAME} and inverted; the camera frame is "
            f"named '{camera_frame}'"
            + ("" if image_topic else " (no img_topic in the file)")
            + ". Its intrinsics come from FAST-LIVO's camera file (another --tf), a Kalibr "
            "camchain with the camera's intrinsics and rostopic, or CameraInfo"
        )
    return LoadedSource(
        kind="fast_lio_config",
        path=path,
        sha256=digest,
        edges=tuple(edges),
        hints=FrameHints(
            topic_frames=topic_frames,
            role_frames={"imu": IMU_FRAME_NAME, "lidar": FAST_LIO_LIDAR_FRAME},
        ),
        notes=tuple(notes),
    )


# ------------------------------------------------------- FAST-LIVO camera (vikit)


def vikit_camera_entry(path: Path) -> dict[str, Any]:
    """A Kalibr-style camera entry (and its ``resolution``) of a vikit ``Pinhole`` file.

    FAST-LIVO reads ``cam_fx``, ``cam_fy``, ``cam_cx``, ``cam_cy`` and the radial-
    tangential ``cam_d0`` to ``cam_d3`` (``cam_d4`` if given) for an image of
    ``cam_width`` x ``cam_height`` pixels.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path}: expected a YAML mapping"
        raise DatasetError(msg)
    model = str(payload.get("cam_model", ""))
    if model.lower() != "pinhole":
        msg = f"{path}: cam_model {model!r} is not supported (only Pinhole)"
        raise DatasetError(msg)
    try:
        intrinsics = [float(payload[key]) for key in ("cam_fx", "cam_fy", "cam_cx", "cam_cy")]
        resolution = [int(payload["cam_width"]), int(payload["cam_height"])]
        coeffs = [float(payload.get(f"cam_d{i}", 0.0)) for i in range(4)]
        if "cam_d4" in payload:
            coeffs.append(float(payload["cam_d4"]))
    except (KeyError, TypeError, ValueError) as exc:
        msg = (
            f"{path}: a Pinhole camera needs numeric cam_width, cam_height, cam_fx, cam_fy, "
            "cam_cx and cam_cy"
        )
        raise DatasetError(msg) from exc
    if min(resolution) <= 0 or min(intrinsics[:2]) <= 0.0:
        msg = f"{path}: cam_width, cam_height, cam_fx and cam_fy must be positive"
        raise DatasetError(msg)
    return {
        "camera_model": "pinhole",
        "intrinsics": intrinsics,
        "distortion_model": "radtan",
        "distortion_coeffs": coeffs,
        "resolution": resolution,
    }


def scale_camera_entry(
    entry: dict[str, Any], width: int, height: int
) -> tuple[dict[str, Any], float]:
    """``(entry for a width x height image, scale)``: pixel units scale with the image.

    ``fx``, ``fy`` scale by ``k = width / resolution[0]``; principal points keep their
    place relative to pixel centres, ``c' = k (c + 0.5) - 0.5``. The distortion acts on
    normalized coordinates and is kept. Raises :class:`ValueError` when the image
    does not keep the calibrated aspect ratio.
    """

    calibrated_width, calibrated_height = (int(value) for value in entry["resolution"])
    if width * calibrated_height != height * calibrated_width:
        msg = (
            f"the images are {width}x{height} but the intrinsics are for "
            f"{calibrated_width}x{calibrated_height}, another aspect ratio"
        )
        raise ValueError(msg)
    scale = width / calibrated_width
    fx, fy, cx, cy = (float(value) for value in entry["intrinsics"])
    scaled = dict(entry)
    scaled["intrinsics"] = [
        scale * fx,
        scale * fy,
        scale * (cx + 0.5) - 0.5,
        scale * (cy + 0.5) - 0.5,
    ]
    scaled["resolution"] = [width, height]
    return scaled, scale


def _load_vikit_camera(path: Path, payload: dict[str, Any], digest: str) -> LoadedSource:
    entry = vikit_camera_entry(path)
    width, height = entry["resolution"]
    notes = [
        f"FAST-LIVO camera intrinsics for {width}x{height} (radtan); they apply to the camera "
        "frame of a FAST-LIVO config given with --tf, scaled to the recorded image size"
    ]
    if "scale" in payload:
        notes.append(
            f"scale {payload['scale']} is FAST-LIVO's own processing scale and is not applied"
        )
    return LoadedSource(kind="vikit_camera", path=path, sha256=digest, edges=(), notes=tuple(notes))


# --------------------------------------------------------------------- RTK-SLAM


def _load_rtk_slam_calib(path: Path, payload: dict[str, Any], digest: str) -> LoadedSource:
    edges, topic_frames, notes = _camchain_edges(path, payload, "rtk_slam_calib")
    role_frames: dict[str, str] = {"imu": IMU_FRAME_NAME}
    lidar_keys = sorted(str(key) for key in payload if _LIDAR_KEY.match(str(key)))
    for key in lidar_keys:
        entry = payload[key]
        if not isinstance(entry, dict) or "T_lidar_imu" not in entry:
            notes.append(f"{key}: no T_lidar_imu; ignored")
            continue
        edges.append(
            StaticEdge(
                parent=IMU_FRAME_NAME,
                child=key,
                transform=_matrix_to_se3(entry["T_lidar_imu"], f"{path} {key}.T_lidar_imu"),
                source="rtk_slam_calib",
            )
        )
        role_frames.setdefault("lidar", key)
    if lidar_keys:
        notes.append(
            "lidarN.T_lidar_imu is read as the repository does (maps LiDAR points into the IMU "
            f"frame), so the edge is T_{IMU_FRAME_NAME}_lidarN"
        )
    offsets = payload.get("reference_offsets")
    if isinstance(offsets, dict) and "gnss_antenna_phase_center" in offsets:
        xyz = _parse_floats(
            " ".join(str(item) for item in offsets["gnss_antenna_phase_center"]),
            3,
            "reference_offsets.gnss_antenna_phase_center",
        )
        edges.append(
            StaticEdge(
                parent=IMU_FRAME_NAME,
                child="gnss_antenna",
                transform=SE3((xyz[0], xyz[1], xyz[2]), (0.0, 0.0, 0.0, 1.0)),
                source="rtk_slam_calib",
            )
        )
        role_frames["gnss"] = "gnss_antenna"
        notes.append("gnss_antenna is a CAD point offset; its orientation is assumed identity")
    if offsets is not None and isinstance(offsets, dict) and "base_center" in offsets:
        notes.append(
            "reference_offsets.base_center has no orientation and is not turned into a vehicle "
            "frame; give a base_link frame through another --tf file to audit vehicle pairs"
        )
    if not edges:
        msg = f"{path}: no camN.T_cam_imu or lidarN.T_lidar_imu entries found"
        raise DatasetError(msg)
    return LoadedSource(
        kind="rtk_slam_calib",
        path=path,
        sha256=digest,
        edges=tuple(edges),
        hints=FrameHints(topic_frames=topic_frames, role_frames=role_frames),
        notes=tuple(notes),
    )


# ------------------------------------------------------------------------ Hilti


def _load_hilti_sensors(path: Path, sensors: dict[str, Any], digest: str) -> LoadedSource:
    edges: list[StaticEdge] = []
    for name, entry in sensors.items():
        if not isinstance(entry, dict) or "parent" not in entry:
            continue
        extrinsics = entry.get("extrinsics")
        if not isinstance(extrinsics, dict):
            msg = f"{path}: sensor '{name}' has no extrinsics"
            raise DatasetError(msg)
        translation = _parse_floats(
            " ".join(str(item) for item in extrinsics.get("translation", [])),
            3,
            f"{name}.translation",
        )
        quaternion = _parse_floats(
            " ".join(str(item) for item in extrinsics.get("quaternion", [])),
            4,
            f"{name}.quaternion",
        )
        edges.append(
            StaticEdge(
                parent=normalize_frame_id(str(entry["parent"])),
                child=normalize_frame_id(str(name)),
                transform=SE3(
                    (translation[0], translation[1], translation[2]),
                    (quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
                ),
                source="hilti_sensors",
            )
        )
    if not edges:
        msg = f"{path}: no sensors with a parent found"
        raise DatasetError(msg)
    roles = {"imu": "imu"} if any(edge.child == "imu" for edge in edges) else {}
    return LoadedSource(
        kind="hilti_sensors",
        path=path,
        sha256=digest,
        edges=tuple(edges),
        hints=FrameHints(role_frames=roles),
        notes=("extrinsics are read as T_parent_sensor with quaternion order x, y, z, w",),
    )
