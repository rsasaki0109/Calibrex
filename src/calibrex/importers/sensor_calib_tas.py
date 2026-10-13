"""Import a sensor_calib_tas result as a ``calibrex check --tf`` frame file.

`sensor_calib_tas <https://github.com/UniBwTAS/sensor_calib_tas>`_ (UniBw TAS,
BSD-3-Clause) calibrates cameras, LiDARs, radars and GNSS against a target. It writes
no calibration file: after solving it publishes every sensor's pose in its reference
sensor's frame twice,

* as static transforms ``reference -> <sensor frame>_calibrated`` on ``/tf_static``,
  next to the vehicle's own (uncalibrated) ``<sensor frame>``;
* as the latched ``sensor_calib_msgs/EstimatedPoses`` message ``~/sensor_poses``, whose
  ``poses`` carry the sensor frame names as they are.

This module reads either one, from a bag recorded during the session or from a YAML
file saved with ``ros2 topic echo --once <node>/sensor_poses``, and writes the poses as
a ``slac.check_frames/v0.1`` file under the sensors' own frame names. Given to
``calibrex check`` with ``--tf``, it overrides the bag's ``tf_static`` for those frames,
so the check judges the target calibration on ordinary driving data. Nothing of
sensor_calib_tas's code is used: only its published message layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from calibrex.check.frame_tree import normalize_frame_id
from calibrex.check.tf_sources import sha256_file
from calibrex.core.calibration_check import (
    CheckFramesEntry,
    CheckFramesFile,
    CheckFramesProvenance,
)
from calibrex.core.provenance import git_commit
from calibrex.data.ros_cdr import decode_ros2_tf_message
from calibrex.data.rosbag2 import TF_MESSAGE_TYPE, iter_topic_messages, list_rosbag2_connections

TOOL = "sensor_calib_tas"
TOOL_LICENSE = "BSD-3-Clause"
CALIBRATED_SUFFIX = "_calibrated"
SourceKind = Literal["bag_tf_static", "estimated_poses_yaml"]


@dataclass(frozen=True)
class TasPose:
    """One sensor's pose in its reference frame: ``T_reference_sensor``."""

    reference_frame: str
    sensor_frame: str
    translation_m: tuple[float, float, float]
    rotation_quat_xyzw: tuple[float, float, float, float]


def read_sensor_calib_tas_poses(source: str | Path) -> tuple[list[TasPose], SourceKind]:
    """The calibrated sensor poses of a bag or of an echoed ``sensor_poses`` YAML file.

    A bag is any file or directory ``calibrex check`` reads (rosbag2, MCAP, ROS 1 bag);
    a ``.yaml``/``.yml`` file is ``ros2 topic echo`` output (the last message is used).
    Raises :class:`ValueError` when the source holds no calibrated pose.
    """

    path = Path(source)
    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
        return _poses_from_echo(path), "estimated_poses_yaml"
    return _poses_from_bag(path), "bag_tf_static"


def sensor_calib_tas_frames(
    source: str | Path, *, tool_version: str | None = None
) -> CheckFramesFile:
    """A ``slac.check_frames`` file with the calibrated poses under their sensors' names."""

    from calibrex.check.runner import bag_input_digest

    path = Path(source)
    poses, kind = read_sensor_calib_tas_poses(path)
    if kind == "estimated_poses_yaml":
        digest, scope = sha256_file(path), "the whole file"
    else:
        digest, scope, _storage = bag_input_digest(path)
    references = sorted({pose.reference_frame for pose in poses})
    notes = [
        f"{len(poses)} sensor pose(s) relative to {', '.join(references)}",
        "the reference sensor's own pose is not part of the result: keep it from the "
        "deployed calibration",
    ]
    if tool_version:
        notes.append(f"{TOOL} version: {tool_version}")
    return CheckFramesFile(
        frames=[
            CheckFramesEntry(
                name=pose.sensor_frame,
                parent=pose.reference_frame,
                translation_m=list(pose.translation_m),
                rotation_quat_xyzw=list(pose.rotation_quat_xyzw),
            )
            for pose in poses
        ],
        provenance=CheckFramesProvenance(
            generator="calibrex.importers.sensor_calib_tas",
            generator_version="0.1",
            git_commit=git_commit(),
            tool=TOOL,
            tool_license=TOOL_LICENSE,
            source_kind=kind,
            source_name=path.name,
            source_sha256=digest,
            source_digest_scope=scope,
            notes=notes,
        ),
    )


def _poses_from_bag(bag: Path) -> list[TasPose]:
    """The latest ``*_calibrated`` transform per sensor on the bag's ``tf_static``.

    Read here rather than with :func:`~calibrex.check.tf_sources.load_bag_tf_static`:
    with its preview on, sensor_calib_tas also publishes the poses under the sensors'
    own names, which then appear under two parents (its reference and the vehicle's
    tree) and would be refused as a candidate tree.
    """

    latest: dict[str, TasPose] = {}
    for connection, _count in list_rosbag2_connections(bag):
        if connection.message_type != TF_MESSAGE_TYPE or not connection.topic.rstrip("/").endswith(
            "tf_static"
        ):
            continue
        for _conn, stamp, data in iter_topic_messages(bag, connection.topic):
            for transform in decode_ros2_tf_message(connection.topic, stamp, data).transforms:
                child = normalize_frame_id(transform.child_frame_id)
                sensor = child.removesuffix(CALIBRATED_SUFFIX)
                reference = normalize_frame_id(transform.frame_id)
                if not sensor or sensor in (child, reference):
                    continue
                latest[sensor] = TasPose(
                    reference_frame=reference,
                    sensor_frame=sensor,
                    translation_m=_triple(transform.translation_m),
                    rotation_quat_xyzw=_quadruple(transform.rotation_xyzw),
                )
    if not latest:
        msg = (
            f"{bag}: no '*{CALIBRATED_SUFFIX}' frame on tf_static; sensor_calib_tas publishes "
            "them once the optimization has run (record /tf_static after solving)"
        )
        raise ValueError(msg)
    return [latest[sensor] for sensor in sorted(latest)]


def _poses_from_echo(path: Path) -> list[TasPose]:
    documents = [
        document
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get("poses")
    ]
    if not documents:
        msg = f"{path}: no EstimatedPoses message with poses (ros2 topic echo ~/sensor_poses)"
        raise ValueError(msg)
    poses: list[TasPose] = []
    for entry in documents[-1]["poses"]:
        try:
            header = entry["header"]
            transform = entry["transform"]
            translation = transform["translation"]
            rotation = transform["rotation"]
            poses.append(
                TasPose(
                    reference_frame=_frame(header["frame_id"]),
                    sensor_frame=_frame(entry["child_frame_id"]).removesuffix(CALIBRATED_SUFFIX),
                    translation_m=_triple([translation[axis] for axis in "xyz"]),
                    rotation_quat_xyzw=_quadruple([rotation[axis] for axis in "xyzw"]),
                )
            )
        except (KeyError, TypeError) as exc:
            msg = f"{path}: a pose is not a geometry_msgs/TransformStamped ({exc!r})"
            raise ValueError(msg) from exc
    poses = [pose for pose in poses if pose.sensor_frame != pose.reference_frame]
    return sorted(poses, key=lambda pose: pose.sensor_frame)


def _frame(value: Any) -> str:
    frame = normalize_frame_id(str(value))
    if not frame:
        raise ValueError("empty frame id")
    return frame


def _triple(values: Any) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in values)
    return x, y, z


def _quadruple(values: Any) -> tuple[float, float, float, float]:
    x, y, z, w = (float(value) for value in values)
    return x, y, z, w
