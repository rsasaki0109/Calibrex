"""Estimate the calibration of a bag that has none (``calibrex estimate``).

This reuses the machinery of ``calibrex check``: topic classification, the pair
planner and the native estimators. Where ``check`` judges a candidate against the
estimate, ``estimate`` keeps the estimate itself, with each axis's uncertainty
and whether the data observed it, and exports the observed frames so that
``calibrex check <other bag> --tf frames.yaml`` can verify them on a different
recording.

Most estimators do not use the candidate transform at all (it is only the
reference of the evidence artifact), so a placeholder identity candidate is
passed to them. LiDAR-LiDAR map registration starts from the candidate, so it
runs only when a rough ``--tf`` prior connects its two frames. A frame whose
transform has an axis the data did not observe is written to the exported tree
only when a ``--tf`` prior supplies that axis (and the YAML says so); otherwise
it is omitted and listed.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from calibrex import __version__
from calibrex.check import estimators
from calibrex.check.estimators import EstimatorRun, invert_transform, transform_matrix
from calibrex.check.frame_tree import StaticEdge, StaticFrameTree, normalize_frame_id
from calibrex.check.hints import pair_hint
from calibrex.check.planner import PAIR_SLOTS, plan_pairs, slot_of
from calibrex.check.progress import CheckProgress, as_progress
from calibrex.check.roles import map_topics_to_frames
from calibrex.check.runner import (
    INVERTED_CONVENTION_PAIRS,
    CheckRunOptions,
    _run_pairs,
    _skipped,
    bag_input_digest,
    classify_bag_topics,
)
from calibrex.check.tf_sources import (
    LoadedSource,
    load_bag_tf_static,
    load_tf_file,
    merge_hints,
    merge_sources,
    sha256_file,
)
from calibrex.core.bag_estimate import (
    BagEstimateArtifact,
    EstimateAxis,
    EstimateExportFile,
    EstimateFill,
    EstimateFrameEntry,
    EstimateFrameOmission,
    EstimateFrames,
    EstimateOptions,
    EstimatePairRecord,
    EstimateSummary,
)
from calibrex.core.calibration_check import (
    CHECK_FRAMES_SCHEMA_VERSION,
    CheckAxisName,
    CheckBagInput,
    CheckCandidateSource,
    CheckPairName,
    CheckPairRecord,
    CheckProvenance,
    CheckTopicRecord,
    CheckTransform,
    OdometryKind,
)
from calibrex.core.exceptions import FrameGraphError
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit
from calibrex.data.rosbag2 import list_rosbag2_connections

PLACEHOLDER_ROOT = "__estimate_placeholder__"
"""Frame that joins sensor frames no prior connects; never exported."""
NEEDS_INITIAL_GUESS = frozenset({"lidar-lidar"})
"""Pairs whose estimator starts from the candidate transform."""
NOT_ESTIMATED_PAIRS = frozenset({"camera-focal"})
"""Pairs that judge intrinsics, not an extrinsic: there is nothing to estimate."""
ESTIMATE_PAIRS: frozenset[str] = frozenset(estimators.WIRED_CHECK_PAIRS - NOT_ESTIMATED_PAIRS)

ROTATION_AXES: tuple[CheckAxisName, ...] = ("roll", "pitch", "yaw")
TRANSLATION_AXES: tuple[CheckAxisName, ...] = ("x", "y", "z")
PAIR_ORDER: dict[str, int] = {name: index for index, name in enumerate(PAIR_SLOTS)}

FRAMES_FILENAME = "frames.yaml"
PUBLISHER_FILENAME = "static_transforms.sh"
LAUNCH_FILENAME = "static_transforms.launch.yaml"
URDF_FILENAME = "joints.urdf.xml"
CAMCHAIN_FILENAME = "camchain-imucam.yaml"
ARTIFACT_FILENAME = "bag_estimate.json"

LIDAR_LIDAR_PRIOR_HINT = (
    "lidar-lidar registration starts from a rough extrinsic: pass a prior with --tf FILE "
    "(a slac.check_frames YAML or URDF that connects the two LiDAR frames to within a few "
    "degrees and decimetres); only that start is used, the estimate comes from the bag"
)


# ----------------------------------------------------------------------- relations


@dataclass(frozen=True)
class Relation:
    """A pair's estimate as a frame relation ``T_parent_child`` for the export tree."""

    pair: CheckPairName
    parent: str
    child: str
    matrix: np.ndarray[Any, Any]
    observed: frozenset[str]
    informed: bool
    """A ``--tf`` prior supplied the values of the axes that are not observed."""
    prior: np.ndarray[Any, Any] | None = None
    """The prior's transform in the same convention (what the unobserved axes come from)."""

    @property
    def translation_free(self) -> bool:
        """No translation axis was observed: the lever arm is entirely the prior's."""

        return not any(name in self.observed for name in TRANSLATION_AXES)

    def edge_matrix(self, node: str) -> np.ndarray[Any, Any]:
        """``T_node_other`` for the export edge leaving ``node``.

        When the data observed no translation axis, the prior's lever arm is kept as
        the position of the other frame in ``node`` (it does not turn with the
        estimated rotation); otherwise the estimate's translation is used.
        """

        forward = self.parent == node
        matrix = self.matrix if forward else invert_transform(self.matrix)
        if self.informed and self.translation_free and self.prior is not None:
            prior = self.prior if forward else invert_transform(self.prior)
            matrix = matrix.copy()
            matrix[:3, 3] = prior[:3, 3]
        return matrix

    @property
    def missing(self) -> list[CheckAxisName]:
        names: tuple[CheckAxisName, ...] = (*ROTATION_AXES, *TRANSLATION_AXES)
        return [name for name in names if name not in self.observed]

    @property
    def complete(self) -> bool:
        return not self.missing or self.informed


def _rotation_matrix(rotvec_rad: Sequence[float]) -> np.ndarray[Any, Any]:
    from scipy.spatial.transform import Rotation

    return np.asarray(Rotation.from_rotvec(np.asarray(rotvec_rad)).as_matrix(), dtype=np.float64)


def _transform_record(matrix: np.ndarray[Any, Any], parent: str, child: str) -> CheckTransform:
    from scipy.spatial.transform import Rotation

    return CheckTransform(
        parent_frame=parent,
        child_frame=child,
        translation_m=[float(v) for v in matrix[:3, 3]],
        rotation_quat_xyzw=[float(v) for v in Rotation.from_matrix(matrix[:3, :3]).as_quat()],
    )


def _estimate_axes(
    outcome: EstimatorRun,
) -> tuple[np.ndarray[Any, Any], list[EstimateAxis], frozenset[str]]:
    """The estimate matrix (observed axes applied to the compared candidate) and its axes.

    ``AxisEstimate.candidate_error`` is the candidate minus the estimate, so the
    estimate is recovered exactly: ``R_est = Exp(e)^T R_candidate`` and
    ``t_est = t_candidate - e``. Axes that are not observed keep the candidate's
    value, so the returned matrix holds the estimate only on observed axes.
    """

    compared = np.asarray(outcome.compared, dtype=np.float64)
    by_name = {item.name: item for item in outcome.estimates}
    observed = frozenset(name for name, item in by_name.items() if item.estimated)
    rotation_error = [
        math.radians(by_name[name].candidate_error) if name in observed else 0.0
        for name in ROTATION_AXES
    ]
    result = np.eye(4)
    result[:3, :3] = _rotation_matrix(rotation_error).T @ compared[:3, :3]
    result[:3, 3] = [
        compared[index, 3] - by_name[name].candidate_error
        if name in observed
        else compared[index, 3]
        for index, name in enumerate(TRANSLATION_AXES)
    ]
    from scipy.spatial.transform import Rotation

    rotvec_deg = np.degrees(Rotation.from_matrix(result[:3, :3]).as_rotvec())
    axes: list[EstimateAxis] = []
    for name in (*ROTATION_AXES, *TRANSLATION_AXES):
        unit: Literal["deg", "m"] = "deg" if name in ROTATION_AXES else "m"
        item = by_name.get(name)
        if item is None:
            axes.append(
                EstimateAxis(
                    name=name,
                    unit=unit,
                    status="not_estimated",
                    reason="this estimator does not estimate this axis",
                )
            )
            continue
        if item.estimated:
            index = ROTATION_AXES.index(name) if unit == "deg" else TRANSLATION_AXES.index(name)
            value = float(rotvec_deg[index]) if unit == "deg" else float(result[index, 3])
            axes.append(
                EstimateAxis(
                    name=name, unit=unit, value=value, std=float(item.std), status="observed"
                )
            )
            continue
        axes.append(
            EstimateAxis(
                name=name,
                unit=unit,
                std=float(item.std) if item.unchecked_code != "no_estimate" else None,
                status=item.unchecked_code or "no_estimate",
                reason=item.unchecked_reason,
            )
        )
    return result, axes, observed


def _withdrawn(axes: Sequence[EstimateAxis], reason: str | None) -> list[EstimateAxis]:
    """Axes of a pair whose estimate is not trusted: nothing is observed."""

    return [
        axis
        if axis.status == "not_estimated"
        else EstimateAxis(
            name=axis.name, unit=axis.unit, status="no_estimate", reason=reason or axis.reason
        )
        for axis in axes
    ]


def _pair_record(
    record: CheckPairRecord, outcome: EstimatorRun | None, *, informed: bool
) -> tuple[EstimatePairRecord, Relation | None]:
    base: dict[str, Any] = {
        "pair": record.pair,
        "sensors": record.sensors,
        "topics": record.topics,
        "frames": record.frames,
        "evidence": record.evidence,
        "evidence_from_cache": record.evidence_from_cache,
        "runtime_s": record.runtime_s,
        "notes": record.notes,
    }
    if outcome is None:
        status = "skipped" if record.status == "skipped" else "failed"  # else: it raised
        return (
            EstimatePairRecord.model_validate(
                {
                    **base,
                    "status": status,
                    "reason_code": record.reason_code,
                    "reason": record.reason,
                }
            ),
            None,
        )
    matrix, axes, observed = _estimate_axes(outcome)
    attempted = [axis for axis in axes if axis.status != "not_estimated"]
    reason_code = None
    reason = None
    if not outcome.solved:
        status = "failed"
        reason_code = "estimator_failed"
        reason = "the estimator did not produce an estimate: " + "; ".join(outcome.policy_reasons)
    elif outcome.policy_status == "fail":
        status = "failed"
        reason_code = "estimator_failed"
        reason = "the estimator failed its own held-out check, so its estimate is not trusted: " + (
            "; ".join(outcome.policy_reasons)
        )
    elif not observed:
        status = "failed"
        reason_code = "no_judgeable_axes"
        reason = "no axis was constrained by the data"
    elif all(axis.status == "observed" for axis in attempted):
        status = "estimated"
    else:
        status = "partial"
    parent, child = (
        (record.frames[1], record.frames[0])
        if record.pair in INVERTED_CONVENTION_PAIRS
        else (record.frames[0], record.frames[1])
    )
    fill: EstimateFill = "prior" if informed else "placeholder_identity"
    usable = status in {"estimated", "partial"}
    pair = EstimatePairRecord.model_validate(
        {
            **base,
            "notes": [*record.notes, *outcome.notes],
            "status": status,
            "reason_code": reason_code,
            "reason": reason,
            "estimator": outcome.estimator,
            "estimator_policy_status": outcome.policy_status,
            "estimator_policy_reasons": list(outcome.policy_reasons),
            "transform": _transform_record(matrix, parent, child) if usable else None,
            "fill": fill if usable else None,
            "axes": axes if usable else _withdrawn(axes, reason),
            "time_offset": outcome.time_offset,
            "deskew": outcome.deskew,
        }
    )
    relation = (
        Relation(
            record.pair,
            parent,
            child,
            matrix,
            frozenset(observed),
            informed,
            np.asarray(outcome.compared, dtype=np.float64),
        )
        if usable
        else None
    )
    return pair, relation


# ------------------------------------------------------------------- frame tree


def _prior_depth(prior: StaticFrameTree | None, frame: str) -> float:
    """Distance of a frame from its root in the prior tree (infinity when the prior lacks it)."""

    if prior is None or frame not in prior:
        return math.inf
    parents = {edge.child: edge.parent for edge in prior.edges}
    depth = 0
    while frame in parents:
        frame = parents[frame]
        depth += 1
    return float(depth)


def _choose_root(
    relations: Sequence[Relation],
    topics: Sequence[CheckTopicRecord],
    vehicle_frame: str | None,
    prior: StaticFrameTree | None,
) -> str | None:
    """The export root: the vehicle frame, else the frame nearest the prior's root, else the IMU.

    Rooting at the prior's shallowest frame keeps the exported edges oriented the way the
    prior's own tree is, so ``--tf frames.yaml`` next to the bag's ``/tf_static`` stays a tree.
    """

    frames = list(dict.fromkeys(f for r in relations for f in (r.parent, r.child)))
    if not frames:
        return None
    if vehicle_frame is not None and vehicle_frame in frames:
        return vehicle_frame
    imu_frames = {t.mapped_frame for t in topics if slot_of(t) == "imu" and t.mapped_frame}
    return min(
        frames,
        key=lambda frame: (
            _prior_depth(prior, frame),
            frame not in imu_frames,
            frames.index(frame),
        ),
    )


def _lookup_relations(relations: Sequence[Relation]) -> dict[str, list[Relation]]:
    by_frame: dict[str, list[Relation]] = {}
    for relation in relations:
        by_frame.setdefault(relation.parent, []).append(relation)
        by_frame.setdefault(relation.child, []).append(relation)
    return by_frame


def build_frame_tree(
    relations: Sequence[Relation],
    pairs: Sequence[EstimatePairRecord],
    topics: Sequence[CheckTopicRecord],
    vehicle_frame: str | None,
    prior: StaticFrameTree | None = None,
) -> tuple[EstimateFrames, list[StaticEdge]]:
    """The exported tree: complete relations reachable from the root, the rest omitted.

    A relation is complete when every axis is observed, or when a ``--tf`` prior
    supplied the axes that are not. Frames reachable only through incomplete
    relations (or through no estimate) are listed as omitted with the reason.
    """

    complete = sorted(
        (r for r in relations if r.complete),
        key=lambda r: (len(r.missing), PAIR_ORDER.get(r.pair, 99)),
    )
    root = _choose_root(complete, topics, vehicle_frame, prior)
    entries: list[EstimateFrameEntry] = []
    edges: list[StaticEdge] = []
    reached: set[str] = set()
    if root is not None:
        reached.add(root)
        by_frame = _lookup_relations(complete)
        queue: deque[str] = deque([root])
        while queue:
            node = queue.popleft()
            for relation in by_frame.get(node, []):
                other = relation.child if relation.parent == node else relation.parent
                if other in reached:
                    continue
                matrix = relation.edge_matrix(node)
                reached.add(other)
                queue.append(other)
                record = _transform_record(matrix, node, other)
                edges.append(
                    StaticEdge(
                        parent=node,
                        child=other,
                        transform=SE3.from_lists(record.translation_m, record.rotation_quat_xyzw),
                        source="frames_yaml",
                    )
                )
                entries.append(
                    EstimateFrameEntry(
                        frame=other,
                        parent=node,
                        pair=relation.pair,
                        axes_from_prior=relation.missing,
                        transform=record,
                    )
                )
    omitted: list[EstimateFrameOmission] = []
    seen: set[str] = set()
    for pair in pairs:
        if pair.status == "skipped" and pair.reason_code in {"not_selected", "method_not_wired"}:
            continue
        for frame in pair.frames:
            if frame in reached or frame in seen:
                continue
            seen.add(frame)
            touching = next((r for r in relations if frame in (r.parent, r.child)), None)
            if touching is not None and not touching.complete:
                reason = (
                    f"{touching.pair}: the data did not observe {', '.join(touching.missing)}; "
                    "give a rough --tf prior for the frame to fill them"
                )
            elif touching is not None:
                reason = f"{touching.pair}: not connected to the root frame '{root}'"
            else:
                why = pair.reason or pair.status
                reason = f"{pair.pair}: no estimate ({why})"
            omitted.append(EstimateFrameOmission(frame=frame, pair=pair.pair, reason=reason))
    return EstimateFrames(root=root, entries=entries, omitted=omitted), edges


# ----------------------------------------------------------------------- exports


def _num(value: float) -> str:
    return f"{value:.9g}"


def frames_yaml_text(
    frames: EstimateFrames,
    *,
    bag_name: str,
    bag_sha256: str,
    topic_frames: Mapping[str, str],
) -> str:
    """A ``slac.check_frames`` YAML of the observed frames (comments say what was measured)."""

    lines = [
        f"# calibrex estimate: frames estimated from bag {bag_name} (sha256 {bag_sha256[:16]}).",
        "# Verify on a different recording: calibrex check <other bag> --tf THIS_FILE",
        f"# Root frame: {frames.root} (not an entry; it is the parent of the first frames).",
        "# Transforms are T_parent_frame (p_parent = T p_frame), quaternion x, y, z, w.",
    ]
    for omission in frames.omitted:
        lines.append(f"# omitted frame {omission.frame}: {omission.reason}")
    lines += [f"schema_version: {CHECK_FRAMES_SCHEMA_VERSION}", "frames:"]
    for entry in frames.entries:
        quality = (
            "all six axes observed by the data"
            if not entry.axes_from_prior
            else f"NOT MEASURED: {', '.join(entry.axes_from_prior)} taken from the --tf prior"
        )
        transform = entry.transform
        lines += [
            f"  - name: {entry.frame}  # {entry.pair}: {quality}",
            f"    parent: {entry.parent}",
            f"    translation_m: [{', '.join(_num(v) for v in transform.translation_m)}]",
            f"    rotation_quat_xyzw: [{', '.join(_num(v) for v in transform.rotation_quat_xyzw)}]",
        ]
    if topic_frames:
        lines.append("topic_frames:")
        lines += [
            f"  {json.dumps(topic)}: {json.dumps(frame)}" for topic, frame in topic_frames.items()
        ]
    return "\n".join(lines) + "\n"


def _publisher_args(entry: EstimateFrameEntry) -> list[str]:
    t = entry.transform
    x, y, z = t.translation_m
    qx, qy, qz, qw = t.rotation_quat_xyzw
    return [
        "--x", _num(x), "--y", _num(y), "--z", _num(z),
        "--qx", _num(qx), "--qy", _num(qy), "--qz", _num(qz), "--qw", _num(qw),
        "--frame-id", entry.parent, "--child-frame-id", entry.frame,
    ]  # fmt: skip


def static_publisher_text(frames: EstimateFrames, *, bag_name: str) -> str:
    """ROS 2 ``static_transform_publisher`` commands, one per exported frame."""

    lines = [
        "#!/usr/bin/env bash",
        f"# calibrex estimate: static transforms estimated from bag {bag_name}.",
        "# Each command publishes T_parent_child on /tf_static. Run them in separate",
        "# terminals, or add each as a node to your launch file.",
    ]
    for entry in frames.entries:
        if entry.axes_from_prior:
            lines.append(
                f"# {entry.frame}: NOT MEASURED axes {', '.join(entry.axes_from_prior)} come "
                "from your --tf prior"
            )
        lines.append(
            "ros2 run tf2_ros static_transform_publisher " + " ".join(_publisher_args(entry))
        )
    for omission in frames.omitted:
        lines.append(f"# omitted {omission.frame}: {omission.reason}")
    return "\n".join(lines) + "\n"


def launch_yaml_text(frames: EstimateFrames, *, bag_name: str) -> str:
    """A ROS 2 YAML launch file with one ``static_transform_publisher`` node per frame."""

    lines = [
        f"# calibrex estimate: static transforms estimated from bag {bag_name}.",
        "# ros2 launch THIS_FILE",
        "launch:",
    ]
    for entry in frames.entries:
        if entry.axes_from_prior:
            lines.append(
                f"  # {entry.frame}: NOT MEASURED axes {', '.join(entry.axes_from_prior)} come "
                "from your --tf prior"
            )
        lines += [
            "  - node:",
            "      pkg: tf2_ros",
            "      exec: static_transform_publisher",
            f"      name: {json.dumps('estimate_' + entry.parent + '_to_' + entry.frame)}",
            f"      args: {json.dumps(' '.join(_publisher_args(entry)))}",
        ]
    if not frames.entries:
        lines[-1] = "launch: []"
    return "\n".join(lines) + "\n"


def urdf_joints_text(frames: EstimateFrames, *, bag_name: str) -> str:
    """URDF fixed-joint snippets (paste into your robot description)."""

    from scipy.spatial.transform import Rotation

    lines = [
        f"<!-- calibrex estimate: fixed joints estimated from bag {bag_name}. -->",
        "<!-- origin is T_parent_child; rpy is the fixed-axis roll, pitch, yaw. -->",
    ]
    for entry in frames.entries:
        roll, pitch, yaw = Rotation.from_quat(entry.transform.rotation_quat_xyzw).as_euler("xyz")
        x, y, z = entry.transform.translation_m
        if entry.axes_from_prior:
            lines.append(
                f"<!-- {entry.frame}: NOT MEASURED axes {', '.join(entry.axes_from_prior)} "
                "come from your --tf prior -->"
            )
        lines += [
            f'<joint name="{entry.parent}_to_{entry.frame}" type="fixed">',
            f'  <parent link="{entry.parent}"/>',
            f'  <child link="{entry.frame}"/>',
            f'  <origin xyz="{_num(x)} {_num(y)} {_num(z)}" '
            f'rpy="{_num(float(roll))} {_num(float(pitch))} {_num(float(yaw))}"/>',
            "</joint>",
        ]
    for omission in frames.omitted:
        lines.append(f"<!-- omitted {omission.frame}: {omission.reason} -->")
    return "\n".join(lines) + "\n"


def camchain_export(
    sources: Sequence[LoadedSource],
    frames: EstimateFrames,
    *,
    bag_name: str,
) -> str | None:
    """A Kalibr camchain-imucam whose ``T_cam_imu`` are the exported camera-IMU transforms.

    Built from the ``--tf`` Kalibr camchain (it carries the intrinsics a camera
    pair needs): each camera whose camera-imu edge is in the exported tree gets its
    ``T_cam_imu`` replaced (inverted from the frames YAML edge, so both exports
    agree); other cameras are dropped. A translation the data did not observe is
    the prior's. ``timeshift_cam_imu`` is removed because its sign convention
    differs from the estimator's clock offset.
    """

    import yaml

    from calibrex.check.tf_sources import camchain_entries

    by_frame = {entry.frame: entry for entry in frames.entries if entry.pair == "camera-imu"}
    by_parent = {entry.parent: entry for entry in frames.entries if entry.pair == "camera-imu"}
    for source in sources:
        if source.kind != "kalibr_camchain" or source.path is None:
            continue
        out: dict[str, Any] = {}
        notes: list[str] = []
        for key, frame, entry in camchain_entries(source.path):
            edge = by_frame.get(frame) or by_parent.get(frame)
            if edge is None:
                continue
            matrix = transform_matrix(
                edge.transform.translation_m, edge.transform.rotation_quat_xyzw
            )
            # T_cam_imu: the edge is T_parent_child between the camera and the IMU frame.
            t_cam_imu = invert_transform(matrix) if edge.frame == frame else matrix
            updated = {k: v for k, v in entry.items() if k != "timeshift_cam_imu"}
            updated["T_cam_imu"] = [[float(v) for v in row] for row in t_cam_imu]
            out[key] = updated
            quality = (
                "all axes observed"
                if not edge.axes_from_prior
                else f"NOT MEASURED {', '.join(edge.axes_from_prior)} (kept from the prior)"
            )
            notes.append(f"{key}: {quality}")
        if not out:
            continue
        header = [
            f"# calibrex estimate: Kalibr camchain-imucam estimated from bag {bag_name}.",
            f"# Intrinsics are copied from {source.path.name}; only T_cam_imu is estimated.",
            "# Cameras without an exported camera-imu edge are omitted.",
            *(f"# {note}" for note in notes),
        ]
        return "\n".join(header) + "\n" + yaml.safe_dump(out, sort_keys=True)
    return None


def write_exports(
    frames: EstimateFrames,
    output_dir: Path,
    *,
    bag_name: str,
    bag_sha256: str,
    topic_frames: Mapping[str, str],
    camchain: str | None = None,
) -> list[EstimateExportFile]:
    """Write the frames YAML, publisher script, launch file, URDF snippets and camchain."""

    output_dir.mkdir(parents=True, exist_ok=True)
    exports: list[EstimateExportFile] = []
    if not frames.entries:
        return exports
    files: list[tuple[str, str, str]] = [
        (
            "frames_yaml",
            FRAMES_FILENAME,
            frames_yaml_text(
                frames, bag_name=bag_name, bag_sha256=bag_sha256, topic_frames=topic_frames
            ),
        ),
        (
            "static_transform_publisher",
            PUBLISHER_FILENAME,
            static_publisher_text(frames, bag_name=bag_name),
        ),
        ("launch_yaml", LAUNCH_FILENAME, launch_yaml_text(frames, bag_name=bag_name)),
        ("urdf_joints", URDF_FILENAME, urdf_joints_text(frames, bag_name=bag_name)),
    ]
    if camchain is not None:
        files.append(("kalibr_camchain", CAMCHAIN_FILENAME, camchain))
    for kind, name, text in files:
        path = output_dir / name
        path.write_text(text, encoding="utf-8")
        exports.append(
            EstimateExportFile(kind=kind, path=name, sha256=sha256_file(path))  # type: ignore[arg-type]
        )
    return exports


# --------------------------------------------------------------------- planning


def _augmented_tree(prior: StaticFrameTree, frames: Sequence[str]) -> StaticFrameTree:
    """The prior tree plus every sensor frame it lacks, all joined to an identity placeholder root.

    The placeholder joins are never exported and never count as a prior: a pair whose frames the
    prior does not connect is run with an identity candidate (and marked ``placeholder_identity``).
    """

    joined = [*prior.roots, *(frame for frame in sorted(set(frames)) if frame not in prior)]
    edges = [
        *prior.edges,
        *(
            StaticEdge(
                parent=PLACEHOLDER_ROOT, child=frame, transform=SE3.identity(), source="frames_yaml"
            )
            for frame in joined
        ),
    ]
    return StaticFrameTree(edges)


def _prior_source_record(source: LoadedSource, bag: Path) -> CheckCandidateSource:
    return CheckCandidateSource(
        kind=source.kind,
        path=str(source.path) if source.path is not None else str(bag),
        sha256=source.sha256,
        frame_count=len(source.edges),
        notes=list(source.notes),
    )


def _step_for(pair: EstimatePairRecord, topics: Sequence[CheckTopicRecord]) -> str | None:
    if pair.status not in {"skipped", "failed"} or pair.reason_code is None:
        return None
    if pair.reason_code == "no_candidate_calibration":
        return LIDAR_LIDAR_PRIOR_HINT if pair.pair in NEEDS_INITIAL_GUESS else None
    hint = pair_hint(
        CheckPairRecord(
            pair=pair.pair,
            status="skipped",
            reason_code=pair.reason_code,
            reason=pair.reason or pair.reason_code,
        ),
        topics,
    )
    if hint is None:
        return None
    return f"{pair.pair}: {hint}" if pair.reason_code == "missing_topic" else hint


def _prior_conflict(prior: StaticFrameTree, edges: Sequence[StaticEdge]) -> str | None:
    """A warning when the exported edges and the prior's own edges do not form one tree."""

    chosen = {edge.child: edge for edge in prior.edges}
    chosen.update({edge.child: edge for edge in edges})
    try:
        StaticFrameTree(chosen.values())
    except FrameGraphError as exc:
        return (
            f"WARNING: frames.yaml conflicts with the prior's frame tree ({exc}); use it "
            "without the prior (for example not next to the bag's /tf_static), or give a "
            "--tf prior whose tree is oriented the same way"
        )
    return None


def next_steps(
    pairs: Sequence[EstimatePairRecord],
    frames: EstimateFrames,
    topics: Sequence[CheckTopicRecord],
    *,
    bag_name: str,
    output_dir: Path,
) -> list[str]:
    steps: list[str] = []

    def add(text: str | None) -> None:
        if text and text not in steps:
            steps.append(text)

    if frames.entries:
        add(
            f"verify on a different recording: calibrex check <other bag> --tf "
            f"{output_dir / FRAMES_FILENAME}"
        )
        add(
            f"deploy: {output_dir / PUBLISHER_FILENAME} (ROS 2 static_transform_publisher), "
            f"{output_dir / LAUNCH_FILENAME} or {output_dir / URDF_FILENAME} (URDF joints)"
        )
    else:
        add(
            "no frame could be exported (no pair observed all six axes of a transform); "
            "give a rough --tf prior for the axes the data cannot observe, then re-run"
        )
    if frames.omitted:
        add(
            "omitted frames ("
            + ", ".join(sorted({o.frame for o in frames.omitted}))
            + "): the data observed only some of their axes or no estimate exists; pass a "
            "rough prior with --tf FILE to fill the unobserved axes (they are then marked "
            "NOT MEASURED in the YAML)"
        )
    for pair in pairs:
        if pair.reason_code in {"method_not_wired", "not_selected"}:
            continue
        add(_step_for(pair, topics))
    return steps


# ------------------------------------------------------------------------- main


def _pair_key(record: CheckPairRecord) -> tuple[object, ...]:
    return (estimators.PAIR_RUN_KEY, record.pair, tuple(record.frames))


def build_bag_estimate(
    bag: str | Path,
    *,
    output_dir: Path,
    run: CheckRunOptions,
    tf_files: Sequence[str | Path] = (),
    vehicle_frame: str | None = None,
    frame_overrides: Mapping[str, str] | None = None,
    topic_kinds: Mapping[str, OdometryKind] | None = None,
    command: Sequence[str] | None = None,
    progress: Callable[[str], None] | CheckProgress | None = None,
) -> BagEstimateArtifact:
    """Estimate every supported pair of ``bag`` and export the observed frames to ``output_dir``."""

    progress = as_progress(progress)

    bag_path = Path(bag)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_sha256, scope, storage_id = bag_input_digest(bag_path)
    connections = list_rosbag2_connections(bag_path)

    sources: list[LoadedSource] = []
    bag_source = load_bag_tf_static(bag_path)
    if bag_source is not None:
        sources.append(bag_source)
    file_sources = [load_tf_file(path) for path in tf_files]
    sources.extend(file_sources)
    prior, _overrides = merge_sources(bag_source, file_sources)
    hints = merge_hints(file_sources)

    candidates, header_frames = classify_bag_topics(bag_path, connections, topic_kinds)
    overrides = dict(frame_overrides or {})
    vehicle = normalize_frame_id(vehicle_frame) if vehicle_frame else None
    # Map against the prior first (so a camchain's topic hints and role defaults win), then
    # give every sensor topic the prior does not know its header frame, joined to a placeholder.
    topics: list[CheckTopicRecord] = []
    for topic in map_topics_to_frames(candidates, header_frames, prior, hints, overrides):
        if (
            topic.role not in {None, "tf_static"}
            and not topic.ignored_reason
            and topic.mapped_frame is None
            and topic.header_frame_id
        ):
            topic = topic.model_copy(
                update={"mapped_frame": topic.header_frame_id, "frame_source": "header"}
            )
        topics.append(topic)
    tree = _augmented_tree(
        prior,
        [t.mapped_frame for t in topics if t.mapped_frame is not None]
        + ([vehicle] if vehicle else []),
    )
    plan = [
        record
        for record in plan_pairs(topics, tree, vehicle_frame=vehicle, wired_pairs=ESTIMATE_PAIRS)
        if record.pair not in NOT_ESTIMATED_PAIRS
    ]

    informed: dict[tuple[str, tuple[str, ...]], bool] = {}
    gated: list[CheckPairRecord] = []
    for record in plan:
        is_informed = len(record.frames) == 2 and prior.connected(*record.frames)
        informed[(record.pair, tuple(record.frames))] = is_informed
        if record.status == "planned" and record.pair in NEEDS_INITIAL_GUESS and not is_informed:
            record = _skipped(
                record,
                "no_candidate_calibration",
                f"{record.pair} map registration needs a rough initial extrinsic and no --tf "
                "prior connects its frames",
            )
        gated.append(record)

    base_dir = output_dir
    evidence_dir = output_dir / "evidence"
    progress(f"estimate: {len([r for r in gated if r.status == 'planned'])} pair(s) to estimate")
    pairs_run, memo = _run_pairs(
        gated,
        run,
        bag=bag_path,
        topics=topics,
        topic_types={c.topic: c.message_type for c, _ in connections},
        sources=sources,
        evidence_dir=evidence_dir,
        base_dir=base_dir,
        progress=progress,
        bag_sha256=input_sha256,
    )

    records: list[EstimatePairRecord] = []
    relations: list[Relation] = []
    for record in pairs_run:
        outcome = memo.get(_pair_key(record))
        pair, relation = _pair_record(
            record,
            outcome,
            informed=informed.get((record.pair, tuple(record.frames)), False),
        )
        records.append(pair)
        if relation is not None:
            relations.append(relation)

    frames, edges = build_frame_tree(relations, records, topics, vehicle, prior)
    conflict = _prior_conflict(prior, edges)
    # The root is not an entry but is a frame of the exported tree: the topics stamped in it
    # (the IMU of an IMU-rooted export) need their mapping too, or the round trip cannot
    # find them in the tree.
    exported = {entry.frame for entry in frames.entries}
    if frames.entries and frames.root is not None:
        exported.add(frames.root)
    topic_frames = {
        t.topic: t.mapped_frame
        for t in topics
        if t.mapped_frame is not None
        and t.mapped_frame in exported
        and normalize_frame_id(t.header_frame_id or "") != t.mapped_frame
    }
    exports = write_exports(
        frames,
        output_dir,
        bag_name=bag_path.name,
        bag_sha256=input_sha256,
        topic_frames=topic_frames,
        camchain=camchain_export(sources, frames, bag_name=bag_path.name),
    )
    steps = next_steps(records, frames, topics, bag_name=bag_path.name, output_dir=output_dir)
    if conflict is not None:
        steps.insert(0, conflict)

    status_counts: dict[str, int] = {}
    skipped_by_reason: dict[str, int] = {}
    for pair in records:
        status_counts[pair.status] = status_counts.get(pair.status, 0) + 1
        if pair.status == "skipped" and pair.reason_code:
            skipped_by_reason[pair.reason_code] = skipped_by_reason.get(pair.reason_code, 0) + 1
    notes = [
        f"{len(connections) - len(topics)} bag topic(s) without a sensor role are not listed",
        "estimators run on the bag's own data; the candidate passed to them is a placeholder "
        "(identity, or the --tf prior where one connects the frames), so the reference fields "
        "of the evidence artifacts do not describe a deployed calibration",
    ]
    if run.max_duration_s is not None:
        notes.append(f"only the first {run.max_duration_s:g} s of the sensor streams were analysed")
    return BagEstimateArtifact(
        bag=CheckBagInput(
            path=str(bag_path),
            storage_identifier=storage_id,
            topic_count=len(connections),
            input_sha256=input_sha256,
            input_digest_scope=scope,
        ),
        vehicle_frame=vehicle,
        prior_sources=[_prior_source_record(source, bag_path) for source in sources],
        topics=topics,
        pairs=records,
        frames=frames,
        exports=exports,
        summary=EstimateSummary(
            pair_count=len(records),
            status_counts=dict(sorted(status_counts.items())),
            skipped_by_reason=dict(sorted(skipped_by_reason.items())),
            frames_written=len(frames.entries),
            frames_omitted=len(frames.omitted),
        ),
        options=EstimateOptions(
            max_duration_s=run.max_duration_s,
            gnss_max_duration_s=run.gnss_max_duration_s,
            pairs=list(run.pairs) if run.pairs is not None else None,
            camera=run.camera,
            imu_lidar_translation=run.imu_lidar_translation,
            acceleration_unit=run.acceleration_unit,
            imu_lidar_deskew=run.imu_lidar_deskew,
            lidar_lidar_deskew=run.lidar_lidar_deskew,
            gnss_lidar_deskew=run.gnss_lidar_deskew,
            topic_kinds=dict(topic_kinds or {}),
            frame_map=overrides,
        ),
        evidence_dir=evidence_dir.relative_to(output_dir).as_posix(),
        next_steps=steps,
        provenance=CheckProvenance(
            generator="calibrex estimate",
            generator_version=__version__,
            git_commit=git_commit(),
            command=list(command or []),
            input_sha256=input_sha256,
            input_digest_scope=scope,
            created_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        ),
    )


# -------------------------------------------------------------------------- text


def _fmt_axis(axis: EstimateAxis) -> str:
    if axis.status == "observed":
        std = f" +- {axis.std:.3g}" if axis.std is not None else ""
        return f"{axis.value:+.4f} {axis.unit}{std}"
    return "-"


def format_estimate_text(artifact: BagEstimateArtifact, output_dir: Path) -> str:
    """Render the estimate: a table per estimated pair, the exported frames and next steps."""

    lines = [
        "calibrex estimate",
        f"bag: {artifact.bag.path} ({artifact.bag.topic_count} topics)",
    ]
    if artifact.prior_sources:
        lines.append(
            "rough priors: "
            + ", ".join(
                f"{s.kind} {s.path} ({s.frame_count} frames)" for s in artifact.prior_sources
            )
        )
    else:
        lines.append("rough priors: none (estimators that need an initial guess are skipped)")
    quiet: dict[str, list[str]] = {}
    shown = 0
    for pair in artifact.pairs:
        if pair.status == "skipped" and pair.reason_code in {"not_selected", "method_not_wired"}:
            quiet.setdefault(pair.reason_code, []).append(pair.pair)
            continue
        shown += 1
        sensors = " / ".join(pair.sensors) if pair.sensors else "-"
        lines.append("")
        label = f"{pair.pair} ({sensors}): {pair.status}"
        if pair.transform is not None:
            label += f"  T_{pair.transform.parent_frame}_{pair.transform.child_frame}"
        lines.append(label)
        if pair.status in {"skipped", "failed"}:
            lines.append(f"  {pair.reason_code}: {pair.reason}")
            continue
        rows = [
            [
                axis.name,
                _fmt_axis(axis),
                axis.status
                + (f" ({axis.reason})" if axis.status != "observed" and axis.reason else ""),
            ]
            for axis in pair.axes
        ]
        width = [max(len(row[i]) for row in rows) for i in (0, 1)]
        for row in rows:
            lines.append(f"  {row[0].ljust(width[0])}  {row[1].ljust(width[1])}  {row[2]}")
        if pair.time_offset is not None:
            std = (
                f" +- {pair.time_offset.std_s * 1e3:.2f}"
                if pair.time_offset.std_s is not None
                else ""
            )
            lines.append(
                f"  time offset {pair.time_offset.estimate_s * 1e3:+.2f}{std} ms "
                f"({pair.time_offset.status})"
            )
        if pair.deskew == "none":
            lines.append(
                "  scans treated as rigid (no per-point time): the time offset includes "
                "the driver's scan-stamp convention"
            )
        if any(axis.status != "observed" for axis in pair.axes if axis.status != "not_estimated"):
            lines.append("  axes that are not observed hold no measurement and are not exported")
    if not shown:
        lines += ["", "no pair could be estimated"]
    for code, names in quiet.items():
        lines.append(f"  {code.replace('_', ' ')}: {', '.join(names)}")
    lines += ["", f"exported frames (root {artifact.frames.root}):"]
    if artifact.frames.entries:
        for entry in artifact.frames.entries:
            note = (
                "all axes observed"
                if not entry.axes_from_prior
                else f"NOT MEASURED {', '.join(entry.axes_from_prior)} (from --tf prior)"
            )
            lines.append(f"  {entry.parent} -> {entry.frame}  [{entry.pair}] {note}")
    else:
        lines.append("  none")
    for omission in artifact.frames.omitted:
        lines.append(f"  omitted {omission.frame}: {omission.reason}")
    if artifact.next_steps:
        lines += ["", "next steps:", *(f"  - {step}" for step in artifact.next_steps)]
    return "\n".join(lines)
