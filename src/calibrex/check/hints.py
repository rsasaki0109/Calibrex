"""Next-step hints for skipped pairs and unmapped topics of ``calibrex check``.

Hints are presentation only: they are rendered in the text output and are not
part of the ``slac.calibration_check`` artifact, so that schema is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, get_args

import yaml

from calibrex.core.calibration_check import (
    CHECK_FRAMES_SCHEMA_VERSION,
    CalibrationCheckArtifact,
    CheckPairRecord,
    CheckReasonCode,
    CheckTopicRecord,
)

TF_FLAG_HINT = (
    "pass the calibration with --tf FILE (URDF, Kalibr camchain-imucam, RTK-SLAM "
    "calib.yaml, or slac.check_frames YAML; --write-frames-template FILE writes a "
    "starter YAML for this bag); with no calibration at all, calibrex estimate BAG "
    "--output DIR estimates one from the bag"
)

TEMPLATE_SCHEMA_VERSION = "slac.check_frames_template/v0.1"

# Message types expected by each sensor slot of ``missing_topic``.
SLOT_MESSAGE_TYPES: dict[str, str] = {
    "imu": "sensor_msgs/msg/Imu",
    "lidar": "sensor_msgs/msg/PointCloud2",
    "camera": "sensor_msgs/msg/Image (plus a CameraInfo for camera-imu)",
    "gnss": "sensor_msgs/msg/NavSatFix",
    "ins": "nav_msgs/msg/Odometry classified as ins (--topic-kind TOPIC=ins)",
    "wheel": "nav_msgs/msg/Odometry or TwistStamped classified as wheel (--topic-kind TOPIC=wheel)",
}

# One actionable hint per reason code; ``{expected}`` is filled for missing_topic.
REASON_HINTS: dict[str, str] = {
    "missing_topic": "the bag lacks a topic for this pair; expected {expected}",
    "frame_not_in_tree": (
        "the sensor frame is not in the candidate tree: add it to --tf, or map the "
        "topic onto a tree frame with --frame-map TOPIC=FRAME"
    ),
    "frames_not_connected": (
        "the two frames are in different trees: add the joining transform to --tf "
        "(a slac.check_frames entry with the missing parent) so they share a root"
    ),
    "degenerate_frames": (
        "both sensors are stamped in one frame: point the topics at their sensor "
        "frames with --frame-map TOPIC=FRAME, or list the sensor frames in --tf"
    ),
    "no_candidate_calibration": TF_FLAG_HINT,
    "no_vehicle_frame": (
        "vehicle pairs are opt-in for ground vehicles: add --vehicle-frame base_link "
        "(use your vehicle frame name)"
    ),
    "method_not_wired": "no check method exists for this pair yet; nothing to do",
    "not_selected": "re-run without --pairs, or add this pair to --pairs",
    "unsupported_sensor": (
        "this sensor type is not supported by the pair's estimator; select another "
        "pair with --pairs"
    ),
    "missing_intrinsics": (
        "the camera has no usable intrinsics: record a calibrated CameraInfo topic, "
        "or pass a Kalibr camchain-imucam file with --tf"
    ),
    "missing_dependency": (
        "an optional dependency is missing: install the extra named in the reason "
        "(for example pip install 'calibrex[opencv]')"
    ),
    "estimator_error": (
        "the estimator crashed: re-run with a shorter --max-duration-s and report "
        "the reason as an issue"
    ),
    "estimator_failed": (
        "the estimator did not converge: the bag may lack excitation or overlap; "
        "try a longer --max-duration-s or a more dynamic bag"
    ),
    "no_judgeable_axes": (
        "no axis was observable in this bag, so none can be judged: use a bag with "
        "more rotation and translation"
    ),
}

ALL_REASON_CODES: tuple[str, ...] = get_args(CheckReasonCode)


def _missing_slots(reason: str) -> list[str]:
    if reason.startswith("no topic for: "):
        slots = reason.removeprefix("no topic for: ").split(" (")[0]
        return [name.strip() for name in slots.split(",") if name.strip()]
    if reason.startswith("needs two distinct "):
        return [reason.split()[3]]
    return []


def pair_hint(pair: CheckPairRecord, topics: Sequence[CheckTopicRecord] = ()) -> str | None:
    """Return the actionable hint for a skipped pair, or ``None`` for other pairs."""

    if pair.status != "skipped" or pair.reason_code is None:
        return None
    template = REASON_HINTS[pair.reason_code]
    if pair.reason_code != "missing_topic":
        return template
    names = _missing_slots(pair.reason or "")
    if (pair.reason or "").startswith("needs two distinct "):
        slot = names[0]
        return (
            f"needs a second distinct {slot} sensor (found one); expected another "
            f"{SLOT_MESSAGE_TYPES.get(slot, slot)} topic with its own frame_id"
        )
    expected = (
        "; ".join(f"{name}: {SLOT_MESSAGE_TYPES.get(name, name)}" for name in names)
        or "a topic for each sensor of the pair"
    )
    hint = template.format(expected=expected)
    unknown = sorted(
        t.topic for t in topics if t.role in {"odometry", "twist"} and t.odometry_kind == "unknown"
    )
    if unknown and any(name in {"ins", "wheel"} for name in names):
        hint += f"; classify an odometry topic with --topic-kind {unknown[0]}=wheel|ins"
    return hint


def topic_hint(topic: CheckTopicRecord, *, tree_frames: Sequence[str] = ()) -> str | None:
    """Return the actionable hint for a sensor topic that maps to no tree frame."""

    if topic.role in {"tf_static", None} or topic.mapped_frame is not None:
        return None
    if topic.ignored_reason:
        return None
    header = topic.header_frame_id
    if not tree_frames:
        if header:
            return (
                f"{topic.topic}: header frame '{header}' has no transform; give it in "
                f"--tf, or map the topic with --frame-map {topic.topic}=FRAME"
            )
        return f"{topic.topic}: no header frame_id; map it with --frame-map {topic.topic}=FRAME"
    example = tree_frames[0]
    known = f"header frame '{header}' is not in the tree" if header else "no header frame_id"
    return (
        f"{topic.topic}: {known}; map it with --frame-map {topic.topic}={example} "
        f"(tree frames include {', '.join(tree_frames[:4])}) or add the frame to --tf"
    )


def _example_map(artifact: CalibrationCheckArtifact) -> str:
    for topic in artifact.topics:
        if topic.header_frame_id and topic.role not in {"tf_static", None}:
            return f"--frame-map {topic.topic}={topic.header_frame_id.strip().lstrip('/')}"
    return "--frame-map /imu=imu_link"


def collect_next_steps(artifact: CalibrationCheckArtifact) -> list[str]:
    """Deduplicated next steps for every skipped pair and unmapped topic."""

    steps: list[str] = []

    def add(text: str | None) -> None:
        if text and text not in steps:
            steps.append(text)

    tree_frames = list(artifact.frame_tree.frames)
    if artifact.candidate_sources:
        for topic in artifact.topics:
            add(topic_hint(topic, tree_frames=tree_frames))
    else:  # without candidates, --tf is the real step; name the frames it must contain
        detected = sorted(
            {
                t.header_frame_id.strip().lstrip("/")
                for t in artifact.topics
                if t.role not in {"tf_static", None} and not t.ignored_reason and t.header_frame_id
            }
        )
        if detected:
            add(
                f"the --tf file must contain the sensor frames {', '.join(detected)} "
                "(topics map to them by header frame_id), or map a topic explicitly "
                f"with --frame-map TOPIC=FRAME (for example {_example_map(artifact)})"
            )
    for pair in artifact.pairs:
        if pair.reason_code in {"method_not_wired", "not_selected"}:
            continue
        hint = pair_hint(pair, artifact.topics)
        if hint is None:
            continue
        add(f"{pair.pair}: {hint}" if pair.reason_code == "missing_topic" else hint)
    return steps


def format_next_steps(artifact: CalibrationCheckArtifact) -> list[str]:
    """Text lines of the ``next steps`` block (empty when nothing needs doing)."""

    steps = collect_next_steps(artifact)
    if not steps:
        return []
    return ["", "next steps:", *(f"  - {step}" for step in steps)]


def frames_template(topics: Sequence[CheckTopicRecord], *, base_frame: str = "base_link") -> str:
    """A starter ``slac.check_frames`` YAML naming the bag's sensor frames.

    The template carries ``schema_version: slac.check_frames_template/v0.1``,
    which ``--tf`` refuses until the user has filled in the TODO transforms and
    changed the version to ``slac.check_frames/v0.1``; a template is never
    mistaken for a calibration.
    """

    role_of: dict[str, str] = {}
    topic_frames: dict[str, str] = {}
    for topic in topics:
        if topic.role in {"tf_static", None} or topic.ignored_reason:
            continue
        frame = (topic.header_frame_id or "").strip().lstrip("/")
        if not frame:
            continue
        role_of.setdefault(frame, topic.role)
        topic_frames[topic.topic] = frame
    lines = [
        "# calibrex check frames template -- NOT a calibration.",
        "# 1. Replace every TODO (translation, rotation) with your measured T_parent_frame",
        "#    (maps frame points into the parent frame; quaternion order x, y, z, w),",
        f"#    and set each parent (the placeholder {base_frame} may not be right).",
        "# 2. Change schema_version below to slac.check_frames/v0.1. --tf refuses the",
        "#    template version and any file that still contains a TODO value.",
        "# 3. Run: calibrex check <bag> --tf THIS_FILE",
        f"schema_version: {TEMPLATE_SCHEMA_VERSION}",
        "frames:",
    ]
    sensor_frames = [frame for frame in role_of if frame != base_frame]
    if not sensor_frames:
        lines.append("  # no sensor frame ids were found in the topic headers; add entries by hand")
        sensor_frames = ["sensor_frame"]
    for frame in sensor_frames:
        lines.extend(
            [
                f"  - name: {frame}  # {role_of.get(frame, 'sensor')}",
                f"    parent: {base_frame}  # TODO: the frame this sensor is measured against",
                "    translation_m: [TODO, TODO, TODO]  # measured x, y, z in metres",
                "    rotation_quat_xyzw: [TODO, TODO, TODO, TODO]  # measured rotation",
            ]
        )
    if topic_frames:
        lines.append(yaml.safe_dump({"topic_frames": topic_frames}, sort_keys=True).rstrip())
    return "\n".join(lines) + "\n"


def write_frames_template(path: str | Path, topics: Sequence[CheckTopicRecord]) -> Path:
    """Write :func:`frames_template` to ``path``."""

    target = Path(path)
    target.write_text(frames_template(topics), encoding="utf-8")
    return target


def _todo_frames(payload: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    frames = payload.get("frames")
    for index, entry in enumerate(frames if isinstance(frames, list) else []):
        if not isinstance(entry, Mapping):
            continue
        leaves: list[Any] = []
        for value in entry.values():
            leaves.extend(value if isinstance(value, list) else [value])
        if any(isinstance(leaf, str) and leaf.strip().upper() == "TODO" for leaf in leaves):
            names.append(str(entry.get("name", f"#{index}")))
    return names


def template_refusal(path: Path, payload: Mapping[str, Any]) -> str | None:
    """The error message for an unedited template (version marker or TODO values)."""

    version = str(payload.get("schema_version", ""))
    todo = _todo_frames(payload) if version.startswith("slac.check_frames") else []
    if todo:
        return (
            f"{path}: still has TODO placeholder values for frame(s) {', '.join(todo)}; this is "
            "a calibrex check frames template, not a calibration. Fill in the measured "
            f"transforms and set schema_version to {CHECK_FRAMES_SCHEMA_VERSION}"
        )
    if version.startswith("slac.check_frames_template/"):
        return (
            f"{path}: this is a calibrex check frames template, not a calibration; fill in "
            f"the transforms and change schema_version to {CHECK_FRAMES_SCHEMA_VERSION}"
        )
    return None
