"""Decide which sensor pairs of a bag can be audited against a candidate tree."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from calibrex.check.frame_tree import StaticFrameTree
from calibrex.core.calibration_check import (
    CHECK_PAIR_NAMES,
    CheckPairName,
    CheckPairRecord,
    CheckReasonCode,
    CheckTopicRecord,
    CheckTransform,
)

# Sensor "slots" a pair draws from. ``ins`` and ``wheel`` are odometry or twist
# topics classified by name or by --topic-kind; ``vehicle`` is a frame, not a topic.
PAIR_SLOTS: dict[CheckPairName, tuple[str, str]] = {
    "imu-lidar": ("imu", "lidar"),
    "lidar-lidar": ("lidar", "lidar"),
    "camera-imu": ("camera", "imu"),
    "camera-focal": ("camera", "imu"),
    "camera-lidar": ("camera", "lidar"),
    "gnss-lidar": ("gnss", "lidar"),
    "gnss-imu": ("gnss", "imu"),
    "lidar-vehicle": ("lidar", "vehicle"),
    "imu-vehicle": ("imu", "vehicle"),
    "ins-lidar": ("ins", "lidar"),
    "lidar-wheel_odometry": ("lidar", "wheel"),
}
ALL_WIRED_PAIRS: frozenset[str] = frozenset(CHECK_PAIR_NAMES)


@dataclass(frozen=True)
class SensorInstance:
    """One physical sensor: topics that share a mapped frame (or one unmapped topic)."""

    slot: str
    label: str
    frame: str | None
    topics: tuple[str, ...]
    header_frames: tuple[str, ...]


def slot_of(record: CheckTopicRecord) -> str | None:
    """The sensor slot a topic fills: a role, or ``ins``/``wheel`` for odometry and twist."""

    if record.ignored_reason is not None:
        return None
    if record.role in {"odometry", "twist"}:
        if record.odometry_kind == "ins":
            return "ins"
        if record.odometry_kind == "wheel":
            return "wheel"
        return None
    if record.role in {"lidar", "imu", "camera", "gnss"}:
        return record.role
    return None


def sensor_instances(topics: Sequence[CheckTopicRecord]) -> dict[str, list[SensorInstance]]:
    """Group sensor topics into instances per slot."""

    grouped: dict[str, dict[str, list[CheckTopicRecord]]] = {}
    for record in topics:
        slot = slot_of(record)
        if slot is None:
            continue
        key = record.mapped_frame if record.mapped_frame is not None else f"topic:{record.topic}"
        grouped.setdefault(slot, {}).setdefault(key, []).append(record)
    result: dict[str, list[SensorInstance]] = {}
    for slot, by_key in grouped.items():
        instances = []
        for _key, records in sorted(by_key.items()):
            frame = records[0].mapped_frame
            headers = sorted({r.header_frame_id for r in records if r.header_frame_id})
            instances.append(
                SensorInstance(
                    slot=slot,
                    label=frame if frame is not None else records[0].topic,
                    frame=frame,
                    topics=tuple(sorted(r.topic for r in records)),
                    header_frames=tuple(headers),
                )
            )
        result[slot] = instances
    return result


def _skip(
    pair: CheckPairName,
    code: CheckReasonCode,
    reason: str,
    instances: Sequence[SensorInstance] = (),
) -> CheckPairRecord:
    return CheckPairRecord(
        pair=pair,
        sensors=[instance.label for instance in instances],
        topics=sorted({topic for instance in instances for topic in instance.topics}),
        frames=[instance.frame for instance in instances if instance.frame is not None],
        status="skipped",
        reason_code=code,
        reason=reason,
    )


def _describe_unmapped(instance: SensorInstance) -> str:
    topic = ", ".join(instance.topics)
    header = f" (header frame '{instance.header_frames[0]}')" if instance.header_frames else ""
    return f"topic {topic}{header} maps to no frame in the candidate tree"


def _unusable_reason(instance: SensorInstance, tree: StaticFrameTree) -> str | None:
    if instance.frame is None:
        return _describe_unmapped(instance)
    if instance.frame not in tree:
        return f"frame '{instance.frame}' of {instance.label} is not in the candidate tree"
    return None


def _evaluate_combo(
    pair: CheckPairName,
    first: SensorInstance,
    second: SensorInstance,
    tree: StaticFrameTree,
) -> CheckPairRecord:
    instances = (first, second)
    assert first.frame is not None and second.frame is not None
    if first.frame == second.frame:
        if "vehicle" in (first.slot, second.slot):
            sensor = second if first.slot == "vehicle" else first
            detail = (
                f"the {sensor.slot} data of {', '.join(sensor.topics)} is stamped in the "
                f"vehicle frame '{first.frame}' itself (typically a cloud or IMU stream "
                "already transformed into base_link), so the candidate extrinsic is the "
                "identity by construction and says nothing about the physical mounting"
            )
        else:
            detail = (
                f"both sensors are stamped in the same frame '{first.frame}' (typically "
                "streams already transformed into base_link), so the candidate extrinsic "
                "is the identity by construction and says nothing about the physical "
                "mounting; point the topics at their sensor frames with --frame-map, or "
                "give the sensor frames in --tf"
            )
        return _skip(pair, "degenerate_frames", detail, instances)
    transform = tree.lookup(first.frame, second.frame)
    if transform is None:
        return _skip(
            pair,
            "frames_not_connected",
            f"frames '{first.frame}' and '{second.frame}' are in different trees",
            instances,
        )
    return CheckPairRecord(
        pair=pair,
        sensors=[first.label, second.label],
        topics=sorted({topic for instance in instances for topic in instance.topics}),
        frames=[first.frame, second.frame],
        candidate_transform=CheckTransform(
            parent_frame=first.frame,
            child_frame=second.frame,
            translation_m=list(transform.translation_m),
            rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        ),
        status="planned",
    )


def plan_pairs(
    topics: Sequence[CheckTopicRecord],
    tree: StaticFrameTree,
    *,
    vehicle_frame: str | None = None,
    wired_pairs: frozenset[str] = ALL_WIRED_PAIRS,
) -> list[CheckPairRecord]:
    """Return one or more pair records for each candidate pair type.

    A pair that cannot run yields one ``skipped`` record with a machine-readable
    reason. Otherwise every combination of sensors yields a ``planned`` record
    carrying ``T_first_second`` composed through the tree, or a ``skipped``
    record when that combination's frames are unmapped or disconnected.
    """

    instances = sensor_instances(topics)
    has_candidates = bool(tree.edges)
    records: list[CheckPairRecord] = []
    for pair in CHECK_PAIR_NAMES:
        first_slot, second_slot = PAIR_SLOTS[pair]
        if pair not in wired_pairs:
            records.append(
                _skip(pair, "method_not_wired", f"no check method is wired for {pair} yet")
            )
            continue
        missing = [
            slot
            for slot in dict.fromkeys((first_slot, second_slot))
            if slot != "vehicle" and not instances.get(slot)
        ]
        if missing:
            missing_reason = "no topic for: " + ", ".join(missing)
            unknown = sorted(
                t.topic
                for t in topics
                if t.role in {"odometry", "twist"} and t.odometry_kind == "unknown"
            )
            if unknown and any(slot in {"ins", "wheel"} for slot in missing):
                missing_reason += (
                    f" (odometry/twist topics of unknown kind: {', '.join(unknown)}; "
                    "classify them with --topic-kind TOPIC=wheel|ins)"
                )
            records.append(_skip(pair, "missing_topic", missing_reason))
            continue
        if first_slot == second_slot and len(instances[first_slot]) < 2:
            records.append(
                _skip(
                    pair,
                    "missing_topic",
                    f"needs two distinct {first_slot} sensors, found one",
                    instances[first_slot],
                )
            )
            continue
        if not has_candidates:
            records.append(
                _skip(pair, "no_candidate_calibration", "no candidate extrinsics were found")
            )
            continue
        if "vehicle" in (first_slot, second_slot) and vehicle_frame is None:
            records.append(
                _skip(
                    pair,
                    "no_vehicle_frame",
                    "vehicle pairs assume ground-vehicle motion and are opt-in: pass "
                    "--vehicle-frame <frame> to check them on a ground vehicle",
                )
            )
            continue
        vehicle = SensorInstance(
            slot="vehicle",
            label=vehicle_frame or "",
            frame=vehicle_frame,
            topics=(),
            header_frames=(),
        )
        # A sensor whose frame is unmapped or absent is reported once, not once per
        # combination; the remaining sensors are combined.
        usable: dict[str, list[SensorInstance]] = {}
        for slot in dict.fromkeys((first_slot, second_slot)):
            candidates = [vehicle] if slot == "vehicle" else instances[slot]
            usable[slot] = []
            for instance in candidates:
                reason = _unusable_reason(instance, tree)
                if reason is None:
                    usable[slot].append(instance)
                else:
                    records.append(_skip(pair, "frame_not_in_tree", reason, [instance]))
        if first_slot == second_slot:
            combos = list(combinations(usable[first_slot], 2))
        else:
            combos = [(a, b) for a in usable[first_slot] for b in usable[second_slot]]
        records.extend(_evaluate_combo(pair, a, b, tree) for a, b in combos)
    return records
