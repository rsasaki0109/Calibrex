"""Build the ``calibrex check`` artifact for one bag (Phase A: plan only)."""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from calibrex import __version__
from calibrex.check import estimators
from calibrex.check.cache import EstimatorCache
from calibrex.check.estimators import (
    CheckSkipError,
    EstimatorRun,
    PairContext,
    RunControls,
    transform_matrix,
)
from calibrex.check.frame_tree import StaticFrameTree, normalize_frame_id
from calibrex.check.planner import ALL_WIRED_PAIRS, PAIR_SLOTS, plan_pairs, slot_of
from calibrex.check.roles import (
    classify_topics,
    map_topics_to_frames,
    read_header_frames,
)
from calibrex.check.tf_sources import (
    LoadedSource,
    load_bag_tf_static,
    load_tf_file,
    merge_hints,
    merge_sources,
    sha256_file,
)
from calibrex.check.verdict import VerdictOptions, judge_pair, worst_verdict
from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckAxisJudgement,
    CheckBagInput,
    CheckCandidateSource,
    CheckEvidenceRef,
    CheckFrameEdge,
    CheckFrameTree,
    CheckOptions,
    CheckPairRecord,
    CheckProvenance,
    CheckTopicRecord,
    CheckTransform,
    OdometryKind,
    summarize_pairs,
)
from calibrex.core.exceptions import DatasetError
from calibrex.core.provenance import git_commit
from calibrex.data.livox_ros2 import ScanStore
from calibrex.data.rosbag2 import list_rosbag2_connections, resolve_storage

INVERTED_CONVENTION_PAIRS = frozenset(
    {"imu-lidar", "lidar-vehicle", "imu-vehicle", "lidar-wheel_odometry"}
)
"""Pairs whose estimator reports ``T_second_first`` (the sensor's pose in the parent frame)."""
BAG_DIGEST_PREFIX_BYTES = 64 * 1024 * 1024
BAG_DIGEST_SCOPE = "metadata.yaml in full; each storage file by name, size and first 64 MiB"


def bag_input_digest(bag: str | Path) -> tuple[str, str, str | None]:
    """Digest a bag without reading a tens-of-gigabyte database in full.

    Returns ``(sha256, scope, storage_identifier)``. Follows the precedent of
    the other bag-based artifacts: metadata in full, each storage file by name,
    size and its first 64 MiB.
    """

    bag_path = Path(bag)
    storage_file, storage_id = resolve_storage(bag_path)
    digest = hashlib.sha256()
    if bag_path.is_dir():
        metadata = bag_path / "metadata.yaml"
        if metadata.is_file():
            digest.update(b"metadata.yaml")
            digest.update(metadata.read_bytes())
        files = sorted(
            [*bag_path.glob("*.db3"), *bag_path.glob("*.mcap")], key=lambda item: item.name
        )
        if storage_file not in files:
            files.append(storage_file)
    else:
        files = [storage_file]
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(str(file.stat().st_size).encode("ascii"))
        with file.open("rb") as stream:
            digest.update(stream.read(BAG_DIGEST_PREFIX_BYTES))
    return digest.hexdigest(), BAG_DIGEST_SCOPE, storage_id


def parse_frame_map(items: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``TOPIC=FRAME`` overrides."""

    overrides: dict[str, str] = {}
    for item in items:
        topic, separator, frame = item.partition("=")
        if not separator or not topic.strip() or not frame.strip():
            msg = f"--frame-map expects TOPIC=FRAME, got {item!r}"
            raise DatasetError(msg)
        overrides[topic.strip()] = normalize_frame_id(frame)
    return overrides


def _source_record(source: LoadedSource, bag: Path) -> CheckCandidateSource:
    path = str(source.path) if source.path is not None else str(bag)
    return CheckCandidateSource(
        kind=source.kind,
        path=path,
        sha256=source.sha256,
        frame_count=len(source.edges),
        notes=list(source.notes),
    )


def _tree_record(tree: StaticFrameTree, overrides: list[str]) -> CheckFrameTree:
    return CheckFrameTree(
        roots=tree.roots,
        frames=tree.frames,
        edges=[
            CheckFrameEdge(
                transform=CheckTransform(
                    parent_frame=edge.parent,
                    child_frame=edge.child,
                    translation_m=list(edge.transform.translation_m),
                    rotation_quat_xyzw=list(edge.transform.rotation_quat_xyzw),
                ),
                source=edge.source,
            )
            for edge in tree.edges
        ],
        overrides=overrides,
    )


@dataclass(frozen=True)
class CheckRunOptions:
    """Verdict thresholds and runtime controls for a full (non-plan) run."""

    verdict: VerdictOptions = field(default_factory=VerdictOptions)
    pairs: tuple[str, ...] | None = None
    max_duration_s: float | None = None
    camera: str | None = None
    imu_lidar_translation: bool = True
    acceleration_unit: Literal["mps2", "g"] = "mps2"
    evidence_dir: Path | None = None
    base_dir: Path | None = None
    cache_dir: Path | None = None
    """Estimator cache directory; ``None`` disables the cache (the CLI defaults to the user dir)."""
    scan_memory_mb: int = 2048
    """Memory for reusing decoded LiDAR scans across passes; 0 re-reads the bag every pass."""


def _quiet(_message: str) -> None:
    return None


def build_calibration_check(
    bag: str | Path,
    *,
    tf_files: Sequence[str | Path] = (),
    vehicle_frame: str | None = None,
    frame_overrides: Mapping[str, str] | None = None,
    topic_kinds: Mapping[str, OdometryKind] | None = None,
    command: Sequence[str] | None = None,
    wired_pairs: frozenset[str] = ALL_WIRED_PAIRS,
    run: CheckRunOptions | None = None,
    progress: Callable[[str], None] = _quiet,
) -> CalibrationCheckArtifact:
    """Read candidate extrinsics, classify topics and plan the pair checks.

    With ``run`` the native estimator of every wired, selected pair is executed
    and the candidate is judged against it (see :mod:`calibrex.check.verdict`);
    without it the result is a plan (every runnable pair ``planned``).
    """

    bag_path = Path(bag)
    input_sha256, scope, storage_id = bag_input_digest(bag_path)
    connections = list_rosbag2_connections(bag_path)

    sources: list[LoadedSource] = []
    bag_source = load_bag_tf_static(bag_path)
    if bag_source is not None:
        sources.append(bag_source)
    file_sources = [load_tf_file(path) for path in tf_files]
    sources.extend(file_sources)
    tree, overrides = merge_sources(bag_source, file_sources)
    hints = merge_hints(file_sources)

    candidates = classify_topics(connections, topic_kinds)
    unknown_topics = sorted(
        topic
        for topic in (topic_kinds or {})
        if not any(
            record.topic == topic and record.role in {"odometry", "twist"} for record in candidates
        )
    )
    if unknown_topics:
        msg = "--topic-kind: not an odometry or twist topic of the bag: " + ", ".join(
            unknown_topics
        )
        raise DatasetError(msg)
    header_frames = read_header_frames(bag_path, candidates)
    topics: list[CheckTopicRecord] = map_topics_to_frames(
        candidates, header_frames, tree, hints, frame_overrides
    )
    vehicle = normalize_frame_id(vehicle_frame) if vehicle_frame else None
    planner_pairs = wired_pairs if run is None else estimators.WIRED_CHECK_PAIRS
    pairs = plan_pairs(topics, tree, vehicle_frame=vehicle, wired_pairs=planner_pairs)

    notes = [f"{len(connections) - len(topics)} bag topic(s) without a sensor role are not listed"]
    options_record: CheckOptions | None = None
    evidence_dir_record: str | None = None
    overall: str | None = None
    if run is None:
        notes.insert(0, "plan only; no pair solver was run")
    else:
        topic_types = {connection.topic: connection.message_type for connection, _ in connections}
        base_dir = run.base_dir or Path.cwd()
        evidence_dir = run.evidence_dir or base_dir / "calibrex_check_evidence"
        pairs = _run_pairs(
            pairs,
            run,
            bag=bag_path,
            topics=topics,
            topic_types=topic_types,
            sources=sources,
            evidence_dir=evidence_dir,
            base_dir=base_dir,
            progress=progress,
            bag_sha256=input_sha256,
        )
        overall = worst_verdict([pair.status for pair in pairs]) or "inconclusive"
        options_record = CheckOptions(
            sigma_k=run.verdict.sigma_k,
            rotation_floor_deg=run.verdict.rotation_floor_deg,
            translation_floor_m=run.verdict.translation_floor_m,
            detection_probe_deg=run.verdict.detection_probe_deg,
            max_duration_s=run.max_duration_s,
            pairs=list(run.pairs) if run.pairs is not None else None,
            camera=run.camera,
            imu_lidar_translation=run.imu_lidar_translation,
            acceleration_unit=run.acceleration_unit,
            topic_kinds=dict(topic_kinds or {}),
        )
        evidence_dir_record = _relative(evidence_dir, base_dir)
        notes.insert(
            0,
            "full run: each wired pair's native estimator was run and the candidate judged "
            "against it (tolerance = max(sigma_k * std, floor); pass within 1x, fail beyond 2x)",
        )
        if run.max_duration_s is not None:
            notes.append(
                f"only the first {run.max_duration_s:g} s of the sensor streams were analysed"
            )
    return CalibrationCheckArtifact(
        plan_only=run is None,
        bag=CheckBagInput(
            path=str(bag_path),
            storage_identifier=storage_id,
            topic_count=len(connections),
            input_sha256=input_sha256,
            input_digest_scope=scope,
        ),
        vehicle_frame=vehicle,
        candidate_sources=[_source_record(source, bag_path) for source in sources],
        topics=topics,
        frame_tree=_tree_record(tree, overrides),
        pairs=pairs,
        summary=summarize_pairs(pairs),
        overall_verdict=overall,
        options=options_record,
        evidence_dir=evidence_dir_record,
        provenance=CheckProvenance(
            generator="calibrex check",
            generator_version=__version__,
            git_commit=git_commit(),
            command=list(command or []),
            input_sha256=input_sha256,
            input_digest_scope=scope,
            created_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        ),
    )


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "x"


def _sensor_topics(
    pair: CheckPairRecord, topics: Sequence[CheckTopicRecord]
) -> dict[str, tuple[str, ...]]:
    """Topics of each pair sensor: keyed ``first``/``second`` and by role (when distinct)."""

    slots = PAIR_SLOTS[pair.pair]

    def of(frame: str, slot: str) -> tuple[str, ...]:
        matched = [
            record for record in topics if record.mapped_frame == frame and slot_of(record) == slot
        ]
        # PointCloud2 before other LiDAR formats, then the busiest topic first.
        matched.sort(
            key=lambda r: (
                (r.message_type != estimators.POINTCLOUD2_TYPE, -(r.message_count or 0))
                if slot == "lidar"
                else (False, -(r.message_count or 0))
            )
        )
        return tuple(record.topic for record in matched)

    result: dict[str, tuple[str, ...]] = {}
    for index, key in enumerate(("first", "second")):
        if index < len(pair.frames):
            result[key] = of(pair.frames[index], slots[index])
    if slots[0] != slots[1]:
        result[slots[0]] = result.get("first", ())
        result[slots[1]] = result.get("second", ())
    return result


def _run_pairs(
    plan: Sequence[CheckPairRecord],
    run: CheckRunOptions,
    *,
    bag: Path,
    topics: Sequence[CheckTopicRecord],
    topic_types: Mapping[str, str],
    sources: Sequence[LoadedSource],
    evidence_dir: Path,
    base_dir: Path,
    progress: Callable[[str], None],
    bag_sha256: str = "",
) -> list[CheckPairRecord]:
    selected = set(run.pairs) if run.pairs is not None else None
    controls = RunControls(
        max_duration_s=run.max_duration_s,
        camera=run.camera,
        imu_lidar_translation=run.imu_lidar_translation,
        acceleration_unit=run.acceleration_unit,
        progress=progress,
        cache=EstimatorCache(run.cache_dir) if run.cache_dir is not None else None,
        bag_sha256=bag_sha256,
        scan_store=ScanStore(run.scan_memory_mb << 20) if run.scan_memory_mb > 0 else None,
    )
    results: list[CheckPairRecord] = []
    for record in plan:
        if record.status == "skipped" and record.reason_code == "method_not_wired":
            if record.pair in estimators.WIRED_CHECK_PAIRS:
                record = record.model_copy(
                    update={
                        "reason_code": "not_selected",
                        "reason": f"{record.pair} was not selected by --pairs",
                    }
                )
            results.append(record)
            continue
        if record.status != "planned":
            results.append(record)
            continue
        if selected is not None and record.pair not in selected:
            results.append(
                _skipped(record, "not_selected", f"{record.pair} was not selected by --pairs")
            )
            continue
        sensor_topics = _sensor_topics(record, topics)
        if run.camera is not None and record.pair == "camera-imu":
            camera_topics = sensor_topics.get("camera", ())
            if run.camera not in camera_topics and run.camera not in record.frames:
                results.append(
                    _skipped(
                        record, "not_selected", f"camera {run.camera} was not selected by --camera"
                    )
                )
                continue
        results.append(
            _run_one(
                record,
                run,
                bag=bag,
                topics=topics,
                topic_types=topic_types,
                sensor_topics=sensor_topics,
                sources=sources,
                controls=controls,
                evidence_dir=evidence_dir,
                base_dir=base_dir,
                progress=progress,
            )
        )
    return results


def _skipped(record: CheckPairRecord, code: str, reason: str) -> CheckPairRecord:
    return CheckPairRecord.model_validate(
        {
            **record.model_dump(),
            "status": "skipped",
            "reason_code": code,
            "reason": reason,
        }
    )


def _label(record: CheckPairRecord) -> str:
    return f"{record.pair} ({' / '.join(record.sensors)})"


def _run_one(
    record: CheckPairRecord,
    run: CheckRunOptions,
    *,
    bag: Path,
    topics: Sequence[CheckTopicRecord],
    topic_types: Mapping[str, str],
    sensor_topics: Mapping[str, tuple[str, ...]],
    sources: Sequence[LoadedSource],
    controls: RunControls,
    evidence_dir: Path,
    base_dir: Path,
    progress: Callable[[str], None],
) -> CheckPairRecord:
    assert record.candidate_transform is not None
    candidate = transform_matrix(
        record.candidate_transform.translation_m, record.candidate_transform.rotation_quat_xyzw
    )
    context = PairContext(
        bag=bag,
        pair=record,
        topic_types=topic_types,
        sensor_topics=sensor_topics,
        candidate=candidate,
        sources=sources,
        controls=controls,
        topics=topics,
    )
    progress(f"check: running {_label(record)}")
    started = time.monotonic()
    try:
        outcome = estimators.ESTIMATORS[record.pair](context)
    except CheckSkipError as exc:
        progress(f"check: {_label(record)} skipped ({exc.code}): {exc.reason}")
        return _skipped(record, exc.code, exc.reason)
    except Exception as exc:  # one failing estimator must not lose the other pairs
        runtime = time.monotonic() - started
        progress(f"check: {_label(record)} failed: {type(exc).__name__}: {exc}")
        return CheckPairRecord.model_validate(
            {
                **record.model_dump(),
                "status": "inconclusive",
                "verdict": "inconclusive",
                "reason_code": "estimator_error",
                "reason": f"the estimator raised {type(exc).__name__}: {exc}",
                "runtime_s": runtime,
            }
        )
    runtime = time.monotonic() - started
    judged = _judge(record, outcome, run.verdict, runtime)
    evidence = _write_evidence(record, outcome, evidence_dir, base_dir)
    update = {**judged, "evidence": [item.model_dump() for item in evidence]}
    if evidence:
        update["evidence_artifact"] = evidence[0].path
    if outcome.evidence_from_cache is not None:
        update["evidence_from_cache"] = outcome.evidence_from_cache
    progress(f"check: {_label(record)} -> {update['status']} ({runtime:.0f} s)")
    return CheckPairRecord.model_validate({**record.model_dump(), **update})


def _transform_fields(matrix: object, parent: str, child: str) -> CheckTransform:
    from scipy.spatial.transform import Rotation

    array = np.asarray(matrix, dtype=float)
    return CheckTransform(
        parent_frame=parent,
        child_frame=child,
        translation_m=[float(value) for value in array[:3, 3]],
        rotation_quat_xyzw=[
            float(value) for value in Rotation.from_matrix(array[:3, :3]).as_quat()
        ],
    )


def _judge(
    record: CheckPairRecord,
    outcome: EstimatorRun,
    options: VerdictOptions,
    runtime_s: float,
) -> dict[str, object]:
    judgement = judge_pair(outcome.estimates, options)
    status: str = judgement.verdict
    reason_code: str | None = None
    reason: str | None = None
    if not outcome.solved:
        status = "inconclusive"
        reason_code = "estimator_failed"
        reason = "the estimator did not produce an estimate: " + "; ".join(outcome.policy_reasons)
    elif outcome.policy_status == "fail":
        status = "inconclusive"
        reason_code = "estimator_failed"
        reason = (
            "the estimator failed its own held-out check, so its estimate is not a "
            "reliable yardstick: " + "; ".join(outcome.policy_reasons)
        )
    elif not judgement.axes:
        reason_code = "no_judgeable_axes"
        reason = "no axis was constrained by the data: " + "; ".join(
            f"{item.name} ({item.reason})" for item in judgement.unchecked
        )
    parent, child = (
        (record.frames[1], record.frames[0])
        if record.pair in INVERTED_CONVENTION_PAIRS
        else (record.frames[0], record.frames[1])
    )
    return {
        "status": status,
        "verdict": status,
        "reason_code": reason_code,
        "reason": reason,
        "compared_transform": _transform_fields(outcome.compared, parent, child).model_dump(),
        "estimator": outcome.estimator,
        "estimator_policy_status": outcome.policy_status,
        "estimator_policy_reasons": list(outcome.policy_reasons),
        "axes": [axis.model_dump() for axis in judgement.axes],
        "unchecked_axes": [axis.model_dump() for axis in judgement.unchecked],
        "coverage": judgement.coverage,
        "time_offset": outcome.time_offset.model_dump() if outcome.time_offset else None,
        "runtime_s": runtime_s,
        "notes": list(outcome.notes),
    }


def _write_evidence(
    record: CheckPairRecord,
    outcome: EstimatorRun,
    evidence_dir: Path,
    base_dir: Path,
) -> list[CheckEvidenceRef]:
    refs: list[CheckEvidenceRef] = []
    stem = _safe_name(f"{record.pair}_{'_'.join(record.sensors)}")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for item in outcome.artifacts:
        suffix = "" if len(outcome.artifacts) == 1 else f"_{_safe_name(item.role)}"
        path = evidence_dir / f"{stem}{suffix}.yaml"
        item.artifact.save(path)
        refs.append(
            CheckEvidenceRef(
                path=_relative(path, base_dir),
                sha256=sha256_file(path),
                schema_version=str(item.artifact.schema_version),
                role=item.role,
                policy_status=item.policy_status,
                from_cache=item.from_cache,
            )
        )
    return refs


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines = ["  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))).rstrip()]
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return lines


def _format_axis(axis: CheckAxisJudgement) -> str:
    unit = "deg" if axis.unit == "deg" else "m"
    return f"{axis.name} {abs(axis.candidate_error):.3g}/{axis.tolerance:.3g} {unit}" + (
        "" if axis.status == "pass" else f" [{axis.status}]"
    )


def _verdict_lines(artifact: CalibrationCheckArtifact) -> list[str]:
    lines: list[str] = []
    rows: list[list[str]] = []
    quiet: dict[str, list[str]] = {}
    for pair in artifact.pairs:
        sensors = " / ".join(pair.sensors) if pair.sensors else "-"
        if pair.status == "skipped" and pair.reason_code in {"not_selected", "method_not_wired"}:
            names = quiet.setdefault(pair.reason_code, [])
            if pair.pair not in names:
                names.append(pair.pair)
            continue
        if pair.status == "skipped":
            rows.append([pair.pair, sensors, "skipped", pair.reason_code or "", "", "", ""])
            continue
        judged = "; ".join(_format_axis(axis) for axis in pair.axes) or "-"
        unchecked = ", ".join(axis.name for axis in pair.unchecked_axes) or "-"
        detect = "; ".join(
            f"{axis.name} {axis.detectable_error:.3g} {axis.unit}" for axis in pair.axes
        )
        verdict: str = pair.status
        if pair.coverage == "partial" and pair.axes:
            verdict += f" (partial: {', '.join(axis.name for axis in pair.axes)} only)"
        rows.append(
            [
                pair.pair,
                sensors,
                verdict,
                pair.reason_code or "",
                judged,
                unchecked,
                detect or "-",
            ]
        )
    lines.extend(
        _table(
            [
                "pair",
                "sensors",
                "verdict",
                "reason",
                "|delta|/tolerance",
                "unchecked",
                "detectable",
            ],
            rows,
        )
    )
    for code, names in quiet.items():
        lines.append(f"  {code.replace('_', ' ')}: {', '.join(names)}")
    for pair in artifact.pairs:
        if pair.status != "skipped" and pair.reason:
            lines.append(f"  {pair.pair}: {pair.reason}")
        if pair.time_offset is not None:
            lines.append(
                f"  {pair.pair}: time offset {pair.time_offset.estimate_s * 1e3:.2f} ms "
                f"({pair.time_offset.status}; reported, not judged)"
            )
        for item in pair.unchecked_axes:
            lines.append(f"  {pair.pair}: unchecked {item.name}: {item.reason}")
    lines.append("")
    lines.append(f"overall verdict: {artifact.overall_verdict}")
    if artifact.summary.partial_pairs:
        lines.append(
            f"WARNING: {artifact.summary.partial_pairs} pair(s) have partial coverage: "
            "unchecked axes were not judged, so a pass covers only the judged axes"
        )
    return lines


def format_check_table(artifact: CalibrationCheckArtifact) -> str:
    """Render the artifact as a human-readable report."""

    lines = [
        f"calibrex check ({'plan' if artifact.plan_only else 'run'})",
        f"bag: {artifact.bag.path} ({artifact.bag.topic_count} topics)",
        "",
        "candidate sources:",
    ]
    if artifact.candidate_sources:
        for source in artifact.candidate_sources:
            digest = f" sha256={source.sha256[:12]}" if source.sha256 else ""
            lines.append(f"  {source.kind}: {source.path} ({source.frame_count} frames){digest}")
            lines.extend(f"    note: {note}" for note in source.notes)
    else:
        lines.append("  none (no /tf_static in the bag and no --tf file)")
    tree = artifact.frame_tree
    lines.append(
        f"frame tree: {len(tree.frames)} frames, {len(tree.roots)} root(s)"
        + (f" [{', '.join(tree.roots)}]" if tree.roots else "")
    )
    lines.extend(f"  override: {item}" for item in tree.overrides)
    lines.append("")
    lines.append("topics:")
    topic_rows = [
        [
            topic.topic,
            topic.role or "-",
            topic.header_frame_id or "-",
            (
                f"{topic.mapped_frame} ({topic.frame_source})"
                if topic.mapped_frame is not None
                else ("UNMAPPED" if topic.role not in {"tf_static", None} else "-")
            ),
        ]
        for topic in artifact.topics
    ]
    lines.extend(_table(["topic", "role", "header frame", "tree frame"], topic_rows))
    unmapped = [
        topic.topic
        for topic in artifact.topics
        if topic.role not in {"tf_static", None} and topic.mapped_frame is None
    ]
    if unmapped:
        lines.append(f"unmapped topics: {', '.join(unmapped)}")
    lines.append("")
    lines.append(
        f"pairs (vehicle frame '{artifact.vehicle_frame}'):"
        if artifact.vehicle_frame
        else "pairs (no --vehicle-frame: vehicle pairs are skipped):"
    )
    if artifact.plan_only:
        pair_rows = [
            [
                pair.pair,
                " / ".join(pair.sensors) if pair.sensors else "-",
                pair.status,
                pair.reason_code or "",
                pair.reason or "",
            ]
            for pair in artifact.pairs
        ]
        lines.extend(_table(["pair", "sensors", "status", "reason", "detail"], pair_rows))
    else:
        lines.extend(_verdict_lines(artifact))
    summary = artifact.summary
    counts = ", ".join(f"{name}={count}" for name, count in summary.status_counts.items())
    lines.append("")
    lines.append(
        f"summary: {summary.pair_count} pair record(s), {summary.runnable_count} runnable"
        + (f" ({counts})" if counts else "")
    )
    return "\n".join(lines)
