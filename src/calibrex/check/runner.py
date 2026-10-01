"""Build the ``calibrex check`` artifact for one bag (Phase A: plan only)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from calibrex import __version__
from calibrex.check.frame_tree import StaticFrameTree, normalize_frame_id
from calibrex.check.planner import ALL_WIRED_PAIRS, plan_pairs
from calibrex.check.roles import classify_topics, map_topics_to_frames, read_header_frames
from calibrex.check.tf_sources import (
    LoadedSource,
    load_bag_tf_static,
    load_tf_file,
    merge_hints,
    merge_sources,
)
from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckBagInput,
    CheckCandidateSource,
    CheckFrameEdge,
    CheckFrameTree,
    CheckProvenance,
    CheckTopicRecord,
    CheckTransform,
    summarize_pairs,
)
from calibrex.core.exceptions import DatasetError
from calibrex.core.provenance import git_commit
from calibrex.data.rosbag2 import list_rosbag2_connections, resolve_storage

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


def build_calibration_check(
    bag: str | Path,
    *,
    tf_files: Sequence[str | Path] = (),
    vehicle_frame: str = "base_link",
    frame_overrides: Mapping[str, str] | None = None,
    command: Sequence[str] | None = None,
    wired_pairs: frozenset[str] = ALL_WIRED_PAIRS,
) -> CalibrationCheckArtifact:
    """Read candidate extrinsics, classify topics and plan the pair checks."""

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

    candidates = classify_topics(connections)
    header_frames = read_header_frames(bag_path, candidates)
    topics: list[CheckTopicRecord] = map_topics_to_frames(
        candidates, header_frames, tree, hints, frame_overrides
    )
    vehicle = normalize_frame_id(vehicle_frame)
    pairs = plan_pairs(topics, tree, vehicle_frame=vehicle, wired_pairs=wired_pairs)

    notes = [
        "Phase A: plan only; no pair solver was run",
        f"{len(connections) - len(topics)} bag topic(s) without a sensor role are not listed",
    ]
    return CalibrationCheckArtifact(
        plan_only=True,
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
    lines.append(f"pairs (vehicle frame '{artifact.vehicle_frame}'):")
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
    summary = artifact.summary
    counts = ", ".join(f"{name}={count}" for name, count in summary.status_counts.items())
    lines.append("")
    lines.append(
        f"summary: {summary.pair_count} pair record(s), {summary.runnable_count} runnable"
        + (f" ({counts})" if counts else "")
    )
    return "\n".join(lines)
