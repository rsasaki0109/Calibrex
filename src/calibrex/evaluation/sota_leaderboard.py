"""Build and render per-pair SOTA standings from audit result artifacts."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from pathlib import Path

from calibrex import __version__
from calibrex.core.camera_lidar_sota_audit import CameraLidarSotaAuditResult
from calibrex.core.io import read_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.sota_audit import SotaAuditResult, SotaModality
from calibrex.core.sota_leaderboard import (
    SotaLeaderboard,
    SotaLeaderboardEntry,
    SotaLeaderboardProvenance,
    SotaPairStanding,
    standing_from_counts,
)


def build_sota_leaderboard(
    audit_paths: Sequence[str | Path],
    *,
    target_pairs: Sequence[str] = (),
    command: list[str] | None = None,
) -> SotaLeaderboard:
    """Collect pair-agnostic or Camera--LiDAR audit results into standings."""

    entries = sorted(
        (_entry(Path(path)) for path in audit_paths),
        key=lambda item: (item.pair_id, item.protocol_id, item.audit_id),
    )
    targets = {_canonical_pair_id(pair) for pair in target_pairs}
    pairs: list[SotaPairStanding] = []
    for pair_id in sorted({item.pair_id for item in entries} | targets):
        verdicts = [item.verdict for item in entries if item.pair_id == pair_id]
        counts = (
            verdicts.count("supported"),
            verdicts.count("refuted"),
            verdicts.count("incomplete"),
        )
        pairs.append(
            SotaPairStanding(
                pair_id=pair_id,
                standing=standing_from_counts(*counts),
                supported_claims=counts[0],
                refuted_claims=counts[1],
                incomplete_claims=counts[2],
                target=pair_id in targets,
            )
        )
    return SotaLeaderboard(
        entries=entries,
        pairs=pairs,
        provenance=SotaLeaderboardProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
        ),
    )


def render_sota_leaderboard_markdown(leaderboard: SotaLeaderboard) -> str:
    """Render standings and every contributing claim as Markdown tables."""

    lines = [
        "| Pair | Standing | Supported | Refuted | Incomplete | Target |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for pair in leaderboard.pairs:
        lines.append(
            f"| `{pair.pair_id}` | {pair.standing} | {pair.supported_claims} | "
            f"{pair.refuted_claims} | {pair.incomplete_claims} | "
            f"{'yes' if pair.target else ''} |"
        )
    if leaderboard.entries:
        lines += [
            "",
            "| Pair | Category | Quantities | Verdict | Gates (achieved/total) | Protocol |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for entry in leaderboard.entries:
            total = sum(entry.requirement_counts.values())
            achieved = entry.requirement_counts.get("achieved", 0)
            lines.append(
                f"| `{entry.pair_id}` | {entry.scope.category} | "
                f"{', '.join(entry.scope.quantities)} | {entry.verdict} | "
                f"{achieved}/{total} | `{entry.protocol_id}` |"
            )
    return "\n".join(lines) + "\n"


def _entry(path: Path) -> SotaLeaderboardEntry:
    payload = read_mapping(path)
    schema_version = payload.get("schema_version")
    result: SotaAuditResult
    if schema_version == "slac.camera_lidar_sota_audit_result/v0.1":
        result = CameraLidarSotaAuditResult.model_validate(payload).as_generic()
    elif schema_version == "slac.sota_audit_result/v0.1":
        result = SotaAuditResult.model_validate(payload)
    else:
        msg = f"{path} is not a SOTA audit result (schema_version {schema_version!r})"
        raise ValueError(msg)
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"audit result is not readable: {path}")
    counts: dict[str, int] = {}
    for requirement in result.requirements:
        counts[requirement.status] = counts.get(requirement.status, 0) + 1
    return SotaLeaderboardEntry.model_validate(
        {
            "pair_id": result.scope.pair_id,
            "scope": result.scope,
            "protocol_id": result.protocol_id,
            "audit_id": result.audit_id,
            "claim_text": result.claim_text,
            "verdict": result.verdict,
            "requirement_counts": counts,
            "achieved_dataset_families": result.achieved_dataset_families,
            "achieved_independent_rig_count": result.achieved_independent_rig_count,
            "audit_path": path.as_posix(),
            "audit_sha256": digest,
            "audit_schema_version": str(schema_version),
        }
    )


def _canonical_pair_id(pair: str) -> str:
    parts = [part for part in pair.split("-") if part]
    known = set(typing.get_args(SotaModality))
    unknown = sorted(set(parts) - known)
    if not parts or unknown:
        msg = (
            f"invalid target pair {pair!r}; use '-'-joined modalities from "
            f"{', '.join(sorted(known))}"
        )
        raise ValueError(msg)
    return "-".join(sorted(parts))
