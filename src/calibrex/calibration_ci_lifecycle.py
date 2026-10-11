"""Record a Calibration CI decision in the append-only lifecycle registry.

``calibrex ci`` decides whether one candidate result passes its falsification
policy. This module carries that decision into a calibration lifecycle
registry without weakening either side:

* the CI decision, the candidate result, the CI assessment and the CI policy
  are copied into the registry under ``ci/<sha256 of calibration-ci.json>/``,
  a content-addressed snapshot that is written once and never overwritten, so
  every event references bytes the registry itself keeps;
* the candidate is evaluated through :func:`evaluate_lifecycle`, with the CI
  decision as its evidence and the CI assessment as its assessment. A PASS
  decision with PASS promotion and smoke artifacts is ADOPT and may be promoted;
  an INCONCLUSIVE (weakly observable) or WARN decision is HOLD and a FAIL
  decision is DO_NOT_ADOPT, and neither ever changes an edge incumbent;
* a FAIL decision on the edge's current incumbent, when that incumbent came from
  a promotion, can roll the edge back to the incumbent before that promotion,
  with the CI decision bound to the rollback event; otherwise it is recorded as
  a rejection.

The registry's own verification (event chain, head, state projection and every
artifact digest) is run after the events are appended and recorded in the
returned artifact.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from calibrex import __version__
from calibrex.calibration_ci import CalibrationCIArtifact
from calibrex.core.assessment import AssessmentStatus
from calibrex.core.exceptions import CalibrexError
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.lifecycle_registry import (
    LifecycleAdmission,
    LifecycleArtifactRef,
    LifecycleEvent,
    evaluate_lifecycle,
    load_registry,
    promote_lifecycle,
    rollback_lifecycle,
)
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel

CALIBRATION_CI_LIFECYCLE_SCHEMA_VERSION: Literal["slac.calibration_ci_lifecycle/v0.1"] = (
    "slac.calibration_ci_lifecycle/v0.1"
)
CI_SNAPSHOT_DIRNAME = "ci"

CalibrationCILifecycleAction = Literal[
    "promote", "adoptable", "hold", "reject", "blocked", "rollback"
]


class CalibrationCILifecycleEventRef(StrictModel):
    """One registry event appended for the CI decision."""

    sequence: int = Field(ge=0)
    event_type: str
    status: str
    event_sha256: str


class CalibrationCILifecycleProvenance(StrictModel):
    """Who recorded the decision, with which Calibrex."""

    generator: str = __name__
    generator_version: str = __version__
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    github_repository: str | None = None
    github_sha: str | None = None
    github_run_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CalibrationCILifecycleArtifact(StrictModel):
    """What a Calibration CI decision did to one lifecycle registry edge."""

    schema_version: Literal["slac.calibration_ci_lifecycle/v0.1"] = (
        CALIBRATION_CI_LIFECYCLE_SCHEMA_VERSION
    )
    action: CalibrationCILifecycleAction
    reason: str
    registry_id: str
    edge_id: str
    ci_status: AssessmentStatus
    ci_decision: LifecycleArtifactRef
    candidate_result: LifecycleArtifactRef
    admission: LifecycleAdmission | None = None
    events: list[CalibrationCILifecycleEventRef] = Field(min_length=1)
    incumbent_before_sha256: str | None = None
    incumbent_after_sha256: str | None = None
    incumbent_changed: bool
    registry_valid: bool
    registry_head_sha256: str | None = None
    provenance: CalibrationCILifecycleProvenance

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def calibration_ci_lifecycle_json_schema() -> dict[str, object]:
    """Return the JSON schema for recorded Calibration CI lifecycle decisions."""

    return CalibrationCILifecycleArtifact.model_json_schema()


def record_calibration_ci(
    ci_decision: str | Path,
    registry_root: str | Path,
    *,
    edge_id: str,
    capture_manifest: str | Path,
    promotion: str | Path | None = None,
    smoke: str | Path | None = None,
    promote: bool = False,
    rollback_on_fail: bool = False,
    operator: str = "unknown",
    timestamp: str | None = None,
    command: list[str] | None = None,
) -> CalibrationCILifecycleArtifact:
    """Append the lifecycle events a Calibration CI decision calls for; see the module."""

    decision_path = Path(ci_decision)
    decision = CalibrationCIArtifact.model_validate(read_mapping(decision_path))
    candidate_source = _ci_path(decision.candidate.path, decision_path)
    observed = sha256_path(candidate_source)
    if observed != decision.candidate.sha256:
        raise CalibrexError(
            f"candidate result {candidate_source} changed after the CI decision "
            f"(declared {decision.candidate.sha256}, observed {observed})"
        )
    registry = load_registry(registry_root)
    root = registry.root
    snapshot = _snapshot(
        root,
        decision_path,
        {
            "candidate": candidate_source,
            "assessment": _ci_path(decision.artifacts.assessment, decision_path),
            "policy": _ci_path(decision.artifacts.ci_policy, decision_path),
        },
    )
    state = registry.state()
    if edge_id not in state.edges:
        raise CalibrexError(f"calibration edge {edge_id!r} is not registered")
    edge = state.edges[edge_id]
    incumbent_before = None if edge.incumbent_result is None else edge.incumbent_result.sha256
    reason_tag = f"calibration CI {snapshot['decision_sha256'][:12]} read {decision.status}"
    command = command or []
    events: list[LifecycleEvent] = []
    admission: LifecycleAdmission | None = None

    promoted_before = any(
        event.event_type == "promote" and edge_id in event.edge_ids for event in registry.events()
    )
    if (
        rollback_on_fail
        and decision.status == "fail"
        and promoted_before
        and incumbent_before is not None
        and incumbent_before == decision.candidate.sha256
    ):
        events.append(
            rollback_lifecycle(
                root,
                edge_id=edge_id,
                evidence=snapshot["decision"],
                operator=operator,
                reason=f"{reason_tag} on the incumbent; roll back to the prior incumbent",
                timestamp=timestamp,
                command=command,
            )
        )
        action: CalibrationCILifecycleAction = "rollback"
        reason = "the CI decision failed the edge's incumbent, which was rolled back"
    else:
        evaluation_path = Path(snapshot["directory"]) / "evaluation.json"
        if evaluation_path.exists():
            raise CalibrexError(
                f"this CI decision was already recorded in the registry ({evaluation_path})"
            )
        evaluation = evaluate_lifecycle(
            root,
            edge_id=edge_id,
            capture_manifest=capture_manifest,
            candidate_result=snapshot["candidate"],
            evidence=snapshot["decision"],
            assessment=snapshot["assessment"],
            policy=snapshot["policy"],
            promotion=promotion,
            smoke=smoke,
            output=evaluation_path,
            operator=operator,
            reason=reason_tag,
            timestamp=timestamp,
            command=command,
        )
        admission = evaluation.admission
        events.append(registry.events()[-1])
        reason = evaluation.reason
        if admission == "ADOPT" and promote:
            events.append(
                promote_lifecycle(
                    root,
                    edge_id=edge_id,
                    evaluation=evaluation_path,
                    operator=operator,
                    reason=f"{reason_tag}; promote the evaluated candidate",
                    timestamp=timestamp,
                    command=command,
                )
            )
            action = "promote"
        elif admission == "ADOPT":
            action = "adoptable"
        elif admission == "HOLD":
            action = "hold"
        elif admission == "BLOCKED":
            action = "blocked"
        else:
            action = "reject"

    after = load_registry(root)
    edge_after = after.state().edges[edge_id]
    incumbent_after = (
        None if edge_after.incumbent_result is None else edge_after.incumbent_result.sha256
    )
    verification = after.verify()
    return CalibrationCILifecycleArtifact(
        action=action,
        reason=reason,
        registry_id=after.manifest().registry_id,
        edge_id=edge_id,
        ci_status=decision.status,
        ci_decision=_ref(root, snapshot["decision"], "calibration_ci"),
        candidate_result=_ref(root, snapshot["candidate"], "result"),
        admission=admission,
        events=[
            CalibrationCILifecycleEventRef(
                sequence=event.sequence,
                event_type=event.event_type,
                status=event.status,
                event_sha256=event.event_sha256,
            )
            for event in events
        ],
        incumbent_before_sha256=incumbent_before,
        incumbent_after_sha256=incumbent_after,
        incumbent_changed=incumbent_before != incumbent_after,
        registry_valid=verification.valid,
        registry_head_sha256=after.head().event_sha256,
        provenance=CalibrationCILifecycleProvenance(
            git_commit=git_commit(),
            command=command,
            github_repository=os.environ.get("GITHUB_REPOSITORY"),
            github_sha=os.environ.get("GITHUB_SHA"),
            github_run_id=os.environ.get("GITHUB_RUN_ID"),
        ),
    )


def _ci_path(path: str, decision_path: Path) -> Path:
    """Resolve a path recorded by ``calibrex ci`` (as given, else next to the decision)."""

    raw = Path(path)
    if raw.is_absolute() or raw.exists():
        return raw
    beside = decision_path.parent / raw.name
    return beside if beside.exists() else raw


def _snapshot(root: Path, decision_path: Path, sources: dict[str, Path]) -> dict[str, str]:
    """Copy the CI decision and its inputs into ``ci/<decision sha256>/``, never overwriting."""

    decision_sha = sha256_path(decision_path)
    if decision_sha is None:
        raise CalibrexError(f"cannot digest the CI decision {decision_path}")
    directory = root / CI_SNAPSHOT_DIRNAME / decision_sha
    directory.mkdir(parents=True, exist_ok=True)
    stored = {"decision": _store(decision_path, directory / "calibration-ci.json")}
    for role, source in sources.items():
        if not source.is_file():
            raise CalibrexError(f"CI {role} artifact {source} does not exist")
        suffix = "".join(source.suffixes) or ".json"
        stored[role] = _store(source, directory / f"{role}{suffix}")
    stored["directory"] = str(directory)
    stored["decision_sha256"] = decision_sha
    return stored


def _store(source: Path, target: Path) -> str:
    """Write ``source`` to ``target`` once; an existing target must hold the same bytes."""

    data = source.read_bytes()
    if target.exists():
        if target.read_bytes() != data:
            raise CalibrexError(f"refusing to overwrite registry snapshot {target}")
        return str(target)
    with target.open("xb") as stream:
        stream.write(data)
    return str(target)


def _ref(root: Path, path: str, kind: str) -> LifecycleArtifactRef:
    resolved = Path(path)
    digest = sha256_path(resolved)
    assert digest is not None
    try:
        portable = resolved.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        portable = str(resolved.resolve())
    return LifecycleArtifactRef(
        kind=kind, path=portable, sha256=digest, size_bytes=resolved.stat().st_size
    )
