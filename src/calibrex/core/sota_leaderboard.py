"""Schema-valid per-sensor-pair standings built from frozen SOTA audits.

The leaderboard never computes a verdict of its own.  Each entry copies the
verdict of one digest-pinned audit result, and each pair's standing only counts
those verdicts, so a refuted claim cannot be hidden behind a supported one.
Declared target pairs without any audit are listed as ``no_claim`` to keep
coverage gaps visible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping
from calibrex.core.result import StrictModel
from calibrex.core.sota_audit import (
    SotaClaimScope,
    SotaRequirementStatus,
    SotaVerdict,
    save_sota_model,
)

SOTA_LEADERBOARD_SCHEMA_VERSION: Literal["slac.sota_leaderboard/v0.1"] = (
    "slac.sota_leaderboard/v0.1"
)
SotaStanding = Literal["supported", "refuted", "incomplete", "no_claim"]


class SotaLeaderboardEntry(StrictModel):
    """One audited claim, copied from a digest-pinned audit result."""

    pair_id: str
    scope: SotaClaimScope
    protocol_id: str
    audit_id: str
    claim_text: str
    verdict: SotaVerdict
    requirement_counts: dict[SotaRequirementStatus, int]
    achieved_dataset_families: list[str]
    achieved_independent_rig_count: int = Field(ge=0)
    audit_path: str
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_schema_version: str

    @model_validator(mode="after")
    def check_pair_id(self) -> SotaLeaderboardEntry:
        """Keep ``pair_id`` derived from the scope."""

        if self.pair_id != self.scope.pair_id:
            msg = f"pair_id {self.pair_id!r} does not match scope {self.scope.pair_id!r}"
            raise ValueError(msg)
        return self


class SotaPairStanding(StrictModel):
    """Claim counts for one sensor pair."""

    pair_id: str
    standing: SotaStanding
    supported_claims: int = Field(ge=0)
    refuted_claims: int = Field(ge=0)
    incomplete_claims: int = Field(ge=0)
    target: bool = False

    @model_validator(mode="after")
    def check_standing(self) -> SotaPairStanding:
        """Derive the standing from the counts, never the other way round."""

        if self.standing != standing_from_counts(
            self.supported_claims, self.refuted_claims, self.incomplete_claims
        ):
            raise ValueError("pair standing must follow from its claim counts")
        return self


class SotaLeaderboardProvenance(StrictModel):
    """Lineage of a generated leaderboard."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SotaLeaderboard(StrictModel):
    """Per-pair SOTA standings with every contributing audit listed."""

    schema_version: Literal["slac.sota_leaderboard/v0.1"] = SOTA_LEADERBOARD_SCHEMA_VERSION
    entries: list[SotaLeaderboardEntry]
    pairs: list[SotaPairStanding]
    provenance: SotaLeaderboardProvenance

    @model_validator(mode="after")
    def check_pairs_match_entries(self) -> SotaLeaderboard:
        """Require each pair's counts to equal its entries' verdicts."""

        pair_ids = [item.pair_id for item in self.pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("leaderboard pair IDs must be unique")
        for pair in self.pairs:
            verdicts = [item.verdict for item in self.entries if item.pair_id == pair.pair_id]
            counts = (
                verdicts.count("supported"),
                verdicts.count("refuted"),
                verdicts.count("incomplete"),
            )
            if counts != (pair.supported_claims, pair.refuted_claims, pair.incomplete_claims):
                raise ValueError(f"pair {pair.pair_id!r} counts do not match its entries")
        missing = {item.pair_id for item in self.entries} - set(pair_ids)
        if missing:
            raise ValueError(f"entries reference pairs without a standing: {sorted(missing)}")
        return self

    def save(self, path: str | Path) -> None:
        """Save the leaderboard."""

        save_sota_model(self, path)


def standing_from_counts(supported: int, refuted: int, incomplete: int) -> SotaStanding:
    """Return ``supported`` if any claim is supported, then refuted, then incomplete."""

    if supported:
        return "supported"
    if refuted:
        return "refuted"
    if incomplete:
        return "incomplete"
    return "no_claim"


def sota_leaderboard_json_schema() -> dict[str, Any]:
    """Return the SOTA leaderboard JSON schema."""

    return SotaLeaderboard.model_json_schema()


def load_sota_leaderboard(path: str | Path) -> SotaLeaderboard:
    """Load and validate a SOTA leaderboard."""

    return SotaLeaderboard.model_validate(read_mapping(Path(path)))
