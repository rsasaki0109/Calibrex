from __future__ import annotations

import json
import subprocess
import typing
from pathlib import Path
from typing import Any

import pytest
import yaml

from calibrex.core.exceptions import CalibrexError
from calibrex.core.schema_registry import (
    RETIRED_SCHEMA_VERSIONS,
    SCHEMA_LEDGER_FILENAME,
    SchemaMigration,
    SchemaRegistryError,
    kind_for_schema_version,
    schema_entries,
    schema_entry,
    schema_ledger,
    upgrade_payload,
)
from calibrex.core.validation import ValidationKind, validate_file, validation_kind_choices

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas"

# Committed files that carry a ``slac.*`` schema_version but are documentation
# assets or catalogs without a published, validatable schema.
UNREGISTERED_SCHEMA_VERSIONS = {
    "slac.public_datasets/v0.1": "public dataset catalog; no published schema",
    "slac.readme_gif_gallery/v0.3": "README gallery provenance asset",
    "slac.readme_time_offset_sweep/v0.1": "README time-sweep provenance asset",
}

# Committed artifacts known to violate their declared schema.  Each entry must
# keep failing; once fixed, delete it so the file is covered again.
KNOWN_INVALID_ARTIFACTS = {
    "examples/raw_replay/synthetic/config.yaml": (
        "digest-bound replay stub declared as slac.config/v0.1 but lacks dataset, "
        "sensors, and frames"
    ),
}


def _migration(
    from_version: str,
    to_version: str,
    kind: str = "trajectory",
) -> SchemaMigration:
    def upgrade(payload: dict[str, Any]) -> dict[str, Any]:
        payload["schema_version"] = to_version
        payload.setdefault("upgraded_from", []).append(from_version)
        return payload

    return SchemaMigration(
        kind=kind,
        from_version=from_version,
        to_version=to_version,
        upgrade=upgrade,
        description=f"test {from_version} -> {to_version}",
    )


def test_registry_kinds_and_filenames_are_unique() -> None:
    kinds = [entry.kind for entry in schema_entries()]
    filenames = [entry.filename for entry in schema_entries()]

    assert len(kinds) == len(set(kinds))
    assert len(filenames) == len(set(filenames))


def test_registry_filenames_match_committed_schemas() -> None:
    committed = {path.name for path in SCHEMA_DIR.glob("*.schema.json")}

    assert {entry.filename for entry in schema_entries()} == committed


def test_current_version_is_accepted_and_versions_have_one_owner() -> None:
    owners: dict[str, str] = {}
    for entry in schema_entries():
        if entry.status == "schema-only":
            continue
        assert entry.current_version in entry.accepted_versions, entry.kind
        if entry.status == "alias":
            continue
        for version in entry.accepted_versions:
            owner = owners.setdefault(version, entry.kind)
            assert owner == entry.kind, f"{version} read by {owner} and {entry.kind}"


def test_aliases_point_at_active_kinds_with_the_same_model() -> None:
    aliases = [entry for entry in schema_entries() if entry.alias_of is not None]

    assert aliases
    for alias in aliases:
        target = schema_entry(alias.alias_of or "")
        assert target.status == "active"
        assert target.model is alias.model


def test_every_accepted_version_auto_detects_to_its_owner() -> None:
    for entry in schema_entries():
        if entry.status != "active":
            continue
        for version in entry.accepted_versions:
            assert kind_for_schema_version(version) == entry.kind


def test_validation_kind_literal_matches_registry() -> None:
    assert set(typing.get_args(ValidationKind)) == set(validation_kind_choices())


def test_committed_ledger_matches_registry() -> None:
    committed = json.loads((SCHEMA_DIR / SCHEMA_LEDGER_FILENAME).read_text(encoding="utf-8"))

    assert committed == schema_ledger(), (
        "schemas/schema_ledger.json is stale; run "
        "`calibrex schema all --output-dir schemas` and review the diff"
    )


def test_retired_versions_are_unreadable_and_explain_the_remedy(tmp_path: Path) -> None:
    for retired in RETIRED_SCHEMA_VERSIONS:
        schema_entry(retired.kind)
        for entry in schema_entries():
            assert retired.schema_version not in entry.accepted_versions

    artifact = tmp_path / "doctor.yaml"
    artifact.write_text("schema_version: slac.doctor/v0.1\nstatus: pass\n", encoding="utf-8")

    with pytest.raises(CalibrexError, match=r"retired in Calibrex 0\.5\.0.*calibrex doctor"):
        validate_file(artifact)


def test_unknown_version_points_at_the_ledger() -> None:
    with pytest.raises(SchemaRegistryError, match="calibrex schema ledger"):
        kind_for_schema_version("slac.never_existed/v9.9")


def test_upgrade_payload_chains_migrations_without_mutating_input() -> None:
    current = schema_entry("trajectory").current_version
    assert current is not None
    migrations = (
        _migration("slac.trajectory/v0.0", "slac.trajectory/v0.0.1"),
        _migration("slac.trajectory/v0.0.1", current),
    )
    source = {"schema_version": "slac.trajectory/v0.0"}

    upgraded, chain = upgrade_payload("trajectory", source, migrations)

    assert source == {"schema_version": "slac.trajectory/v0.0"}
    assert upgraded["schema_version"] == current
    assert chain == ("slac.trajectory/v0.0", "slac.trajectory/v0.0.1")
    assert kind_for_schema_version("slac.trajectory/v0.0", migrations) == "trajectory"


def test_upgrade_payload_leaves_accepted_versions_untouched() -> None:
    current = schema_entry("trajectory").current_version
    source = {"schema_version": current, "value": 1}

    upgraded, chain = upgrade_payload("trajectory", source, ())

    assert upgraded is source
    assert chain == ()


def test_upgrade_payload_rejects_wrong_target_version() -> None:
    broken = SchemaMigration(
        kind="trajectory",
        from_version="slac.trajectory/v0.0",
        to_version="slac.trajectory/v0.1",
        upgrade=lambda payload: {**payload, "schema_version": "slac.trajectory/v7"},
        description="broken",
    )

    with pytest.raises(SchemaRegistryError, match="produced schema_version"):
        upgrade_payload("trajectory", {"schema_version": "slac.trajectory/v0.0"}, (broken,))


def test_digest_bound_kinds_cannot_be_migrated() -> None:
    digest_bound = next(entry for entry in schema_entries() if entry.digest_method is not None)
    current = digest_bound.current_version
    assert current is not None
    migration = _migration("slac.legacy/v0.0", current, kind=digest_bound.kind)

    with pytest.raises(SchemaRegistryError, match="digest-bound"):
        upgrade_payload(digest_bound.kind, {"schema_version": "slac.legacy/v0.0"}, (migration,))


def _committed_slac_artifacts() -> list[tuple[str, str]]:
    listed = subprocess.run(
        ["git", "ls-files", "*.yaml", "*.yml", "*.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        pytest.skip("git is required to enumerate committed artifacts")
    artifacts: list[tuple[str, str]] = []
    for name in listed.stdout.split():
        if name.startswith(("schemas/", ".github/", "site/")):
            continue
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        except (UnicodeDecodeError, ValueError, yaml.YAMLError):
            continue
        if not isinstance(payload, dict):
            continue
        version = payload.get("schema_version")
        if isinstance(version, str) and version.startswith("slac."):
            artifacts.append((name, version))
    return artifacts


def test_every_committed_artifact_is_readable() -> None:
    artifacts = _committed_slac_artifacts()
    assert artifacts

    unreadable: list[str] = []
    for name, version in artifacts:
        if version in UNREGISTERED_SCHEMA_VERSIONS:
            continue
        try:
            validate_file(REPO_ROOT / name)
        except (CalibrexError, ValueError) as exc:
            if name in KNOWN_INVALID_ARTIFACTS:
                continue
            unreadable.append(f"{name} ({version}): {str(exc).splitlines()[0]}")
        else:
            assert name not in KNOWN_INVALID_ARTIFACTS, (
                f"{name} validates now; remove it from KNOWN_INVALID_ARTIFACTS"
            )

    assert not unreadable, "\n".join(unreadable)


def test_unregistered_versions_are_still_unregistered() -> None:
    for version in UNREGISTERED_SCHEMA_VERSIONS:
        with pytest.raises(SchemaRegistryError):
            kind_for_schema_version(version)
