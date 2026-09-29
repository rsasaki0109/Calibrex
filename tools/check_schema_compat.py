"""Check that released Calibrex schema versions stay readable.

Compares the working tree against a released git ref (the latest tag by
default) and fails when

* a schema version published at the ref is neither read by a current model,
  upgraded by a registered migration, nor listed as retired;
* a schema whose ``schema_version`` did not change was narrowed (new required
  properties, removed properties, tighter constraints, removed enum values or
  types) unless the narrowing is acknowledged in
  ``tools/schema_compat_acknowledged.yaml``; or
* a ``slac.*`` artifact committed at the ref can no longer be read, unless
  acknowledged as ``artifact:<path>`` (for example, one already invalid when
  it was released).

Compatible widenings (new optional properties, extra enum values) are reported
but never fail the check.  Run with ``--base <ref>`` to compare against another
release.  CI checkouts need tags (``fetch-depth: 0``).
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from calibrex.core.exceptions import CalibrexError
from calibrex.core.schema_registry import (
    SchemaRegistryError,
    kind_for_schema_version,
    retired_schema_version,
)
from calibrex.core.validation import validate_file

REPO_ROOT = Path(__file__).resolve().parents[1]
ACKNOWLEDGED_PATH = REPO_ROOT / "tools" / "schema_compat_acknowledged.yaml"
UNREGISTERED_PREFIXES = (
    "slac.public_datasets/",
    "slac.readme_gif_gallery/",
    "slac.readme_time_offset_sweep/",
)
NARROWING_KEYWORDS = (
    "pattern",
    "format",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "not",
)


@dataclass(frozen=True)
class Finding:
    """One compatibility observation."""

    level: str  # "breaking", "review", or "compatible"
    schema_file: str
    location: str
    detail: str

    @property
    def key(self) -> str:
        return f"{self.schema_file}#{self.location}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="released git ref; defaults to the latest tag")
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    args = parser.parse_args(argv)

    base = args.base or _latest_tag()
    acknowledged = _load_acknowledged()
    findings: list[Finding] = []
    errors: list[str] = []
    seen_artifacts: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="calibrex-schema-compat-") as scratch:
        base_root = Path(scratch)
        _extract(base, base_root)
        for base_schema in sorted((base_root / "schemas").glob("*.schema.json")):
            old = json.loads(base_schema.read_text(encoding="utf-8"))
            current_path = REPO_ROOT / "schemas" / base_schema.name
            for version in _schema_versions(old):
                problem = _unreadable_version(version)
                if problem is not None:
                    errors.append(f"{base_schema.name}: released {version} {problem}")
            if not current_path.exists():
                continue
            new = json.loads(current_path.read_text(encoding="utf-8"))
            if old == new or _schema_versions(old) != _schema_versions(new):
                continue
            findings.extend(_compare(base_schema.name, old, new))
        errors.extend(_unreadable_artifacts(base_root, acknowledged, seen_artifacts))

    unacknowledged = [
        finding
        for finding in findings
        if finding.level == "breaking" and finding.key not in acknowledged
    ]
    stale = sorted(set(acknowledged) - {finding.key for finding in findings} - seen_artifacts)
    errors.extend(
        f"{finding.key}: {finding.detail} (bump the schema version, keep the old one "
        "readable, or acknowledge it in tools/schema_compat_acknowledged.yaml)"
        for finding in unacknowledged
    )
    errors.extend(f"{key}: acknowledged narrowing no longer occurs; remove it" for key in stale)

    if args.json:
        payload = {
            "base": base,
            "errors": errors,
            "findings": [
                finding.__dict__ | {"acknowledged": finding.key in acknowledged}
                for finding in findings
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"schema compatibility against {base}")
        for finding in findings:
            mark = " (acknowledged)" if finding.key in acknowledged else ""
            print(f"  [{finding.level}] {finding.key}: {finding.detail}{mark}")
        for error in errors:
            print(f"ERROR {error}")
        print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


def _latest_tag() -> str:
    completed = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit("no git tag found; fetch tags (fetch-depth: 0) or pass --base")
    return completed.stdout.strip()


def _extract(ref: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", ref],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if archive.returncode != 0:
        raise SystemExit(f"cannot archive {ref}: {archive.stderr.decode(errors='replace')}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(destination, filter="data")


def _load_acknowledged() -> dict[str, str]:
    if not ACKNOWLEDGED_PATH.exists():
        return {}
    payload = yaml.safe_load(ACKNOWLEDGED_PATH.read_text(encoding="utf-8")) or {}
    entries = payload.get("acknowledged", [])
    return {str(entry["key"]): str(entry["reason"]) for entry in entries}


def _schema_versions(schema: Mapping[str, Any]) -> tuple[str, ...]:
    field = schema.get("properties", {}).get("schema_version", {})
    if "const" in field:
        return (str(field["const"]),)
    return tuple(str(value) for value in field.get("enum", ()))


def _unreadable_version(version: str) -> str | None:
    try:
        kind_for_schema_version(version)
    except SchemaRegistryError:
        if retired_schema_version(version) is not None:
            return None
        return "is no longer readable, migrated, or retired"
    return None


def _unreadable_artifacts(
    base_root: Path,
    acknowledged: Mapping[str, str],
    seen: set[str],
) -> list[str]:
    errors: list[str] = []
    for path in sorted(base_root.rglob("*")):
        relative = path.relative_to(base_root)
        if path.suffix not in {".yaml", ".yml", ".json"} or relative.parts[0] in {
            "schemas",
            ".github",
            "site",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        except (UnicodeDecodeError, ValueError, yaml.YAMLError):
            continue
        if not isinstance(payload, dict):
            continue
        version = payload.get("schema_version")
        if not isinstance(version, str) or not version.startswith("slac."):
            continue
        if version.startswith(UNREGISTERED_PREFIXES) or retired_schema_version(version):
            continue
        try:
            validate_file(path)
        except (CalibrexError, ValueError, OSError) as exc:
            reason = str(exc).splitlines()[0]
            key = f"artifact:{relative.as_posix()}"
            if key in acknowledged:
                seen.add(key)
                continue
            errors.append(f"released artifact {relative} ({version}) is unreadable: {reason}")
    return errors


def _definitions(schema: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    yield "<root>", schema
    yield from sorted(schema.get("$defs", {}).items())


def _compare(schema_file: str, old: Mapping[str, Any], new: Mapping[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    new_definitions = dict(_definitions(new))
    for name, old_definition in _definitions(old):
        new_definition = new_definitions.get(name)
        if new_definition is None:
            findings.append(Finding("review", schema_file, name, "definition removed"))
            continue
        findings.extend(_compare_object(schema_file, name, old_definition, new_definition))
    return findings


def _compare_object(
    schema_file: str,
    location: str,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    old_properties = old.get("properties", {})
    new_properties = new.get("properties", {})
    added_required = sorted(set(new.get("required", ())) - set(old.get("required", ())))
    if added_required:
        findings.append(
            Finding("breaking", schema_file, location, f"new required {added_required}")
        )
    removed = sorted(set(old_properties) - set(new_properties))
    if removed and new.get("additionalProperties") is False:
        findings.append(Finding("breaking", schema_file, location, f"removed {removed}"))
    if old.get("additionalProperties") is not False and new.get("additionalProperties") is False:
        findings.append(
            Finding("breaking", schema_file, location, "additional properties now forbidden")
        )
    added = sorted(set(new_properties) - set(old_properties) - set(added_required))
    if added:
        findings.append(Finding("compatible", schema_file, location, f"optional added {added}"))
    for name in sorted(set(old_properties) & set(new_properties)):
        if name == "schema_version":
            continue
        findings.extend(
            _compare_value(
                schema_file,
                f"{location}.{name}",
                old_properties[name],
                new_properties[name],
            )
        )
    return findings


def _compare_value(
    schema_file: str,
    location: str,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
) -> list[Finding]:
    if _strip_annotations(old) == _strip_annotations(new):
        return []
    breaking: list[str] = []
    for keyword in NARROWING_KEYWORDS:
        if keyword in new and new.get(keyword) != old.get(keyword):
            breaking.append(f"{keyword} {'added' if keyword not in old else 'changed'}")
    if "enum" in old and "enum" in new:
        dropped = [value for value in old["enum"] if value not in new["enum"]]
        if dropped:
            breaking.append(f"enum values removed {dropped}")
    elif "enum" in new and "enum" not in old:
        breaking.append("enum restriction added")
    if "const" in new and new.get("const") != old.get("const"):
        breaking.append("const added or changed")
    old_types, new_types = _types(old), _types(new)
    if old_types and new_types and not old_types <= new_types:
        breaking.append(f"types removed {sorted(old_types - new_types)}")
    for branch_keyword in ("anyOf", "oneOf"):
        for index, branch in enumerate(new.get(branch_keyword, ())):
            if isinstance(branch, Mapping):
                breaking.extend(
                    f"{branch_keyword}[{index}] {keyword} added"
                    for keyword in NARROWING_KEYWORDS
                    if keyword in branch and not _keyword_anywhere(old, keyword, branch[keyword])
                )
    if breaking:
        return [Finding("breaking", schema_file, location, "; ".join(breaking))]
    if "enum" in old and "enum" in new:
        return [Finding("compatible", schema_file, location, "enum widened")]
    return [Finding("review", schema_file, location, "definition changed")]


def _strip_annotations(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_annotations(item)
            for key, item in value.items()
            if key not in {"title", "description", "examples"}
        }
    if isinstance(value, list):
        return [_strip_annotations(item) for item in value]
    return value


def _types(schema: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    declared = schema.get("type")
    if isinstance(declared, str):
        found.add(declared)
    elif isinstance(declared, list):
        found.update(str(item) for item in declared)
    for branch_keyword in ("anyOf", "oneOf"):
        for branch in schema.get(branch_keyword, ()):
            if isinstance(branch, Mapping):
                found.update(_types(branch))
                if "$ref" in branch:
                    found.add(f"ref:{branch['$ref']}")
    if "$ref" in schema:
        found.add(f"ref:{schema['$ref']}")
    return found


def _keyword_anywhere(schema: Any, keyword: str, value: Any) -> bool:
    if isinstance(schema, Mapping):
        if schema.get(keyword) == value:
            return True
        return any(_keyword_anywhere(item, keyword, value) for item in schema.values())
    if isinstance(schema, list):
        return any(_keyword_anywhere(item, keyword, value) for item in schema)
    return False


if __name__ == "__main__":
    sys.exit(main())
