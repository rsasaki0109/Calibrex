from __future__ import annotations

from typing import Any

from tools import check_schema_compat as compat


def _levels(old: dict[str, Any], new: dict[str, Any]) -> dict[str, str]:
    return {
        finding.location: finding.level for finding in compat._compare("x.schema.json", old, new)
    }


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"schema_version": {"const": "slac.x/v0.1"}, **properties},
        "required": required or [],
    }


def test_new_optional_property_is_compatible() -> None:
    old = _object({"a": {"type": "string"}})
    new = _object({"a": {"type": "string"}, "b": {"type": "integer"}})

    assert _levels(old, new) == {"<root>": "compatible"}


def test_new_required_property_is_breaking() -> None:
    old = _object({"a": {"type": "string"}})
    new = _object({"a": {"type": "string"}, "b": {"type": "integer"}}, required=["b"])

    assert _levels(old, new)["<root>"] == "breaking"


def test_removed_property_on_closed_object_is_breaking() -> None:
    old = _object({"a": {"type": "string"}, "b": {"type": "integer"}})
    new = _object({"a": {"type": "string"}})

    assert _levels(old, new)["<root>"] == "breaking"


def test_added_pattern_inside_nullable_branch_is_breaking() -> None:
    old = _object({"digest": {"anyOf": [{"type": "string"}, {"type": "null"}]}})
    new = _object(
        {"digest": {"anyOf": [{"type": "string", "pattern": "^sha256:"}, {"type": "null"}]}}
    )

    assert _levels(old, new)["<root>.digest"] == "breaking"


def test_enum_widening_is_compatible_and_narrowing_is_breaking() -> None:
    old = _object({"kind": {"enum": ["a", "b"]}})

    assert _levels(old, _object({"kind": {"enum": ["a", "b", "c"]}})) == {
        "<root>.kind": "compatible"
    }
    assert _levels(old, _object({"kind": {"enum": ["a"]}})) == {"<root>.kind": "breaking"}


def test_removed_type_is_breaking() -> None:
    old = _object({"value": {"type": ["string", "number"]}})
    new = _object({"value": {"type": "string"}})

    assert _levels(old, new) == {"<root>.value": "breaking"}


def test_description_only_change_is_ignored() -> None:
    old = _object({"a": {"type": "string", "description": "old"}})
    new = _object({"a": {"type": "string", "description": "new"}})

    assert _levels(old, new) == {}


def test_schema_versions_read_const_and_enum() -> None:
    assert compat._schema_versions(_object({})) == ("slac.x/v0.1",)
    assert compat._schema_versions(
        {"properties": {"schema_version": {"enum": ["slac.x/v0.1", "slac.x/v0.2"]}}}
    ) == ("slac.x/v0.1", "slac.x/v0.2")


def test_retired_version_is_not_reported_as_unreadable() -> None:
    assert compat._unreadable_version("slac.doctor/v0.1") is None
    assert compat._unreadable_version("slac.never_existed/v0.1") is not None
