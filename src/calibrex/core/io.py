"""Small YAML/JSON IO helpers used by CLI and models."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

import yaml

_SAFE_DUMP = yaml.safe_dump
_SAFE_DUMPER = yaml.SafeDumper
_C_DUMP = yaml.dump
_SAFE_REPRESENTERS = dict(yaml.SafeDumper.yaml_representers)
_SAFE_MULTI_REPRESENTERS = dict(yaml.SafeDumper.yaml_multi_representers)
_SAFE_IGNORE_ALIASES = yaml.SafeDumper.ignore_aliases


def _plain_yaml_data(data: object) -> bool:
    """Avoid speculative representation of user objects or invalid Unicode."""
    pending: list[Any] = [data]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        kind = type(value)
        if kind is str:
            if not value.isascii():
                try:
                    value.encode("utf-8")
                except UnicodeError:
                    return False
        elif value is None or kind in (bool, int, float):
            continue
        elif kind in (dict, list, tuple):
            if id(value) not in seen:
                seen.add(id(value))
                if kind is dict:
                    pending.extend(value.keys())
                    pending.extend(value.values())
                else:
                    pending.extend(value)
        else:
            return False
    return True


def _write_yaml(data: dict[str, Any], stream: TextIO) -> None:
    dumper = getattr(yaml, "CSafeDumper", None)
    tables = ("yaml_representers", "yaml_multi_representers",
              "yaml_implicit_resolvers", "yaml_path_resolvers")
    defaults = ((_SAFE_REPRESENTERS, yaml.SafeDumper.yaml_representers),
                (_SAFE_MULTI_REPRESENTERS, yaml.SafeDumper.yaml_multi_representers))
    if (
        dumper is not None
        and yaml.safe_dump is _SAFE_DUMP
        and yaml.SafeDumper is _SAFE_DUMPER
        and yaml.SafeDumper.ignore_aliases is _SAFE_IGNORE_ALIASES
        and all(getattr(dumper, table) is getattr(yaml.SafeDumper, table) for table in tables)
        and all(
            len(actual) == len(expected) and all(actual.get(k) is v for k, v in expected.items())
            for expected, actual in defaults
        )
        and _plain_yaml_data(data)
    ):
        try:
            # Buffer emission so a C-emitter failure cannot leave a partial
            # prefix before falling back to the original Python dumper.
            document = _C_DUMP(data, Dumper=dumper, sort_keys=False, allow_unicode=False)
        except (yaml.YAMLError, UnicodeError):
            pass
        else:
            stream.write(document)
            return
    yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=False)


def _read_yaml(stream: TextIO) -> Any:
    """Use LibYAML's parser with the same safe constructor and resolver tables."""
    loader = getattr(yaml, "CSafeLoader", None)
    tables = (
        "yaml_constructors",
        "yaml_multi_constructors",
        "yaml_implicit_resolvers",
        "yaml_path_resolvers",
    )
    if loader is None or any(
        getattr(loader, table) is not getattr(yaml.SafeLoader, table) for table in tables
    ):
        # Preserve Python SafeLoader registrations made by an embedding caller.
        return yaml.safe_load(stream)
    try:
        return yaml.load(stream, Loader=loader)
    except yaml.YAMLError:
        # The parsers differ on a few edge cases. Retain the original Python
        # parser's acceptance and diagnostics when LibYAML cannot parse a file.
        stream.seek(0)
        return yaml.safe_load(stream)


def read_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON file and return its top-level mapping."""

    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream) if path.suffix.lower() == ".json" else _read_yaml(stream)
    if not isinstance(data, dict):
        msg = f"{path} must contain a mapping at the top level"
        raise ValueError(msg)
    return data


def write_mapping(path: Path, data: dict[str, Any]) -> None:
    """Write a mapping as YAML or JSON based on the target suffix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" keeps LF on Windows so artifact bytes (and their SHA-256)
    # match across platforms, as in write_mapping_atomic.
    with path.open("w", encoding="utf-8", newline="") as stream:
        if path.suffix.lower() == ".json":
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        else:
            _write_yaml(data, stream)


def write_mapping_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write a YAML/JSON mapping with a replace-on-success commit.

    Result and evidence artifacts are often the only durable record of a
    guarded operation.  Writing through a sibling temporary file prevents a
    killed process or a full disk from leaving a truncated artifact at the
    destination.  ``os.replace`` also replaces a destination symlink itself
    rather than following it.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = None
            if path.suffix.lower() == ".json":
                json.dump(data, stream, indent=2, sort_keys=True)
                stream.write("\n")
            else:
                _write_yaml(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        # Directory fsync is useful on POSIX for crash durability.  Windows
        # does not permit opening a directory this way, so it is best effort.
        try:
            directory_fd = os.open(path.parent, getattr(os, "O_DIRECTORY", 0))
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def write_text(path: Path, text: str) -> None:
    """Write text, creating parent directories first."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
