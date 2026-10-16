"""Small YAML/JSON IO helpers used by CLI and models."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

import yaml


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
            yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=False)


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
                yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=False)
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
