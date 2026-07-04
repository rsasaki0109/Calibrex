"""Small YAML/JSON IO helpers used by CLI and models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def read_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON file and return its top-level mapping."""

    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream) if path.suffix.lower() == ".json" else yaml.safe_load(stream)
    if not isinstance(data, dict):
        msg = f"{path} must contain a mapping at the top level"
        raise ValueError(msg)
    return data


def write_mapping(path: Path, data: dict[str, Any]) -> None:
    """Write a mapping as YAML or JSON based on the target suffix."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        if path.suffix.lower() == ".json":
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
        else:
            yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=False)


def write_text(path: Path, text: str) -> None:
    """Write text, creating parent directories first."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
