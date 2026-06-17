"""Provenance helpers for reproducible result files."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_path(path: Path) -> str | None:
    """Return SHA256 for a file, or a deterministic directory manifest hash."""

    if not path.exists():
        return None
    digest = hashlib.sha256()
    if path.is_file():
        _hash_file(path, digest)
        return digest.hexdigest()
    if path.is_dir():
        for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(file_path.relative_to(path)).encode("utf-8"))
            _hash_file(file_path, digest)
        return digest.hexdigest()
    return None


def git_commit() -> str | None:
    """Return current git commit if available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _hash_file(path: Path, digest: hashlib._Hash) -> None:
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
