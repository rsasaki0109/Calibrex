"""Release bodies written from CHANGELOG.md by ``tools/release_notes.py``."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools import release_notes

ROOT = Path(__file__).resolve().parents[2]

CHANGELOG = """# Changelog

## Unreleased

- Something later.

## 1.2.0 - 2026-01-01

Highlights:

- **First highlight**, wrapped
  onto a second line.
- Second highlight.

- An ordinary entry that is not a highlight.

## 1.1.0 - 2025-12-01

- No highlights here.
"""


def test_the_highlights_list_ends_at_its_blank_line() -> None:
    assert release_notes.changelog_highlights(CHANGELOG, "1.2.0") == [
        "- **First highlight**, wrapped",
        "  onto a second line.",
        "- Second highlight.",
    ]


@pytest.mark.parametrize(
    ("version", "message"), [("1.1.0", "no Highlights"), ("9.9.9", "no section")]
)
def test_a_missing_section_or_list_is_an_error(version: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        release_notes.changelog_highlights(CHANGELOG, version)


def test_the_body_keeps_the_generated_list_and_drops_the_draft_note() -> None:
    draft = (
        "Review CHANGELOG.md before publishing this draft.\n\n"
        "## What's Changed\n* a PR by @someone in #1\n"
    )
    body = release_notes.release_body(CHANGELOG, "1.2.0", "v1.2.0", draft)
    assert body.startswith("## Highlights\n\n- **First highlight**")
    assert "releases/download/v1.2.0/calibrex-1.2.0-py3-none-any.whl" in body
    assert "blob/v1.2.0/CHANGELOG.md" in body
    assert body.rstrip().endswith("## What's Changed\n* a PR by @someone in #1")
    assert "before publishing this draft" not in body
    assert "What's Changed" not in release_notes.release_body(CHANGELOG, "1.2.0", "v1.2.0")


def test_the_repository_changelog_has_highlights_for_the_released_versions() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for version in ("0.6.0", "0.5.1"):
        highlights = release_notes.changelog_highlights(changelog, version)
        assert highlights[0].startswith("- ")
