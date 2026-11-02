#!/usr/bin/env python3
"""Write the body of a GitHub release from CHANGELOG.md.

The body is the version's ``Highlights:`` list, the wheel install command, a link to the
changelog at the tag, and the "What's Changed" list GitHub generated for the draft. The
``Publish release`` workflow uses it before it publishes a draft.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPOSITORY = "rsasaki0109/Calibrex"
GENERATED_MARKER = "## What's Changed"


def changelog_highlights(changelog: str, version: str) -> list[str]:
    """The lines of the ``Highlights:`` list in the CHANGELOG section of ``version``.

    The list runs from the first line after ``Highlights:`` to the first blank line after
    it has started; the section's other entries follow that blank line.
    """

    lines = changelog.splitlines()
    heading = re.compile(rf"^## {re.escape(version)}(\s|$)")
    start = next((i for i, line in enumerate(lines) if heading.match(line)), None)
    if start is None:
        raise ValueError(f"CHANGELOG.md has no section for {version}")
    highlights: list[str] = []
    started = False
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        if not started:
            started = line.strip() == "Highlights:"
            continue
        if not line.strip():
            if highlights:
                break
            continue
        highlights.append(line)
    if not highlights:
        raise ValueError(f"the CHANGELOG section for {version} has no Highlights list")
    return highlights


def generated_section(draft_body: str) -> str:
    """The "What's Changed" part GitHub generated for the draft, or ``""``."""

    index = draft_body.find(GENERATED_MARKER)
    return draft_body[index:].strip() if index >= 0 else ""


def release_body(changelog: str, version: str, tag: str, draft_body: str = "") -> str:
    """The release body: highlights, install command, changelog link, generated notes."""

    wheel = (
        f"https://github.com/{REPOSITORY}/releases/download/{tag}/"
        f"calibrex-{version}-py3-none-any.whl"
    )
    parts = [
        "## Highlights",
        "",
        *changelog_highlights(changelog, version),
        "",
        "Install:",
        "",
        "```bash",
        f'python -m pip install "{wheel}"',
        "```",
        "",
        f"See [CHANGELOG.md](https://github.com/{REPOSITORY}/blob/{tag}/CHANGELOG.md) for details.",
    ]
    generated = generated_section(draft_body)
    if generated:
        parts += ["", "", generated]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="package version, e.g. 0.6.0")
    parser.add_argument("--tag", required=True, help="release tag, e.g. v0.6.0")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--draft-body", type=Path, help="file holding the draft's body")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    draft = args.draft_body.read_text(encoding="utf-8") if args.draft_body else ""
    changelog = args.changelog.read_text(encoding="utf-8")
    args.output.write_text(release_body(changelog, args.version, args.tag, draft), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
