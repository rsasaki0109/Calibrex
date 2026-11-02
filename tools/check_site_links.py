#!/usr/bin/env python3
"""Fail when a page of the built docs site links to a file that is not in the site.

``mkdocs build --strict`` validates Markdown links, but not URLs in raw HTML: an
``<img src="../assets/x.gif">`` that works on GitHub, next to the Markdown file, points
one level too shallow on the site, where ``docs/tutorials/workflow.md`` becomes
``tutorials/workflow/index.html``. This walks every built HTML page and resolves each
relative ``src`` and ``href`` against it. URLs with a scheme, fragments, and absolute
paths outside ``--base-path`` (the site's path on its host) are not checked.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

URL_ATTRIBUTE = re.compile(r'\s(?:src|href)="([^"]+)"')


def target_path(site: Path, page: Path, url: str, base_path: str) -> Path | None:
    """The file ``url`` on ``page`` refers to, or ``None`` when it is not checked."""

    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path:
        return None
    path = unquote(parts.path)
    if path.startswith("/"):
        if not path.startswith(base_path):
            return None
        target = site / path[len(base_path) :]
    else:
        target = page.parent / path
    if path.endswith("/") or target.is_dir():
        target = target / "index.html"
    return target


def broken_links(site: Path, base_path: str = "/") -> dict[str, list[str]]:
    """``{page: [url, ...]}`` for every relative URL whose file is missing."""

    base = base_path if base_path.endswith("/") else base_path + "/"
    broken: dict[str, list[str]] = {}
    for page in sorted(site.rglob("*.html")):
        text = page.read_text(encoding="utf-8", errors="replace")
        for url in URL_ATTRIBUTE.findall(text):
            target = target_path(site, page, url, base)
            if target is not None and not target.exists():
                broken.setdefault(page.relative_to(site).as_posix(), []).append(url)
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site", type=Path, help="the built site directory")
    parser.add_argument(
        "--base-path", default="/", help="the site's path on its host, e.g. /Calibrex/"
    )
    args = parser.parse_args(argv)
    broken = broken_links(args.site, args.base_path)
    for page, urls in broken.items():
        for url in urls:
            print(f"{page}: {url}")
    count = sum(len(urls) for urls in broken.values())
    print("OK" if not count else f"{count} broken link(s)")
    return 1 if count else 0


if __name__ == "__main__":
    raise SystemExit(main())
