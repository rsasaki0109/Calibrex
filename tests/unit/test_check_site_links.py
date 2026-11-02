"""Relative links in the built docs site, as ``tools/check_site_links.py`` resolves them."""

from __future__ import annotations

from pathlib import Path

from tools import check_site_links


def _page(site: Path, relative: str, body: str) -> None:
    path = site / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")


def test_a_raw_html_path_written_for_github_is_broken_on_the_site(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "x.gif").write_bytes(b"GIF89a")
    _page(tmp_path, "index.html", '<img src="assets/x.gif">')
    _page(
        tmp_path,
        "tutorials/workflow/index.html",
        '<img src="../assets/x.gif"> <img src="../../assets/x.gif">'
        ' <a href="../calibrex_drift/#control">drift</a>'
        ' <code>&lt;img src="../assets/missing.gif"&gt;</code>',
    )
    _page(tmp_path, "tutorials/calibrex_drift/index.html", "")

    assert check_site_links.broken_links(tmp_path) == {
        "tutorials/workflow/index.html": ["../assets/x.gif"]
    }


def test_external_fragment_and_base_path_urls(tmp_path: Path) -> None:
    _page(
        tmp_path,
        "404.html",
        '<a href="https://example.org/a.md">x</a> <a href="#top">top</a>'
        ' <a href="/Calibrex/">home</a> <a href="/Calibrex/missing/">gone</a>'
        ' <a href="/elsewhere/">other site</a>',
    )
    _page(tmp_path, "index.html", "")

    assert check_site_links.broken_links(tmp_path, "/Calibrex") == {
        "404.html": ["/Calibrex/missing/"]
    }
    assert check_site_links.main([str(tmp_path), "--base-path", "/Calibrex/"]) == 1
