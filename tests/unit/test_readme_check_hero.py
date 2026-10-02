from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "generate_readme_check_hero", ROOT / "tools" / "generate_readme_check_hero.py"
)
assert _SPEC is not None and _SPEC.loader is not None
hero = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hero)

COMMITTED = ROOT / "docs" / "assets" / "readme-check-hero.svg"


def test_committed_hero_matches_demo_artifacts() -> None:
    assert COMMITTED.read_text(encoding="utf-8") == hero.render_hero(ROOT)


def test_hero_is_deterministic_well_formed_and_bound_to_sources() -> None:
    svg = hero.render_hero(ROOT)
    assert svg == hero.render_hero(ROOT)
    root = ElementTree.fromstring(svg)
    meta = next(e for e in root.iter() if e.tag.endswith("metadata"))
    provenance = json.loads(meta.text or "")
    for rel, digest in provenance["sources"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
    assert "<script" not in svg
    assert len(svg.encode("utf-8")) < 40_000


def test_hero_numbers_come_from_summary() -> None:
    summary = json.loads(
        (ROOT / "docs/assets/calibrex_check_demo/summary.json").read_text(encoding="utf-8")
    )
    yaw = summary["variants"]["yaw1"]["pairs"]["lidar-vehicle"]["axes"]["yaw"]
    svg = hero.render_hero(ROOT)
    assert f"{abs(yaw['candidate_error_deg']):.3g}" in svg
    assert summary["variants"]["yaw1"]["overall_verdict"].upper() in svg


def test_readme_embeds_hero_with_existing_file() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/assets/readme-check-hero.svg" in readme
    for target in re.findall(r'(?:src|href)="(docs/[^"#]+)"', readme):
        assert (ROOT / target).exists(), target
