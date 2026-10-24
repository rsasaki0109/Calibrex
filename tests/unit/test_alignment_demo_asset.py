"""The browser sample must remain a reproducible, truthful export of committed data."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_alignment_demo import build  # noqa: E402


def test_display_asset_matches_its_real_sources_and_generator() -> None:
    data = json.loads((ROOT / "docs/app/samples/alignment_demo.json").read_text())
    assert data == build()
    for source in data["provenance"]["sources"]:
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    assert data["provenance"]["generator_sha256"] == hashlib.sha256(
        (TOOLS / "build_alignment_demo.py").read_bytes()
    ).hexdigest()
    assert len(data["before"]) == len(data["after"])
    assert data["dataset"]["license"] == "CC BY-NC-SA 4.0"
