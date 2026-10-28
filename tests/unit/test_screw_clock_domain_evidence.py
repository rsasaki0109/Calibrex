"""Check retained paired development artifacts and their source bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from calibrex.core.validation import validate_file

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/screw_clock_domain"


def test_retained_clock_domain_comparison() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for version in ("baseline", "current"):
        protocol = json.loads((ROOT / version / "protocol.json").read_text())
        assert protocol["seeds"] == [20261010, 1, 2]
        assert protocol["base"] == manifest["base"]
        for seed in protocol["seeds"]:
            path = ROOT / version / f"result-{seed}.json"
            assert validate_file(path).valid
            result = json.loads(path.read_text())
            assert result["policy_status"] == "pass"
            assert len(result["controls"]) == 32
            assert all(control["detected"] for control in result["controls"])
    protocol = json.loads((ROOT / "current/protocol.json").read_text())
    for snapshot in (ROOT / "measured_source").rglob("*.txt"):
        name = str(snapshot.relative_to(ROOT / "measured_source"))[:-4]
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == protocol["source_sha256"][name]
