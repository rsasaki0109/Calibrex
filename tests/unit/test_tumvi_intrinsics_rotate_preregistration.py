"""Pin the protocol documented before the evaluation run."""

from __future__ import annotations

import re

import yaml
from tools import tumvi_intrinsics_rotate_audit as audit

PLAN = yaml.safe_load(audit.PLAN.read_text())


def test_frozen_agreement_gates_and_missing_case_policy() -> None:
    claim = PLAN["claims"]["tumvi-rotate-agreement"]
    assert claim["gates"] == {"max_focal_error_pct": 5, "max_principal_error_px": 5}
    assert claim["min_passing_cases"] == 5
    assert "contribute zero passes" in PLAN["method"]["inconclusive"]
    assert re.fullmatch(r"[0-9a-f]{64}", PLAN["frozen_source_sha256"])


def test_session_and_reference_pins() -> None:
    assert PLAN["sessions"]["development_spent"] == ["room1"]
    for split, sessions in (("development", audit.DEVELOPMENT), ("evaluation", audit.SESSIONS)):
        assert list(PLAN["sessions"][split]) == list(sessions)
        for session, (size, etag) in sessions.items():
            pin = PLAN["sessions"][split][session]
            assert PLAN["sessions"]["host"] + "/" + pin["bag"] == audit.bag_url(session)
            assert pin["size_bytes"] == size
            assert pin["etag"] == etag
    assert audit.ROOT / PLAN["reference"]["camchain"] == audit.CAMCHAIN
    assert PLAN["reference"]["camchain_sha256"] == audit.CAMCHAIN_SHA256


def test_window_recipe() -> None:
    method = PLAN["method"]
    assert method["cameras"] == list(audit.CAMERAS)
    assert method["duration_s"] == audit.DURATION_S == 10
    assert method["step_s"] == audit.STEP_S == 2
    assert method["min_rotation_span_deg"] == audit.MIN_AXIS_DEG == 20
    assert method["pose_topic"] == audit.POSE_TOPIC
