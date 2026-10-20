"""Keep every published measurement-loss development outcome intact and readable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from calibrex.core.validation import validate_file

_ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/screw_weighted_objective"
_CASES = (
    (20261010, "converged", 28, "pass",
     "a3046d0bdb7aa44666dec1f3f20e76eb7bd02d50ea72a1cd196cb85b78e1c62b"),
    (20261011, "max_iterations", 31, "fail",
     "84672c64cfeac1f509f0cd8d001349e067875f4e8210f138d5e923aeff40491c"),
    (20261012, "converged", 23, "pass",
     "f539bf5a6dc67546c82be356bd813e77ece94b2a2468a3c105f3a6187426fd60"),
)


@pytest.mark.parametrize("seed,status,iterations,policy,digest", _CASES)
def test_complete_development_outcome_stays_valid(
    seed: int, status: str, iterations: int, policy: str, digest: str
) -> None:
    path = _ASSETS / f"seed-{seed}.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert validate_file(path, kind="continuous-time-lidar-imu-joint").valid
    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["seed"], data["fit_status"], data["iterations"], data["policy_status"]) == (
        seed, status, iterations, policy
    )
    assert data["provenance"]["seed"] == seed
    assert data["provenance"]["git_commit"] == "63475165"
    assert data["provenance"]["command"][:3] == [
        "calibrex", "trajectory", "recover-lidar-imu-joint"
    ]


def test_entire_declared_cohort_is_retained() -> None:
    assert {path.name for path in _ASSETS.glob("seed-*.json")} == {
        f"seed-{seed}.json" for seed, *_ in _CASES
    }
