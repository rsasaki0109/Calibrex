"""The clock development record preserves every outcome and frozen source."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

import jsonschema
import numpy as np
import pytest

from calibrex.core.validation import validate_file

_ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/screw_gyro_clock"
_SEEDS = (20261010, 20261011, 20261012)
_BASE = "c8001467f88eee880c6f21169ba858c572bdb068"
_SOURCE = "3c2998cf84cbd1916a313ead72269c4e62cc5695"
_ARCHIVE_SHA = "5a79db53d3988d122b2639699225be6f3869cf6b8300cb7793312efebb8d4a9d"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def evidence() -> dict[str, bytes]:
    path = _ASSETS / "complete-development.tar.gz"
    assert _sha(path.read_bytes()) == _ARCHIVE_SHA
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        assert len({member.name for member in members}) == len(members)
        for member in members:
            assert member.isfile() and not PurePosixPath(member.name).is_absolute()
            assert ".." not in PurePosixPath(member.name).parts
        files = {member.name: archive.extractfile(member).read() for member in members}
    index = json.loads(files["index.json"])
    assert set(files) == set(index["files"]) | {"index.json"}
    for name, metadata in index["files"].items():
        assert metadata == {"size": len(files[name]), "sha256": _sha(files[name])}
    receipt = json.loads((_ASSETS / "receipt.json").read_bytes())
    jsonschema.validate(receipt, json.loads((_ASSETS / "receipt.schema.json").read_bytes()))
    assert receipt["sha256"] == _ARCHIVE_SHA
    assert receipt["bytes"] == path.stat().st_size
    assert receipt["indexed_files"] == len(index["files"]) == 880
    assert receipt["index_sha256"] == _sha(files["index.json"])
    assert receipt["summary_sha256"] == _sha(files["summary.json"])
    assert receipt["source_commits"] == {"baseline": _BASE, "candidate": _SOURCE}
    assert files["summary.json"] == (_ASSETS / "summary.json").read_bytes()
    assert files["summary.schema.json"] == (_ASSETS / "summary.schema.json").read_bytes()
    jsonschema.validate(json.loads(files["summary.json"]), json.loads(files["summary.schema.json"]))
    assert index["provenance"]["producer_sha256"] == _sha(files["packaging.py.txt"])
    return files


@pytest.mark.parametrize("seed,iterations,status", [
    (20261010, 20, "converged"), (20261011, 32, "max_iterations"), (20261012, 19, "converged"),
])
def test_native_outcomes_remain_complete_including_the_iteration_limit(
    evidence: dict[str, bytes], seed: int, iterations: int, status: str,
) -> None:
    path = _ASSETS / f"seed-{seed}.json"
    assert path.read_bytes() == evidence[f"candidate/seed-{seed}/native.json"]
    assert validate_file(path, kind="continuous-time-lidar-imu-joint").valid
    result = json.loads(path.read_bytes())
    assert (result["seed"], result["iterations"], result["fit_status"]) == (
        seed, iterations, status,
    )
    assert result["provenance"]["git_commit"] == _SOURCE[:8]
    assert result["policy_status"] == "pass"
    assert len(result["controls"]) == 32 and all(row["detected"] for row in result["controls"])


@pytest.mark.parametrize("variant", ["baseline", "candidate"])
def test_both_complete_source_capsules_are_bound(evidence: dict[str, bytes], variant: str) -> None:
    prefix = f"factors/{variant}/"
    manifest = json.loads(evidence[prefix + "manifest.json"])
    sources = {name.removeprefix(prefix): _sha(raw)
               for name, raw in evidence.items() if name.startswith(prefix + "src/")}
    assert sources == manifest and len(sources) == 402
    if variant == "candidate":
        assert manifest == json.loads(evidence["candidate/source-manifest.json"])


def test_all_saved_state_factors_keep_residuals_and_nonclock_bits(
    evidence: dict[str, bytes],
) -> None:
    before, after = (json.loads(evidence[f"factors/{variant}.json"])
                     for variant in ("baseline", "candidate"))
    assert len(before["factors"]) == len(after["factors"]) == 63
    for a, b in zip(before["factors"], after["factors"], strict=True):
        for row in (a, b):
            payload = {key: value for key, value in row.items()
                       if key not in ("nonclock_sha256", "jacobian_clock")}
            raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
            assert row["nonclock_sha256"] == _sha(raw)
        assert a["nonclock_sha256"] == b["nonclock_sha256"]
    parity = json.loads(evidence["factors/result.json"])
    jsonschema.validate(parity, json.loads(evidence["factors/result.schema.json"]))
    assert parity["changed_clock_blocks"] == 63 and parity["new_optimizer_steps"] == 0
    assert parity["saved_states"] == 9
    assert parity["provenance"]["producer_sha256"] == _sha(evidence["factor-comparison.py.txt"])


@pytest.mark.parametrize("seed", _SEEDS)
def test_complete_problem_options_and_fit_schemas_are_retained(
    evidence: dict[str, bytes], seed: int,
) -> None:
    before = json.loads(evidence[f"baseline/seed-{seed}/input.json"])
    after = json.loads(evidence[f"candidate/seed-{seed}/input.json"])
    for value in (before, after):
        jsonschema.validate(value, json.loads(evidence["candidate/input.schema.json"]))
    assert before["problem"] == after["problem"] and before["options"] == after["options"]
    before_input = {key: before[key] for key in ("problem", "options")}
    after_input = {key: after[key] for key in ("problem", "options")}
    assert json.dumps(before_input, sort_keys=True) == json.dumps(after_input, sort_keys=True)
    old_fit = json.loads(evidence[f"baseline/seed-{seed}/fit.json"])
    origin = old_fit.pop("provenance")
    for value in ({"payload": old_fit, "provenance": origin},
                  json.loads(evidence[f"candidate/seed-{seed}/fit.json"])):
        jsonschema.validate(value, json.loads(evidence["candidate/fit.schema.json"]))


@pytest.mark.parametrize("seed", _SEEDS)
def test_final_model_retains_the_reported_gradient(evidence: dict[str, bytes], seed: int) -> None:
    model = json.loads(evidence[f"candidate/seed-{seed}/model.json"])
    fitted = json.loads(evidence[f"candidate/seed-{seed}/fit.json"])["payload"]
    jsonschema.validate(model, json.loads(evidence["candidate/model.schema.json"]))
    normal, gradient = np.asarray(model["normal"]), np.asarray(model["half_gradient"])
    assert normal.shape == (gradient.size, gradient.size) == (58, 58)
    assert np.all(np.isfinite(normal)) and np.all(np.isfinite(gradient))
    observed = float(np.max(np.abs(gradient)))
    assert observed == model["final_gradient_norm"] == fitted["gradient_norm"]


def test_summary_records_each_metric_and_remaining_limits(evidence: dict[str, bytes]) -> None:
    summary = json.loads(evidence["summary.json"])
    assert summary["candidate_iteration_limit_cases"] == [20261011]
    for case in summary["cases"]:
        seed = case["seed"]
        for variant in ("baseline", "candidate"):
            native = json.loads(evidence[f"{variant}/seed-{seed}/native.json"])
            fitted = json.loads(evidence[f"{variant}/seed-{seed}/fit.json"])
            if variant == "candidate":
                fitted = fitted["payload"]
            for key, value in case[variant].items():
                assert value == (fitted[key] if key in fitted else native[key])
        assert case["policies"] == ["pass", "pass"] and case["detected_controls"] == [32, 32]
    assert "32 iterations" in " ".join(summary["attempts"])
    assert "constrained optimizer" in " ".join(summary["limits"])


def test_initial_failures_and_original_producers_are_retained(evidence: dict[str, bytes]) -> None:
    summary = json.loads(evidence["summary.json"])
    checks = {row["name"]: row for row in summary["checks"]}
    assert checks["baseline-tests-v2"]["tests"] == 79
    assert checks["baseline-tests-v2"]["failure"] == 47
    assert checks["new-tests-v2"]["failure"] == 0
    assert checks["targeted-v2"]["tests"] == 405 and checks["targeted-v2"]["failure"] == 0
    protocol = json.loads(evidence["candidate/protocol.json"])
    assert protocol["provenance"]["launcher_sha256"] == _sha(evidence["generator.py.txt"])
    assert protocol["provenance"]["source_commit"] == _SOURCE
