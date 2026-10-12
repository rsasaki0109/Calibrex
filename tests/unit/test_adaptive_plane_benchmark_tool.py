"""Protect complete-artifact comparisons and provenance guards."""

import json
from copy import deepcopy

import pytest
from tools import benchmark_adaptive_plane_search as benchmark


def test_canonical_preserves_every_numerical_value_status_and_lineage():
    payload = {
        "status": "rejected",
        "estimated_time_offset_sec": -0.0,
        "iterations": [{"accepted": False, "candidate_train_rmse_m": [None, 0.001]}],
        "provenance": {
            "git_commit": "measured",
            "source_sha256": {"data\\recording": "verified"},
        },
        "dataset_path": "data\\recording",
    }
    original = deepcopy(payload)
    canonical = benchmark._canonical(payload)
    normalized = json.loads(canonical)
    assert payload == original
    assert normalized["status"] == "rejected"
    assert normalized["iterations"] == original["iterations"]
    assert normalized["provenance"]["git_commit"] == "measured"
    assert normalized["provenance"]["source_sha256"] == {"data/recording": "verified"}
    assert normalized["dataset_path"] == "data/recording"
    assert '"estimated_time_offset_sec":-0.0' in canonical
    payload["iterations"][0]["accepted"] = True
    assert benchmark._canonical(payload) != canonical


@pytest.mark.parametrize("option", ["--repetitions", "--omp-threads"])
def test_invalid_budget_is_rejected_before_reading_inputs_or_creating_outputs(tmp_path, option):
    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="must be positive"):
        benchmark.main(["--output-dir", str(destination), option, "0"])
    assert not destination.exists()


def test_nonempty_outputs_are_preserved(tmp_path):
    destination = tmp_path / "output"
    destination.mkdir()
    evidence = destination / "existing.json"
    evidence.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        benchmark.main(["--output-dir", str(destination)])
    assert evidence.read_text(encoding="utf-8") == "preserve"


def test_digest_streams_and_supports_official_md5(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"abc")
    assert benchmark._digest(path, "md5") == "900150983cd24fb0d6963f7d28e17f72"
    assert benchmark._digest(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
