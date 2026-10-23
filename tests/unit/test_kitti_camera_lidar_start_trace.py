"""Pinned numerical diagnosis of start-selection failures on spent KITTI inputs."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.core.camera_lidar_edge import CameraLidarEdgeControl, load_camera_lidar_edge
from calibrex.evaluation.camera_lidar_edge import EdgeAlignmentOptions, _replaces_current

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/assets/kitti_camera_lidar_v3/native_development_20110928_0002"
TRACE = BASE / "start_trace"
replay = _load("kitti_camera_lidar_v3_replay")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_trace_receipts_bind_sources_and_preserve_full_evidence() -> None:
    manifest = _json(TRACE / "manifest.json")
    assert manifest["provenance"]["sensor_payload_published"] is False
    assert manifest["provenance"]["generator_sha256"] == _sha(TRACE / "publication-source.py.txt")
    names = [item["path"] for item in manifest["files"]]
    assert len(names) == len(set(names)) == 8
    assert set(names) == {p.name for p in TRACE.iterdir() if p.name != "manifest.json"}
    for item in manifest["files"]:
        path = TRACE / item["path"]
        assert path.resolve().is_relative_to(TRACE.resolve())
        assert path.suffix in {".json", ".txt"}
        assert path.stat().st_size == item["bytes"] and _sha(path) == item["sha256"]
    report = _json(TRACE / "report.json")
    jsonschema.validate(report, _json(TRACE / "report.schema.json"))
    assert report["status"] == "PASS_DIAGNOSTIC_ONLY" and report["error"] is None
    provenance = report["provenance"]
    assert provenance["spent_data_only"] and not provenance["new_accuracy_claim"]
    assert not provenance["new_evaluation_started"]
    for key, name in [("estimator_sha256", "estimator-source.py.txt"),
                      ("replay_sha256", "replay-source.py.txt"),
                      ("parent_report_sha256", "report.json")]:
        assert provenance[key] == _sha(BASE / "attempt_1" / name)
    assert provenance["generator_sha256"] == _sha(TRACE / "producer-source.py.txt")
    assert {(r["run"], r["rule"]) for r in report["runs"]} == {
        (run, rule) for run in ("inject_y+3", "inject_z+3") for rule in ("current", "proposed")
    }
    for run in report["runs"]:
        capture = BASE / "attempt_1/capture" / run["run"] / "snapshot.json"
        assert run["snapshot_sha256"] == _sha(capture)
        assert run["arrays_sha256"] == _json(capture)["arrays_sha256"]
        expected = (BASE / "attempt_1/comparison/2011_09_28_0002"
                    / run["rule"] / (run["run"] + ".json"))
        assert run["expected_artifact_sha256"] == _sha(expected)
        artifact = TRACE / run["artifact"]
        assert _sha(artifact) == run["artifact_sha256"]
        assert run["full_evidence_parity_excluding_provenance"]
        assert replay._same(
            load_camera_lidar_edge(artifact).model_dump(exclude={"provenance"}),
            load_camera_lidar_edge(expected).model_dump(exclude={"provenance"}),
        )


def test_no_scalar_margin_separates_the_two_traced_multi_axis_fits() -> None:
    report = _json(TRACE / "report.json")
    records = {r["run"]: r for r in report["runs"] if r["rule"] == "current"}
    fits = {run: [e for e in record["trace"] if e["event"] == "fit"]
            for run, record in records.items()}
    for run, record in records.items():
        assert len(fits[run]) == 3
        assert [f["observed"] for f in fits[run]] == [[], [1], [0, 1]]
        assert [e["accepted"] for e in record["trace"] if e["event"] == "adoption"] == [False, True]
    pitch, yaw = fits["inject_y+3"][-1], fits["inject_z+3"][-1]
    pitch_min = min(pitch["controls"][axis]["min_t_statistic"] for axis in pitch["observed"])
    yaw_min = min(yaw["controls"][axis]["min_t_statistic"] for axis in yaw["observed"])
    assert pitch_min == pytest.approx(2.621130200574456)
    assert yaw_min == pytest.approx(2.649341460141496)
    assert pitch_min < yaw_min
    for margin in (None, pitch_min, np.nextafter(pitch_min, np.inf), yaw_min, 3.5):
        opts = replace(EdgeAlignmentOptions(), start_scan_min_control_t=margin)
        accepted = [
            _replaces_current(
                [], fit["observed"], np.array(fit["theta"]),
                [CameraLidarEdgeControl.model_validate(c) for c in fit["controls"]], opts,
            )
            for fit in (pitch, yaw)
        ]
        assert accepted != [True, False]
        if margin == 3.5:
            assert accepted == [False, False]
    assert fits["inject_y+3"][1]["theta"][:3] == pytest.approx([0, -2.85, 0])
    # A small yaw correction error is not evidence that yaw is observable.
    assert fits["inject_z+3"][1]["theta"][:3] == pytest.approx([0, 0.05, -2.95])
    assert 2 not in fits["inject_z+3"][1]["observed"]
