"""Integrity and limits of the fixed spent-data conditional-axis probe batch."""
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest
from tests.unit.test_score_hilti_camera_camera import _load

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/kitti_camera_lidar_v3"
PUBLIC = ASSETS / "conditional_axis_probe"
probe = _load("kitti_camera_lidar_axis_probe")
replay = _load("kitti_camera_lidar_v3_replay")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batch_publication_is_byte_bound_and_schema_valid() -> None:
    manifest = _json(PUBLIC / "manifest.json")
    jsonschema.validate(manifest, _json(PUBLIC / "manifest.schema.json"))
    names = [f["path"] for f in manifest["files"]]
    assert len(names) == len(set(names)) == 21
    assert set(names) == {p.relative_to(PUBLIC).as_posix() for p in PUBLIC.rglob("*")
                          if p.is_file() and p.name != "manifest.json"}
    for item in manifest["files"]:
        path = PUBLIC / item["path"]
        assert path.resolve().is_relative_to(PUBLIC.resolve())
        assert path.suffix in {".json", ".txt"}
        assert _sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
    assert manifest["provenance"]["sensor_payload_published"] is False
    assert manifest["provenance"]["generator_sha256"] == _sha(PUBLIC / "publication-source.py.txt")
    batch = _json(PUBLIC / "report.json")
    jsonschema.validate(batch, _json(PUBLIC / "report.schema.json"))
    assert batch["status"] == "complete_diagnostic_only" and batch["errors"] == {}
    provenance = batch["provenance"]
    assert not provenance["new_accuracy_claim"] and not provenance["new_evaluation_started"]
    assert not provenance["threshold_selection"]
    assert provenance["generator_sha256"] == _sha(PUBLIC / "producer-source.py.txt")
    for source, digest in provenance["frozen_sources_sha256"].items():
        name = source.replace("\\", "/").split("/")[-1]
        capsule = {
            "kitti_camera_lidar_axis_probe.py": PUBLIC / "probe-source.py.txt",
            "kitti_camera_lidar_v3_replay.py": ASSETS / (
                "native_development_20110928_0002/attempt_1/replay-source.py.txt"),
            "camera_lidar_edge.py": ASSETS / (
                "native_development_20110928_0002/attempt_1/estimator-source.py.txt"),
        }[name]
        assert digest == _sha(capsule)


def test_all_fourteen_spent_inputs_and_all_axes_are_retained() -> None:
    batch = _json(PUBLIC / "report.json")
    assert {(r["drive"], r["run"]) for r in batch["runs"]} == {
        (drive, run) for drive in ("2011_09_26_0005", "2011_09_28_0002") for run in replay.RUNS
    }
    for run in batch["runs"]:
        path = PUBLIC / run["report"]
        assert _sha(path) == run["report_sha256"]
        report = _json(path)
        jsonschema.validate(report, _json(PUBLIC / "probe.schema.json"))
        parsed = probe.ProbeReport.model_validate(report)
        assert parsed.status == "complete" and parsed.error is None
        assert [p.axis for p in parsed.probes] == [0, 1, 2]
        parent = (ASSETS / "native_pilot_0005/attempt_2" if run["drive"] == "2011_09_26_0005"
                  else ASSETS / "native_development_20110928_0002/attempt_1")
        snapshot = parent / "capture" / run["run"] / "snapshot.json"
        assert parsed.provenance["snapshot_sha256"] == _sha(snapshot)
        assert parsed.provenance["arrays_sha256"] == _json(snapshot)["arrays_sha256"]
        assert not parsed.provenance["new_accuracy_claim"]
        assert not parsed.provenance["new_evaluation_started"]
        assert parsed.provenance["generator_sha256"] == _sha(PUBLIC / "probe-source.py.txt")
        for p in parsed.probes:
            assert len(p.nuisance) == 4 and p.reference_error_deg is not None
            assert {n.fixed_axis for n in p.nuisance} == {0, 1, 2} - {p.axis}
            assert {n.fixed_offset_deg for n in p.nuisance} == {-2.0, 2.0}


def test_small_conditional_error_does_not_remove_nuisance_sensitivity() -> None:
    batch = _json(PUBLIC / "report.json")
    qualifying = []
    for run in batch["runs"]:
        report = probe.ProbeReport.model_validate(_json(PUBLIC / run["report"]))
        qualifying.extend(p for p in report.probes if p.conditional_control.detected
                          and p.conditional_std_deg <= 0.5 and not p.at_search_bound)
    assert len(qualifying) == 18
    assert max(abs(p.reference_error_deg) for p in qualifying) == pytest.approx(0.15)
    pitch = probe.ProbeReport.model_validate(
        _json(PUBLIC / "2011_09_28_0002/inject_y+3/report.json"),
    ).probes[1]
    assert pitch.correction_deg == pytest.approx(-2.85)
    assert pitch.conditional_control.detected
    assert pitch.conditional_std_deg < 0.05
    assert pitch.nuisance_correction_span_deg == pytest.approx(7.8)
    yaw = probe.ProbeReport.model_validate(
        _json(PUBLIC / "2011_09_28_0002/inject_z+3/report.json"),
    ).probes[2]
    assert yaw.reference_error_deg == pytest.approx(0.05)
    assert not yaw.conditional_control.detected  # Correct angle alone is not observability.
