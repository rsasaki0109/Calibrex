"""Paired multi-start probes remove the retained false control without changing base fits."""
import jsonschema
import pytest
from tests.unit.test_kitti_camera_lidar_profile_evidence import ASSETS, _json, _sha, probe, replay

PUBLIC = ASSETS / "profiled_controls_multistart"
LOCAL = ASSETS / "profiled_controls"


def test_multistart_publication_sources_and_all_bytes_are_bound() -> None:
    manifest = _json(PUBLIC / "manifest.json")
    jsonschema.validate(manifest, _json(PUBLIC / "manifest.schema.json"))
    names = [f["path"] for f in manifest["files"]]
    assert len(names) == len(set(names)) == 24
    assert set(names) == {p.relative_to(PUBLIC).as_posix() for p in PUBLIC.rglob("*")
                          if p.is_file() and p.name != "manifest.json"}
    for item in manifest["files"]:
        path = PUBLIC / item["path"]
        assert path.resolve().is_relative_to(PUBLIC.resolve())
        assert path.suffix in {".json", ".txt"}
        assert _sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
    assert not manifest["provenance"]["sensor_payload_published"]
    assert manifest["provenance"]["generator_sha256"] == _sha(PUBLIC / "publication-source.py.txt")
    batch = _json(PUBLIC / "report.json")
    jsonschema.validate(batch, _json(PUBLIC / "report.schema.json"))
    assert batch["status"] == "complete_diagnostic_only" and batch["errors"] == {}
    provenance = batch["provenance"]
    assert provenance["git_commit"].startswith("24543be7")
    assert provenance["source_committed_before_experiment"]
    assert provenance["nuisance_search"] == "profile_multistart"
    assert not provenance["threshold_selection"]
    assert not provenance["new_accuracy_claim"] and not provenance["new_evaluation_started"]
    assert provenance["generator_sha256"] == _sha(PUBLIC / "producer-source.py.txt")
    for name, source in provenance["sources"].items():
        assert source["sha256"] == _sha(PUBLIC / f"{name}-source.py.txt")


def test_all_fourteen_inputs_preserve_base_and_direct_control_evidence() -> None:
    batch = _json(PUBLIC / "report.json")
    assert {(r["drive"], r["run"]) for r in batch["runs"]} == {
        (drive, run) for drive in ("2011_09_26_0005", "2011_09_28_0002") for run in replay.RUNS
    }
    for run in batch["runs"]:
        path = PUBLIC / run["report"]
        assert _sha(path) == run["report_sha256"]
        previous = LOCAL / run["report"]
        assert _sha(previous) == run["local_report_sha256"] and run["base_and_direct_parity"]
        data = _json(path)
        jsonschema.validate(data, _json(PUBLIC / "profile.schema.json"))
        after = probe.ProfileReport.model_validate(data)
        before = probe.ProfileReport.model_validate(_json(previous))
        assert after.status == "complete" and after.error is None
        assert after.provenance["nuisance_search"] == "profile_multistart"
        assert after.provenance["generator_sha256"] == _sha(PUBLIC / "profile-source.py.txt")
        for key in ("snapshot_sha256", "arrays_sha256", "conditional_report_sha256", "options"):
            assert after.provenance[key] == before.provenance[key]
        assert [a.axis for a in after.axes] == [0, 1, 2]
        for old, new in zip(before.axes, after.axes, strict=True):
            assert replay._same(old.direct_control.model_dump(), new.direct_control.model_dump())
            assert len(new.folds) == 6
            for local, wide in zip(old.folds, new.folds, strict=True):
                for field in ("base_theta", "shifted_theta", "direct_drops"):
                    assert replay._same(getattr(local, field), getattr(wide, field))
                for direct, fitted in zip(wide.shifted_theta, wide.profiled_theta, strict=True):
                    assert direct[new.axis] == fitted[new.axis]
                    assert direct[3:] == fitted[3:] == wide.base_theta[3:]


def test_only_the_known_wrong_roll_control_changes_detection() -> None:
    changed = []
    counts: dict[str, list[int]] = {}
    for run in _json(PUBLIC / "report.json")["runs"]:
        before = probe.ProfileReport.model_validate(_json(LOCAL / run["report"]))
        after = probe.ProfileReport.model_validate(_json(PUBLIC / run["report"]))
        totals = counts.setdefault(run["drive"], [0, 0])
        for old, new in zip(before.axes, after.axes, strict=True):
            totals[0] += old.profiled_control.detected
            totals[1] += new.profiled_control.detected
            if old.profiled_control.detected != new.profiled_control.detected:
                changed.append((run["drive"], run["run"], new.axis))
                assert new.profiled_control.min_t_statistic == pytest.approx(1.7323722509750514)
    assert changed == [("2011_09_28_0002", "inject_x+3", 0)]
    assert counts == {"2011_09_26_0005": [20, 20], "2011_09_28_0002": [1, 0]}
    # Broadening removes a false control, but does not restore the lost pitch detection.
    pitch = probe.ProfileReport.model_validate(
        _json(PUBLIC / "2011_09_28_0002/inject_y+3/report.json"),
    ).axes[1]
    assert not pitch.profiled_control.detected
