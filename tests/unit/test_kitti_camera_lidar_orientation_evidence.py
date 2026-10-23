"""Keep the preselected negative orientation comparison and scalar-path parity explicit."""
import jsonschema
from tests.unit.test_kitti_camera_lidar_profile_evidence import ASSETS, _json, _sha, probe, replay

PUBLIC = ASSETS / "orientation_proxy"


def test_orientation_publication_is_byte_bound_and_preselected() -> None:
    manifest = _json(PUBLIC / "manifest.json")
    jsonschema.validate(manifest, _json(PUBLIC / "manifest.schema.json"))
    names = [f["path"] for f in manifest["files"]]
    assert len(names) == len(set(names)) == 22
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
    p = batch["provenance"]
    assert p["git_commit"].startswith("c237f78d") and p["source_committed_before_experiment"]
    assert p["expected_inputs"] == 6 and p["expected_axis_controls"] == 18
    assert p["execution"]["workers"] == 2 and p["execution"]["blas_threads_per_worker"] == 1
    assert not p["threshold_selection"] and not p["new_accuracy_claim"]
    assert not p["new_evaluation_started"]
    assert p["generator_sha256"] == _sha(PUBLIC / "producer-source.py.txt")
    for name, source in p["sources"].items():
        assert source["sha256"] == _sha(PUBLIC / f"{name}-source.py.txt")


def test_all_six_inputs_match_scalar_inputs_but_lose_positive_profiled_controls() -> None:
    batch = _json(PUBLIC / "report.json")
    assert {(r["drive"], r["run"]) for r in batch["runs"]} == {
        (drive, run) for drive in ("2011_09_26_0005", "2011_09_28_0002")
        for run in ("baseline", "inject_y+3", "inject_x+3")
    }
    counts: dict[str, list[int]] = {}
    for record in batch["runs"]:
        path = PUBLIC / record["report"]
        assert _sha(path) == record["report_sha256"] and record["input_and_options_parity"]
        old = ASSETS / "profiled_controls_multistart" / record["report"]
        assert _sha(old) == record["scalar_report_sha256"]
        data = _json(path)
        jsonschema.validate(data, _json(PUBLIC / "profile.schema.json"))
        result = probe.ProfileReport.model_validate(data)
        previous = probe.ProfileReport.model_validate(_json(old))
        assert result.status == "complete" and result.error is None
        p = result.provenance
        assert p["orientation_model"] == "projected_azimuth_gradient"
        assert p["spatial_weights"] == "original"
        assert p["orientation_source_sha256"] == _sha(PUBLIC / "orientation-source.py.txt")
        assert p["generator_sha256"] == _sha(PUBLIC / "profile-source.py.txt")
        for key in ("snapshot_sha256", "arrays_sha256", "conditional_report_sha256", "options"):
            assert p[key] == previous.provenance[key]
        assert [a.axis for a in result.axes] == [0, 1, 2]
        assert all(len(a.folds) == 6 for a in result.axes)
        totals = counts.setdefault(record["drive"], [0, 0])
        totals[0] += sum(a.profiled_control.detected for a in previous.axes)
        totals[1] += sum(a.profiled_control.detected for a in result.axes)
    assert counts == {"2011_09_26_0005": [8, 0], "2011_09_28_0002": [0, 0]}


def test_default_scalar_mode_retains_full_previous_numerical_evidence() -> None:
    root = PUBLIC / "default_confirmation"
    receipt = _json(root / "confirmation.json")
    jsonschema.validate(receipt, _json(root / "confirmation.schema.json"))
    assert receipt["status"] == "pass" and receipt["error"] is None
    p = receipt["provenance"]
    assert p["generator_sha256"] == _sha(root / "producer-source.py.txt")
    assert p["profile_source_sha256"] == _sha(PUBLIC / "profile-source.py.txt")
    assert p["actual_report_sha256"] == _sha(root / "report.json")
    assert p["numerical_parity_atol"] == 1e-9
    assert not p["new_accuracy_claim"] and not p["new_evaluation_started"]
    expected = ASSETS / "profiled_controls_multistart/2011_09_26_0005/baseline/report.json"
    assert p["expected_report_sha256"] == _sha(expected)
    data = _json(root / "report.json")
    jsonschema.validate(data, _json(root / "report.schema.json"))
    result = probe.ProfileReport.model_validate(data)
    previous = probe.ProfileReport.model_validate(_json(expected))
    assert result.provenance["orientation_model"] == "none"
    assert result.provenance["spatial_weights"] == "original"
    assert replay._same(result.model_dump(exclude={"provenance"}),
                        previous.model_dump(exclude={"provenance"}))
