"""The failed support assumption and negative spatial ablation remain reproducible."""
import jsonschema
import pytest
from tests.unit.test_kitti_camera_lidar_profile_evidence import ASSETS, _json, _sha, probe, replay
from tests.unit.test_kitti_camera_lidar_spatial_weights import spatial

PUBLIC = ASSETS / "spatial_weights"


def test_both_attempts_and_all_source_bytes_are_published_verbatim() -> None:
    manifest = _json(PUBLIC / "manifest.json")
    jsonschema.validate(manifest, _json(PUBLIC / "manifest.schema.json"))
    names = [f["path"] for f in manifest["files"]]
    assert len(names) == len(set(names)) == 49
    assert set(names) == {p.relative_to(PUBLIC).as_posix() for p in PUBLIC.rglob("*")
                          if p.is_file() and p.name != "manifest.json"}
    for item in manifest["files"]:
        path = PUBLIC / item["path"]
        assert path.resolve().is_relative_to(PUBLIC.resolve())
        assert path.suffix in {".json", ".txt"}
        assert _sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
    assert not manifest["provenance"]["sensor_payload_published"]
    assert manifest["provenance"]["failure_retained"]
    assert manifest["provenance"]["generator_sha256"] == _sha(PUBLIC / "publication-source.py.txt")
    for attempt, prefix in ((1, "50c77d37"), (2, "67ff8c5d")):
        root = PUBLIC / f"attempt_{attempt}"
        batch = _json(root / "report.json")
        jsonschema.validate(batch, _json(root / "report.schema.json"))
        p = batch["provenance"]
        assert p["git_commit"].startswith(prefix) and p["source_committed_before_experiment"]
        assert p["spatial_weights"] == "candidate_fixed_2x2" and p["frame_mass_preserved"]
        assert not p["threshold_selection"] and not p["new_accuracy_claim"]
        assert not p["new_evaluation_started"]
        assert p["generator_sha256"] == _sha(root / "producer-source.py.txt")
        for name, source in p["sources"].items():
            assert source["sha256"] == _sha(root / f"{name}-source.py.txt")


def test_margin_failure_is_retained_and_retry_reuses_all_fourteen_inputs() -> None:
    first, retry = PUBLIC / "attempt_1", PUBLIC / "attempt_2"
    failed = _json(first / "report.json")
    batch = _json(retry / "report.json")
    assert failed["status"] == "failed" and failed["runs"] == []
    assert batch["status"] == "complete_diagnostic_only" and batch["errors"] == {}
    expected = {f"{drive}/{run}" for drive in ("2011_09_26_0005", "2011_09_28_0002")
                for run in replay.RUNS}
    assert set(failed["errors"]) == expected
    assert all("inside the candidate image" in e for e in failed["errors"].values())
    assert {r["report"].removesuffix("/report.json") for r in batch["runs"]} == expected
    assert batch["provenance"]["prior_failed_report_sha256"] == _sha(first / "report.json")
    for record in batch["runs"]:
        path = retry / record["report"]
        assert _sha(path) == record["report_sha256"] and record["input_and_options_parity"]
        old = ASSETS / "profiled_controls_multistart" / record["report"]
        assert _sha(old) == record["unbalanced_report_sha256"]
        before = probe.ProfileReport.model_validate(_json(old))
        bad = probe.ProfileReport.model_validate(_json(first / record["report"]))
        result = probe.ProfileReport.model_validate(_json(path))
        assert bad.status == "failed" and bad.axes == []
        assert result.status == "complete" and result.error is None
        for root, value in ((first, bad), (retry, result)):
            jsonschema.validate(value.model_dump(mode="json"), _json(root / "profile.schema.json"))
            assert value.provenance["generator_sha256"] == _sha(root / "profile-source.py.txt")
            for key in ("snapshot_sha256", "arrays_sha256", "conditional_report_sha256", "options"):
                assert value.provenance[key] == before.provenance[key]
        assert result.provenance["spatial_source_sha256"] == _sha(retry / "spatial-source.py.txt")
        support = result.provenance["tile_support"]
        assert len(support) == result.provenance["prepared_frames"]
        assert any(row["outside_point_count"] > 0 for row in support)
        for row in support:
            jsonschema.validate(row, _json(PUBLIC / "tile.schema.json"))
            parsed = spatial.TileSupport.model_validate(row)
            assert parsed.outside_original_weight == parsed.outside_balanced_weight
            assert all(count >= 0 for count in parsed.point_counts)
            assert sum(parsed.balanced_weight) == pytest.approx(sum(parsed.original_weight))
            occupied = sum(count > 0 for count in parsed.point_counts)
            for count, mass in zip(parsed.point_counts, parsed.balanced_weight, strict=True):
                assert mass == pytest.approx(sum(parsed.original_weight) / occupied if count else 0)


def test_spatial_weights_do_not_restore_pitch_and_lose_useful_yaw_controls() -> None:
    root = PUBLIC / "attempt_2"
    counts: dict[str, list[int]] = {}
    for record in _json(root / "report.json")["runs"]:
        previous = probe.ProfileReport.model_validate(
            _json(ASSETS / "profiled_controls_multistart" / record["report"]),
        )
        balanced = probe.ProfileReport.model_validate(_json(root / record["report"]))
        totals = counts.setdefault(record["drive"], [0, 0])
        totals[0] += sum(a.profiled_control.detected for a in previous.axes)
        totals[1] += sum(a.profiled_control.detected for a in balanced.axes)
    assert counts == {"2011_09_26_0005": [20, 12], "2011_09_28_0002": [0, 0]}
    for drive, run, axis in [("2011_09_28_0002", "inject_y+3", 1),
                              ("2011_09_28_0002", "inject_x+3", 0),
                              ("2011_09_26_0005", "baseline", 2),
                              ("2011_09_26_0005", "inject_z+3", 2)]:
        result = probe.ProfileReport.model_validate(_json(root / drive / run / "report.json"))
        assert not result.axes[axis].profiled_control.detected
