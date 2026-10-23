"""Byte-bound spent-data controls retain both lost detections and false confidence."""
import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.test_score_hilti_camera_camera import _load

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/kitti_camera_lidar_v3"
PUBLIC = ASSETS / "profiled_controls"
probe = _load("kitti_camera_lidar_profile_probe")
replay = _load("kitti_camera_lidar_v3_replay")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_published_bytes_schemas_and_frozen_sources_are_bound() -> None:
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
    p = batch["provenance"]
    assert p["git_commit"].startswith("7a6ea534") and p["source_committed_before_experiment"]
    assert not p["new_accuracy_claim"] and not p["new_evaluation_started"]
    assert not p["threshold_selection"]
    assert p["generator_sha256"] == _sha(PUBLIC / "producer-source.py.txt")
    for name, source in p["sources"].items():
        assert source["sha256"] == _sha(PUBLIC / f"{name}-source.py.txt")


def test_fourteen_inputs_keep_all_fold_poses_and_previous_input_bindings() -> None:
    batch = _json(PUBLIC / "report.json")
    assert {(r["drive"], r["run"]) for r in batch["runs"]} == {
        (drive, run) for drive in ("2011_09_26_0005", "2011_09_28_0002") for run in replay.RUNS
    }
    for run in batch["runs"]:
        path = PUBLIC / run["report"]
        assert _sha(path) == run["report_sha256"]
        data = _json(path)
        jsonschema.validate(data, _json(PUBLIC / "profile.schema.json"))
        result = probe.ProfileReport.model_validate(data)
        assert result.status == "complete" and result.error is None
        assert [a.axis for a in result.axes] == [0, 1, 2]
        old = ASSETS / "conditional_axis_probe" / run["report"]
        assert result.provenance["conditional_report_sha256"] == _sha(old)
        previous = _json(old)["provenance"]
        for key in ("snapshot_sha256", "arrays_sha256", "drive", "run"):
            assert result.provenance[key] == previous[key]
        assert not result.provenance["new_accuracy_claim"]
        assert not result.provenance["new_evaluation_started"]
        assert result.provenance["generator_sha256"] == _sha(PUBLIC / "profile-source.py.txt")
        for axis in result.axes:
            assert len(axis.folds) == 6
            for fold in axis.folds:
                assert len(fold.shifted_theta) == len(fold.profiled_theta) == 2
                for sign, shifted, profiled in zip(
                    (-1, 1), fold.shifted_theta, fold.profiled_theta, strict=True,
                ):
                    assert shifted[axis.axis] == pytest.approx(
                        fold.base_theta[axis.axis] + sign * 2.0,
                    )
                    assert shifted[axis.axis] == profiled[axis.axis]
                    assert shifted[3:] == profiled[3:] == fold.base_theta[3:]


def test_profile_controls_lose_pitch_but_still_pass_a_wrong_roll_basin() -> None:
    batch = _json(PUBLIC / "report.json")
    axes = [axis for r in batch["runs"]
            for axis in probe.ProfileReport.model_validate(_json(PUBLIC / r["report"])).axes]
    assert sum(a.direct_control.detected for a in axes) == 25
    assert sum(a.profiled_control.detected for a in axes) == 21
    pitch = probe.ProfileReport.model_validate(
        _json(PUBLIC / "2011_09_28_0002/inject_y+3/report.json"),
    ).axes[1]
    assert pitch.direct_control.detected and not pitch.profiled_control.detected
    assert pitch.profiled_control.min_t_statistic == pytest.approx(0.391, abs=0.001)
    roll = probe.ProfileReport.model_validate(
        _json(PUBLIC / "2011_09_28_0002/inject_x+3/report.json"),
    ).axes[0]
    assert roll.direct_control.detected and roll.profiled_control.detected
    injected = Rotation.from_rotvec(np.radians([3.0, 0, 0]))
    # These are saved fold-fit reference errors, not a final calibration verdict.
    errors = [np.degrees((Rotation.from_rotvec(np.radians(f.base_theta[:3]))
                         * injected).as_rotvec())[0] for f in roll.folds]
    assert max(abs(e) for e in errors) == pytest.approx(1.5, abs=0.01)
