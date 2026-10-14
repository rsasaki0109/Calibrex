"""Gates, trial counting and the drive rules of the KITTI camera-lidar scorer."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tests.unit.test_score_hilti_camera_camera import _load

scorer = _load("score_kitti_camera_lidar")
ROOT = Path(__file__).resolve().parents[2]
DRIVES = [f"d{index:02d}" for index in range(20)]
REFERENCE = Rotation.from_euler("xyz", [-90.0, 0.3, -90.5], degrees=True).as_matrix()


def _turned(spec: str | None) -> np.ndarray:
    if spec is None:
        return REFERENCE
    name, degrees = scorer.turn(spec)
    vector = np.radians(degrees) * np.eye(3)[scorer.AXES.index(name)]
    return np.asarray(Rotation.from_rotvec(vector).as_matrix() @ REFERENCE)


def _run(
    spec: str | None = None,
    errors_deg: tuple[float, float, float] = (0.1, -0.1, 0.05),
    judged: tuple[str, ...] = ("roll", "pitch", "yaw"),
    **changes: Any,
) -> Any:
    """A run started at KITTI's rotation turned by ``spec``, ending ``errors_deg`` off it."""

    estimate = Rotation.from_rotvec(np.radians(errors_deg)).as_matrix() @ REFERENCE
    candidate = _turned(spec)
    readings = scorer.axis_errors_deg(candidate, estimate)
    # calibrex check's rule at its 0.5 deg floor: pass within it, fail beyond twice it
    statuses = {
        name: "pass" if abs(readings[i]) <= 0.5 else ("warn" if abs(readings[i]) <= 1.0 else "fail")
        for i, name in enumerate(scorer.AXES)
        if name in judged
    }
    worst = max(statuses.values(), key=("pass", "warn", "fail").index, default="inconclusive")
    verdict = worst
    fields: dict[str, Any] = {
        "verdict": verdict,
        "candidate": candidate,
        "estimate": estimate,
        "judged": statuses,
        **changes,
    }
    return scorer.RunInput(**fields)


def _drive(judged: tuple[str, ...] = ("roll", "pitch", "yaw"), **turned: Any) -> Any:
    runs = {spec: _run(spec, judged=judged) for spec in scorer.INJECTIONS}
    runs.update({spec.replace("_", ":"): run for spec, run in turned.items()})
    return scorer.DriveInput(baseline=_run(judged=judged), turned=runs)


def _all() -> dict[str, Any]:
    return {name: _drive() for name in DRIVES}


def _score(drives: dict[str, Any]) -> Any:
    return scorer.score_drives(drives, list(drives))


def test_a_clean_set_is_supported() -> None:
    result = _score(_all())
    assert result["verdict"] == "supported"
    assert {item["status"] for item in result["requirements"].values()} == {"pass"}
    assert result["trials"] == {
        "3deg": {"counted": 60, "flagged": 60, "judged_right": 0},
        "1deg": {"counted": 60, "flagged": 60, "judged_right": 0},
    }
    assert result["requirements"]["judged-axis-accuracy"]["observed"] == pytest.approx(0.1)
    assert result["requirements"]["turned-judged-axis-accuracy"]["observed"] == pytest.approx(0.1)


@pytest.mark.parametrize(("error", "status"), [(0.75, "pass"), (0.7501, "fail")])
def test_a_judged_baseline_axis_is_gated(error: float, status: str) -> None:
    drives = _all()
    drives["d03"] = scorer.DriveInput(
        baseline=_run(errors_deg=(0.1, -error, 0.05), judged=("roll", "yaw")),
        turned=drives["d03"].turned,
    )
    # a judged axis whose std is 0.3 deg has a 0.9 deg tolerance: the check passes it
    drives["d04"] = scorer.DriveInput(
        baseline=_run(errors_deg=(0.1, -error, 0.05), verdict="pass"),
        turned=drives["d04"].turned,
    )
    result = _score(drives)
    assert result["requirements"]["judged-axis-accuracy"]["status"] == status
    # d03's pitch is not judged: its error is never gated, and its pitch trials do not count
    assert result["scores"]["d03"]["trials"]["y:3"]["counted"] is False
    assert result["verdict"] == ("supported" if status == "pass" else "refuted")


@pytest.mark.parametrize("verdict", ["warn", "fail"])
def test_a_flagged_vendor_calibration_refutes(verdict: str) -> None:
    drives = _all()
    drives["d07"] = scorer.DriveInput(baseline=_run(verdict=verdict), turned=drives["d07"].turned)
    result = _score(drives)
    assert result["requirements"]["vendor-calibration-not-flagged"]["status"] == "fail"
    assert result["verdict"] == "refuted"


def test_an_inconclusive_vendor_calibration_is_not_a_false_alarm() -> None:
    drives = _all()
    drives["d07"] = _drive(judged=())
    result = _score(drives)
    assert result["scores"]["d07"]["state"] == "scored"
    assert result["scores"]["d07"]["verdict"] == "inconclusive"
    assert result["requirements"]["vendor-calibration-not-flagged"]["status"] == "pass"
    assert result["judged_axes"] == 57
    assert not any(trial["counted"] for trial in result["scores"]["d07"]["trials"].values())


def test_coverage_below_the_minimum_is_inconclusive_not_refuted() -> None:
    drives = {name: _drive(judged=("pitch",)) for name in DRIVES}
    result = _score(drives)
    assert result["judged_fraction"] == pytest.approx(1 / 3)
    assert result["requirements"]["judged-axis-coverage"]["status"] == "fail"
    assert result["verdict"] == "inconclusive"


def _missed(spec: str, verdict: str) -> Any:
    """A turned run that does not judge the turned axis, ending with ``verdict``."""

    name, _ = scorer.turn(spec)
    run = _run(spec)
    judged = {key: value for key, value in run.judged.items() if key != name}
    return scorer.RunInput(
        verdict=verdict, candidate=run.candidate, estimate=run.estimate, judged=judged
    )


@pytest.mark.parametrize(("misses", "status"), [(6, "pass"), (7, "fail")])
def test_the_3deg_flagged_rate_is_gated(misses: int, status: str) -> None:
    drives = _all()
    for name in DRIVES[:misses]:
        drives[name] = _drive(z_3=_missed("z:3", "inconclusive"))
    result = _score(drives)
    assert result["requirements"]["turn-3deg-flagged"]["observed"] == misses / 60
    assert result["requirements"]["turn-3deg-flagged"]["status"] == status
    assert result["requirements"]["turn-3deg-never-judged-right"]["status"] == "pass"


def test_a_pass_with_the_turned_axis_unjudged_is_a_miss_not_a_wrong_verdict() -> None:
    drives = _all()
    drives["d11"] = _drive(x_3=_missed("x:3", "pass"))
    result = _score(drives)
    trial = result["scores"]["d11"]["trials"]["x:3"]
    assert (trial["outcome"], trial["verdict"]) == ("unjudged", "pass")
    assert result["requirements"]["turn-3deg-flagged"]["observed"] == 1 / 60
    assert result["requirements"]["turn-3deg-never-judged-right"]["status"] == "pass"
    assert result["verdict"] == "supported"


def test_a_3deg_turn_judged_right_refutes() -> None:
    drives = _all()
    # the estimate stayed at the turned candidate: the check reads 0 on roll and passes it
    drives["d11"] = _drive(x_3=_run("x:3", errors_deg=(3.0, -0.1, 0.05)))
    result = _score(drives)
    trial = result["scores"]["d11"]["trials"]["x:3"]
    assert trial["outcome"] == "judged right"
    assert trial["reading_deg"] == pytest.approx(0.0, abs=1e-3)
    requirements = result["requirements"]
    assert requirements["turn-3deg-never-judged-right"]["status"] == "fail"
    assert requirements["turned-judged-axis-accuracy"]["observed"] == pytest.approx(3.0)
    assert result["verdict"] == "refuted"


@pytest.mark.parametrize(("error", "status"), [(1.5, "pass"), (1.5001, "fail")])
def test_every_axis_judged_in_a_turned_run_is_gated(error: float, status: str) -> None:
    drives = _all()
    # yaw judged 1.5 deg off KITTI in the roll-turned run of a drive whose roll is not judged
    # on its baseline: the trial does not count, the turned run's accuracy still does
    run = _run("x:1", errors_deg=(0.8, -0.1, error))
    drives["d06"] = _drive(judged=("pitch", "yaw"), x_1=run)
    result = _score(drives)
    assert result["scores"]["d06"]["trials"]["x:1"]["counted"] is False
    assert result["requirements"]["turned-judged-axis-accuracy"]["status"] == status


@pytest.mark.parametrize(("misses", "status"), [(15, "pass"), (16, "fail")])
def test_the_1deg_flagged_rate_is_gated(misses: int, status: str) -> None:
    drives = _all()
    for index in range(misses):
        name = DRIVES[index % len(DRIVES)]
        spec = ("x:1", "y:1", "z:1")[index // len(DRIVES)]
        turned = dict(drives[name].turned)
        turned[spec] = _missed(spec, "pass")
        drives[name] = scorer.DriveInput(baseline=drives[name].baseline, turned=turned)
    result = _score(drives)
    assert result["requirements"]["turn-1deg-flagged"]["status"] == status


def test_a_refused_baseline_is_not_usable_and_a_broken_run_fails() -> None:
    drives = _all()
    drives["d00"] = scorer.DriveInput(
        baseline=scorer.RunInput(verdict="inconclusive", solver_status="insufficient_samples"),
        turned=drives["d00"].turned,
    )
    result = _score(drives)
    assert result["scores"]["d00"]["state"] == "inconclusive"
    assert result["verdict"] == "supported"  # 19 drives scored
    turned = dict(drives["d01"].turned)
    turned["y:1"] = scorer.RunInput(error="exit code 1")
    drives["d01"] = scorer.DriveInput(baseline=drives["d01"].baseline, turned=turned)
    drives["d02"] = scorer.DriveInput(
        error="download: size or ETag differs from the pre-registration"
    )
    result = _score(drives)
    assert result["scores"]["d01"]["state"] == "failed"
    assert result["requirements"]["no-failed-drives"]["observed"] == 2.0
    assert result["verdict"] == "refuted"


def test_too_few_scored_drives_are_inconclusive() -> None:
    drives = _all()
    for name in DRIVES[: len(DRIVES) - scorer.SCORED_DRIVES_MIN + 1]:
        drives[name] = scorer.DriveInput(
            baseline=scorer.RunInput(verdict="inconclusive", solver_status="insufficient_samples"),
            turned=drives[name].turned,
        )
    result = _score(drives)
    assert len(result["scored_drives"]) == scorer.SCORED_DRIVES_MIN - 1
    assert result["requirements"]["scored-drives-coverage"]["status"] == "fail"
    assert result["verdict"] == "inconclusive"


def test_a_candidate_that_is_not_the_turned_reference_is_rejected() -> None:
    drives = _all()
    wrong = _run("y:3")
    drives["d05"] = _drive(
        **{
            "y_3": scorer.RunInput(
                verdict="fail",
                candidate=_turned("x:3"),
                estimate=wrong.estimate,
                judged=wrong.judged,
            )
        }
    )
    with pytest.raises(ValueError, match="turned candidate"):
        _score(drives)


COMMITTED = ROOT / "docs/assets/kitti_camera_lidar/development/runs/0005"


def _committed_reference() -> np.ndarray:
    payload = json.loads((COMMITTED / "baseline.json").read_text(encoding="utf-8"))
    (record,) = [item for item in payload["pairs"] if item["pair"] == "camera-lidar"]
    quat = record["candidate_transform"]["rotation_quat_xyzw"]
    return np.asarray(Rotation.from_quat(quat).as_matrix())


def test_a_committed_drive_is_read_with_its_own_verdicts() -> None:
    drive = scorer.load_drive(COMMITTED, _committed_reference())
    score = scorer.score_drive(drive)
    assert score.state == "scored" and score.verdict == "pass"
    assert all(axis["judged"] for axis in score.axes.values())
    assert max(abs(axis["error_deg"]) for axis in score.axes.values()) < 0.15
    assert {spec: trial["outcome"] for spec, trial in score.trials.items()} == dict.fromkeys(
        scorer.INJECTIONS, "flagged"
    )
    assert score.trials["x:3"]["reading_deg"] == pytest.approx(3.0, abs=0.01)
    assert score.trials["z:1"]["turned_axis_status"] == "warn"


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "0005"
    shutil.copytree(COMMITTED, target)
    return target


def test_a_drive_converted_with_another_time_model_is_rejected(tmp_path: Path) -> None:
    drive = _copy(tmp_path)
    conversion = json.loads((drive / "conversion.json").read_text(encoding="utf-8"))
    conversion["velodyne_time"] = "none"
    (drive / "conversion.json").write_text(json.dumps(conversion), encoding="utf-8")
    with pytest.raises(ValueError, match="velodyne_time"):
        scorer.load_drive(drive, _committed_reference())


def test_an_artifact_from_another_recipe_is_rejected(tmp_path: Path) -> None:
    drive = _copy(tmp_path)
    (evidence,) = (drive / "inject_y+3_evidence").glob("*.yaml")
    payload = yaml.safe_load(evidence.read_text(encoding="utf-8"))
    payload["options"]["edge_background_sigma_px"] = None
    evidence.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="edge_background_sigma_px"):
        scorer.load_drive(drive, _committed_reference())


def test_a_baseline_candidate_off_kitti_is_rejected() -> None:
    turned = Rotation.from_rotvec(np.radians([0.0, 0.01, 0.0])).as_matrix()
    with pytest.raises(ValueError, match="off KITTI"):
        scorer.load_drive(COMMITTED, turned @ _committed_reference())


def test_a_truncated_artifact_fails_its_drive(tmp_path: Path) -> None:
    drive = _copy(tmp_path)
    text = (drive / "inject_z+1.json").read_text(encoding="utf-8")
    (drive / "inject_z+1.json").write_text(text[: len(text) // 2], encoding="utf-8")
    score = scorer.score_drive(scorer.load_drive(drive, _committed_reference()))
    assert score.state == "failed" and score.reason.startswith("z:1: artifact unreadable")


def test_a_drive_that_did_not_run_fails(tmp_path: Path) -> None:
    (tmp_path / "0011").mkdir()
    (tmp_path / "0011" / "download.json").write_text(
        json.dumps({"ok": False, "reason": "size or ETag differs from the pre-registration"}),
        encoding="utf-8",
    )
    failed = scorer.score_drive(scorer.load_drive(tmp_path / "0011", REFERENCE))
    assert failed.state == "failed" and "ETag" in failed.reason
    missing = scorer.score_drive(scorer.load_drive(tmp_path / "0013", REFERENCE))
    assert missing.state == "failed" and "manifest" in missing.reason


def test_kitti_rotation_is_the_rectification_times_velo_to_cam(tmp_path: Path) -> None:
    rect = Rotation.from_euler("xyz", [0.4, -0.2, 0.1], degrees=True).as_matrix()
    rotation = Rotation.from_euler("xyz", [-90.0, 0.5, -90.0], degrees=True).as_matrix()
    numbers = " ".join(f"{value:.12e}" for value in rect.ravel())
    (tmp_path / "calib_cam_to_cam.txt").write_text(f"R_rect_00: {numbers}\n", encoding="utf-8")
    numbers = " ".join(f"{value:.12e}" for value in rotation.ravel())
    (tmp_path / "calib_velo_to_cam.txt").write_text(f"R: {numbers}\nT: 0 0 0\n", encoding="utf-8")
    np.testing.assert_allclose(scorer.kitti_rotation(tmp_path), rect @ rotation, atol=1e-12)
