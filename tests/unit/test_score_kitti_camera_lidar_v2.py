"""Gates, trial counting, the drive rules and the pins of the KITTI camera-lidar v2 scorer."""

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tests.unit.test_score_hilti_camera_camera import _load

scorer = _load("score_kitti_camera_lidar_v2")
ROOT = Path(__file__).resolve().parents[2]
DAYS = ("2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03")
DRIVES = [f"{DAYS[index % 4]}_{index:04d}" for index in range(20)]
REFERENCE = Rotation.from_euler("xyz", [-90.0, 0.3, -90.5], degrees=True).as_matrix()
TRIALS = 3 * len(DRIVES)  # trials of one size when every axis is judged


def _allowed(maximum: float) -> int:
    """The most misses of ``TRIALS`` the rate gate ``maximum`` allows."""

    return math.floor(maximum * TRIALS + 1e-9)


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
    verdict = max(statuses.values(), key=("pass", "warn", "fail").index, default="inconclusive")
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


def test_the_recipe_is_the_v2_estimator() -> None:
    assert scorer.RECIPE["min_control_t"] == 2.5
    assert scorer.RECIPE["start_scan_min_observed"] == 2
    assert scorer.MAX_FRAMES == 600 and scorer.VELODYNE_TIME == "azimuth"


def test_a_clean_set_is_supported() -> None:
    result = _score(_all())
    assert result["verdict"] == "supported"
    assert {item["status"] for item in result["requirements"].values()} == {"pass"}
    assert result["trials"] == {
        "3deg": {"counted": TRIALS, "flagged": TRIALS, "judged_right": 0},
        "1deg": {"counted": TRIALS, "flagged": TRIALS, "judged_right": 0},
    }
    assert result["requirements"]["judged-axis-accuracy"]["observed"] == pytest.approx(0.1)


@pytest.mark.parametrize("over", [False, True])
def test_a_judged_baseline_axis_is_gated(over: bool) -> None:
    error = scorer.AXIS_ERROR_MAX_DEG + (1e-4 if over else 0.0)
    drives = _all()
    drives[DRIVES[3]] = scorer.DriveInput(
        baseline=_run(errors_deg=(0.1, -error, 0.05), judged=("roll", "yaw")),
        turned=drives[DRIVES[3]].turned,
    )
    # a judged axis with a wide reported std has a wide tolerance: the check passes it
    drives[DRIVES[4]] = scorer.DriveInput(
        baseline=_run(errors_deg=(0.1, -error, 0.05), verdict="pass"),
        turned=drives[DRIVES[4]].turned,
    )
    result = _score(drives)
    status = "fail" if over else "pass"
    assert result["requirements"]["judged-axis-accuracy"]["status"] == status
    # the unjudged pitch of DRIVES[3] is never gated, and its pitch trials do not count
    assert result["scores"][DRIVES[3]]["trials"]["y:3"]["counted"] is False
    assert result["verdict"] == ("refuted" if over else "supported")


@pytest.mark.parametrize("over", [False, True])
def test_a_judged_turned_axis_is_gated(over: bool) -> None:
    error = scorer.TURNED_AXIS_ERROR_MAX_DEG + (1e-4 if over else 0.0)
    drives = _all()
    drives[DRIVES[2]] = _drive(y_1=_run("y:1", errors_deg=(error, 0.2, 0.05)))
    result = _score(drives)
    status = "fail" if over else "pass"
    assert result["requirements"]["turned-judged-axis-accuracy"]["status"] == status


@pytest.mark.parametrize("verdict", ["warn", "fail"])
def test_a_flagged_vendor_calibration_refutes(verdict: str) -> None:
    drives = _all()
    drives[DRIVES[7]] = scorer.DriveInput(
        baseline=_run(verdict=verdict), turned=drives[DRIVES[7]].turned
    )
    result = _score(drives)
    assert result["requirements"]["vendor-calibration-not-flagged"]["status"] == "fail"
    assert result["verdict"] == "refuted"


def test_an_inconclusive_vendor_calibration_is_not_a_false_alarm() -> None:
    drives = _all()
    drives[DRIVES[7]] = _drive(judged=())
    result = _score(drives)
    assert result["scores"][DRIVES[7]]["state"] == "scored"
    assert result["scores"][DRIVES[7]]["verdict"] == "inconclusive"
    assert result["requirements"]["vendor-calibration-not-flagged"]["status"] == "pass"
    assert result["judged_axes"] == TRIALS - 3
    assert not any(trial["counted"] for trial in result["scores"][DRIVES[7]]["trials"].values())


def test_coverage_below_the_minimum_is_inconclusive_not_refuted() -> None:
    drives = {name: _drive(judged=("pitch",)) for name in DRIVES}
    result = _score(drives)
    assert result["judged_fraction"] == pytest.approx(1 / 3)
    assert scorer.JUDGED_FRACTION_MIN > 1 / 3
    assert result["requirements"]["judged-axis-coverage"]["status"] == "fail"
    assert result["verdict"] == "inconclusive"


def _missed(spec: str, verdict: str) -> Any:
    """A turned run that does not flag the turned axis, ending with ``verdict``."""

    name, _ = scorer.turn(spec)
    run = _run(spec)
    judged = {key: value for key, value in run.judged.items() if key != name}
    return scorer.RunInput(
        verdict=verdict, candidate=run.candidate, estimate=run.estimate, judged=judged
    )


def _with_misses(misses: int, degrees: str, verdict: str) -> dict[str, Any]:
    drives = _all()
    for index in range(misses):
        name = DRIVES[index % len(DRIVES)]
        spec = f"{'xyz'[index // len(DRIVES)]}:{degrees}"
        turned = dict(drives[name].turned)
        turned[spec] = _missed(spec, verdict)
        drives[name] = scorer.DriveInput(baseline=drives[name].baseline, turned=turned)
    return drives


@pytest.mark.parametrize("extra", [0, 1])
def test_the_3deg_flagged_rate_is_gated(extra: int) -> None:
    misses = _allowed(scorer.MISSED_3DEG_MAX) + extra
    result = _score(_with_misses(misses, "3", "inconclusive"))
    gate = result["requirements"]["turn-3deg-flagged"]
    assert gate["observed"] == pytest.approx(misses / TRIALS)
    assert gate["status"] == ("fail" if extra else "pass")
    assert result["requirements"]["turn-3deg-never-judged-right"]["status"] == "pass"


def test_a_3deg_turn_judged_right_refutes() -> None:
    drives = _all()
    drives[DRIVES[11]] = _drive(x_3=_missed("x:3", "pass"))
    drives[DRIVES[11]].turned["x:3"].judged["roll"] = "pass"
    result = _score(drives)
    assert result["scores"][DRIVES[11]]["trials"]["x:3"]["outcome"] == "judged right"
    assert result["requirements"]["turn-3deg-never-judged-right"]["status"] == "fail"
    assert result["verdict"] == "refuted"


@pytest.mark.parametrize("extra", [0, 1])
def test_the_1deg_flagged_rate_is_gated(extra: int) -> None:
    misses = _allowed(scorer.MISSED_1DEG_MAX) + extra
    result = _score(_with_misses(misses, "1", "pass"))
    assert result["requirements"]["turn-1deg-flagged"]["status"] == ("fail" if extra else "pass")


def test_a_refused_baseline_is_not_usable_and_a_broken_run_fails() -> None:
    drives = _all()
    drives[DRIVES[0]] = scorer.DriveInput(
        baseline=scorer.RunInput(verdict="inconclusive", solver_status="insufficient_samples"),
        turned=drives[DRIVES[0]].turned,
    )
    result = _score(drives)
    assert result["scores"][DRIVES[0]]["state"] == "inconclusive"
    assert result["verdict"] == "supported"
    turned = dict(drives[DRIVES[1]].turned)
    turned["y:1"] = scorer.RunInput(error="exit code 1")
    drives[DRIVES[1]] = scorer.DriveInput(baseline=drives[DRIVES[1]].baseline, turned=turned)
    drives[DRIVES[2]] = scorer.DriveInput(error="fetch: curl exit 22")
    result = _score(drives)
    assert result["scores"][DRIVES[1]]["state"] == "failed"
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
    turned = scorer.RunInput(
        verdict="fail", candidate=_turned("x:3"), estimate=wrong.estimate, judged=wrong.judged
    )
    drives[DRIVES[5]] = _drive(y_3=turned)
    with pytest.raises(ValueError, match="turned candidate"):
        _score(drives)


COMMITTED = ROOT / "docs/assets/kitti_camera_lidar_v2/development/runs/2011_09_26_0005"


def _committed_reference() -> np.ndarray:
    payload = json.loads((COMMITTED / "baseline.json").read_text(encoding="utf-8"))
    (record,) = [item for item in payload["pairs"] if item["pair"] == "camera-lidar"]
    quat = record["candidate_transform"]["rotation_quat_xyzw"]
    return np.asarray(Rotation.from_quat(quat).as_matrix())


def _download() -> dict[str, Any]:
    return dict(json.loads((COMMITTED / "download.json").read_text(encoding="utf-8")))


def _pin() -> dict[str, Any]:
    download = _download()
    return {"size_bytes": download["size_bytes"], "etag": download["etag"]}


def test_a_committed_drive_is_read_with_its_own_verdicts_and_fetch() -> None:
    drive = scorer.load_drive(COMMITTED, _committed_reference(), _pin())
    score = scorer.score_drive(drive)
    assert score.state == "scored" and score.verdict == "pass"
    assert all(axis["judged"] for axis in score.axes.values())
    assert max(abs(axis["error_deg"]) for axis in score.axes.values()) < 0.15
    assert {spec: trial["outcome"] for spec, trial in score.trials.items()} == dict.fromkeys(
        scorer.INJECTIONS, "flagged"
    )
    assert score.fetch["etag"] == _pin()["etag"]
    assert score.fetch["frames"] == {"image_02": 154, "velodyne_points": 154, "oxts": 154}


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "2011_09_26_0005"
    shutil.copytree(COMMITTED, target)
    return target


def _edit_download(drive: Path, **changes: Any) -> None:
    download = _download()
    download.update(changes)
    (drive / "download.json").write_text(json.dumps(download), encoding="utf-8")


def test_an_archive_other_than_the_pinned_one_fails_its_drive(tmp_path: Path) -> None:
    drive = _copy(tmp_path)
    _edit_download(drive, etag="0" * 32 + "-78")
    score = scorer.score_drive(scorer.load_drive(drive, _committed_reference(), _pin()))
    assert score.state == "failed" and "pinned size and ETag" in score.reason
    # development runs have no pins
    assert scorer.score_drive(scorer.load_drive(drive, _committed_reference())).state == "scored"


def test_an_incomplete_fetch_fails_its_drive(tmp_path: Path) -> None:
    drive = _copy(tmp_path)
    _edit_download(drive, frames={"image_02": 154, "velodyne_points": 153, "oxts": 154})
    score = scorer.score_drive(scorer.load_drive(drive, _committed_reference(), _pin()))
    assert score.state == "failed" and score.reason == "fetch incomplete: velodyne_points"


@pytest.mark.parametrize("change", [{"max_frames": 800}, {"crc_verified": False}])
def test_a_drive_not_fetched_as_pre_registered_is_rejected(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    drive = _copy(tmp_path)
    _edit_download(drive, **change)
    with pytest.raises(ValueError, match="not fetched as pre-registered"):
        scorer.load_drive(drive, _committed_reference())


def test_a_drive_converted_with_another_time_model_is_rejected(tmp_path: Path) -> None:
    drive = _copy(tmp_path)
    conversion = json.loads((drive / "conversion.json").read_text(encoding="utf-8"))
    conversion["velodyne_time"] = "none"
    (drive / "conversion.json").write_text(json.dumps(conversion), encoding="utf-8")
    with pytest.raises(ValueError, match="velodyne_time"):
        scorer.load_drive(drive, _committed_reference())


@pytest.mark.parametrize(("key", "value"), [("min_control_t", 2.0), ("start_scan_min_observed", 1)])
def test_an_artifact_from_another_recipe_is_rejected(tmp_path: Path, key: str, value: Any) -> None:
    drive = _copy(tmp_path)
    (evidence,) = (drive / "inject_y+3_evidence").glob("*.yaml")
    payload = yaml.safe_load(evidence.read_text(encoding="utf-8"))
    payload["options"][key] = value
    evidence.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        scorer.load_drive(drive, _committed_reference())


def test_a_baseline_candidate_off_the_days_calibration_is_rejected() -> None:
    turned = Rotation.from_rotvec(np.radians([0.0, 0.01, 0.0])).as_matrix()
    with pytest.raises(ValueError, match="off KITTI"):
        scorer.load_drive(COMMITTED, turned @ _committed_reference())


def test_a_truncated_artifact_fails_its_drive(tmp_path: Path) -> None:
    drive = _copy(tmp_path)
    text = (drive / "inject_z+1.json").read_text(encoding="utf-8")
    (drive / "inject_z+1.json").write_text(text[: len(text) // 2], encoding="utf-8")
    score = scorer.score_drive(scorer.load_drive(drive, _committed_reference()))
    assert score.state == "failed" and score.reason.startswith("z:1: artifact unreadable")


def test_a_drive_that_was_not_fetched_or_run_fails(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "download.json").write_text(
        json.dumps({"ok": False, "reason": "size 1 and ETag x differ from the pin 2 y"}),
        encoding="utf-8",
    )
    failed = scorer.score_drive(scorer.load_drive(tmp_path / "a", REFERENCE))
    assert failed.state == "failed" and "differ from the pin" in failed.reason
    missing = scorer.score_drive(scorer.load_drive(tmp_path / "b", REFERENCE))
    assert missing.state == "failed" and "fetch record missing" in missing.reason
    drive = _copy(tmp_path)
    (drive / "manifest.json").unlink()
    unrun = scorer.score_drive(scorer.load_drive(drive, _committed_reference()))
    assert unrun.state == "failed" and "manifest missing" in unrun.reason


def test_kitti_rotation_is_the_rectification_times_velo_to_cam(tmp_path: Path) -> None:
    rect = Rotation.from_euler("xyz", [0.4, -0.2, 0.1], degrees=True).as_matrix()
    rotation = Rotation.from_euler("xyz", [-90.0, 0.5, -90.0], degrees=True).as_matrix()
    _write_calibration(tmp_path, rect, rotation)
    np.testing.assert_allclose(scorer.kitti_rotation(tmp_path), rect @ rotation, atol=1e-12)


def _write_calibration(directory: Path, rect: np.ndarray, rotation: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    numbers = " ".join(f"{value:.15e}" for value in rect.ravel())
    (directory / "calib_cam_to_cam.txt").write_text(f"R_rect_00: {numbers}\n", encoding="utf-8")
    numbers = " ".join(f"{value:.15e}" for value in rotation.ravel())
    text = f"R: {numbers}\nT: 0 0 0\n"
    (directory / "calib_velo_to_cam.txt").write_text(text, encoding="utf-8")


def _pre_registered(tmp_path: Path, pin: dict[str, Any], calibration_sha: str) -> list[str]:
    """A run directory holding the committed drive, its day's calibration and a plan."""

    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copytree(COMMITTED, runs / "2011_09_26_0005")
    record = {"2011_09_26": {"ok": True, "sha256": "c" * 64}}
    (runs / "calibration_download.json").write_text(json.dumps(record), encoding="utf-8")
    _write_calibration(tmp_path / "calib" / "2011_09_26", np.eye(3), _committed_reference())
    plan = {
        "recordings": {
            "evaluation": ["2011_09_26_0005"],
            "archives": {"2011_09_26_0005": pin},
            "calibration_archive_sha256": {"2011_09_26": calibration_sha},
        }
    }
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan), encoding="utf-8")
    arguments = [str(runs), str(tmp_path / "calib"), str(tmp_path / "out")]
    return [*arguments, "--preregistration", str(tmp_path / "plan.yaml")]


def test_a_pre_registered_run_is_scored_against_its_pins(tmp_path: Path) -> None:
    assert scorer.main(_pre_registered(tmp_path, _pin(), "c" * 64)) == 0
    scores = json.loads((tmp_path / "out/scores.json").read_text(encoding="utf-8"))
    assert scores["drives"] == ["2011_09_26_0005"]
    assert scores["scores"]["2011_09_26_0005"]["state"] == "scored"
    provenance = scores["provenance"]
    assert provenance["preregistration"]["sha256"] and provenance["calibration_download"]
    assert provenance["thresholds"]["recipe"]["min_control_t"] == 2.5


def test_a_pre_registered_drive_whose_archive_differs_fails(tmp_path: Path) -> None:
    pin = {**_pin(), "size_bytes": _pin()["size_bytes"] + 1}
    assert scorer.main(_pre_registered(tmp_path, pin, "c" * 64)) == 0
    scores = json.loads((tmp_path / "out/scores.json").read_text(encoding="utf-8"))
    assert scores["scores"]["2011_09_26_0005"]["state"] == "failed"
    assert scores["requirements"]["no-failed-drives"]["status"] == "fail"


def test_nothing_is_scored_when_a_calibration_differs_from_its_pin(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="calibration archive does not match its pin"):
        scorer.main(_pre_registered(tmp_path, _pin(), "d" * 64))
