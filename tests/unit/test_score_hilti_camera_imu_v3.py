"""Per-axis verdicts, gate boundaries and the usable-unit rule of the camera-IMU v3 scorer."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tests.unit.test_score_hilti_camera_camera import _load

scorer = _load("score_hilti_camera_imu_v3")
ROOT = Path(__file__).resolve().parents[2]
RECORDINGS = ["r1", "r2", "r3", "r4", "r5"]
REFERENCE = {
    "cam0": Rotation.from_euler("xyz", [90.0, -1.0, 90.0], degrees=True).as_matrix(),
    "cam1": Rotation.from_euler("xyz", [90.0, -1.3, 90.2], degrees=True).as_matrix(),
}
UNJUDGED = scorer.AxisInput(status="unobservable", std=0.4, control_detected=True)
UNDETECTED = scorer.AxisInput(status="estimated", std=0.2, control_detected=False)


def _unit(
    camera: str = "cam0",
    errors_deg: tuple[float, float, float] = (0.1, -0.2, 0.1),
    time_error_ms: float = 0.1,
    axes: tuple[Any, Any, Any] | None = None,
    **changes: Any,
) -> Any:
    rotation = Rotation.from_rotvec(np.radians(errors_deg)).as_matrix() @ REFERENCE[camera]
    fields = {
        "rotation": rotation,
        "reference_rotation": REFERENCE[camera],
        "time_offset_s": 0.002 + time_error_ms / 1000.0,
        "reference_shift_s": 0.002,
        "windows_used": 20,
        "pairs": 1000,
        "failed_pairs": 10,
        **({"axes": axes} if axes is not None else {}),
        **changes,
    }
    return scorer.UnitInput(**fields)


def _all(recordings: list[str] = RECORDINGS) -> dict[str, dict[str, Any]]:
    return {camera: {r: _unit(camera) for r in recordings} for camera in ("cam0", "cam1")}


def _score(units: dict[str, dict[str, Any]], recordings: list[str] = RECORDINGS) -> Any:
    return scorer.score_units(units, recordings)


def test_a_clean_set_is_supported() -> None:
    result = _score(_all())
    assert result["verdict"] == "supported"
    assert {item["status"] for item in result["requirements"].values()} == {"pass"}
    assert result["judged_fraction"] == 1.0
    assert result["requirements"]["judged-axis-accuracy"]["observed"] == pytest.approx(0.2)


@pytest.mark.parametrize(("error", "status"), [(1.0, "pass"), (1.0001, "fail")])
def test_a_judged_axis_is_gated_at_1_deg(error: float, status: str) -> None:
    units = _all()
    for camera in ("cam0", "cam1"):  # both cameras, so the relative rotation stays small
        units[camera]["r3"] = _unit(camera, errors_deg=(0.1, -error, 0.1))
    result = _score(units)
    assert result["requirements"]["judged-axis-accuracy"]["status"] == status
    assert result["verdict"] == ("supported" if status == "pass" else "refuted")


@pytest.mark.parametrize("axis", [UNJUDGED, UNDETECTED])
def test_an_unjudged_axis_is_not_gated_but_counts_against_coverage(axis: Any) -> None:
    units = _all()
    judged = scorer.AxisInput()
    units["cam0"]["r2"] = _unit(errors_deg=(0.1, -3.0, 0.1), axes=(judged, axis, judged))
    result = _score(units)
    assert result["requirements"]["judged-axis-accuracy"]["status"] == "pass"
    assert result["judged_axes"] == 29
    assert result["judged_fraction"] == pytest.approx(29 / 30)
    # the axis is left out of consistency and of the relative rotation on r2
    assert result["consistency"]["cam0-pitch"]["recordings"] == ["r1", "r3", "r4", "r5"]
    assert "r2-pitch" not in result["relative_cam0_cam1"]
    assert result["verdict"] == "supported"


def test_an_axis_without_a_control_is_not_judged() -> None:
    axis = scorer.AxisInput(status="estimated", std=0.1, control_detected=None)
    assert not axis.judged


def test_coverage_below_half_is_inconclusive_not_refuted() -> None:
    units = _all()
    judged = scorer.AxisInput()
    for camera in ("cam0", "cam1"):
        for recording in RECORDINGS:
            units[camera][recording] = _unit(camera, axes=(judged, UNJUDGED, UNJUDGED))
    result = _score(units)
    assert result["judged_fraction"] == pytest.approx(1 / 3)
    assert result["requirements"]["judged-axis-coverage"]["status"] == "fail"
    assert result["verdict"] == "inconclusive"


def test_a_missing_artifact_fails_and_a_refusal_is_not_usable(tmp_path: Path) -> None:
    log = tmp_path / "cam0_r1.log"
    log.write_text(
        "ValueError: camera tracking produced 2 windows; at least 3 are needed\n", encoding="utf-8"
    )
    refused = scorer.load_unit(tmp_path / "cam0_r1.yaml", log, {}, "cam0")
    assert refused.refused_windows == 2
    assert scorer.score_unit(refused).state == "inconclusive"
    missing = scorer.load_unit(tmp_path / "cam0_r2.yaml", tmp_path / "cam0_r2.log", {}, "cam0")
    assert scorer.score_unit(missing).state == "failed"
    units = _all()
    units["cam0"]["r1"] = refused
    assert _score(units)["verdict"] == "supported"  # four recordings are scored for both
    units["cam0"]["r2"] = missing
    result = _score(units)
    assert result["requirements"]["no-failed-units"]["status"] == "fail"
    assert result["verdict"] == "refuted"


@pytest.mark.parametrize(
    ("changes", "state"),
    [
        ({"windows_used": 7}, "inconclusive"),
        ({"windows_used": 8}, "scored"),
        ({"failed_pairs": 260}, "inconclusive"),
        ({"policy_status": "fail"}, "failed"),
        ({"windows_used": 5, "policy_status": "fail"}, "inconclusive"),
        ({"solver_status": "insufficient_intervals", "rotation": None}, "failed"),
    ],
)
def test_the_usable_unit_rule_comes_before_the_failure_policy(
    changes: dict[str, Any], state: str
) -> None:
    assert scorer.score_unit(_unit(**changes)).state == state


@pytest.mark.parametrize(("error", "status"), [(0.75, "pass"), (-0.7501, "fail")])
def test_a_judged_clock_offset_is_gated_at_0_75_ms(error: float, status: str) -> None:
    units = _all()
    units["cam0"]["r4"] = _unit(time_error_ms=error)
    assert _score(units)["requirements"]["judged-time-offset"]["status"] == status
    unjudged = _unit(time_error_ms=5.0, time=UNJUDGED)
    units["cam0"]["r4"] = unjudged
    assert _score(units)["requirements"]["judged-time-offset"]["status"] == "pass"


@pytest.mark.parametrize(("spread", "status"), [(1.0, "pass"), (1.01, "fail")])
def test_a_judged_axis_spans_at_most_1_deg_across_recordings(spread: float, status: str) -> None:
    units = _all()
    units["cam1"]["r1"] = _unit("cam1", errors_deg=(0.1, -0.2 + spread / 2, 0.1))
    units["cam1"]["r5"] = _unit("cam1", errors_deg=(0.1, -0.2 - spread / 2, 0.1))
    result = _score(units)
    assert result["consistency"]["cam1-pitch"]["range_deg"] == pytest.approx(spread)
    # r1 and r5 of cam1 also move the relative rotation; keep the gate on consistency
    assert result["requirements"]["judged-axis-consistency"]["status"] == status


@pytest.mark.parametrize(("error", "status"), [(0.7, "pass"), (0.8, "fail")])
def test_the_relative_rotation_is_gated_per_axis_judged_for_both(error: float, status: str) -> None:
    units = _all()
    units["cam0"]["r2"] = _unit("cam0", errors_deg=(0.1, -0.2, 0.1 + error))
    result = _score(units)
    assert result["relative_cam0_cam1"]["r2-yaw"]["relative_deg"] == pytest.approx(error, abs=5e-3)
    assert result["requirements"]["judged-axis-relative"]["status"] == status
    # an axis judged for one camera only is not compared
    judged = scorer.AxisInput()
    units["cam1"]["r2"] = _unit("cam1", axes=(judged, judged, UNJUDGED))
    result = _score(units)
    assert "r2-yaw" not in result["relative_cam0_cam1"]
    assert result["requirements"]["judged-axis-relative"]["status"] == "pass"


def test_two_recordings_scored_for_both_cameras_are_inconclusive() -> None:
    units = _all()
    for recording in ("r3", "r4", "r5"):
        units["cam1"][recording] = _unit("cam1", windows_used=3)
    result = _score(units)
    assert result["recordings_scored_for_both_cameras"] == ["r1", "r2"]
    assert result["requirements"]["scored-recordings-coverage"]["status"] == "fail"
    assert result["verdict"] == "inconclusive"


def test_nothing_judged_leaves_the_gates_unevaluated() -> None:
    units = {
        camera: {r: _unit(camera, axes=(UNJUDGED,) * 3, time=UNJUDGED) for r in RECORDINGS}
        for camera in ("cam0", "cam1")
    }
    result = _score(units)
    requirements = result["requirements"]
    assert requirements["judged-axis-accuracy"]["status"] == "not_evaluated"
    assert requirements["judged-time-offset"]["status"] == "not_evaluated"
    assert requirements["judged-axis-coverage"]["status"] == "fail"
    assert result["verdict"] == "inconclusive"


def test_a_committed_artifact_is_read_with_its_own_verdicts() -> None:
    artifact = ROOT / "docs/assets/hilti2022_camera_sota/camera_imu_v2/runs/cam0_exp15.yaml"
    payload = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    transform = np.eye(4)  # the estimate itself as the reference: every error is zero
    transform[:3, :3] = Rotation.from_quat(payload["rotation_quat_xyzw"]).as_matrix()
    camchain = {
        "cam0": {"T_cam_imu": transform.tolist(), "timeshift_cam_imu": 0.0019071204237431304}
    }
    unit = scorer.load_unit(artifact, artifact.with_suffix(".log"), camchain, "cam0")
    score = scorer.score_unit(unit)
    assert score.state == "scored"
    # roll judged; pitch's control was not detected; yaw is unobservable (std 0.328 deg)
    assert [axis["judged"] for axis in score.axes.values()] == [True, False, False]
    assert all(axis["error_deg"] == pytest.approx(0.0, abs=1e-9) for axis in score.axes.values())
    assert score.time["judged"]
    assert score.time["error_ms"] == pytest.approx(0.0243591, abs=1e-6)


def test_an_artifact_from_another_recipe_is_rejected(tmp_path: Path) -> None:
    source = ROOT / "docs/assets/hilti2022_camera_sota/camera_imu_v2/runs/cam0_exp15.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["options"]["rotation_control_deg"] = 2.0
    other = tmp_path / "cam0_r1.yaml"
    other.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="rotation_control_deg"):
        scorer.load_unit(other, tmp_path / "cam0_r1.log", {"cam0": {}}, "cam0")
