"""Gate boundaries, the usable-unit rule, and verdict logic of the Hilti camera-IMU scorer."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scorer = _load("score_hilti_camera_imu")
builder = _load("build_hilti_camera_imu_audit")

REFERENCE = Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True).as_matrix()
REFERENCE_1 = Rotation.from_euler("xyz", [-40.0, 5.0, 15.0], degrees=True).as_matrix()


def _rotated(reference: np.ndarray, degrees: float, axis: int = 2) -> np.ndarray:
    vector = np.zeros(3)
    vector[axis] = np.radians(degrees)
    return Rotation.from_rotvec(vector).as_matrix() @ reference


def _unit(camera: int = 0, offset_deg: float = 0.2, **changes: object) -> "scorer.UnitInput":
    reference = (REFERENCE, REFERENCE_1)[camera]
    unit = scorer.UnitInput(
        rotation=_rotated(reference, offset_deg),
        time_offset_s=0.0019,
        reference_rotation=reference,
        reference_shift_s=0.0019,
        windows_used=14,
        pairs=1000,
        failed_pairs=10,
        axis_std_deg=(0.1, 0.1, 0.1),
    )
    return replace(unit, **changes)


def _all(recordings: list[str], **changes: object) -> dict:
    return {
        camera: {recording: _unit(index, **changes) for recording in recordings}
        for index, camera in enumerate(("cam0", "cam1"))
    }


RECORDINGS = ["exp01", "exp02", "exp03", "exp04"]


def test_all_pass_is_supported() -> None:
    result = scorer.score_units(_all(RECORDINGS), RECORDINGS)
    assert result["verdict"] == "supported"
    assert all(item["status"] == "pass" for item in result["requirements"].values())
    assert result["units"]["cam0"]["exp01"]["rotation_error_deg"] == pytest.approx(0.2)
    assert result["consistency_deg"]["cam0"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(("offset_deg", "status"), [(1.0, "pass"), (1.001, "fail")])
def test_rotation_boundary(offset_deg: float, status: str) -> None:
    units = _all(RECORDINGS)
    units["cam0"]["exp01"] = _unit(0, offset_deg=offset_deg)
    result = scorer.score_units(units, RECORDINGS)
    assert result["requirements"]["rotation-vs-kalibr"]["status"] == status


@pytest.mark.parametrize(
    ("shift_ms", "status"), [(0.5, "pass"), (-0.5, "pass"), (0.51, "fail"), (-0.51, "fail")]
)
def test_time_boundary_is_absolute(shift_ms: float, status: str) -> None:
    units = _all(RECORDINGS)
    units["cam1"]["exp02"] = _unit(1, time_offset_s=0.0019 + shift_ms / 1000.0)
    result = scorer.score_units(units, RECORDINGS)
    assert result["requirements"]["time-offset-vs-kalibr"]["status"] == status


def test_constrained_rules() -> None:
    def check(**changes: object) -> str:
        units = _all(RECORDINGS)
        units["cam0"]["exp01"] = _unit(0, **changes)
        return scorer.score_units(units, RECORDINGS)["requirements"]["constrained"]["status"]

    assert check(controls_detected=(True, True, False)) == "pass"  # 2 of 3
    assert check(controls_detected=(True, False, False)) == "fail"
    assert check(axis_std_deg=(0.3, 0.1, 0.1)) == "pass"  # boundary is inclusive
    assert check(axis_std_deg=(0.1, 0.1, 0.31)) == "fail"
    assert check(axis_status=("estimated", "unobservable", "estimated")) == "fail"


def test_usable_unit_rule_makes_units_inconclusive_not_failed() -> None:
    for changes in (
        {"windows_used": 7},
        {"pairs": 100, "failed_pairs": 26},
    ):
        units = _all(RECORDINGS)
        units["cam0"]["exp01"] = _unit(0, offset_deg=5.0, **changes)  # would fail if scored
        result = scorer.score_units(units, RECORDINGS)
        assert result["units"]["cam0"]["exp01"]["state"] == "inconclusive"
        assert result["requirements"]["rotation-vs-kalibr"]["status"] == "pass"
        assert result["verdict"] == "supported"  # exp02-exp04 still scored for both cameras
    for changes in ({"windows_used": 8}, {"pairs": 100, "failed_pairs": 25}):
        units = _all(RECORDINGS)
        units["cam0"]["exp01"] = _unit(0, **changes)
        assert scorer.score_units(units, RECORDINGS)["units"]["cam0"]["exp01"]["state"] == "scored"


def test_failed_unit_refutes() -> None:
    for unit in (
        scorer.UnitInput(error="boom"),
        _unit(0, policy_status="fail"),
        _unit(0, solver_status="failed", rotation=None),
    ):
        units = _all(RECORDINGS)
        units["cam0"]["exp01"] = unit
        result = scorer.score_units(units, RECORDINGS)
        assert result["units"]["cam0"]["exp01"]["state"] == "failed"
        assert result["requirements"]["no-failed-units"]["status"] == "fail"
        assert result["verdict"] == "refuted"


def test_consistency_is_largest_pairwise_angle() -> None:
    units = _all(RECORDINGS)
    for recording, offset in zip(RECORDINGS, (0.0, 0.3, 0.8, 0.5), strict=True):
        units["cam0"][recording] = _unit(0, offset_deg=offset)
    result = scorer.score_units(units, RECORDINGS)
    assert result["consistency_deg"]["cam0"] == pytest.approx(0.8)
    assert result["requirements"]["cross-recording-consistency"]["status"] == "fail"
    assert result["verdict"] == "refuted"
    units["cam0"]["exp03"] = _unit(0, offset_deg=0.75)
    result = scorer.score_units(units, RECORDINGS)
    assert result["requirements"]["cross-recording-consistency"]["status"] == "pass"


def test_relative_rotation_cancels_common_offset_and_gates() -> None:
    units = _all(RECORDINGS)
    # The same IMU-frame offset on both cameras leaves the relative rotation unchanged.
    common = Rotation.from_euler("xyz", [0.3, 0.2, -0.4], degrees=True).as_matrix()
    for recording in RECORDINGS:
        for camera, index in (("cam0", 0), ("cam1", 1)):
            reference = (REFERENCE, REFERENCE_1)[index]
            units[camera][recording] = replace(
                units[camera][recording], rotation=reference @ common
            )
    result = scorer.score_units(units, RECORDINGS)
    assert result["relative_cam0_cam1_deg"]["exp01"] == pytest.approx(0.0, abs=1e-6)
    units["cam1"]["exp04"] = _unit(1, offset_deg=0.6)  # opposite-signed errors are impossible here
    units["cam0"]["exp04"] = _unit(0, offset_deg=0.0)
    result = scorer.score_units(units, RECORDINGS)
    assert result["relative_cam0_cam1_deg"]["exp04"] == pytest.approx(0.6, abs=1e-6)
    assert result["requirements"]["relative-cam0-cam1"]["status"] == "fail"


def test_verdict_inconclusive_with_two_recordings_and_supported_with_three() -> None:
    two = scorer.score_units(_all(RECORDINGS[:2]), RECORDINGS[:2])
    assert all(item["status"] == "pass" for item in two["requirements"].values())
    assert two["verdict"] == "inconclusive"
    units = _all(RECORDINGS)
    units["cam0"]["exp01"] = _unit(0, windows_used=3)
    units["cam1"]["exp02"] = _unit(1, windows_used=3)
    result = scorer.score_units(units, RECORDINGS)
    assert result["recordings_scored_for_both_cameras"] == ["exp03", "exp04"]
    assert result["verdict"] == "inconclusive"
    units = _all(RECORDINGS)
    units["cam0"]["exp01"] = _unit(0, windows_used=3)
    assert scorer.score_units(units, RECORDINGS)["verdict"] == "supported"


def test_ungated_cameras_are_reported_but_not_gated() -> None:
    units = _all(RECORDINGS)
    units["cam2"] = {recording: _unit(0, offset_deg=5.0) for recording in RECORDINGS}
    result = scorer.score_units(units, RECORDINGS)
    assert result["units"]["cam2"]["exp01"]["rotation_error_deg"] == pytest.approx(5.0)
    assert result["verdict"] == "supported"


def test_builder_maps_scores_to_audit(tmp_path: Path) -> None:
    import json

    preregistration = tmp_path / "prereg.yaml"
    preregistration.write_text("claim: test\n", encoding="utf-8")

    def run(units: dict, recordings: list[str]) -> str:
        scores_path = tmp_path / "scores.json"
        scores_path.write_text(json.dumps(scorer.score_units(units, recordings)), encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
        return str(builder.build(scores, scores_path, preregistration, out).verdict)

    assert run(_all(RECORDINGS), RECORDINGS) == "supported"
    assert run(_all(RECORDINGS[:2]), RECORDINGS[:2]) == "incomplete"
    bad = _all(RECORDINGS)
    bad["cam1"]["exp03"] = _unit(1, offset_deg=1.5)
    assert run(bad, RECORDINGS) == "refuted"
    failed = _all(RECORDINGS)
    failed["cam0"]["exp02"] = scorer.UnitInput(error="boom")
    assert run(failed, RECORDINGS) == "refuted"
