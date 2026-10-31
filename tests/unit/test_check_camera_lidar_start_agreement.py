"""Opt-in camera-LiDAR cross-start disagreement warning through check, estimate and drift."""

import json
import shutil
from pathlib import Path

import pytest
from tests.unit.camera_lidar_scene import true_transform
from tests.unit.test_check_camera_lidar import write_bag

from calibrex.check.estimators import camera_lidar_axis_estimates
from calibrex.check.runner import CheckRunOptions
from calibrex.check.verdict import AxisEstimate, VerdictOptions, judge_pair
from calibrex.cli.main import main
from calibrex.core.bag_estimate import EstimateOptions
from calibrex.core.calibration_check import CheckOptions
from calibrex.core.camera_lidar_edge import load_camera_lidar_edge
from calibrex.core.validation import validate_file

FLAG = "--camera-lidar-start-agreement-deg"
FIELD = "camera_lidar_start_agreement_deg"
OPTIONS = VerdictOptions()


@pytest.fixture(scope="module")
def bag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_bag(tmp_path_factory.mktemp("start-agreement") / "scene", true_transform())


def _run(command: str, bag: Path, tmp_path: Path, *extra: str) -> tuple[dict, list[Path]]:
    output = tmp_path / "check.json" if command == "check" else tmp_path / "result"
    args = [command, str(bag)]
    if command == "drift":
        other = tmp_path / "other"
        shutil.copytree(bag, other)
        args.append(str(other))
    args += ["--pairs", "camera-lidar", "--output", str(output), "--no-cache", "--quiet", *extra]
    if command == "check":
        args += ["--fail-on", "never"]
        result_path = output
    elif command == "estimate":
        result_path = output / "bag_estimate.json"
    else:
        result_path = output / "calibration_drift.json"
    assert main(args) == 0
    assert validate_file(result_path).valid
    return json.loads(result_path.read_text()), list(tmp_path.rglob("camera-lidar*.yaml"))


@pytest.mark.parametrize("command", ["check", "estimate", "drift"])
def test_cli_records_the_rule_in_results_and_estimator_evidence(
    command: str, bag: Path, tmp_path: Path
) -> None:
    payload, evidence = _run(command, bag, tmp_path, FLAG, "0.5")
    assert payload["options"][FIELD] == 0.5
    assert len(evidence) == (2 if command == "drift" else 1)
    for path in evidence:
        assert load_camera_lidar_edge(path).options["start_agreement_deg"] == 0.5


def test_rule_off_leaves_estimator_evidence_unchanged(bag: Path, tmp_path: Path) -> None:
    payload, (path,) = _run("check", bag, tmp_path)
    assert payload["options"].get(FIELD) is None
    artifact = load_camera_lidar_edge(path)
    assert "start_agreement_deg" not in artifact.options
    assert artifact.start_disagreement_deg is None


@pytest.mark.parametrize("command", ["check", "estimate", "drift"])
@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_invalid_cli_bound_fails_before_a_bag_is_opened(
    command: str, value: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "result"
    args = [command, str(tmp_path / "absent")]
    if command == "drift":
        args.append(str(tmp_path / "also-absent"))
    args += ["--output", str(output), FLAG, value]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert FLAG in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("value", [0.0, -0.5, float("nan"), float("inf")])
def test_runtime_and_result_models_reject_nonpositive_or_nonfinite_bounds(value: float) -> None:
    with pytest.raises(ValueError, match=FIELD):
        CheckRunOptions(**{FIELD: value})
    with pytest.raises(ValueError):
        EstimateOptions(**{FIELD: value})
    with pytest.raises(ValueError):
        CheckOptions(
            sigma_k=3.0,
            rotation_floor_deg=0.5,
            translation_floor_m=0.02,
            detection_probe_deg=1.0,
            **{FIELD: value},
        )


def _disputed(name: str = "roll") -> AxisEstimate:
    return AxisEstimate(name, "deg", 1.5, 0.2, False, "starts disagree", "start_disagreement")  # type: ignore[arg-type]


def test_disagreement_turns_a_pass_or_inconclusive_pair_into_warn_but_keeps_fail() -> None:
    passing = AxisEstimate("pitch", "deg", 0.1, 0.1, True)
    assert judge_pair([passing, _disputed()], OPTIONS).verdict == "warn"
    only = judge_pair([_disputed(), _disputed("yaw")], OPTIONS)
    assert only.verdict == "warn" and only.axes == ()
    assert [item.reason_code for item in only.unchecked] == ["start_disagreement"] * 2
    failing = AxisEstimate("pitch", "deg", 3.0, 0.1, True)
    assert judge_pair([failing, _disputed()], OPTIONS).verdict == "fail"
    unobservable = AxisEstimate("yaw", "deg", 0.0, 3.0, False, "flat", "unobservable")
    assert judge_pair([unobservable], OPTIONS).verdict == "inconclusive"


def test_withheld_artifact_axis_becomes_a_start_disagreement_unchecked_axis(
    bag: Path, tmp_path: Path
) -> None:
    _, (path,) = _run("check", bag, tmp_path, FLAG, "0.5")
    artifact = load_camera_lidar_edge(path)
    dofs = [
        record.model_copy(update={"status": "unobservable"}) if record.name == "yaw" else record
        for record in artifact.dofs
    ]
    withheld = artifact.model_copy(
        update={
            "dofs": dofs,
            "calibrated_dofs": [n for n in artifact.calibrated_dofs if n != "yaw"],
            "start_disagreement_deg": {"yaw": 1.25},
        }
    )
    candidate = true_transform()
    estimates = {item.name: item for item in camera_lidar_axis_estimates(withheld, candidate)}
    assert estimates["yaw"].unchecked_code == "start_disagreement"
    assert "1.25 deg (above 0.5 deg)" in (estimates["yaw"].unchecked_reason or "")
    assert estimates["pitch"].unchecked_code != "start_disagreement"
