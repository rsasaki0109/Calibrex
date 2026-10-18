"""Opt-in far-start controls through the real CLI, artifacts and cache."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.unit.camera_lidar_scene import true_transform
from tests.unit.test_check_camera_lidar import write_bag

from calibrex.check.runner import CheckRunOptions
from calibrex.cli.main import main
from calibrex.core.bag_estimate import EstimateOptions
from calibrex.core.calibration_check import CheckOptions
from calibrex.core.camera_lidar_edge import (
    DEFAULT_MIN_CONTROL_T,
    CameraLidarEdgeControl,
    load_camera_lidar_edge,
)
from calibrex.core.validation import validate_file
from calibrex.evaluation import camera_lidar_edge as edge

FLAG = "--camera-lidar-far-control-t"
FIELD = "camera_lidar_start_scan_min_control_t"


@pytest.fixture(scope="module")
def bag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_bag(tmp_path_factory.mktemp("far-control") / "scene", true_transform())


@pytest.mark.parametrize("command", ["check", "estimate", "drift"])
def test_cli_records_the_margin_in_results_and_estimator_evidence(
    command: str,
    bag: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "check.json" if command == "check" else tmp_path / "result"
    args = [command, str(bag)]
    if command == "drift":
        other = tmp_path / "other"
        shutil.copytree(bag, other)
        args.append(str(other))
    args += [
        "--pairs",
        "camera-lidar",
        "--output",
        str(output),
        FLAG,
        "3.5",
        "--no-cache",
        "--quiet",
    ]
    if command == "check":
        args += ["--fail-on", "never"]
        result_path = output
    elif command == "estimate":
        result_path = output / "bag_estimate.json"
    else:
        result_path = output / "calibration_drift.json"
    assert main(args) == 0
    payload = json.loads(result_path.read_text())
    assert payload["options"][FIELD] == 3.5
    assert validate_file(result_path).valid
    evidence = list(tmp_path.rglob("camera-lidar*.yaml"))
    assert len(evidence) == (2 if command == "drift" else 1)
    for path in evidence:
        artifact = load_camera_lidar_edge(path)
        assert artifact.options["start_scan_min_control_t"] == 3.5


@pytest.mark.parametrize("command", ["check", "estimate", "drift"])
@pytest.mark.parametrize("value", ["nan", "inf", "2.49", "-1"])
def test_invalid_cli_margin_fails_before_a_bag_is_opened(
    command: str,
    value: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
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


@pytest.mark.parametrize("value", [None, 2.5, 3.5])
def test_runtime_and_result_models_accept_compatible_margins(value: float | None) -> None:
    assert DEFAULT_MIN_CONTROL_T == edge.EdgeAlignmentOptions().min_control_t == 2.5
    assert CheckRunOptions(**{FIELD: value}).camera_lidar_start_scan_min_control_t == value
    assert EstimateOptions(**{FIELD: value}).camera_lidar_start_scan_min_control_t == value
    options = CheckOptions(
        sigma_k=3.0,
        rotation_floor_deg=0.5,
        translation_floor_m=0.02,
        detection_probe_deg=1.0,
        **{FIELD: value},
    )
    assert options.camera_lidar_start_scan_min_control_t == value


@pytest.mark.parametrize("value", [2.49, float("nan"), float("inf")])
def test_runtime_and_result_models_reject_weak_or_nonfinite_margins(value: float) -> None:
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


def test_cli_keeps_current_fit_and_separates_both_rules_in_the_cache(
    bag: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    far = np.array([-1.5, -0.15, -5.3, 0.0, 0.0, 0.0])

    def control(t: float) -> CameraLidarEdgeControl:
        return CameraLidarEdgeControl(
            amount=2.0,
            unit="deg",
            held_out_blocks=6,
            detected_blocks=6,
            mean_objective_drop=0.01,
            min_t_statistic=t,
            detected=t >= 2.5,
        )

    current_controls = [control(t) for t in (0.4, 3.0, 1.0, 0.1, 0.1, 0.1)]
    far_controls = [control(t) for t in (2.65, 3.25, 1.23, 0.1, 0.1, 0.1)]
    monkeypatch.setattr(edge, "_profile_start", lambda *_: np.ones(6))
    monkeypatch.setattr(edge, "_profile_starts", lambda *_: [np.ones(6)])
    monkeypatch.setattr(
        edge,
        "fit_extrinsic",
        lambda _, start, opts: (current if not start.any() else far, 1.0),
    )

    def evidence(*args: Any) -> tuple[np.ndarray, np.ndarray, list[CameraLidarEdgeControl]]:
        theta = args[3]
        controls = current_controls if theta is current else far_controls
        return np.tile(theta, (6, 1)), np.full(6, 0.025), controls

    monkeypatch.setattr(edge, "_evidence", evidence)
    for index, (margin, hit, verdict) in enumerate(
        [
            (None, False, "fail"),
            (None, True, "fail"),
            (3.5, False, "pass"),
            (3.5, True, "pass"),
            (None, True, "fail"),
        ]
    ):
        folder = tmp_path / str(index)
        output = folder / "check.json"
        args = [
            "check",
            str(bag),
            "--pairs",
            "camera-lidar",
            "--output",
            str(output),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--fail-on",
            "never",
            "--quiet",
        ]
        if margin is not None:
            args += [FLAG, str(margin)]
        assert main(args) == 0
        payload = json.loads(output.read_text())
        (pair,) = [item for item in payload["pairs"] if item["pair"] == "camera-lidar"]
        assert payload["options"].get(FIELD) == margin
        assert pair["evidence_from_cache"] is hit
        assert pair["verdict"] == verdict
        artifact = load_camera_lidar_edge(folder / pair["evidence"][0]["path"])
        assert artifact.options["start_scan_min_control_t"] == margin
        assert artifact.dofs[1].known_bad_control == (
            current_controls[1] if margin is not None else far_controls[1]
        )
        assert artifact.calibrated_dofs == (["pitch"] if margin is not None else ["roll", "pitch"])
