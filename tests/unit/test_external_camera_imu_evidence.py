import json
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.external_camera_imu_evidence import (
    CameraImuEvidenceThresholds,
    CameraImuMotionRecording,
    ExternalCameraImuEvidenceArtifact,
    load_external_camera_imu_evidence,
)
from calibrex.core.io import read_mapping, write_mapping
from calibrex.evaluation.external_camera_imu import (
    SyntheticCameraImuFixture,
    evaluate_external_camera_imu,
    write_synthetic_camera_imu_fixture,
)
from calibrex.importers.kalibr import import_kalibr_camchain


def _import(
    fixture: SyntheticCameraImuFixture,
    output: Path,
    *,
    isolation: bool = True,
) -> Path:
    artifact = import_kalibr_camchain(
        fixture.camchain_path,
        input_artifacts=(fixture.fitting_recording_path,),
        tool_version="synthetic",
        source_commit="0" * 40,
        training_isolation_declared=isolation,
        training_isolation_evidence="separate seeded recordings" if isolation else None,
    )
    artifact.save(output)
    return output


def _evaluate(tmp_path: Path, **fixture_options: object) -> ExternalCameraImuEvidenceArtifact:
    fixture = write_synthetic_camera_imu_fixture(tmp_path, **fixture_options)  # type: ignore[arg-type]
    run = _import(fixture, tmp_path / "external-run.json")
    return evaluate_external_camera_imu(run, fixture.evaluation_recording_path)


def test_true_kalibr_candidate_passes_with_all_signed_controls_detected(
    tmp_path: Path,
) -> None:
    evidence = _evaluate(tmp_path)

    assert evidence.status == "pass"
    assert evidence.candidate is not None
    assert evidence.candidate.time_offset_declared is True
    assert evidence.holdout_rate_rmse_rad_s is not None
    assert evidence.holdout_rate_rmse_rad_s < evidence.thresholds.max_holdout_rate_rmse_rad_s
    assert len(evidence.controls) == 8
    assert {control.kind for control in evidence.controls} == {"rotation", "time"}
    assert all(control.detected for control in evidence.controls)
    assert evidence.control_detection_fraction == 1.0
    assert evidence.split is not None
    assert evidence.split.train_interval_count >= evidence.thresholds.min_train_intervals
    assert evidence.split.holdout_interval_count >= evidence.thresholds.min_holdout_intervals
    assert evidence.external_run.tool_name == "kalibr"
    assert evidence.external_run.license_spdx == "BSD-4-Clause"
    assert evidence.recording.synthetic is True
    assert set(evidence.provenance.input_sha256) == {"external_run", "recording"}
    assert evidence.external_metrics_used is False
    ExternalCameraImuEvidenceArtifact.model_validate(evidence.model_dump(mode="json"))
    lever_arm = evidence.lever_arm
    assert lever_arm is not None and lever_arm.status == "pass"
    assert len(lever_arm.controls) == 6
    assert all(control.detected for control in lever_arm.controls)
    assert lever_arm.holdout_specific_force_rmse_m_s2 is not None
    assert (
        lever_arm.holdout_specific_force_rmse_m_s2
        < evidence.thresholds.max_holdout_specific_force_rmse_m_s2
    )
    # Bias and gravity are fitted on training windows only and land near the truth.
    assert lever_arm.gravity_magnitude_m_s2 == pytest.approx(9.81, abs=0.05)
    assert lever_arm.estimated_accel_bias_m_s2 == pytest.approx([0.05, -0.03, 0.08], abs=0.02)
    assert lever_arm.excitation_s2 is not None
    assert lever_arm.excitation_s2 >= evidence.thresholds.min_lever_arm_excitation_s2


@pytest.mark.parametrize(
    "fixture_options",
    [
        {"rotation_error_deg": (0.0, 0.0, 3.0)},
        {"rotation_error_deg": (-3.0, 0.0, 0.0)},
        {"timeshift_error_sec": 0.02},
        {"timeshift_error_sec": -0.02},
    ],
)
def test_wrong_kalibr_candidate_fails_on_holdout(
    tmp_path: Path,
    fixture_options: dict[str, object],
) -> None:
    evidence = _evaluate(tmp_path, **fixture_options)

    assert evidence.status == "fail"
    assert evidence.reasons


@pytest.mark.parametrize(
    "translation_error_m",
    [(0.05, 0.0, 0.0), (-0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, -0.05, 0.0), (0.0, 0.0, 0.05)],
)
def test_five_centimetre_lever_arm_error_fails(
    tmp_path: Path, translation_error_m: tuple[float, float, float]
) -> None:
    evidence = _evaluate(tmp_path, translation_error_m=translation_error_m)

    # Rotation and time are right, so only the accelerometer can reject it: a
    # signed control towards the truth lowers the holdout residual.
    assert evidence.status == "fail"
    assert evidence.control_detection_fraction == 1.0
    assert evidence.lever_arm is not None and evidence.lever_arm.status == "fail"
    assert any(reason.startswith("lever arm:") for reason in evidence.reasons)
    assert "fits better than the candidate" in " ".join(evidence.lever_arm.reasons)


def test_insensitive_lever_arm_controls_are_inconclusive_not_fail(tmp_path: Path) -> None:
    fixture = write_synthetic_camera_imu_fixture(tmp_path)
    run = _import(fixture, tmp_path / "external-run.json")
    # Demanding a 1 m/s^2 rise makes every control "not detected" without any
    # control fitting better: the recording cannot test the translation.
    gates = CameraImuEvidenceThresholds(lever_arm_control_min_increase_m_s2=1.0)

    evidence = evaluate_external_camera_imu(
        run, fixture.evaluation_recording_path, thresholds=gates
    )

    assert evidence.lever_arm is not None and evidence.lever_arm.status == "inconclusive"
    assert "cannot falsify the translation" in evidence.lever_arm.reasons[0]
    assert evidence.status == "warn"
    assert any("lever arm not checked" in warning for warning in evidence.warnings)


def test_recording_without_accelerometer_leaves_translation_unevaluated(
    tmp_path: Path,
) -> None:
    evidence = _evaluate(tmp_path, accelerometer=False, translation_error_m=(0.05, 0.0, 0.0))

    assert evidence.status == "pass"
    assert evidence.lever_arm is not None
    assert evidence.lever_arm.status == "not_evaluated"
    assert "no accelerometer" in evidence.lever_arm.reasons[0]


def test_yaw_only_motion_is_inconclusive_not_pass(tmp_path: Path) -> None:
    evidence = _evaluate(tmp_path, motion="yaw_only")

    assert evidence.status == "inconclusive"
    assert evidence.excitation is not None
    assert (
        evidence.excitation.min_rate_excitation_rad_s
        < evidence.thresholds.min_rate_excitation_rad_s
    )
    assert "not observable" in " ".join(evidence.reasons)
    # Rotation about one axis leaves the lever arm along that axis unobservable.
    assert evidence.lever_arm is not None and evidence.lever_arm.status == "inconclusive"


def test_fitting_recording_reuse_is_blocked(tmp_path: Path) -> None:
    fixture = write_synthetic_camera_imu_fixture(tmp_path)
    run = _import(fixture, tmp_path / "external-run.json")

    evidence = evaluate_external_camera_imu(run, fixture.fitting_recording_path)

    assert evidence.status == "blocked"
    assert "not be independent" in evidence.reasons[0]
    assert evidence.controls == []


def test_undeclared_isolation_and_time_offset_downgrade_to_warn(tmp_path: Path) -> None:
    fixture = write_synthetic_camera_imu_fixture(tmp_path)
    camchain = read_mapping(fixture.camchain_path)
    camchain["cam0"].pop("timeshift_cam_imu")
    write_mapping(fixture.camchain_path, camchain)
    run = _import(fixture, tmp_path / "external-run.json", isolation=False)

    evidence = evaluate_external_camera_imu(run, fixture.evaluation_recording_path)

    # The true shift is 4 ms, below the 10 ms control, so the candidate still
    # scores inside budget; the missing declarations must still block PASS.
    assert evidence.status == "warn"
    assert evidence.candidate is not None
    assert evidence.candidate.time_offset_declared is False
    joined = " ".join(evidence.warnings)
    assert "isolation" in joined
    assert "dt_imu_minus_cam0" in joined


def test_unsuccessful_external_run_is_blocked(tmp_path: Path) -> None:
    fixture = write_synthetic_camera_imu_fixture(tmp_path)
    artifact = import_kalibr_camchain(fixture.camchain_path, expected_source_sha256="0" * 64)
    run = tmp_path / "external-run.json"
    artifact.save(run)

    evidence = evaluate_external_camera_imu(run, fixture.evaluation_recording_path)

    assert evidence.status == "blocked"
    assert "digest_mismatch" in evidence.reasons[0]


def test_camera_mismatch_and_missing_inputs_are_blocked(tmp_path: Path) -> None:
    fixture = write_synthetic_camera_imu_fixture(tmp_path)
    run = _import(fixture, tmp_path / "external-run.json")

    mismatch = evaluate_external_camera_imu(
        run, fixture.evaluation_recording_path, camera_name="cam1"
    )
    missing = evaluate_external_camera_imu(run, tmp_path / "missing.json")

    assert mismatch.status == "blocked"
    assert "does not match" in mismatch.reasons[0]
    assert missing.status == "blocked"
    assert missing.recording.sha256 is None


def test_recording_rejects_unsorted_timestamps(tmp_path: Path) -> None:
    fixture = write_synthetic_camera_imu_fixture(tmp_path)
    payload = read_mapping(fixture.evaluation_recording_path)
    payload["imu_gyro"][1]["timestamp_sec"] = payload["imu_gyro"][0]["timestamp_sec"]

    with pytest.raises(ValueError, match="strictly increasing"):
        CameraImuMotionRecording.model_validate(payload)
    payload = read_mapping(fixture.evaluation_recording_path)
    payload["imu_accel"][2]["timestamp_sec"] = payload["imu_accel"][1]["timestamp_sec"]
    with pytest.raises(ValueError, match="imu_accel timestamps"):
        CameraImuMotionRecording.model_validate(payload)


def test_cli_synthesize_import_evaluate_and_validate(tmp_path: Path, capsys) -> None:
    fixture_dir = tmp_path / "fixture"
    assert (
        main(
            [
                "external-run",
                "synthesize-camera-imu-fixture",
                "--output-dir",
                str(fixture_dir),
                "--json",
            ]
        )
        == 0
    )
    paths = json.loads(capsys.readouterr().out)
    run = tmp_path / "external-run.json"
    assert (
        main(
            [
                "external-run",
                "import-kalibr",
                paths["camchain"],
                "--output",
                str(run),
                "--input-artifact",
                paths["fitting_recording"],
                "--tool-version",
                "synthetic",
                "--source-commit",
                "0" * 40,
                "--training-isolation-declared",
                "--training-isolation-evidence",
                "separate seeded recordings",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    evidence_path = tmp_path / "evidence.json"
    assert (
        main(
            [
                "external-run",
                "evaluate-camera-imu",
                str(run),
                "--recording",
                paths["evaluation_recording"],
                "--output",
                str(evidence_path),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"
    evidence = load_external_camera_imu_evidence(evidence_path)
    assert evidence.provenance.command[:3] == [
        "calibrex",
        "external-run",
        "evaluate-camera-imu",
    ]
    assert main(["validate", str(evidence_path), "--json"]) == 0
    assert main(["validate", paths["evaluation_recording"], "--json"]) == 0

    bad_dir = tmp_path / "bad"
    assert (
        main(
            [
                "external-run",
                "synthesize-camera-imu-fixture",
                "--output-dir",
                str(bad_dir),
                "--translation-error-m",
                "0",
                "0.05",
                "0",
            ]
        )
        == 0
    )
    bad_run = tmp_path / "bad-run.json"
    _import(
        SyntheticCameraImuFixture(
            bad_dir / "camchain-imucam.yaml",
            bad_dir / "fitting-recording.json",
            bad_dir / "evaluation-recording.json",
        ),
        bad_run,
    )
    capsys.readouterr()
    assert (
        main(
            [
                "external-run",
                "evaluate-camera-imu",
                str(bad_run),
                "--recording",
                str(bad_dir / "evaluation-recording.json"),
                "--output",
                str(tmp_path / "bad-evidence.json"),
            ]
        )
        == 1
    )
