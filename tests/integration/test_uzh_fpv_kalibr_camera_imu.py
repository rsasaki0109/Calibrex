"""A public Kalibr camchain evaluated on a separately captured recording (opt-in).

UZH-FPV publishes the Kalibr camera-IMU calibration of its Snapdragon forward camera,
fitted on ``indoor_forward_calib_snapdragon_imu.bag``. The evaluation recording is the
separately captured ``outdoor_forward_calib_snapdragon_imu.bag``, posed against its
AprilGrid by ``tools/aprilgrid_camera_imu_recording.py`` (needs OpenCV); the commands
are in ``examples/public_datasets/uzh_fpv_kalibr_camera_imu/README.md``.

The dataset page states no license; the data stay outside the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from calibrex.evaluation.external_camera_imu import evaluate_external_camera_imu
from calibrex.importers.kalibr import import_kalibr_camchain

ROOT = Path(os.environ.get("CALIBREX_UZH_FPV_KALIBR", "data/public/uzh_fpv_kalibr_camera_imu"))
CAMCHAIN = (
    ROOT
    / "indoor_forward_calib_snapdragon"
    / "camchain-imucam-..indoor_forward_calib_snapdragon_imu.yaml"
)
FITTING_BAG = ROOT / "indoor_forward_calib_snapdragon_imu.bag"
RECORDING = ROOT / "outdoor_forward_calib_snapdragon_recording.json"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (CAMCHAIN.is_file() and FITTING_BAG.is_file() and RECORDING.is_file()),
        reason="prepare the UZH-FPV recording (see the module docstring)",
    ),
]


def _evaluate(camchain: Path, tmp_path: Path) -> object:
    run = import_kalibr_camchain(
        camchain,
        input_artifacts=(FITTING_BAG,),
        tool_version="Kalibr (UZH-FPV release; version not stated)",
        training_isolation_declared=True,
        training_isolation_evidence=(
            "fitted on indoor_forward_calib_snapdragon_imu.bag; evaluated on "
            "outdoor_forward_calib_snapdragon_imu.bag"
        ),
    )
    path = tmp_path / "external-run.json"
    run.save(path)
    return evaluate_external_camera_imu(path, RECORDING)


def test_published_translation_passes_and_rates_are_noise_limited(tmp_path: Path) -> None:
    evidence = _evaluate(CAMCHAIN, tmp_path)

    lever_arm = evidence.lever_arm  # type: ignore[attr-defined]
    assert lever_arm.status == "pass"
    assert lever_arm.control_detection_fraction == 1.0
    assert lever_arm.gravity_magnitude_m_s2 == pytest.approx(9.81, abs=0.2)
    # Frame-to-frame camera rates from AprilGrid poses are too noisy for the
    # 0.05 rad/s angular-rate budget, so rotation and time are not admissible here.
    assert evidence.status == "fail"  # type: ignore[attr-defined]
    assert "angular-rate RMSE" in evidence.reasons[0]  # type: ignore[attr-defined]


@pytest.mark.parametrize(("axis", "sign"), [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)])
def test_five_centimetre_translation_errors_fail(tmp_path: Path, axis: int, sign: int) -> None:
    camchain = yaml.safe_load(CAMCHAIN.read_text(encoding="utf-8"))
    camchain["cam0"]["T_cam_imu"][axis][3] += 0.05 * sign
    perturbed = tmp_path / "camchain-imucam.yaml"
    perturbed.write_text(yaml.safe_dump(camchain), encoding="utf-8")

    evidence = _evaluate(perturbed, tmp_path)

    assert evidence.lever_arm.status == "fail"  # type: ignore[attr-defined]
