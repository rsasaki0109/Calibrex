from pathlib import Path

from calibrex.core.provenance import sha256_path
from calibrex.importers.kalibr import import_kalibr_camchain


def _write_camchain(path: Path) -> None:
    path.write_text(
        """
cam0:
  camera_model: pinhole
  intrinsics: [461.629, 460.152, 362.680, 246.049]
  distortion_model: radtan
  distortion_coeffs: [-0.27, 0.06, 0.0, 0.0]
  T_cam_imu:
    - [1.0, 0.0, 0.0, 0.1]
    - [0.0, 1.0, 0.0, 0.2]
    - [0.0, 0.0, 1.0, 0.3]
    - [0.0, 0.0, 0.0, 1.0]
  timeshift_cam_imu: -0.00008121
  rostopic: /cam0/image_raw
  resolution: [752, 480]
cam1:
  camera_model: omni
  intrinsics: [0.8, 833.0, 830.0, 373.0, 253.0]
  distortion_model: radtan
  distortion_coeffs: [-0.33, 0.13, 0.0, 0.0]
  T_cn_cnm1:
    - [1.0, 0.0, 0.0, -0.11]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  T_cam_imu:
    - [1.0, 0.0, 0.0, -0.01]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  timeshift_cam_imu: -0.00008681
  rostopic: /cam1/image_raw
  resolution: [752, 480]
""".strip(),
        encoding="utf-8",
    )


def test_import_kalibr_camchain_preserves_spatiotemporal_conventions(
    tmp_path: Path,
) -> None:
    camchain = tmp_path / "camchain-imucam.yaml"
    fitting_input = tmp_path / "recording.bag"
    _write_camchain(camchain)
    fitting_input.write_bytes(b"fixture")

    artifact = import_kalibr_camchain(
        camchain,
        input_artifacts=(fitting_input,),
        tool_version="test",
        source_commit="0123456789abcdef",
        training_isolation_declared=True,
        training_isolation_evidence="fixture split declared before import",
    )

    assert artifact.status == "success"
    assert artifact.tool.license_spdx == "BSD-4-Clause"
    assert artifact.execution.mode == "imported"
    assert artifact.parsed_outputs.transforms["T_cam0_imu"].parent == "cam0"
    assert artifact.parsed_outputs.transforms["T_cam0_imu"].child == "imu"
    assert artifact.parsed_outputs.transforms["T_cam0_imu"].translation_m == [
        0.1,
        0.2,
        0.3,
    ]
    assert artifact.parsed_outputs.transforms["T_cam1_cam0"].translation_m == [
        -0.11,
        0.0,
        0.0,
    ]
    assert artifact.parsed_outputs.time_offsets_seconds["dt_imu_minus_cam0"] == (
        -0.00008121
    )
    assert artifact.parsed_outputs.intrinsics["cam0"]["camera_model"] == "pinhole"
    assert artifact.parsed_outputs.external_metrics_comparable is False
    assert {
        (item.role, item.sha256)
        for item in artifact.artifacts
    } == {
        ("input", sha256_path(fitting_input)),
        ("output", sha256_path(camchain)),
    }


def test_import_kalibr_camchain_materializes_digest_mismatch(
    tmp_path: Path,
) -> None:
    camchain = tmp_path / "camchain.yaml"
    _write_camchain(camchain)

    artifact = import_kalibr_camchain(
        camchain,
        expected_source_sha256="0" * 64,
    )

    assert artifact.status == "digest_mismatch"
    assert artifact.parsed_outputs.transforms == {}
    assert "SHA-256 mismatch" in " ".join(artifact.warnings)


def test_import_kalibr_camchain_materializes_malformed_output(
    tmp_path: Path,
) -> None:
    camchain = tmp_path / "broken.yaml"
    camchain.write_text("cam0: [", encoding="utf-8")

    artifact = import_kalibr_camchain(camchain)

    assert artifact.status == "invalid_output"
    assert "could not be parsed" in " ".join(artifact.warnings)


def test_import_kalibr_camchain_reports_missing_source(tmp_path: Path) -> None:
    artifact = import_kalibr_camchain(tmp_path / "missing.yaml")

    assert artifact.status == "unavailable"
    assert artifact.provenance.source_artifact_sha256 is None
