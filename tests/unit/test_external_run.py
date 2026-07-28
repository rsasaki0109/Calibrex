from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from calibrex.core.external_run import (
    ExternalArtifactDigest,
    ExternalCalibrationRunArtifact,
    ExternalDataIsolation,
    ExternalExecution,
    ExternalLicenseBoundary,
    ExternalParsedOutputs,
    ExternalRunProvenance,
    ExternalToolIdentity,
    ExternalTransformOutput,
    external_run_json_schema,
    load_external_run,
)


def _external_run(
    *,
    license_spdx: str | None = "MIT",
    license_boundary: ExternalLicenseBoundary = "imported",
) -> ExternalCalibrationRunArtifact:
    return ExternalCalibrationRunArtifact(
        run_id="external-fixture",
        adapter_name="fixture_adapter",
        adapter_version="calibrex.fixture_adapter/v0.1",
        tool=ExternalToolIdentity(
            name="fixture-tool",
            version="1.2.3",
            source_repository="https://example.test/fixture",
            source_commit="0123456789abcdef",
            license_spdx=license_spdx,
            license_boundary=license_boundary,
        ),
        execution=ExternalExecution(mode="imported"),
        artifacts=[
            ExternalArtifactDigest(
                role="output",
                path="external-result.yaml",
                sha256="a" * 64,
                size_bytes=123,
                media_type="application/yaml",
            )
        ],
        frame_convention="T_parent_child",
        time_convention="seconds; positive sensor timestamp minus reference timestamp",
        train_data_isolation=ExternalDataIsolation(
            declared=True,
            evidence="fixture was fitted without Calibrex holdout IDs",
        ),
        status="success",
        parsed_outputs=ExternalParsedOutputs(
            transforms={
                "T_camera0_lidar0": ExternalTransformOutput(
                    parent="camera0",
                    child="lidar0",
                    translation_m=[1.0, 2.0, 3.0],
                    rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                )
            },
            external_metrics={"external_fitness": 0.99},
        ),
        provenance=ExternalRunProvenance(calibrex_version="test"),
    )


def test_external_run_is_schema_valid_and_round_trips(tmp_path: Path) -> None:
    artifact = _external_run()
    output = tmp_path / "external-run.json"

    artifact.save(output)
    loaded = load_external_run(output)

    assert loaded == artifact
    assert loaded.parsed_outputs.external_metrics_comparable is False
    jsonschema.validate(loaded.model_dump(mode="json"), external_run_json_schema())


def test_external_run_converts_to_legacy_adapter_result_with_lineage() -> None:
    artifact = _external_run()

    converted = artifact.to_solver_adapter_result()
    provenance = artifact.transform_provenance("T_camera0_lidar0")

    assert converted.backend == "fixture_adapter"
    assert converted.status == "success"
    assert converted.transforms["T_camera0_lidar0"].translation_m == (1.0, 2.0, 3.0)
    assert converted.provenance["external_calibration_run_schema_version"] == (
        "slac.external_calibration_run/v0.1"
    )
    assert provenance.tool_name == "fixture-tool"
    assert provenance.license_spdx == "MIT"
    assert "output_sha256=" + ("a" * 64) in provenance.notes


def test_external_run_rejects_gpl_in_process_boundary() -> None:
    with pytest.raises(ValidationError, match="GPL external tools must use"):
        _external_run(
            license_spdx="GPL-3.0-only",
            license_boundary="in_process",
        )


def test_external_run_requires_container_digest() -> None:
    with pytest.raises(ValidationError, match="container_digest"):
        ExternalExecution(mode="container")


def test_external_run_allows_unknown_license_at_import_boundary() -> None:
    artifact = _external_run(license_spdx=None, license_boundary="imported")

    assert artifact.tool.license_spdx is None
    assert "external tool license is unknown" in artifact.warnings
