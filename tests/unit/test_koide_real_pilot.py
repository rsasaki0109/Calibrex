from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from calibrex.cli.main import main
from calibrex.core.external_run import (
    ExternalCalibrationRunArtifact,
    ExternalDataIsolation,
    ExternalExecution,
    ExternalRunProvenance,
    ExternalToolIdentity,
)
from calibrex.core.koide_real_pilot import (
    KoideExecutionStageRequest,
    KoideRealPilotFinalization,
    KoideRealPilotProvenance,
    KoideRealPilotRequest,
    KoideRealPilotVerification,
    KoideSplitManifest,
    _verify_external_run,
    build_koide_real_pilot_request,
    finalize_koide_real_pilot,
    koide_real_pilot_finalization_json_schema,
    koide_real_pilot_request_json_schema,
    koide_real_pilot_verification_json_schema,
    load_koide_real_pilot_request,
    load_koide_real_pilot_verification,
)
from calibrex.evaluation.koide_pilot import (
    KoidePilotArtifact,
    KoidePilotExecutionManifest,
    KoidePilotKnownBadControl,
    KoidePilotMetricSet,
    KoidePilotProvenance,
    KoidePilotReferenceDelta,
)


def _request() -> KoideRealPilotRequest:
    return build_koide_real_pilot_request(command=["pytest", "koide-real"])


def _test_verification(request: KoideRealPilotRequest) -> KoideRealPilotVerification:
    all_gates: dict[str, Any] = {
        "dataset_verified": True,
        "dataset_archive_verified": True,
        "split_manifest_verified": True,
        "config_verified": True,
        "native_output_verified": True,
        "external_run_verified": True,
        "readiness_verified": True,
        "execution_verified": True,
        "execution_evidence_verified": True,
        "execution_log_verified": True,
        "environment_verified": True,
        "known_bad_protocol_verified": True,
        "kpi_protocol_verified": True,
    }
    return KoideRealPilotVerification(
        request_id=request.request_id,
        request_sha256=request.request_sha256,
        status="READY_FOR_PILOT",
        evidence_label="test",
        official_evidence=False,
        real_data_observed=False,
        **all_gates,
        native_output_sha256="a" * 64,
        native_output_size_bytes=1,
        external_run_sha256="b" * 64,
        external_run_size_bytes=1,
        input_manifest_sha256="c" * 64,
        config_sha256="d" * 64,
        readiness_sha256="e" * 64,
        execution_log_sha256="f" * 64,
        environment_sha256="1" * 64,
        execution_command_sha256="2" * 64,
        source_commit=request.execution.source_commit,
        container_digest=request.execution.image_digest,
        reasons=["test-labelled fake external evidence; never an official result"],
        provenance=KoideRealPilotProvenance(
            request_sha256=request.request_sha256,
            command=["pytest", "fake-external-evidence"],
        ),
        artifact_sha256="0" * 64,
    ).with_artifact_digest()


def _fake_pilot(
    request: KoideRealPilotRequest,
    verification: KoideRealPilotVerification,
    *,
    passed: bool,
) -> KoidePilotArtifact:
    metric_value = 0.9 if passed else 0.0
    metrics = KoidePilotMetricSet(
        candidate_id="fake-koide-candidate",
        train_frame_ids=list(request.split.fitting_frame_ids),
        holdout_frame_ids=list(request.split.holdout_frame_ids),
        train_frame_count=len(request.split.fitting_frame_ids),
        holdout_frame_count=len(request.split.holdout_frame_ids),
        scored_frame_count=len(request.split.frame_ids),
        train_projection_ratio=metric_value,
        holdout_projection_ratio=metric_value,
        train_edge_alignment=metric_value,
        holdout_edge_alignment=metric_value,
        train_depth_edge_alignment=metric_value,
        holdout_depth_edge_alignment=metric_value,
        training_isolation_declared=True,
    )
    controls = [
        KoidePilotKnownBadControl(
            control_id=control.control_id,
            dof=control.dof,
            sign=control.sign,
            amount=control.amount,
            unit=control.unit,
            detected=passed,
            status="pass" if passed else "fail",
            reason="test control" if passed else "test control failure",
        )
        for control in request.known_bad_controls
    ]
    return KoidePilotArtifact(
        pilot_id="fake-koide-pilot",
        status="PASS" if passed else "FAIL",
        adoption_decision="ADOPT" if passed else "DO_NOT_ADOPT",
        admissible=passed,
        reason="test-only fake pilot",
        dataset_path="synthetic/test-sequence",
        camera_frame="camera0",
        lidar_frame="lidar0",
        selected_frame_ids=list(request.split.frame_ids),
        train_frame_ids=list(request.split.fitting_frame_ids),
        holdout_frame_ids=list(request.split.holdout_frame_ids),
        thresholds=request.acceptance.thresholds,
        readiness_status="ready",
        external_run_status="success",
        candidate_metrics=metrics,
        candidate_to_reference=KoidePilotReferenceDelta(
            declared=True,
            translation_delta_m=0.0 if passed else 1.0,
            rotation_delta_deg=0.0 if passed else 10.0,
            reason="test reference",
        ),
        known_bad_controls=controls,
        known_bad_detected_count=12 if passed else 0,
        known_bad_detectable_fraction=1.0 if passed else 0.0,
        execution=KoidePilotExecutionManifest(
            command=["docker", "run", "fake"],
            status="attempted",
            attempted=True,
            reason="test-only external execution record",
            docker_available=True,
            engine="docker",
            source_commit=request.execution.source_commit,
            container_digest=request.execution.image_digest,
        ),
        provenance=KoidePilotProvenance(
            dataset_path="synthetic/test-sequence",
            candidate_output_sha256=verification.native_output_sha256,
            input_manifest_sha256=verification.input_manifest_sha256,
        ),
        artifact_sha256="0" * 64,
    ).with_artifact_digest()


def test_blocked_request_and_verify_missing_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path = tmp_path / "request.yaml"
    verification_path = tmp_path / "verification.yaml"
    request = _request()
    request.save(request_path)

    assert (
        main(
            [
                "camera-lidar",
                "koide-real-verify",
                str(request_path),
                "--output",
                str(verification_path),
                "--json",
            ]
        )
        == 1
    )
    output = json.loads(capsys.readouterr().out)
    verification = load_koide_real_pilot_verification(verification_path)
    assert output["status"] == "BLOCKED"
    assert verification.status == "BLOCKED"
    assert verification.official_evidence is False
    assert any("missing" in reason for reason in verification.reasons)


def test_request_tamper_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "request.yaml"
    _request().save(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["request_id"] = "tampered"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="self-digest mismatch"):
        load_koide_real_pilot_request(path)


def test_split_rejects_fitting_holdout_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        KoideSplitManifest(
            frame_ids=["0000000000", "0000000001"],
            fitting_frame_ids=["0000000000"],
            holdout_frame_ids=["0000000000", "0000000001"],
        )


def test_superglue_is_rejected_at_commercial_execution_boundary() -> None:
    with pytest.raises(ValueError, match="SuperGlue"):
        KoideExecutionStageRequest(
            name="initial_guess",
            argv=["find_matches_superglue.py"],
            official_documentation_url="https://example.invalid/test",
        )


def test_source_and_container_mismatch_are_blockers() -> None:
    request = _request()
    external = ExternalCalibrationRunArtifact(
        run_id="mismatch",
        adapter_name="test",
        adapter_version="test",
        tool=ExternalToolIdentity(
            name="direct_visual_lidar_calibration",
            source_repository=request.execution.source_repository,
            source_commit="0" * 40,
            license_spdx="MIT",
            license_boundary="container",
        ),
        execution=ExternalExecution(
            mode="container",
            command=["docker"],
            container_digest="other/image@sha256:" + "0" * 64,
            attempted=True,
            network_mode="none",
        ),
        train_data_isolation=ExternalDataIsolation(
            declared=True,
            evidence="test split",
            training_data_ids_sha256="0" * 64,
            holdout_data_ids_sha256="0" * 64,
        ),
        status="success",
        frame_convention="test",
        time_convention="test",
        provenance=ExternalRunProvenance(calibrex_version="test"),
    )
    reasons = _verify_external_run(
        external,
        external_path=Path("external.yaml"),
        request=request,
        candidate_path=None,
        native_sha=None,
        manifest=None,
        evidence_label="test",
    )
    assert any("source commit" in reason for reason in reasons)
    assert any("container digest" in reason for reason in reasons)


def test_fully_supplied_fake_evidence_finalizes_test_only(tmp_path: Path) -> None:
    request_path = tmp_path / "request.yaml"
    verification_path = tmp_path / "verification.yaml"
    pilot_path = tmp_path / "pilot.yaml"
    finalization_path = tmp_path / "finalization.yaml"
    request = _request()
    request.save(request_path)
    verification = _test_verification(request)
    verification.save(verification_path)
    _fake_pilot(request, verification, passed=True).save(pilot_path)

    finalization = finalize_koide_real_pilot(
        request_path,
        verification_path,
        pilot_path,
        output_path=finalization_path,
        evidence_label="test",
        command=["pytest", "fake-finalize"],
    )
    assert finalization.status == "TEST_ONLY"
    assert finalization.execution_state == "EXECUTED"
    assert finalization.official_result_claim is False
    assert finalization.holdout_kpis_verified is True
    assert finalization.known_bad_controls_verified is True
    assert KoideRealPilotFinalization.model_validate(
        yaml.safe_load(finalization_path.read_text(encoding="utf-8"))
    )


def test_known_bad_and_kpi_failure_cannot_finalize(tmp_path: Path) -> None:
    request_path = tmp_path / "request.yaml"
    verification_path = tmp_path / "verification.yaml"
    pilot_path = tmp_path / "pilot.yaml"
    request = _request()
    request.save(request_path)
    verification = _test_verification(request)
    verification.save(verification_path)
    _fake_pilot(request, verification, passed=False).save(pilot_path)

    finalization = finalize_koide_real_pilot(
        request_path,
        verification_path,
        pilot_path,
        evidence_label="test",
    )
    assert finalization.status == "FAIL"
    assert finalization.holdout_kpis_verified is False
    assert finalization.known_bad_controls_verified is False
    assert "KPI failure" in finalization.reasons[0]
    assert "known-bad/KPI failure" in finalization.reasons[0]


@pytest.mark.parametrize(
    ("filename", "schema"),
    [
        ("koide_real_pilot_request.schema.json", koide_real_pilot_request_json_schema),
        ("koide_real_pilot_verification.schema.json", koide_real_pilot_verification_json_schema),
        ("koide_real_pilot_finalization.schema.json", koide_real_pilot_finalization_json_schema),
    ],
)
def test_static_real_koide_schemas_are_exact(
    filename: str, schema: Callable[[], dict[str, Any]]
) -> None:
    checked_in = json.loads(Path("schemas", filename).read_text(encoding="utf-8"))
    assert checked_in == schema()
    jsonschema.Draft202012Validator.check_schema(checked_in)
