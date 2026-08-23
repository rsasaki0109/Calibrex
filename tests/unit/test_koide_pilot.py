from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.validation import validate_file
from calibrex.evaluation.koide_pilot import (
    KoidePilotThresholds,
    export_koide_pilot_autoware,
    load_koide_pilot,
    run_koide_pilot,
)
from calibrex.export.autoware import AutowareExportError
from calibrex.importers.koide import import_koide_result

FIXTURE = Path(
    "examples/public_datasets/kitti_lidar_camera_evidence/2011_09_26/2011_09_26_drive_0005_sync"
)
CONFIG = Path("examples/public_datasets/kitti_lidar_camera_evidence/config.yaml")


def _native(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "results": {
                    "T_lidar_camera": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
                }
            }
        ),
        encoding="utf-8",
    )


def _ready_readiness(tmp_path: Path) -> Path:
    from calibrex.core.koide_readiness import evaluate_koide_readiness_from_config

    artifact = evaluate_koide_readiness_from_config(CONFIG)
    ready = artifact.model_copy(update={"status": "ready"}).with_artifact_digest()
    output = tmp_path / "readiness.yaml"
    ready.save(output)
    return output


def test_a2d2_diagnostic_does_not_claim_unsupported_kitti_execution(tmp_path: Path) -> None:
    readiness = _ready_readiness(tmp_path)
    artifact = run_koide_pilot(
        tmp_path / "a2d2-diagnostic-only",
        None,
        output_directory=tmp_path,
        config_path=CONFIG,
        readiness_path=readiness,
        official_command=("docker", "run", "--rm", "koide@sha256:" + "a" * 64),
    )

    assert artifact.status == "BLOCKED"
    assert artifact.adoption_decision == "BLOCKED"
    assert artifact.execution.attempted is False
    assert artifact.execution.status in {"not_attempted", "unavailable"}
    assert artifact.execution.command[0:2] == ["docker", "run"]
    assert "A2D2 is unsupported" in artifact.reason
    assert "readiness diagnostic only" in artifact.reason
    assert "KITTI input manifest" in artifact.reason
    assert load_koide_pilot(tmp_path / "pilot.json").artifact_sha256 == artifact.artifact_sha256


def test_native_candidate_is_evaluated_but_never_adopted_without_provenance(
    tmp_path: Path,
) -> None:
    native = tmp_path / "calib.json"
    _native(native)
    readiness = _ready_readiness(tmp_path)

    artifact = run_koide_pilot(
        FIXTURE,
        native,
        output_directory=tmp_path / "pilot",
        config_path=CONFIG,
        readiness_path=readiness,
        thresholds=KoidePilotThresholds(
            max_frames=2,
            max_points=800,
            min_holdout_projection_ratio=0.0,
            min_holdout_edge_alignment=0.0,
            min_holdout_depth_edge_alignment=0.0,
        ),
    )

    assert artifact.external_run_status == "success"
    assert artifact.candidate_metrics is not None
    assert artifact.candidate_metrics.holdout_frame_ids
    assert len(artifact.known_bad_controls) == 12
    assert artifact.status == "WARN"
    assert artifact.adoption_decision == "DO_NOT_ADOPT"
    assert "provenance" in artifact.reason
    # This tiny fixture proves plumbing only; it is not an official Koide run.
    assert artifact.provenance.candidate_output_sha256


def test_pilot_self_digest_tampering_fails_loader_and_validation(tmp_path: Path) -> None:
    readiness = _ready_readiness(tmp_path)
    run_koide_pilot(
        FIXTURE,
        None,
        output_directory=tmp_path / "pilot",
        config_path=CONFIG,
        readiness_path=readiness,
    )
    pilot_path = tmp_path / "pilot" / "pilot.json"
    payload = read_mapping(pilot_path)
    payload["reason"] = "tampered"
    write_mapping(pilot_path, payload)

    with pytest.raises(ValueError, match="self-digest"):
        load_koide_pilot(pilot_path)
    with pytest.raises(ValueError, match="self-digest"):
        validate_file(pilot_path, "koide-pilot")
    from calibrex.cli.main import main

    assert main(["validate", str(pilot_path), "--kind", "koide-pilot", "--json"]) == 2


def test_autoware_export_rejects_blocked_pilot(tmp_path: Path) -> None:
    readiness = _ready_readiness(tmp_path)
    run_koide_pilot(
        FIXTURE,
        None,
        output_directory=tmp_path / "pilot",
        config_path=CONFIG,
        readiness_path=readiness,
    )

    with pytest.raises(AutowareExportError, match="not admissible"):
        export_koide_pilot_autoware(
            tmp_path / "pilot" / "pilot.json",
            tmp_path / "autoware.yaml",
            base_frame="camera0",
        )


def test_external_candidate_that_binds_holdout_is_blocked(tmp_path: Path) -> None:
    native = tmp_path / "calib.json"
    _native(native)
    readiness = _ready_readiness(tmp_path)
    external = import_koide_result(
        native,
        lidar_frame="lidar0",
        camera_frame="camera0",
        input_artifacts=(
            FIXTURE / "image_02" / "data" / "0000000000.png",
            FIXTURE / "velodyne_points" / "data" / "0000000000.bin",
        ),
        tool_version="fixture",
        source_commit="a" * 40,
    )
    external_path = tmp_path / "external-run.yaml"
    external.save(external_path)

    artifact = run_koide_pilot(
        FIXTURE,
        external_path,
        output_directory=tmp_path / "pilot",
        config_path=CONFIG,
        readiness_path=readiness,
        thresholds=KoidePilotThresholds(max_frames=2, max_points=800),
    )
    assert artifact.status == "BLOCKED"
    assert artifact.adoption_decision == "BLOCKED"
    assert "overlap holdout" in artifact.reason
