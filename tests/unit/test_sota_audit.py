from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from calibrex.cli.main import main
from calibrex.core.camera_lidar_sota_audit import (
    CameraLidarClaimRequirement,
    CameraLidarSotaAuditProtocol,
    CameraLidarSotaAuditProvenance,
    CameraLidarSotaCategory,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.sota_audit import (
    SotaAuditProtocol,
    SotaAuditProvenance,
    SotaAuditResult,
    SotaClaimRequirement,
    SotaClaimScope,
    load_sota_audit_result,
)
from calibrex.core.sota_leaderboard import SotaLeaderboard, load_sota_leaderboard
from calibrex.core.validation import validate_file
from calibrex.evaluation.camera_lidar_sota_audit import audit_camera_lidar_sota_claim
from calibrex.evaluation.sota_audit import audit_sota_claim
from calibrex.evaluation.sota_leaderboard import (
    build_sota_leaderboard,
    render_sota_leaderboard_markdown,
)

_FIXTURES = Path(__file__).with_name("test_camera_lidar_sota_audit.py")
_SPEC = importlib.util.spec_from_file_location("camera_lidar_sota_audit_fixtures", _FIXTURES)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_benchmark = _MODULE._benchmark


def _evidence(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "benchmark.yaml"
    _benchmark().save(path)
    digest = sha256_path(path)
    assert digest is not None
    return path, digest


def _requirements(kind: str, path: Path, digest: str) -> list[dict[str, Any]]:
    """Gates over the fixture benchmark, whose candidate hit mean is 0.5."""

    gate: dict[str, Any] = {
        "requirement_id": "hit-gate",
        "phase": "phase1",
        "description": "candidate hit rate",
        "dataset_family": "fixture",
        "independent_rig": True,
        "rig_id": "fixture-rig",
        "evidence_path": str(path),
        "evidence_sha256": digest,
        "method_id": "candidate",
        "metric": "hit",
        "statistic": "metric_mean",
        "comparison": "greater_equal",
        "threshold": {"supported": 0.4, "refuted": 1.0}.get(kind, 0.4),
    }
    requirements = [gate]
    if kind == "incomplete":
        requirements.append(
            {
                "requirement_id": "field-trial",
                "phase": "phase2",
                "description": "field trial evidence not yet collected",
            }
        )
    return requirements


def _camera_lidar_protocol(
    kind: str,
    path: Path,
    digest: str,
    category: CameraLidarSotaCategory = "training_free_targetless",
) -> CameraLidarSotaAuditProtocol:
    return CameraLidarSotaAuditProtocol(
        protocol_id=f"fixture-{kind}",
        declared_category=category,
        claim_text=f"fixture {kind} claim",
        minimum_dataset_families=1,
        minimum_independent_rigs=1,
        requirements=[
            CameraLidarClaimRequirement.model_validate(item)
            for item in _requirements(kind, path, digest)
        ],
        provenance=CameraLidarSotaAuditProvenance(generator="pytest", generator_version="1"),
    )


def _generic_protocol(
    kind: str,
    path: Path,
    digest: str,
    *,
    modalities: list[str],
    protocol_id: str | None = None,
) -> SotaAuditProtocol:
    return SotaAuditProtocol.model_validate(
        {
            "protocol_id": protocol_id or f"generic-{kind}",
            "scope": {
                "modalities": modalities,
                "quantities": ["rotation", "translation"],
                "category": "trajectory_hand_eye",
            },
            "claim_text": f"generic {kind} claim",
            "minimum_dataset_families": 1,
            "minimum_independent_rigs": 1,
            "requirements": [
                SotaClaimRequirement.model_validate(item)
                for item in _requirements(kind, path, digest)
            ],
            "provenance": SotaAuditProvenance(generator="pytest", generator_version="1"),
        }
    )


@pytest.mark.parametrize("kind", ["supported", "refuted", "incomplete"])
def test_generic_engine_reproduces_camera_lidar_verdicts(tmp_path: Path, kind: str) -> None:
    path, digest = _evidence(tmp_path)
    protocol = _camera_lidar_protocol(kind, path, digest)
    camera_lidar_path = tmp_path / "camera-lidar-protocol.yaml"
    generic_path = tmp_path / "generic-protocol.yaml"
    protocol.save(camera_lidar_path)
    protocol.as_generic().save(generic_path)

    camera_lidar = audit_camera_lidar_sota_claim(camera_lidar_path)
    generic = audit_sota_claim(generic_path)

    assert camera_lidar.verdict == generic.verdict == kind
    assert camera_lidar.as_generic().scope == generic.scope
    assert generic.scope.pair_id == "camera-lidar"
    assert [
        (item.requirement_id, item.status, item.observed_value)
        for item in camera_lidar.requirements
    ] == [(item.requirement_id, item.status, item.observed_value) for item in generic.requirements]
    assert camera_lidar.achieved_dataset_families == generic.achieved_dataset_families
    assert camera_lidar.achieved_independent_rig_count == generic.achieved_independent_rig_count


def test_spatiotemporal_camera_lidar_scope_includes_time_offset(tmp_path: Path) -> None:
    path, digest = _evidence(tmp_path)
    protocol = _camera_lidar_protocol("supported", path, digest, "spatiotemporal_targetless")

    assert protocol.as_generic().scope.quantities == ["rotation", "translation", "time_offset"]


def test_cli_audits_a_non_camera_pair(tmp_path: Path) -> None:
    path, digest = _evidence(tmp_path)
    protocol_path = tmp_path / "gnss-lidar-protocol.yaml"
    output = tmp_path / "gnss-lidar-audit.yaml"
    _generic_protocol("supported", path, digest, modalities=["lidar", "gnss"]).save(protocol_path)

    exit_code = main(["sota", "audit", str(protocol_path), "--output", str(output)])

    assert exit_code == 0
    report = validate_file(output)
    assert report.kind == "sota-audit-result"
    result = load_sota_audit_result(output)
    assert result.verdict == "supported"
    assert result.scope.pair_id == "gnss-lidar"


def test_cli_audit_exit_code_is_nonzero_for_refuted_claims(tmp_path: Path) -> None:
    path, digest = _evidence(tmp_path)
    protocol_path = tmp_path / "protocol.yaml"
    _generic_protocol("refuted", path, digest, modalities=["imu", "lidar"]).save(protocol_path)

    exit_code = main(
        ["sota", "audit", str(protocol_path), "--output", str(tmp_path / "audit.yaml")]
    )

    assert exit_code == 2


def test_scope_allows_same_type_pairs_and_single_sensor_claims() -> None:
    multi_lidar = SotaClaimScope(
        modalities=["lidar", "lidar"], quantities=["rotation", "translation"], category="x"
    )
    intrinsics = SotaClaimScope(modalities=["camera"], quantities=["intrinsics"], category="x")

    assert multi_lidar.pair_id == "lidar-lidar"
    assert intrinsics.pair_id == "camera"


def test_scope_rejects_duplicate_quantities_and_unknown_modalities() -> None:
    with pytest.raises(ValueError, match="quantities must be unique"):
        SotaClaimScope(modalities=["lidar"], quantities=["rotation", "rotation"], category="x")
    with pytest.raises(ValueError):
        SotaClaimScope.model_validate(
            {"modalities": ["sonar"], "quantities": ["rotation"], "category": "x"}
        )
    assert (
        SotaClaimScope(
            modalities=["vehicle", "lidar"], quantities=["rotation"], category="x"
        ).pair_id
        == "lidar-vehicle"
    )


def test_forged_supported_verdict_is_rejected(tmp_path: Path) -> None:
    path, digest = _evidence(tmp_path)
    protocol_path = tmp_path / "protocol.yaml"
    _generic_protocol("refuted", path, digest, modalities=["gnss", "imu"]).save(protocol_path)
    forged = audit_sota_claim(protocol_path).model_dump(mode="python")
    forged["verdict"] = "supported"

    with pytest.raises(ValueError, match="supported SOTA verdict"):
        SotaAuditResult.model_validate(forged)


def _audit_file(tmp_path: Path, name: str, protocol: Any) -> Path:
    protocol_path = tmp_path / f"{name}-protocol.yaml"
    output = tmp_path / f"{name}-audit.yaml"
    protocol.save(protocol_path)
    if isinstance(protocol, CameraLidarSotaAuditProtocol):
        audit_camera_lidar_sota_claim(protocol_path).save(output)
    else:
        audit_sota_claim(protocol_path).save(output)
    return output


def test_leaderboard_keeps_refutations_and_target_gaps_visible(tmp_path: Path) -> None:
    path, digest = _evidence(tmp_path)
    audits = [
        _audit_file(tmp_path, "cl", _camera_lidar_protocol("supported", path, digest)),
        _audit_file(
            tmp_path,
            "gl-good",
            _generic_protocol(
                "supported", path, digest, modalities=["gnss", "lidar"], protocol_id="gl-a"
            ),
        ),
        _audit_file(
            tmp_path,
            "gl-bad",
            _generic_protocol(
                "refuted", path, digest, modalities=["lidar", "gnss"], protocol_id="gl-b"
            ),
        ),
        _audit_file(
            tmp_path,
            "il",
            _generic_protocol("incomplete", path, digest, modalities=["imu", "lidar"]),
        ),
    ]

    leaderboard = build_sota_leaderboard(audits, target_pairs=["vehicle-lidar", "gnss-lidar"])

    standings = {item.pair_id: item for item in leaderboard.pairs}
    assert {pair: item.standing for pair, item in standings.items()} == {
        "camera-lidar": "supported",
        "gnss-lidar": "supported",
        "imu-lidar": "incomplete",
        "lidar-vehicle": "no_claim",
    }
    assert (standings["gnss-lidar"].supported_claims, standings["gnss-lidar"].refuted_claims) == (
        1,
        1,
    )
    assert standings["lidar-vehicle"].target and standings["gnss-lidar"].target
    assert not standings["camera-lidar"].target
    markdown = render_sota_leaderboard_markdown(leaderboard)
    assert "| `gnss-lidar` | supported | 1 | 1 | 0 | yes |" in markdown
    assert "| `lidar-vehicle` | no_claim | 0 | 0 | 0 | yes |" in markdown
    assert "`gl-b`" in markdown

    forged = leaderboard.model_dump(mode="python")
    forged["pairs"][1]["refuted_claims"] = 0
    with pytest.raises(ValueError, match="counts do not match"):
        SotaLeaderboard.model_validate(forged)


def test_leaderboard_rejects_unknown_target_pairs_and_non_audits(tmp_path: Path) -> None:
    path, _ = _evidence(tmp_path)

    with pytest.raises(ValueError, match="invalid target pair"):
        build_sota_leaderboard([], target_pairs=["lidar-sonar"])
    with pytest.raises(ValueError, match="not a SOTA audit result"):
        build_sota_leaderboard([path])


def test_cli_leaderboard_writes_schema_valid_artifact_and_markdown(tmp_path: Path) -> None:
    path, digest = _evidence(tmp_path)
    audit = _audit_file(
        tmp_path, "gl", _generic_protocol("supported", path, digest, modalities=["gnss", "lidar"])
    )
    output = tmp_path / "leaderboard.yaml"
    markdown = tmp_path / "leaderboard.md"

    exit_code = main(
        [
            "sota",
            "leaderboard",
            str(audit),
            "--target-pair",
            "imu-lidar",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ]
    )

    assert exit_code == 0
    assert validate_file(output).kind == "sota-leaderboard"
    assert [item.pair_id for item in load_sota_leaderboard(output).pairs] == [
        "gnss-lidar",
        "imu-lidar",
    ]
    assert "| `imu-lidar` | no_claim |" in markdown.read_text(encoding="utf-8")
