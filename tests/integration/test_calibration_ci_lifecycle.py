"""Calibration CI decisions recorded in the append-only lifecycle registry.

The candidates are real: the bundled KITTI raw camera-LiDAR evidence demo
passes its falsification policy, the precomputed example result is
INCONCLUSIVE (no recomputed raw observations), and a tightened policy (known-bad
controls must pass 90 % instead of 50 %) fails the same KITTI candidate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from calibrex.calibration_ci_lifecycle import record_calibration_ci
from calibrex.cli.main import main
from calibrex.core.capture_manifest import SensorIdentity, inspect_capture
from calibrex.core.exceptions import CalibrexError
from calibrex.core.lifecycle_registry import (
    init_registry,
    load_registry,
    register_calibration_edge,
    register_sensor,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.validation import validate_file

INCONCLUSIVE_RESULT = Path("examples/precomputed/result.yaml")


@pytest.fixture(scope="module")
def kitti_demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("kitti_demo")
    assert (
        main(
            [
                "demo",
                "kitti-lidar-camera-evidence",
                "--output-dir",
                str(output),
                "--strict-assessment",
                "--json",
            ]
        )
        == 0
    )
    return output


def _ci(candidate: Path, output_dir: Path, *, policy: Path | None = None) -> Path:
    arguments = ["ci", str(candidate), "--output-dir", str(output_dir), "--json"]
    if policy is not None:
        arguments += ["--policy", str(policy)]
    assert main(arguments) == 0
    return output_dir / "calibration-ci.json"


def _strict_policy(kitti_demo: Path, path: Path) -> Path:
    payload = json.loads((kitti_demo / "policy.json").read_text(encoding="utf-8"))
    parameters = payload["policy"]["parameters"]
    parameters["min_known_bad_pass_fraction"] = 0.9
    parameters["min_supported_known_bad_pass_fraction"] = 0.9
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _registry(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    root = tmp_path / "registry"
    init_registry(root, registry_id="fleet", operator="ci", timestamp="2026-10-11T00:00:00Z")
    register_sensor(
        root,
        sensor_id="camera-0",
        vehicle_id="car-1",
        sensor_kit_id="kit-1",
        serial="cam-serial",
        model="kitti-camera",
        firmware="1.0",
        mount="front",
        install=True,
        operator="ci",
        timestamp="2026-10-11T00:00:01Z",
    )
    incumbent = tmp_path / "incumbent.yaml"
    shutil.copyfile(INCONCLUSIVE_RESULT, incumbent)
    register_calibration_edge(
        root,
        edge_id="camera0-lidar0",
        vehicle_id="car-1",
        sensor_kit_id="kit-1",
        parent_frame="camera0",
        child_frame="lidar0",
        incumbent_result=incumbent,
        operator="ci",
        timestamp="2026-10-11T00:00:02Z",
    )
    source = tmp_path / "capture.bin"
    source.write_bytes(b"capture")
    capture = tmp_path / "capture.yaml"
    inspect_capture(
        source,
        source_format="files",
        capture_id="capture-1",
        session_id="session-1",
        vehicle_id="car-1",
        sensor_kit_id="kit-1",
        sensors=[
            SensorIdentity(
                sensor_id="camera-0",
                sensor_type="camera",
                serial="cam-serial",
                model="kitti-camera",
                firmware="1.0",
                mount="front",
                frame_id="camera0",
            )
        ],
    ).model_copy(update={"status": "ready"}).with_artifact_digest().save(capture)
    promotion = tmp_path / "promotion.json"
    promotion.write_text(json.dumps({"status": "PASS", "decision": "ADOPT"}), encoding="utf-8")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        json.dumps({"status": "PASS", "admission_label": "production-admissible"}),
        encoding="utf-8",
    )
    return root, capture, {"promotion": promotion, "smoke": smoke, "incumbent": incumbent}


def _incumbent_sha(root: Path) -> str | None:
    ref = load_registry(root).state().edges["camera0-lidar0"].incumbent_result
    return None if ref is None else ref.sha256


def test_ci_decisions_hold_promote_reject_and_roll_back_an_edge(
    tmp_path: Path, kitti_demo: Path
) -> None:
    root, capture, extra = _registry(tmp_path)
    initial = _incumbent_sha(root)
    assert initial == sha256_path(extra["incumbent"])
    candidate = kitti_demo / "result.yaml"

    # A weakly observable (INCONCLUSIVE) candidate is held and never moves the head,
    # even when promotion is requested and promotion/smoke evidence passes.
    weak = record_calibration_ci(
        _ci(INCONCLUSIVE_RESULT, tmp_path / "ci-weak"),
        root,
        edge_id="camera0-lidar0",
        capture_manifest=capture,
        promotion=extra["promotion"],
        smoke=extra["smoke"],
        promote=True,
        operator="ci",
    )
    assert weak.ci_status == "inconclusive"
    assert (weak.action, weak.admission) == ("hold", "HOLD")
    assert not weak.incumbent_changed and _incumbent_sha(root) == initial
    assert [event.event_type for event in weak.events] == ["hold"]
    assert weak.registry_valid

    # A passing candidate with passing promotion and smoke evidence is promoted.
    passing_decision = _ci(candidate, tmp_path / "ci-pass")
    adopted = record_calibration_ci(
        passing_decision,
        root,
        edge_id="camera0-lidar0",
        capture_manifest=capture,
        promotion=extra["promotion"],
        smoke=extra["smoke"],
        promote=True,
        operator="ci",
    )
    assert (adopted.ci_status, adopted.action, adopted.admission) == ("pass", "promote", "ADOPT")
    assert [event.event_type for event in adopted.events] == ["evaluate", "promote"]
    assert adopted.incumbent_changed
    assert _incumbent_sha(root) == sha256_path(candidate)
    assert adopted.registry_valid
    # The evaluation event references the CI decision by digest, in a snapshot the
    # registry keeps.
    evaluate_event = load_registry(root).events()[adopted.events[0].sequence]
    assert evaluate_event.artifacts.evidence is not None
    assert evaluate_event.artifacts.evidence.sha256 == sha256_path(passing_decision)
    assert evaluate_event.artifacts.evidence.path.startswith("ci/")

    # A tightened policy fails the same candidate. Without --rollback-on-fail the
    # decision is only rejected; the incumbent stays.
    strict = _strict_policy(kitti_demo, tmp_path / "strict-policy.json")
    failing = _ci(candidate, tmp_path / "ci-fail", policy=strict)
    rejected = record_calibration_ci(
        failing, root, edge_id="camera0-lidar0", capture_manifest=capture, operator="ci"
    )
    assert (rejected.ci_status, rejected.action, rejected.admission) == (
        "fail",
        "reject",
        "DO_NOT_ADOPT",
    )
    assert not rejected.incumbent_changed and _incumbent_sha(root) == sha256_path(candidate)

    # With --rollback-on-fail a failing decision about the promoted incumbent rolls
    # the edge back to the incumbent before that promotion, bound to the decision.
    failing_again = _ci(candidate, tmp_path / "ci-fail-again", policy=strict)
    rolled = record_calibration_ci(
        failing_again,
        root,
        edge_id="camera0-lidar0",
        capture_manifest=capture,
        rollback_on_fail=True,
        operator="ci",
    )
    assert rolled.action == "rollback"
    assert [event.event_type for event in rolled.events] == ["rollback"]
    assert _incumbent_sha(root) == initial
    rollback_event = load_registry(root).events()[-1]
    assert rollback_event.artifacts.evidence is not None
    assert rollback_event.artifacts.evidence.sha256 == sha256_path(failing_again)
    assert rolled.registry_valid and load_registry(root).verify().valid


def test_a_decision_is_recorded_once_and_snapshots_are_never_overwritten(
    tmp_path: Path, kitti_demo: Path
) -> None:
    root, capture, _ = _registry(tmp_path)
    decision = _ci(kitti_demo / "result.yaml", tmp_path / "ci")
    record_calibration_ci(decision, root, edge_id="camera0-lidar0", capture_manifest=capture)
    events = len(load_registry(root).events())
    with pytest.raises(CalibrexError, match="already recorded"):
        record_calibration_ci(decision, root, edge_id="camera0-lidar0", capture_manifest=capture)
    assert len(load_registry(root).events()) == events

    snapshot = root / "ci" / str(sha256_path(decision))
    (snapshot / "evaluation.json").unlink()
    (snapshot / "assessment.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CalibrexError, match="refusing to overwrite"):
        record_calibration_ci(decision, root, edge_id="camera0-lidar0", capture_manifest=capture)
    # The tampered snapshot is caught by the registry's own verification.
    assert not load_registry(root).verify().valid


def test_a_candidate_changed_after_the_decision_is_refused(
    tmp_path: Path, kitti_demo: Path
) -> None:
    root, capture, _ = _registry(tmp_path)
    candidate = tmp_path / "candidate.yaml"
    shutil.copyfile(kitti_demo / "result.yaml", candidate)
    decision = _ci(candidate, tmp_path / "ci")
    candidate.write_text(candidate.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    with pytest.raises(CalibrexError, match="changed after the CI decision"):
        record_calibration_ci(decision, root, edge_id="camera0-lidar0", capture_manifest=capture)
    assert all(event.event_type != "evaluate" for event in load_registry(root).events())


def test_cli_record_ci_writes_a_schema_valid_record(
    tmp_path: Path, kitti_demo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, capture, extra = _registry(tmp_path)
    decision = _ci(kitti_demo / "result.yaml", tmp_path / "ci")
    capsys.readouterr()
    output = tmp_path / "record.json"
    assert (
        main(
            [
                "lifecycle",
                "record-ci",
                str(decision),
                "--registry-root",
                str(root),
                "--edge-id",
                "camera0-lidar0",
                "--capture-manifest",
                str(capture),
                "--promotion",
                str(extra["promotion"]),
                "--smoke",
                str(extra["smoke"]),
                "--promote",
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "slac.calibration_ci_lifecycle/v0.1"
    assert payload["action"] == "promote"
    assert payload["provenance"]["command"][:3] == ["calibrex", "lifecycle", "record-ci"]
    assert validate_file(output, kind="calibration-ci-lifecycle").valid
    assert main(["lifecycle", "verify", "--registry-root", str(root), "--json"]) == 0


def test_example_workflow_commands_parse() -> None:
    import shlex

    import yaml

    from calibrex.cli.main import _build_parser

    workflow = yaml.safe_load(
        Path("examples/ci/calibration-ci-lifecycle.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["record"]["steps"]
    record = next(step for step in steps if step.get("name", "").startswith("Record"))
    script = record["run"].replace("\\\n", " ")
    command_line = next(line for line in script.splitlines() if "calibrex lifecycle" in line)
    parser = _build_parser()
    for mode in ("--promote", "--rollback-on-fail"):
        line = command_line.replace("$mode", mode).replace("${{ github.run_id }}", "1")
        arguments = shlex.split(line)[1:]
        parsed = parser.parse_args(arguments)
        assert parsed.edge_id == "lidar-front-camera"
        assert parsed.promote is (mode == "--promote")
        assert parsed.rollback_on_fail is (mode == "--rollback-on-fail")
    verify = next(step for step in steps if step.get("name", "").startswith("Verify"))
    parser.parse_args(shlex.split(verify["run"])[1:])
