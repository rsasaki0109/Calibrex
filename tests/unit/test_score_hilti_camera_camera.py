"""Units, trials, gate boundaries and verdicts of the Hilti camera-camera scorer."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scorer = _load("score_hilti_camera_camera")
PAIRS = scorer.GATED_PAIRS + scorer.REPORTED_PAIRS


def _record(left: str, right: str, status: str, errors: dict[str, float]) -> dict[str, Any]:
    return {
        "pair": "camera-camera",
        "frames": [scorer.frame(left), scorer.frame(right)],
        "status": status,
        "axes": [{"name": name, "candidate_error": value} for name, value in errors.items()],
    }


def _artifact(
    errors: dict[str, dict[str, float]] | None = None,
    statuses: dict[str, str] | None = None,
    attributed: str | None = None,
) -> dict[str, Any]:
    errors = errors or {}
    statuses = statuses or {}
    pairs = []
    for left, right in PAIRS:
        key = f"{left}-{right}"
        status = statuses.get(key, "pass")
        values = {} if status in ("inconclusive", "skipped") else errors.get(key, {"roll": 0.1})
        pairs.append(_record(left, right, status, values))
    group = {
        "frames": [scorer.frame(f"cam{k}") for k in range(5)],
        "status": "attributed" if attributed else "consistent",
        "frame": scorer.frame(attributed) if attributed else None,
    }
    return {"pairs": pairs, "rotation_attribution": {"groups": [group]}}


def _write(directory: Path, runs: dict[str, dict[str, Any] | None]) -> None:
    directory.mkdir(parents=True)
    manifest = {"runs": {}}
    for name, artifact in runs.items():
        manifest["runs"][name] = {"exit_code": 0 if artifact is not None else 3}
        if artifact is not None:
            (directory / f"{name}.json").write_text(json.dumps(artifact), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _injected(camera: str) -> dict[str, Any]:
    statuses = {f"{a}-{b}": "fail" for a, b in scorer.GATED_PAIRS if camera in (a, b)}
    return _artifact(statuses=statuses, attributed=camera)


def _good_recording(directory: Path, **baseline: Any) -> None:
    _write(
        directory,
        {
            "baseline": _artifact(**baseline),
            "inject_cam3_y+1": _injected("cam3"),
            "inject_cam1_y+1": _injected("cam1"),
        },
    )


def test_a_clean_set_of_recordings_is_supported(tmp_path: Path) -> None:
    for recording in ("a", "b", "c"):
        _good_recording(tmp_path / recording)
    result = scorer.score(tmp_path, ["a", "b", "c"])
    assert result["verdict"] == "supported"
    assert result["summary"]["scored_units"] == 15
    assert result["summary"]["detection_rate"] == 1.0
    assert result["summary"]["localization_rate"] == 1.0
    assert {item["status"] for item in result["requirements"].values()} == {"pass"}


@pytest.mark.parametrize(("error", "status"), [(0.75, "pass"), (0.7501, "fail")])
def test_the_accuracy_gate_holds_at_its_threshold(
    tmp_path: Path, error: float, status: str
) -> None:
    _good_recording(tmp_path / "a", errors={"cam0-cam3": {"pitch": -error, "yaw": 0.1}})
    result = scorer.score(tmp_path, ["a"])
    assert result["requirements"]["accuracy-vs-kalibr"]["status"] == status
    assert result["verdict"] == ("supported" if status == "pass" else "refuted")
    assert result["units"]["a"]["cam0-cam3"]["max_abs_error_deg"] == pytest.approx(error)


def test_a_reported_pair_is_never_gated(tmp_path: Path) -> None:
    _good_recording(tmp_path / "a", errors={"cam2-cam3": {"roll": 5.0}})
    result = scorer.score(tmp_path, ["a"])
    assert result["verdict"] == "supported"
    assert result["reported_units"]["a"]["cam2-cam3"]["max_abs_error_deg"] == 5.0


def test_a_run_that_crashed_fails_its_units(tmp_path: Path) -> None:
    _good_recording(tmp_path / "a")
    _write(tmp_path / "b", {"baseline": None})
    result = scorer.score(tmp_path, ["a", "b"])
    assert result["summary"]["failed_units"] == 5
    assert result["requirements"]["no-failed-units"]["status"] == "fail"
    assert result["verdict"] == "refuted"
    # its trials are not evaluable: nothing was scored on its baseline
    assert not result["trials"]["b"]["inject_cam3_y+1"]["evaluable"]


def test_too_few_scored_units_is_inconclusive(tmp_path: Path) -> None:
    inconclusive = dict.fromkeys(("cam0-cam3", "cam1-cam3", "cam0-cam4"), "inconclusive")
    _good_recording(tmp_path / "a", statuses=inconclusive)
    result = scorer.score(tmp_path, ["a"])
    assert result["summary"]["scored_units"] == 2
    assert result["requirements"]["scored-units-coverage"]["status"] == "fail"
    assert result["verdict"] == "inconclusive"


def test_a_missed_turn_refutes_detection(tmp_path: Path) -> None:
    _write(
        tmp_path / "a",
        {
            "baseline": _artifact(),
            "inject_cam3_y+1": _artifact(attributed="cam3"),  # every pair still passes
            "inject_cam1_y+1": _injected("cam1"),
        },
    )
    result = scorer.score(tmp_path, ["a"])
    trial = result["trials"]["a"]["inject_cam3_y+1"]
    assert trial["evaluable"] and not trial["detected"] and trial["localized"]
    assert result["requirements"]["detection"]["status"] == "fail"
    assert result["verdict"] == "refuted"


def test_localization_needs_the_turned_camera_named(tmp_path: Path) -> None:
    for recording in ("a", "b", "c", "d", "e"):
        runs = {
            "baseline": _artifact(),
            "inject_cam3_y+1": _injected("cam3"),
            "inject_cam1_y+1": _injected("cam1"),
        }
        if recording in ("a", "b"):  # named the wrong camera, or nothing
            runs["inject_cam1_y+1"] = _artifact(
                statuses={"cam0-cam1": "fail"}, attributed="cam0" if recording == "a" else None
            )
        _write(tmp_path / recording, runs)
    result = scorer.score(tmp_path, ["a", "b", "c", "d", "e"])
    assert result["summary"]["localization_rate"] == pytest.approx(0.8)
    assert result["requirements"]["localization"]["status"] == "pass"
    assert result["requirements"]["detection"]["status"] == "pass"


def test_a_turn_with_no_scored_pair_is_not_evaluable(tmp_path: Path) -> None:
    statuses = dict.fromkeys(("cam0-cam3", "cam1-cam3"), "inconclusive")
    _good_recording(tmp_path / "a", statuses=statuses)
    trial = scorer.score(tmp_path, ["a"])["trials"]["a"]["inject_cam3_y+1"]
    assert not trial["evaluable"]
    assert trial["detected"] is None


builder = _load("build_hilti_camera_camera_audit")


@pytest.mark.parametrize(
    ("case", "verdict"),
    [
        ("clean", "supported"),
        ("inaccurate", "refuted"),
        ("crashed", "refuted"),
        ("missed", "refuted"),
        ("thin", "incomplete"),
    ],
)
def test_the_builder_s_audit_agrees_with_the_scorer(
    tmp_path: Path, case: str, verdict: str
) -> None:
    runs = tmp_path / "runs"
    recordings = ["a", "b", "c"]
    for recording in recordings:
        _good_recording(runs / recording)
    if case == "inaccurate":
        (runs / "b" / "baseline.json").write_text(
            json.dumps(_artifact(errors={"cam0-cam1": {"roll": 0.9}})), encoding="utf-8"
        )
    elif case == "crashed":
        (runs / "c" / "baseline.json").unlink()
    elif case == "missed":
        (runs / "a" / "inject_cam1_y+1.json").write_text(
            json.dumps(_artifact(attributed="cam1")), encoding="utf-8"
        )
    elif case == "thin":
        thin = dict.fromkeys(("cam0-cam3", "cam1-cam3", "cam0-cam4", "cam1-cam4"), "inconclusive")
        for recording in recordings:
            (runs / recording / "baseline.json").write_text(
                json.dumps(_artifact(statuses=thin)), encoding="utf-8"
            )
    scores = scorer.score(runs, recordings)
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    preregistration = tmp_path / "prereg.yaml"
    preregistration.write_text("claim: test\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    result = builder.build(scores, scores_path, preregistration, out)
    assert result.verdict == verdict
    expected = {"supported": "supported", "refuted": "refuted", "incomplete": "inconclusive"}
    assert scores["verdict"] == expected[verdict]
