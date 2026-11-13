"""INS-LiDAR audit v2: what is gated, what a missing case means, and how a run resumes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tests.unit.test_boreas_ins_lidar_metadata import provenance, transport
from tests.unit.test_ins_lidar_sota_inventory import central, pool
from tools import build_ins_lidar_sota_v2_audit as builder
from tools import ins_lidar_sota_inventory as kitti
from tools import ins_lidar_sota_v2_evaluate as driver
from tools import ins_lidar_sota_v2_inventory as inventory
from tools import ins_lidar_sota_v2_protocol as protocol
from tools.boreas_ins_lidar_metadata import inventory as boreas_inventory
from tools.score_ins_lidar_sota import CaseSpec

V1_RUNS = Path("docs/assets/ins_lidar_sota_readiness/evaluation/runs")


def test_kitti_inventory_takes_every_eligible_recording_but_the_first_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = pool(tmp_path)  # 17 eligible recordings, 2011_09_29_0002 .. 0018
    monkeypatch.setattr(kitti.fetch, "head", lambda _: (1000, "pin"))
    monkeypatch.setattr(kitti.fetch, "central_directory", central)
    monkeypatch.setattr(kitti.fetch, "spans", lambda *args: [(0, 99)])
    spent = {"2011_09_29_0002", "2011_09_29_0005"}

    result = kitti.collect(
        metadata,
        source_root=tmp_path,
        choose=lambda names: [name for name in names if name not in spent],
        selection_rule=inventory.KITTI_SELECTION_RULE,
    )

    assert len(result.selected) == 15 and not spent & {r.case_id for r in result.selected}
    assert result.selection_rule == inventory.KITTI_SELECTION_RULE
    assert len(result.eligible_recording_ids) == 17  # the spent ones stay listed as eligible


def test_boreas_inventory_skips_the_first_audit_and_keeps_incomplete_recordings() -> None:
    names = [f"boreas-2021-01-{day:02d}-10-00/" for day in range(1, 9)]
    missing = names[4] + "applanix/"
    fetch, _calls = transport(missing=missing)

    def listing(url: str) -> bytes:
        if "prefix=&" in url or url.endswith("prefix="):
            from tests.unit.test_boreas_ins_lidar_metadata import listing as page

            return page(prefixes=names)
        return fetch(url)

    result = boreas_inventory(
        provenance(),
        fetch=listing,
        frame_limit=2,
        skip=3,
        complete=3,
        selection_rule=inventory.BOREAS_SELECTION_RULE,
    )

    # prefixes 4.. in order; the one without a trajectory is kept and does not count
    assert [r.recording_id for r in result.recordings] == [n[:-1] for n in names[3:7]]
    assert [bool(r.missing_required) for r in result.recordings] == [False, True, False, False]
    assert {r.role for r in result.recordings} == {"evaluation_candidate"}


def test_the_contract_declares_ten_unseen_cases_and_no_looser_gate_than_v1() -> None:
    from tools import ins_lidar_sota_protocol as v1

    cases = [case for case, _family, _rig in protocol.EXPECTED_CASES]
    assert len(cases) == len(set(cases)) == 10
    assert not set(cases) & {case for case, _f, _r in v1.EXPECTED_CASES}
    assert protocol.DEVELOPMENT_CASES == v1.EXPECTED_CASES  # v1's evaluation is spent
    assert protocol.ERROR_MAX_DEG <= v1.ERROR_MAX_DEG
    assert protocol.UNOBSERVED_FRACTION_MAX <= v1.UNOBSERVED_FRACTION_MAX
    assert protocol.CONTROL_MISS_FRACTION_MAX <= v1.CONTROL_MISS_FRACTION_MAX
    assert {family for _c, family, _r in protocol.EXPECTED_CASES} == {"kitti_raw", "boreas"}
    assert "is not a failure" in protocol.CLAIM


def _row(case_id: str, family: str, *, estimated: int, errors=(), misses=0, completed=True):
    return {
        "case": {"case_id": case_id, "dataset_family": family},
        "completed": completed,
        "estimated_axes": estimated,
        "errors_deg": list(errors),
        "control_misses": misses,
        "reasons": [] if completed else ["method did not converge"],
    }


def test_a_recording_without_an_estimated_axis_counts_in_coverage_only() -> None:
    rows = [
        _row("a", "kitti_raw", estimated=0),
        _row("b", "kitti_raw", estimated=2, errors=(0.1, 0.3), misses=1),
        _row("c", "kitti_raw", estimated=3, errors=(0.2, 0.1, 0.05)),
    ]
    states = dict.fromkeys("abc", "scored")

    trials = builder.family_rows(rows, states)

    assert trials["unobserved_rotation_fraction"] == {"a": 1.0, "b": pytest.approx(1 / 3), "c": 0.0}
    assert trials["rotation_reference_difference_deg"] == {"b": 0.3, "c": 0.2}  # no trial for a
    assert trials["missed_observed_control_fraction"] == {"b": 0.5, "c": 0.0}


def test_a_method_failure_is_a_failed_trial_and_a_missing_case_is_no_trial() -> None:
    rows = [
        _row("a", "boreas", estimated=3, errors=(0.1, 0.1, 0.1)),
        _row("b", "boreas", estimated=0, completed=False),  # ran, did not converge
        _row("c", "boreas", estimated=0, completed=False),  # crashed: no artifact
        _row("d", "boreas", estimated=0, completed=False),  # never acquired
    ]
    states = {"a": "scored", "b": "scored", "c": "method_failure", "d": "missing"}

    coverage = builder.family_rows(rows, states)["unobserved_rotation_fraction"]

    assert coverage["a"] == 0.0
    assert isinstance(coverage["b"], str) and isinstance(coverage["c"], str)
    assert "d" not in coverage


def test_case_states_follow_the_method_attempt() -> None:
    manifest = {
        "cases": [
            {"case_id": "passed", "artifact_path": "passed.yaml"},
            {"case_id": "failed", "artifact_path": None},
            {"case_id": "interrupted", "artifact_path": None},
            {"case_id": "not_acquired", "artifact_path": None},
        ],
        "attempts": {
            "passed": {"status": "PASS"},
            "failed": {"status": "FAIL"},
            "interrupted": {"status": "RUNNING"},
        },
    }

    assert builder.case_states(manifest) == {
        "passed": "scored",
        "failed": "method_failure",
        "interrupted": "missing",
        "not_acquired": "missing",
    }


def _development_manifest(tmp_path: Path, *, drop: str | None = None) -> Path:
    """v1's published evaluation artifacts as a development run of the v2 driver."""

    plan = driver.development_plan()
    runs = tmp_path / "runs"
    runs.mkdir()
    cases, attempts = [], {}
    for case in plan.evaluation:
        spec = CaseSpec(**case.model_dump(), drive_ids=case.drive_ids)
        if case.case_id != drop:
            data = (V1_RUNS / f"{case.case_id}.yaml").read_bytes()
            (runs / f"{case.case_id}.yaml").write_bytes(data)
            spec = spec.model_copy(
                update={
                    "artifact_path": f"{case.case_id}.yaml",
                    "artifact_sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            attempts[case.case_id] = {"status": "PASS"}
        cases.append(spec.model_dump())
    manifest = {
        "schema_version": "calibrex.ins_lidar_sota_run/v0.2",
        "preregistration_sha256": plan.identity_sha256,
        "frozen_code_commit": plan.frozen_code_commit,
        "development": True,
        "cases": cases,
        "input_receipts": {},
        "attempts": attempts,
    }
    path = runs / "run.json"
    path.write_text(json.dumps(manifest))
    return path


def test_v1s_results_are_supported_under_the_v2_rules(tmp_path: Path) -> None:
    result = builder.build(_development_manifest(tmp_path), None, tmp_path / "audit")

    # v1 was refuted only by 2011_09_29_0071, where no axis was observable
    assert result["verdict"] == "supported" and result["achieved"] == 8
    assert result["families"]["kitti_raw"]["estimated_axes"] == 5
    assert result["families"]["kitti_raw"]["maximum_reference_difference_deg"] < 0.29


def test_a_case_that_was_never_acquired_leaves_the_audit_incomplete(tmp_path: Path) -> None:
    manifest = _development_manifest(tmp_path, drop="2011_09_30_0027")

    result = builder.build(manifest, None, tmp_path / "audit")

    assert result["case_states"]["2011_09_30_0027"] == "missing"
    assert result["verdict"] == "incomplete"
    assert result["statuses"]["kitti_raw-agreement"] == "incomplete"
    assert result["statuses"]["boreas-agreement"] == "achieved"


def test_raw_inputs_are_deleted_and_an_interrupted_result_is_set_aside(tmp_path: Path) -> None:
    work = tmp_path / "case"
    (work / "data").mkdir(parents=True)
    (work / "data" / "scan.bin").write_bytes(b"x" * 100)
    (work / "acquisition.json").write_text("{}")
    receipt = {"files": [{"path": "data/scan.bin"}, {"path": "../outside"}]}

    assert driver.delete_raw_inputs(receipt, work) == 100
    assert not (work / "data" / "scan.bin").exists()
    assert (work / "acquisition.json").exists()  # the receipt keeps the digests

    log = tmp_path / "case.log"
    log.write_text("partial")
    assert driver._set_aside(log, 1) == "case.interrupted-1.log"
    assert not log.exists() and (tmp_path / "case.interrupted-1.log").read_text() == "partial"
    assert driver._set_aside(log, 2) is None
