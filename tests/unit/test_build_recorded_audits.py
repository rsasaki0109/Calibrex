"""The audit records built from committed run files reproduce, and keep their verdicts."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml
from tools import build_recorded_audits as builder

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "spec", [builder.BOARD, builder.NCLT, builder.TUMVI], ids=lambda s: s.protocol_id
)
def test_a_fresh_build_equals_the_committed_records(spec, tmp_path: Path) -> None:
    built = builder.build(dataclasses.replace(spec, output=tmp_path))
    committed = sorted(spec.output.glob("*.yaml"))
    assert [path.name for path in committed] == sorted(p.name for p in tmp_path.glob("*.yaml"))
    here = spec.output.relative_to(ROOT).as_posix()
    for path in committed:
        fresh = (tmp_path / path.name).read_text().replace(str(tmp_path), here)
        assert fresh == path.read_text(), path.name
    assert built["verdict"] == ("supported" if spec is builder.TUMVI else "refuted")


def test_the_verdicts_are_the_recorded_ones() -> None:
    nclt = yaml.safe_load(
        (builder.NCLT.output / "nclt-lidar-wheel-odometry-v1_sota_audit.yaml").read_text()
    )
    statuses = {r["requirement_id"]: r["status"] for r in nclt["requirements"]}
    # the pre-registered accuracy claim was refuted, the check-honesty claim held
    assert statuses["accuracy-rotation"] == "contradicted"
    assert all(statuses[k] == "achieved" for k in statuses if k.startswith("honesty-"))
    board = yaml.safe_load(
        (builder.BOARD.output / "kitti-board-camera-lidar-v1_sota_audit.yaml").read_text()
    )
    assert board["verdict"] == "refuted"
    assert {r["status"] for r in board["requirements"]} <= {"contradicted", "incomplete"}


def test_the_leaderboard_lists_both() -> None:
    import json

    board = json.loads((ROOT / "docs/assets/sota_leaderboard.json").read_text())
    paths = {entry["audit_path"] for entry in board["entries"]}
    assert {
        "docs/assets/nclt_lidar_wheel/audit/nclt-lidar-wheel-odometry-v1_sota_audit.yaml",
        "docs/assets/kitti_board_camera_lidar/audit/kitti-board-camera-lidar-v1_sota_audit.yaml",
    } <= paths
