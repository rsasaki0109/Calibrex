from __future__ import annotations

import argparse

import pytest

from calibrex.cli.help import COMMAND_GROUPS
from calibrex.cli.main import _build_parser, main

# Snapshot of every top-level command registered before the help regrouping.
EXPECTED_COMMANDS = [
    "doctor",
    "schema",
    "validate",
    "benchmark",
    "ci",
    "replay",
    "multi-lidar-service",
    "camera-imu-service",
    "radar-service",
    "verify",
    "assess",
    "evidence",
    "init",
    "calibrate",
    "compile",
    "evaluate",
    "render",
    "report",
    "compare",
    "report-compare",
    "window-consistency",
    "trajectory-window-drift",
    "calibration-readiness",
    "continuous-time-lidar-pair",
    "metrics",
    "public-datasets",
    "demo",
    "convert",
    "kitti",
    "gnss-lidar",
    "imu-lidar",
    "camera-imu",
    "lidar-lidar",
    "lidar-vehicle",
    "imu-vehicle",
    "gnss-imu",
    "lidar-wheel",
    "ins-lidar",
    "sota",
    "camera-lidar",
    "external-run",
    "trajectory",
    "lifecycle",
    "visualize",
    "capture",
    "autoware",
    "check",
    "estimate",
    "inspect",
    "export",
]


def _commands() -> list[str]:
    parser = _build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return list(action.choices)
    raise AssertionError("no subparsers")


def test_every_previous_command_is_still_registered() -> None:
    assert set(EXPECTED_COMMANDS) <= set(_commands())


def test_every_command_is_in_a_help_group() -> None:
    grouped = [name for _, names in COMMAND_GROUPS for name in names]
    assert sorted(grouped) == sorted(set(grouped))
    assert set(_commands()) <= set(grouped)


def test_top_level_help_has_sections_and_no_giant_choices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for header in ("Start here:", "Per-pair calibration:", "Evidence and CI:", "examples:"):
        assert header in out
    assert "<command>" in out
    assert "{doctor," not in out
    assert "calibrex check bag/ --plan" in out


def test_check_help_has_argument_groups(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["check", "--help"])
    out = capsys.readouterr().out
    for header in ("basic:", "topic mapping:", "duration and performance:", "advanced thresholds"):
        assert header in out


def test_check_flags_keep_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["check", "bag", "--plan"])
    assert args.plan is True
    assert args.sigma_k == 3.0
    assert args.fail_on == "fail"


def test_unknown_command_suggests_closest(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["chek", "bag"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unknown command 'chek'" in err
    assert "Did you mean: check" in err
