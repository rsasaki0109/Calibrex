"""Grouped ``--help`` output and unknown-command suggestions for the Calibrex CLI.

This only changes how the command list is *presented*. Every subcommand keeps its
name and dispatch (``set_defaults(func=...)``); commands not listed in
``COMMAND_GROUPS`` are shown under "Other commands" so a new command is never hidden.
"""

from __future__ import annotations

import argparse
import difflib
import re
from collections.abc import Iterable
from typing import NoReturn

COMMAND_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Start here",
        ("check", "estimate", "doctor", "demo", "inspect", "init", "calibrate", "render"),
    ),
    (
        "Per-pair calibration",
        (
            "imu-lidar",
            "camera-imu",
            "lidar-lidar",
            "lidar-vehicle",
            "imu-vehicle",
            "gnss-lidar",
            "gnss-imu",
            "lidar-wheel",
            "ins-lidar",
            "camera-lidar",
        ),
    ),
    (
        "Evidence and CI",
        (
            "validate",
            "verify",
            "assess",
            "evidence",
            "ci",
            "evaluate",
            "compare",
            "report-compare",
            "window-consistency",
            "calibration-readiness",
            "lifecycle",
            "schema",
            "metrics",
            "report",
        ),
    ),
    (
        "Data and conversion",
        (
            "convert",
            "kitti",
            "public-datasets",
            "capture",
            "export",
            "autoware",
            "visualize",
            "external-run",
        ),
    ),
    (
        "Research, benchmarking and services",
        (
            "benchmark",
            "sota",
            "compile",
            "trajectory",
            "trajectory-window-drift",
            "continuous-time-lidar-pair",
            "replay",
            "multi-lidar-service",
            "camera-imu-service",
            "radar-service",
        ),
    ),
)

HELP_EPILOG = """\
examples:
  calibrex check bag/ --plan            list which sensor pairs a bag can check
  calibrex check bag/ --html out.html   audit the deployed calibration against a bag
  calibrex doctor                       verify the environment
  calibrex <command> --help             options of one command
"""


def _grouped(names: Iterable[str]) -> list[tuple[str, list[str]]]:
    remaining = list(names)
    sections: list[tuple[str, list[str]]] = []
    for title, members in COMMAND_GROUPS:
        present = [name for name in members if name in remaining]
        if present:
            sections.append((title, present))
    placed = {name for _, members in sections for name in members}
    other = [name for name in remaining if name not in placed]
    if other:
        sections.append(("Other commands", other))
    return sections


class GroupedHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Render the top-level subcommand list in titled sections."""

    def _format_action(self, action: argparse.Action) -> str:
        if not isinstance(action, argparse._SubParsersAction):
            return super()._format_action(action)
        helps = {choice.dest: choice for choice in action._choices_actions}
        parts: list[str] = []
        for title, names in _grouped(action.choices):
            parts.append(f"{title}:\n")
            self._indent()
            for name in names:
                sub = helps.get(name)
                if sub is None:
                    parts.append(f"{' ' * self._current_indent}{name}\n")
                else:
                    parts.append(self._format_action(sub))
            self._dedent()
            parts.append("\n")
        return "".join(parts)


class CalibrexArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that suggests the closest command for a mistyped one."""

    def format_help(self) -> str:
        text = super().format_help()
        if self.formatter_class is GroupedHelpFormatter:
            # The grouped command list carries its own section titles.
            text = text.replace("positional arguments:\n", "", 1)
        return text

    def error(self, message: str) -> NoReturn:
        match = re.search(r"invalid choice: '([^']*)'", message)
        choices: list[str] = []
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                choices = list(action.choices)
        if match and choices:
            typed = match.group(1)
            close = difflib.get_close_matches(typed, choices, n=3, cutoff=0.7)
            message = f"unknown command '{typed}'"
            if close:
                message += f". Did you mean: {', '.join(close)}?"
            message += f"\nRun '{self.prog} --help' to list the commands."
        super().error(message)


CHECK_ARGUMENT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "basic",
        (
            "--tf",
            "--vehicle-frame",
            "--plan",
            "--write-frames-template",
            "--pairs",
            "--output",
            "--html",
            "--json",
            "--quiet",
            "--fail-on",
        ),
    ),
    ("topic mapping", ("--frame-map", "--topic-kind", "--camera")),
    (
        "duration and performance",
        (
            "--max-duration-s",
            "--gnss-max-duration-s",
            "--cache-dir",
            "--no-cache",
            "--scan-memory-mb",
            "--evidence-dir",
        ),
    ),
)
CHECK_ADVANCED_TITLE = "advanced thresholds and estimator modes"


def group_arguments(
    parser: argparse.ArgumentParser,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    rest_title: str,
) -> None:
    """Regroup an already-populated parser's optional arguments for ``--help``.

    Flags not named in ``groups`` (including ones added later) land in ``rest_title``.
    Names, defaults and parsing are untouched; only help sections change.
    """

    movable = [
        a for a in parser._actions if a.option_strings and not isinstance(a, argparse._HelpAction)
    ]
    for default_group in parser._action_groups:
        default_group._group_actions = [a for a in default_group._group_actions if a not in movable]
    by_flag = {a.option_strings[0]: a for a in movable}
    used: set[str] = set()
    for title, flags in groups:
        group = parser.add_argument_group(title)
        for flag in flags:
            action = by_flag.get(flag)
            if action is not None:
                group._group_actions.append(action)
                used.add(flag)
    rest = [a for flag, a in by_flag.items() if flag not in used]
    if rest:
        parser.add_argument_group(rest_title)._group_actions.extend(rest)
