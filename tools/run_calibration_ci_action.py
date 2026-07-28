#!/usr/bin/env python3
"""Execute ``calibrex ci`` and export GitHub Action outputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run the installed CLI, publish outputs, and preserve its exit code."""

    candidate = _required_env("CALIBREX_ACTION_CANDIDATE")
    output_dir = Path(_required_env("CALIBREX_ACTION_OUTPUT_DIR"))
    command = [
        sys.executable,
        "-m",
        "calibrex.cli.main",
        "ci",
        candidate,
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    baseline = os.environ.get("CALIBREX_ACTION_BASELINE", "").strip()
    policy = os.environ.get("CALIBREX_ACTION_POLICY", "").strip()
    if baseline:
        command.extend(["--baseline", baseline])
    if policy:
        command.extend(["--policy", policy])
    if not _boolean_env("CALIBREX_ACTION_ENFORCE_PROTOCOL", default=True):
        command.append("--allow-incompatible-protocol")
    if _boolean_env("CALIBREX_ACTION_ENFORCE", default=True):
        command.append("--enforce")

    completed = subprocess.run(command, check=False)
    artifact_path = output_dir / "calibration-ci.json"
    if not artifact_path.is_file():
        return completed.returncode or 2

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifacts = artifact.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    outputs = {
        "status": str(artifact.get("status", "unknown")),
        "artifact": str(artifact_path),
        "summary": str(artifacts.get("markdown_summary", "")),
        "evidence_card": str(artifacts.get("evidence_card", "")),
        "comparison": str(artifacts.get("comparison", "")),
    }
    _write_outputs(outputs)
    _append_step_summary(Path(outputs["summary"]))
    return completed.returncode


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def _boolean_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise SystemExit(f"{name} must be true or false, got {value!r}")


def _write_outputs(outputs: dict[str, str]) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with Path(github_output).open("a", encoding="utf-8") as stream:
        for name, value in outputs.items():
            if "\n" in value or "\r" in value:
                raise SystemExit(f"action output {name!r} contains a newline")
            stream.write(f"{name}={value}\n")


def _append_step_summary(path: Path) -> None:
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not github_summary or not path.is_file():
        return
    with Path(github_summary).open("a", encoding="utf-8") as stream:
        stream.write(path.read_text(encoding="utf-8"))
        stream.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
