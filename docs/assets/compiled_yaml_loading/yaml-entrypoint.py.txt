"""Measure complete YAML-input CLIs using the shared frozen-source harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.audit_lazy_runtime_benchmark import audit as audit_cli
from tools.benchmark_lazy_runtime_loading import main as benchmark_cli
from tools.benchmark_lazy_runtime_loading import sha


def audit(root: Path) -> dict:
    """Audit YAML input identities, full results and unchanged numerical sources."""
    result = audit_cli(root, expected_runtime_changes=frozenset({"calibrex/core/io.py"}))
    protocol = json.loads((root / "protocol.json").read_bytes())
    conversion = json.loads((root / "input-transcoding.json").read_bytes())
    fixture = json.loads((root / "fixture-protocol.json").read_bytes())
    if (
        conversion["format"] != "yaml"
        or conversion["libyaml"] is not True
        or conversion["input_sha256"] != protocol["provenance"]["inputs"]
        or conversion["original_sha256"]
        != {
            "correspondence": fixture["correspondence_sha256"],
            "problem": fixture["problem_sha256"],
        }
        or conversion["provenance"]
        != {
            "launcher_sha256": protocol["provenance"]["launcher_sha256"],
            "protocol_sha256": sha(root / "protocol.json"),
        }
        or sha(root / "fixture-builder.py.txt") != fixture["builder_sha256"]
        or any(not name.endswith(".yaml") for name in protocol["commands"]["refine"][2:4])
    ):
        raise ValueError("YAML fixture conversion binding differs")
    for record in json.loads((root / "attempts.json").read_bytes()):
        if record["case"] != "refine":
            continue
        folder = root / f"refine-{record['repeat']:02d}-{record['variant']}"
        value = json.loads((folder / "result.json").read_bytes())
        commit = value["provenance"]["git_commit"]
        if (
            not isinstance(commit, str)
            or len(commit) < 7
            or not protocol["commits"][record["variant"]].startswith(commit)
            or len(value["trace"]) != 325
            or value["options"]["max_evaluations"] != 400
        ):
            raise ValueError("refinement workload or source identity differs")
    return result


def main() -> None:
    if "--audit" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--audit", type=Path, required=True)
        print(json.dumps(audit(parser.parse_args().audit), indent=2))
    else:
        sys.argv.extend(["--input-format", "yaml"])
        benchmark_cli()


if __name__ == "__main__":
    main()
