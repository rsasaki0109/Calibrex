"""Audit both committed CPU implementations without executing archived code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_online_grid_reuse as bench  # noqa: E402
from tools.audit_online_edge_cuda_archive import _local_references, audit_archive  # noqa: E402


def audit(path: Path, temporary_root: Path, *, verify_git: bool = False) -> dict:
    """Use the bounded archive reader and validate both sources and all clocks."""

    def check(folder: Path) -> dict:
        for directory in ("baseline/schemas", "candidate/schemas", "toolchain/tools/schemas"):
            for schema in (folder / directory).glob("*.json"):
                _local_references(bench.base.read(schema))
        return bench.audit(folder, verify_git=verify_git)

    value = audit_archive(path, temporary_root, _audit=check)
    value["provenance"]["grid_archive_auditor_sha256"] = bench.base.sha(Path(__file__))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--temporary-root", type=Path, default=Path("outputs/online-grid-audits"))
    parser.add_argument("--verify-git", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = audit(args.archive, args.temporary_root, verify_git=args.verify_git)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if value["passes_solve_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
