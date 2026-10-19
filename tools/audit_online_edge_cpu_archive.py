"""Audit indexed CPU execution evidence without optional runtime imports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import benchmark_online_edge_cpu as cpu  # noqa: E402
from tools.audit_online_edge_cuda_archive import audit_archive  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--temporary-root", type=Path, default=Path("outputs/online-edge-cpu-audits")
    )
    parser.add_argument("--bind-commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_archive(
        args.archive,
        args.temporary_root,
        bind_commit=args.bind_commit,
        _audit=cpu.audit,
        _toolchain=cpu.TOOLCHAIN,
    )
    result["provenance"]["cpu_archive_auditor_sha256"] = cpu.base.sha(Path(__file__))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["passes_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
