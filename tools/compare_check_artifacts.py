#!/usr/bin/env python3
"""Compare two ``calibrex check`` / ``estimate`` outputs for result equality.

Used to prove a speed-up did not change results: it loads the two top-level
YAML/JSON artifacts and every evidence artifact beside them, ignores volatile
fields (timestamps, runtimes, paths, digests of evidence files, command lines,
cache bookkeeping) and reports every other difference.  Floats are compared
bit-for-bit by default; ``--rtol`` allows a relative tolerance and reports the
largest difference found either way.

Example::

    python tools/compare_check_artifacts.py before.yaml after.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

VOLATILE_KEYS = frozenset(
    {
        "runtime_s",
        "created_at",
        "created_utc",
        "generated_at",
        "timestamp",
        "command",
        "git_commit",
        "path",
        "evidence_artifact",
        "evidence_dir",
        "sha256",
        "output",
        "cache_dir",
        "cache_hit",
        "cache_key",
        "wall_s",
        "elapsed_s",
    }
)


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


class Comparison:
    """Accumulate differences between two nested structures."""

    def __init__(self, rtol: float) -> None:
        self.rtol = rtol
        self.differences: list[str] = []
        self.floats = 0
        self.max_rel = 0.0

    def walk(self, left: Any, right: Any, where: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right), key=str):
                if key in VOLATILE_KEYS:
                    continue
                if key not in left or key not in right:
                    self.differences.append(f"{where}/{key}: present on one side only")
                    continue
                self.walk(left[key], right[key], f"{where}/{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                self.differences.append(f"{where}: length {len(left)} vs {len(right)}")
                return
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                self.walk(a, b, f"{where}[{index}]")
        elif isinstance(left, float) and isinstance(right, float):
            self.floats += 1
            if left == right or (math.isnan(left) and math.isnan(right)):
                return
            scale = max(abs(left), abs(right))
            rel = abs(left - right) / scale if scale > 0.0 else 0.0
            self.max_rel = max(self.max_rel, rel)
            if rel > self.rtol:
                self.differences.append(f"{where}: {left!r} vs {right!r} (rel {rel:.3e})")
        elif left != right:
            self.differences.append(f"{where}: {left!r} vs {right!r}")


def _evidence_files(top: Path) -> dict[str, Path]:
    folder = top.with_name(top.name + "_evidence")
    if not folder.is_dir():
        folder = top.parent / (top.stem + "_evidence")
    if not folder.is_dir():
        return {}
    return {item.name: item for item in sorted(folder.iterdir()) if item.is_file()}


def compare(left: Path, right: Path, rtol: float = 0.0) -> Comparison:
    """Compare two artifacts and their evidence folders."""

    result = Comparison(rtol)
    result.walk(_load(left), _load(right), left.name)
    left_files, right_files = _evidence_files(left), _evidence_files(right)
    for name in sorted(set(left_files) | set(right_files)):
        if name not in left_files or name not in right_files:
            result.differences.append(f"evidence {name}: present on one side only")
            continue
        result.walk(_load(left_files[name]), _load(right_files[name]), f"evidence/{name}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--rtol", type=float, default=0.0, help="relative float tolerance")
    args = parser.parse_args()
    result = compare(args.before, args.after, args.rtol)
    for line in result.differences:
        print("DIFF", line)
    print(
        f"{args.before.name} vs {args.after.name}: {result.floats} floats compared, "
        f"max relative difference {result.max_rel:.3e}, "
        f"{len(result.differences)} difference(s): "
        + ("IDENTICAL" if not result.differences else "DIFFERENT")
    )
    return 0 if not result.differences else 1


if __name__ == "__main__":
    sys.exit(main())
