"""Audit complete ICP controls, cold observations, source snapshots and failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def audit(root: Path) -> dict[str, int]:
    index = json.loads((root / "publication-index.json").read_bytes())
    for name, digest in index["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    complete = 0
    for name, run in index["runs"].items():
        anchor = None
        for worker in run["workers"]:
            measurement = json.loads((root / worker["measurement"]).read_bytes())
            raw = (root / worker["result"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == measurement["result_sha256"]
            model = json.loads(raw)
            assert model["status"] == "converged"
            assert len(model["multi_start_trials"]) == 6
            assert model["rematching_diagnostics"] is not None
            assert len(model["train_source_ids"]) + len(model["holdout_source_ids"]) == 127158
            if run["compare"]:
                anchor = raw if anchor is None else anchor
                assert anchor == raw, name
            provenance = measurement["provenance"]
            identity = provenance["source"]
            for source, digest in identity["sources"].items():
                captured = root / run["sources"] / (Path(source).name + ".txt")
                assert hashlib.sha256(captured.read_bytes()).hexdigest() == digest
            assert provenance["cupy_imported"] == (measurement["method"] == "cuda")
            if name == "final":
                assert all(entry["num_threads"] == 1 for entry in provenance["blas_runtime"]
                           if entry["user_api"] == "blas")
                assert set(provenance["blas_environment"].values()) == {"1"}
            complete += 1
        if name == "final":
            summary = json.loads((root / "final/summary.json").read_bytes())
            assert len(summary["trials"]) == 6
            for method in ("native", "cuda"):
                rows = [t for t in summary["trials"] if t["method"] == method]
                assert statistics.median(t["seconds"] for t in rows) == summary[
                    "medians_s"][method]
                assert statistics.median(t["subprocess_wall_seconds"] for t in rows) == summary[
                    "whole_medians_s"][method]
                for row in rows:
                    assert row["provenance"]["source"] == summary["provenance"]
            assert summary["whole_speedup"] == (summary["whole_medians_s"]["native"] /
                                                summary["whole_medians_s"]["cuda"])
    assert complete == index["complete_models"]
    return {"complete_models": complete, "digest_files": len(index["files"]),
            "retained_failures": len(index["failures"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs/assets/cuda_point_to_point_icp"))
    args = parser.parse_args()
    print(json.dumps(audit(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
