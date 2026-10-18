"""Audit complete historical-source ICP experiments without importing CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import jsonschema


def _read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def audit(root: Path) -> dict[str, object]:
    index = _read(root / "publication-index.json")
    base = root.resolve()
    for name, expected in index["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(base):
            raise ValueError("archive file is outside its root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"archive digest mismatch: {name}")
    schema = _read(root / "comparison.schema.json")
    count = 0
    comparisons = {}
    for name in index["formal_cohorts"]:
        cohort = root / name
        protocol, summary = _read(cohort / "protocol.json"), _read(cohort / "summary.json")
        for model in (protocol, summary, _read(cohort / "attempts.json")):
            jsonschema.validate(model, schema)
        assert protocol["provenance"] == summary["provenance"]
        trials = summary["trials"]
        assert len(trials) == 2 * protocol["repeats"]
        prior = _read(cohort / "sources/prior/manifest.json")
        expected_sources = protocol["provenance"]["current"]["sources"]
        for variant, sources in (("prior", prior["files"]), ("current", expected_sources)):
            for relative, digest in sources.items():
                filename = Path(relative).name + (".txt" if Path(relative).suffix == ".py" else "")
                path = cohort / "sources" / variant / filename
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        anchor = None
        for trial in trials:
            variant, repeat = trial["variant"], trial["repeat"]
            assert trial["order"] == protocol["planned_orders"][repeat - 1]
            assert trial["exit_code"] == 0
            child = cohort / f"{variant}.{repeat:02d}"
            measurement = _read(child / "measurement.json")
            assert trial["measurement"] == measurement
            binding = _read(child / "result-reference.json")
            raw = (root / binding["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == measurement["result_sha256"]
            if anchor is None:
                anchor = raw
            assert raw == anchor
            provenance = measurement["provenance"]
            assert provenance["checkout_source_guard"] == protocol["provenance"]["current"]
            expected = prior["files"] if variant == "prior" else expected_sources
            execution = provenance["source"]["sources"]
            assert execution and all(expected[path] == digest for path, digest in execution.items())
            assert provenance["source_binding"] == (
                "frozen_prior" if variant == "prior" else "current_checkout"
            )
            assert all(
                item["num_threads"] == 1
                for item in provenance["blas_runtime"]
                if item["user_api"] == "blas"
            )
            if protocol["backend"] == "native":
                assert not provenance["cupy_imported"]
            count += 1
        whole = {
            v: statistics.median(t["whole_seconds"] for t in trials if t["variant"] == v)
            for v in ("prior", "candidate")
        }
        solver = {
            v: statistics.median(t["measurement"]["seconds"] for t in trials if t["variant"] == v)
            for v in ("prior", "candidate")
        }
        assert whole == summary["whole_medians_s"] and solver == summary["solver_medians_s"]
        assert whole["prior"] / whole["candidate"] == summary["whole_speedup"]
        assert solver["prior"] / solver["candidate"] == summary["solver_speedup"]
        assert summary["full_result_equal"] is True
        comparisons[name] = {"whole_medians_s": whole, "whole_speedup": summary["whole_speedup"]}
    return {
        "digest_files": len(index["files"]),
        "complete_workers": count,
        "full_result_equal": True,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs/assets/icp_neighbor_experiments"))
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))


if __name__ == "__main__":
    main()
