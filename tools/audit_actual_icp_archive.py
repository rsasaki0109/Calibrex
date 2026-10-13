"""Validate real-frame provenance, native parity, evaluations and all timings."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import jsonschema


def audit(root: Path) -> dict[str, int]:
    index = json.loads((root / "publication-index.json").read_bytes())
    for name, digest in index["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"archive digest mismatch: {name}")
    models = trials = 0
    for label, run in index["runs"].items():
        summary = json.loads((root / run["summary"]).read_bytes())
        anchor = evaluation = None
        for worker in run["workers"]:
            row = json.loads((root / worker["measurement"]).read_bytes())
            raw = (root / worker["result"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row["result_sha256"]
            anchor = raw if anchor is None else anchor
            assert raw == anchor
            evaluation = row["evaluation"] if evaluation is None else evaluation
            assert row["evaluation"] == evaluation
            model = json.loads(raw)
            assert model["status"] in ("converged", "max_iterations")
            assert len(model["multi_start_trials"]) == 6
            assert model["rematching_diagnostics"] is not None
            assert len(model["train_source_ids"]) + len(model["holdout_source_ids"]) == row[
                "point_count"]
            assert model["holdout_rmse_m"] == row["evaluation"]["final_holdout_rmse_m"]
            provenance = row["provenance"]
            assert provenance["cupy_imported"] == (row["method"] == "cuda")
            assert set(provenance["blas_environment"].values()) == {"1"}
            assert all(entry["num_threads"] == 1 for entry in provenance["blas_runtime"]
                       if entry["user_api"] == "blas")
            assert provenance["source"] == summary["provenance"]
            for source, digest in provenance["source"]["sources"].items():
                capture = root / run["sources"] / (Path(source).name + ".txt")
                assert hashlib.sha256(capture.read_bytes()).hexdigest() == digest
            fixture_raw = (root / run["fixture"]).read_bytes()
            assert hashlib.sha256(fixture_raw).hexdigest() == provenance["fixture_sha256"]
            fixture = json.loads(fixture_raw)
            jsonschema.validate(fixture, json.loads((root / run["sources"] /
                                "icp_pair_fixture.schema.json.txt").read_bytes()))
            assert fixture == provenance["fixture"]
            for source, digest in fixture["provenance"]["source_sha256"].items():
                capture = root / "fixture_sources" / Path(source.replace("\\", "/")).name
                capture = capture.with_name(capture.name + ".txt")
                assert hashlib.sha256(capture.read_bytes()).hexdigest() == digest
            for source, digest in fixture["provenance"]["input_sha256"].items():
                if source.endswith(".txt"):
                    assert hashlib.sha256((root / "navigation" / Path(source).name).read_bytes(
                        )).hexdigest() == digest
            if label == "primary-115":
                assert model["status"] == "converged"
                assert fixture["options"]["max_iterations"] == 50
            if label == "small-115" and row["method"] == "cuda":
                assert provenance["adapter"]["gpu_query_calls"] == 0
            models += 1
        assert len(run["workers"]) == len(summary["trials"])
        for method in ("native", "cuda"):
            rows = [t for t in summary["trials"] if t["method"] == method]
            assert statistics.median(t["seconds"] for t in rows) == summary["medians_s"][method]
            assert statistics.median(t["subprocess_wall_seconds"] for t in rows) == summary[
                "whole_medians_s"][method]
            trials += len(rows)
        assert summary["whole_speedup"] == (summary["whole_medians_s"]["native"] /
                                            summary["whole_medians_s"]["cuda"])
    assert models == trials == index["complete_models"]
    return {"cohorts": len(index["runs"]), "complete_models": models,
            "digest_files": len(index["files"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("docs/assets/cuda_point_to_point_icp/actual_frames"))
    print(json.dumps(audit(parser.parse_args().root), sort_keys=True))


if __name__ == "__main__":
    main()
