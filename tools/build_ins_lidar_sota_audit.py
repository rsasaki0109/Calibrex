"""Build the frozen INS--LiDAR rotation audit, preserving every declared case.

The second benchmark method is a mathematical zero anchor required by the
benchmark schema, not an empirical competitor. Only Calibrex is gated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ins_lidar_sota_acquire import file_digest, verify_input_receipt
from tools.ins_lidar_sota_protocol import (
    CLAIM,
    CONTROL_MISS_FRACTION_MAX,
    ERROR_MAX_DEG,
    EXPECTED_CASES,
    METHOD_ID,
    PROTOCOL_ID,
    SCOPE,
    UNOBSERVED_FRACTION_MAX,
    RunManifest,
    load_preregistration,
    verify_frozen_source,
)
from tools.score_ins_lidar_sota import CaseSpec, score_artifacts

from calibrex import __version__
from calibrex.core.benchmark import (
    BenchmarkDefinition,
    BenchmarkMethodDefinition,
    BenchmarkMetricDefinition,
    BenchmarkProtocol,
    BenchmarkProvenance,
    BenchmarkSplit,
    BenchmarkTrial,
    BenchmarkTrialProvenance,
    aggregate_benchmark_definition,
)
from calibrex.core.sota_audit import (
    SotaAuditProtocol,
    SotaAuditProvenance,
    SotaClaimRequirement,
    SotaClaimScope,
)
from calibrex.evaluation.sota_audit import audit_sota_claim

METHOD = BenchmarkMethodDefinition(
    method_id=METHOD_ID,
    label="Native joint trajectory hand-eye",
    implementation="calibrex_native",
    tool_name="calibrex",
)
ANCHOR = BenchmarkMethodDefinition(
    method_id="reference_zero_anchor",
    label="Mathematical zero anchor (not an empirical calibration method)",
    implementation="imported",
    tool_name="dataset_reference_zero",
)


def _benchmark(
    family: str,
    metric_name: str,
    rows: dict[str, float | str],
    digest: str,
    verified: bool,
    sources: dict[str, str],
    frozen: str,
) -> BenchmarkDefinition:
    splits = [
        BenchmarkSplit(
            split_id=key,
            seed=0,
            fit_count=1,
            holdout_count=1,
            fit_ids_sha256=hashlib.sha256((key + "/fit").encode()).hexdigest(),
            holdout_ids_sha256=hashlib.sha256((key + "/holdout").encode()).hexdigest(),
        )
        for key in rows
    ]
    provenance = BenchmarkTrialProvenance(
        command="frozen INS-LiDAR audit case", config_sha256=digest, input_sha256=digest
    )
    trials = []
    for key, value in rows.items():
        trials.append(
            BenchmarkTrial(
                method_id=ANCHOR.method_id,
                split_id=key,
                status="success",
                metrics={metric_name: 0.0},
                provenance=provenance,
            )
        )
        trials.append(
            BenchmarkTrial(
                method_id=METHOD_ID,
                split_id=key,
                status="failed" if isinstance(value, str) else "success",
                metrics={} if isinstance(value, str) else {metric_name: value},
                failure_reason=value if isinstance(value, str) else None,
                provenance=provenance,
            )
        )
    return BenchmarkDefinition(
        benchmark_id=f"{PROTOCOL_ID}-{family}-{metric_name}",
        title=f"INS-LiDAR {family}: {metric_name}",
        protocol=BenchmarkProtocol(
            protocol_id=PROTOCOL_ID,
            dataset_id=family,
            dataset_source_sha256=digest,
            data_license="CC BY 4.0" if family == "boreas" else "CC BY-NC-SA 3.0",
            split_policy=(
                "One recording per split; fit_count/holdout_count count run-level cases. "
                "Actual train/holdout blocks remain in the pinned native artifacts."
            ),
            splits=splits,
            initial_estimate_policy="Native defaults; no INS or reference-extrinsic odometry prior",
            tuning_policy="Method and gates frozen before evaluation; no tuning on these cases",
            failure_policy=(
                "All cases retained. No estimate fails agreement; missing results have "
                "zero coverage and control-miss fraction one."
            ),
        ),
        metrics=[
            BenchmarkMetricDefinition(
                name=metric_name,
                label=metric_name,
                unit="deg" if metric_name == "rotation_reference_difference_deg" else "ratio",
                direction="lower",
                primary=True,
                interpretation=("Lower is better; dataset calibration is a reference estimate."),
            )
        ],
        methods=[ANCHOR, METHOD],
        trials=trials,
        reference_method_id=ANCHOR.method_id,
        bootstrap_samples=2000,
        bootstrap_seed=0,
        limitations=[
            "The zero anchor is not an empirical competitor or a targetless method.",
            "Only estimated rotation axes are gated; coverage retains all axes and cases.",
            "Different KITTI dates share one platform; Boreas supplies the second platform.",
            "No absolute clock, translation, full-extrinsic or general superiority claim.",
        ],
        provenance=BenchmarkProvenance(
            generator="tools.build_ins_lidar_sota_audit",
            generator_version=__version__,
            git_commit=frozen,
            command="python tools/build_ins_lidar_sota_audit.py RUN PREREG OUT",
            source_artifacts=sources,
            data_verified=verified,
        ),
    )


def build(
    manifest_path: Path, prereg_path: Path, out: Path, *, source_root: Path
) -> dict[str, Any]:
    """Verify source/input custody, build three benchmarks per family, then audit all gates."""
    prereg = load_preregistration(prereg_path, source_root)
    verify_frozen_source(prereg, source_root, require_main=False)
    manifest = RunManifest.model_validate_json(manifest_path.read_text()).model_dump(mode="json")
    if (
        manifest["preregistration_sha256"] != file_digest(prereg_path)
        or manifest["frozen_code_commit"] != prereg.frozen_code_commit
    ):
        raise ValueError("run does not match the frozen pre-registration")
    cases = [CaseSpec.model_validate(c) for c in manifest["cases"]]
    identities = tuple((c.case_id, c.dataset_family, c.rig_id) for c in cases)
    if identities != EXPECTED_CASES:
        raise ValueError("run dropped or substituted a declared case")
    for declared, case in zip(prereg.evaluation, cases, strict=True):
        if declared.drive_ids != case.drive_ids:
            raise ValueError("declared recording IDs differ")
    verified = manifest.get("input_data_verified") is True
    for case in cases:
        info = manifest.get("input_receipts", {}).get(case.case_id)
        if info is None:
            verified = False
            continue
        path = Path(info["path"])
        if file_digest(path) != info["sha256"]:
            raise ValueError("input acquisition receipt changed")
        receipt = json.loads(path.read_text())
        if (
            receipt["case_id"] != case.case_id
            or receipt["verified"] is not True
            or receipt["provenance"]["frozen_code_commit"] != prereg.frozen_code_commit
            or receipt["provenance"]["source_metadata_sha256"]
            != prereg.metadata[case.dataset_family].sha256
        ):
            raise ValueError("input acquisition identity differs")
        verify_input_receipt(receipt, Path(info["work_dir"]))
    scores = score_artifacts(cases, manifest_path.parent)
    sources = {
        str(manifest_path): file_digest(manifest_path),
        str(prereg_path): file_digest(prereg_path),
    }
    if out.exists():
        raise ValueError("preserve the previous audit output; choose a new directory")
    out.mkdir(parents=True)
    (out / "scores.json").write_text(
        json.dumps(
            {
                "schema_version": "calibrex.ins_lidar_audit_scores/v0.1",
                "provenance": {
                    "generator": "tools.build_ins_lidar_sota_audit",
                    "frozen_code_commit": prereg.frozen_code_commit,
                    "source_sha256": sources,
                    "data_verified": verified,
                },
                **scores,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    requirements = []
    for family, rig in (("kitti_raw", "kitti-platform"), ("boreas", "boreas-platform")):
        group = [r for r in scores["cases"] if r["case"]["dataset_family"] == family]
        agreement = {
            r["case"]["case_id"]: (
                max(r["errors_deg"])
                if r["completed"] and r["errors_deg"]
                else "no scorable rotation estimate"
            )
            for r in group
        }
        unobserved = {r["case"]["case_id"]: 1 - r["estimated_axes"] / 3 for r in group}
        controls = {
            r["case"]["case_id"]: (
                r["control_misses"] / r["estimated_axes"] if r["estimated_axes"] else 1.0
            )
            for r in group
        }
        definitions = {}
        for metric, rows in (
            ("rotation_reference_difference_deg", agreement),
            ("unobserved_rotation_fraction", unobserved),
            ("missed_observed_control_fraction", controls),
        ):
            definition = _benchmark(
                family,
                metric,
                rows,
                file_digest(manifest_path),
                verified,
                sources,
                prereg.frozen_code_commit,
            )
            artifact = aggregate_benchmark_definition(definition)
            path = out / f"{family}_{metric}.yaml"
            artifact.save(path)
            definitions[metric] = path
        gates = [
            ("completion", "rotation_reference_difference_deg", "failure_rate", 0.0),
            ("agreement", "rotation_reference_difference_deg", "metric_maximum", ERROR_MAX_DEG),
            ("coverage", "unobserved_rotation_fraction", "metric_mean", UNOBSERVED_FRACTION_MAX),
            (
                "controls",
                "missed_observed_control_fraction",
                "metric_maximum",
                CONTROL_MISS_FRACTION_MAX,
            ),
        ]
        for name, metric, statistic, threshold in gates:
            path = definitions[metric]
            requirements.append(
                SotaClaimRequirement(
                    requirement_id=f"{family}-{name}",
                    phase="phase2",
                    description=f"{family}: {name} under the frozen per-axis contract",
                    dataset_family=family,
                    independent_rig=True,
                    rig_id=rig,
                    evidence_path=path.name,
                    evidence_sha256=file_digest(path),
                    method_id=METHOD_ID,
                    metric=metric if statistic != "failure_rate" else None,
                    statistic=statistic,
                    comparison="less_equal",
                    threshold=threshold,
                )
            )
    protocol = SotaAuditProtocol(
        protocol_id=PROTOCOL_ID,
        scope=SotaClaimScope.model_validate(SCOPE),
        claim_text=CLAIM,
        minimum_dataset_families=2,
        minimum_independent_rigs=2,
        requirements=requirements,
        provenance=SotaAuditProvenance(
            generator="tools.build_ins_lidar_sota_audit",
            generator_version=__version__,
            git_commit=prereg.frozen_code_commit,
            command=[
                "python",
                "tools/build_ins_lidar_sota_audit.py",
                str(manifest_path),
                str(prereg_path),
                str(out),
            ],
            source_sha256=sources,
        ),
    )
    protocol_path = out / "ins_lidar_sota_protocol.yaml"
    protocol.save(protocol_path)
    audit = audit_sota_claim(protocol_path)
    audit.save(out / "ins_lidar_sota_audit.yaml")
    return {
        "verdict": audit.verdict,
        "data_verified": verified,
        "scores": scores,
        "achieved": sum(r.status == "achieved" for r in audit.requirements),
        "requirements": len(audit.requirements),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("preregistration", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    result = build(
        args.manifest,
        args.preregistration,
        args.output,
        source_root=Path(__file__).resolve().parents[1],
    )
    print(json.dumps({k: v for k, v in result.items() if k != "scores"}))
    return 0 if result["verdict"] == "supported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
