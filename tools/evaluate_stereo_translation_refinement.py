"""Run the frozen translation experiment; reference scoring occurs after each fit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from tools.run_camera_lidar_development_matrix import digest, pose_metrics


class RefinementCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    bundle: str
    d2d_sha256: str


class RefinementPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: str
    sampling: Literal["uniform", "point_priority"] = "uniform"
    timeout_seconds: int
    cases: list[RefinementCase]
    source_sha256: dict[str, str]


class RefinementEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    plan_sha256: str
    driver_sha256: str
    cases: list[dict[str, Any]]


def evaluate(plan_path: Path, output: Path) -> None:
    """Preserve all eight cases, raw candidates, fallbacks and stage failures."""
    plan = RefinementPlan.model_validate_json(plan_path.read_text())
    if len(plan.cases) != 8 or len({c.case_id for c in plan.cases}) != 8:
        raise ValueError("the frozen experiment requires eight distinct cases")
    for p, expected in plan.source_sha256.items():
        if digest(Path(p)) != expected:
            raise ValueError(f"frozen source changed: {p}")
    output.mkdir(parents=True, exist_ok=False)
    records = [
        {"case_id": c.case_id, "status": "not_completed", "stage_recovered": False}
        for c in plan.cases
    ]
    report = RefinementEvaluation(
        plan_sha256=digest(plan_path), driver_sha256=digest(Path(__file__)), cases=records
    )

    def save() -> None:
        report.cases = records
        (output / "evaluation.json").write_text(report.model_dump_json(indent=2) + "\n")
        (output / "evaluation.schema.json").write_text(
            json.dumps(RefinementEvaluation.model_json_schema(), indent=2) + "\n"
        )

    try:
        for case, record in zip(plan.cases, records, strict=True):
            bundle = Path(case.bundle)
            if digest(bundle / "d2d.json") != case.d2d_sha256:
                raise ValueError("D2D initializer differs from the frozen baseline")
            destination = output / case.case_id
            destination.mkdir()
            command = [
                "taskset",
                "-c",
                ",".join(map(str, sorted(os.sched_getaffinity(0))[:4])),
                sys.executable,
                "tools/run_stereo_translation_refinement.py",
                str(bundle),
                str(destination / "refinement.json"),
                "--sampling",
                plan.sampling,
            ]
            with (destination / "execution.log").open("w") as log:
                try:
                    code = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "4"},
                        timeout=plan.timeout_seconds,
                    ).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            # Only the scorer reads the reference, after the refinement subprocess ended.
            seed_path = bundle / "seed.json"
            reference = np.array(json.loads(seed_path.read_text())["reference_T_camera_lidar"])
            initial = np.array(
                json.loads((bundle / "d2d.json").read_text())["output_T_camera_lidar"]
            )
            fit = json.loads((destination / "refinement.json").read_text()) if code == 0 else None
            accepted = bool(fit is not None and fit["result"]["accepted"])
            candidate = np.array(fit["result"]["candidate_T_camera_lidar"]) if fit else None
            returned = np.array(fit["result"]["output_T_camera_lidar"]) if fit else initial
            if fit:
                np.testing.assert_array_equal(initial[:3, :3], returned[:3, :3])
            before = pose_metrics(initial, reference, True)
            after = pose_metrics(returned, reference, True)
            raw = pose_metrics(candidate, reference, code == 0)
            record.update(
                returncode=code,
                status=fit["result"]["status"] if fit else "execution_failed",
                accepted=accepted,
                baseline=before,
                returned=after,
                raw_candidate=raw,
                stage_recovered=accepted and after["recovered"],
                translation_improved=after["translation_m"] < before["translation_m"],
                translation_delta_m=after["translation_m"] - before["translation_m"],
                reference_file_sha256=digest(seed_path),
            )
            save()
            print(
                case.case_id,
                record["status"],
                record["translation_delta_m"],
                after["recovered"],
                flush=True,
            )
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    evaluate(a.plan, a.output)
