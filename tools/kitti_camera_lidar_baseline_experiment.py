"""One fixed ablation: release the tested axis from each training-only control seed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from tools import kitti_camera_lidar_profile_probe as probe
from tools import kitti_camera_lidar_v3_replay as replay
from tools.kitti_camera_lidar_initialization_compare import DRIVES, ComparisonReport, sha

from calibrex.core.provenance import git_commit

ROOT = Path(__file__).resolve().parents[1]


def run(captures: Path, output: Path, *, boundary_tangent: bool = False,
        nearest_edge: bool = False) -> ComparisonReport:
    """Keep all fourteen inputs and failures; report controls and fold reference errors."""
    if boundary_tangent and nearest_edge:
        raise ValueError("choose one ablation")
    output.mkdir(parents=True, exist_ok=False)
    report = ComparisonReport(
        provenance={
            "experiment": (
                "nearest_edge_distance" if nearest_edge else
                "local_boundary_tangent" if boundary_tangent else "one_pass_control_seeded_baseline"
            ),
            "git_commit": git_commit(),
            "generator_sha256": sha(Path(__file__)),
            "probe_sha256": sha(Path(probe.__file__)),
            "estimator_sha256": sha(Path(probe.edge.__file__)),
            "new_evaluation_started": False,
            "new_accuracy_claim": False,
            "hypothesis": (
                "Bounded nearest-ridge distance reduces distant incorrect edge matches."
                if nearest_edge else "Local boundary directions reject incorrect edge alignments."
                if boundary_tangent
                else "A baseline trapped in a local basin may underestimate pitch controls."
            ),
            "selection": ("all fourteen reacquired spent inputs, paired objective ablation, "
                          "original search settings"),
        }
    )
    (output / "report.schema.json").write_text(json.dumps(report.model_json_schema(), indent=2))

    def write() -> None:
        (output / "report.json").write_text(report.model_dump_json(indent=2))

    write()
    for drive in DRIVES:
        for name, injection in replay.RUNS.items():
            row: dict[str, Any] = {"drive": drive, "run": name, "arms": {}}
            report.inputs.append(row)
            try:
                snapshot = captures / drive / name
                meta, _, _, candidate, _ = replay.load_snapshot(snapshot)
                if (meta.drive, meta.run) != (drive, name):
                    raise ValueError("wrong input identity")
                row.update(
                    arrays_sha256=meta.arrays_sha256,
                    snapshot_sha256=sha(snapshot / "snapshot.json"),
                )
                vector = np.zeros(3)
                if injection:
                    axis, degrees = injection.split(":")
                    vector["xyz".index(axis)] = np.radians(float(degrees))
                reference = Rotation.from_rotvec(-vector).as_matrix() @ candidate[:3, :3]
                for arm, enabled in [
                    ("original", False),
                    ("nearest_edge" if nearest_edge else
                     "boundary_tangent" if boundary_tangent else "control_seeded", True),
                ]:
                    tick = time.monotonic()
                    dest = output / drive / name / arm
                    result = probe.profile_snapshot(
                        snapshot,
                        None,
                        dest,
                        multi_start=True,
                        training_only_start=True,
                        control_seeded_baseline=enabled and not (boundary_tangent or nearest_edge),
                        nearest_edge=enabled and nearest_edge,
                        boundary_tangent=enabled and boundary_tangent,
                    )
                    values = []
                    for item in result.axes:
                        errors = []
                        for fold in item.folds:
                            rotation = probe.edge._transform(candidate, np.array(fold.base_theta))
                            delta = Rotation.from_matrix(rotation[:3, :3] @ reference.T).as_rotvec()
                            errors.append(float(np.degrees(delta)[item.axis]))
                        values.append(
                            {
                                "axis": item.axis,
                                "control": item.profiled_control.model_dump(mode="json"),
                                "fold_active_axis_reference_errors_deg": errors,
                            }
                        )
                    row["arms"][arm] = {
                        "elapsed_s": time.monotonic() - tick,
                        "axes": values,
                        "report_sha256": sha(dest / "report.json"),
                    }
                    if not enabled:
                        previous = ROOT / "docs/assets/camera_lidar_initialization_20261027"
                        path = previous / drive / name / "training_only/report.json"
                        prior = probe.ProfileReport.model_validate_json(path.read_bytes())
                        row["original_parity"] = replay._same(
                            [a.model_dump(mode="json") for a in result.axes],
                            [a.model_dump(mode="json") for a in prior.axes],
                        )
                        row["previous_report_sha256"] = sha(path)
                        if not row["original_parity"]:
                            raise ValueError("original arm numerical parity failed")
                    write()
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            write()
            print(drive, name, row.get("error", "complete"), flush=True)
    report.status = "blocked" if any("error" in r for r in report.inputs) else "complete"
    write()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary-tangent", action="store_true")
    args = parser.parse_args()
    return (
        0
        if run(args.captures, args.output, boundary_tangent=args.boundary_tangent).status
        == "complete"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
