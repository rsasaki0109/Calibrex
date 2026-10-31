"""Execute the frozen four-case joint experiment from saved single-frame inputs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tools.run_camera_lidar_development_matrix import (
    DevelopmentMatrixPlan,
    MatrixResult,
    digest,
    docker_environment,
    pose_metrics,
)


def run(plan_path: Path, source: Path, output: Path) -> None:
    """Reuse exact arrays, fit each method once, and retain all scheduled cases."""
    plan = DevelopmentMatrixPlan.model_validate_json(plan_path.read_text())
    assert plan.expected_cases == len(plan.drives) * len(plan.perturbations) == 4
    for name, expected in plan.source_sha256.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"changed frozen source: {name}")
    output.mkdir(exist_ok=False)
    records = [
        {"case_id": f"{d.id}-{p.id}", "drive": d.id, "seed": p.id, "methods": {}}
        for d in plan.drives
        for p in plan.perturbations
    ]
    report = MatrixResult(
        plan_sha256=digest(plan_path), driver_sha256=digest(Path(__file__)), records=records
    )
    environment = docker_environment()
    docker = ["docker", "--host=unix:///var/run/docker.sock"]
    container = "calibrex-koide-runtime-20261028"
    mounts = json.loads(
        subprocess.check_output(
            [*docker, "inspect", "--format", "{{json .Mounts}}", container],
            env=environment,
            text=True,
        )
    )
    if not any(
        m["Destination"] == "/data" and Path(m["Source"]).resolve() == output.parent.resolve()
        for m in mounts
    ):
        raise ValueError("output.parent must be the existing /data bind mount")
    original = output / "original-entrypoint.sh"
    subprocess.run(
        [*docker, "cp", f"{container}:/ros_entrypoint.sh", str(original)],
        env=environment,
        check=True,
    )
    wrapper = output / "runtime-entrypoint.sh"
    wrapper.write_text(f"#!/bin/bash\nexec /bin/bash /data/{output.name}/job.sh\n")
    wrapper.chmod(0o755)

    def save() -> None:
        report.records = records
        (output / "matrix.json").write_text(report.model_dump_json(indent=2) + "\n")
        (output / "matrix.schema.json").write_text(
            json.dumps(MatrixResult.model_json_schema(), indent=2) + "\n"
        )

    try:
        subprocess.run(
            [*docker, "cp", str(wrapper), f"{container}:/ros_entrypoint.sh"],
            env=environment,
            check=True,
        )
        for record in records:
            drive = next(d for d in plan.drives if d.id == record["drive"])
            case = output / record["case_id"]
            case.mkdir()
            names = []
            input_map = {}
            camera = None
            seed = None
            for frame in drive.frames:
                shared = source / drive.id / f"frame-{frame}"
                frontend = json.loads((shared / "frontend.json").read_text())
                config = json.loads((shared / "calib.json").read_text())
                if camera is not None and camera != config["camera"]:
                    raise ValueError("joint frames must share the exact camera model")
                camera = config["camera"]
                name = f"frame-{frame}"
                names.append(name)
                for filename, suffix in [
                    ("aligned.png", ".png"),
                    ("aligned.ply", ".ply"),
                    ("depth.npy", ".depth.npy"),
                ]:
                    path = shared / filename
                    if digest(path) != frontend["output_sha256"][filename]:
                        raise ValueError(f"changed source data: {path}")
                    destination = case / (name + suffix)
                    shutil.copyfile(path, destination)
                    input_map[destination.name] = {
                        "source": str(path),
                        "sha256": digest(destination),
                    }
                single = source / f"{drive.id}-{frame:010d}-{record['seed']}"
                current = json.loads((single / "seed.json").read_text())
                if seed is not None and seed != current:
                    raise ValueError("single-frame seeds must be identical")
                seed = current
            config = {"camera": camera, "meta": {"bag_names": names}}
            (case / "calib.json").write_text(json.dumps(config, indent=2) + "\n")
            shutil.copyfile(case / "calib.json", case / "calib-frontend.json")
            (case / "frontend.json").write_text(json.dumps(input_map, indent=2) + "\n")
            (case / "seed.json").write_text(json.dumps(seed, indent=2) + "\n")
            reference = np.array(seed["reference_T_camera_lidar"])
            initial = np.array(seed["initial_T_camera_lidar"])
            record["initial"] = pose_metrics(initial, reference, True)
            command = [
                "taskset",
                "-c",
                ",".join(map(str, sorted(os.sched_getaffinity(0))[:4])),
                sys.executable,
                "tools/run_joint_stereo_d2d.py",
                str(case),
                str(case / "seed.json"),
                str(case / "d2d.json"),
            ]
            with (case / "d2d.log").open("w") as log:
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
            native = json.loads((case / "d2d.json").read_text()) if code == 0 else None
            status = native["result"]["status"] if native else "execution_failed"
            support = native["result"]["final_evaluation"]["evaluated_frame_count"] if native else 0
            record["methods"]["d2d"] = {
                "returncode": code,
                "status": status,
                "supporting_frames": support,
                **pose_metrics(
                    np.array(native["output_T_camera_lidar"]) if native else None,
                    reference,
                    code == 0 and status == "converged" and support == 3,
                ),
            }
            inverse = np.linalg.inv(initial)
            config["results"] = {
                "init_T_lidar_camera": [
                    *inverse[:3, 3].tolist(),
                    *Rotation.from_matrix(inverse[:3, :3]).as_quat().tolist(),
                ]
            }
            (case / "calib.json").write_text(json.dumps(config, indent=2) + "\n")
            shutil.copyfile(case / "calib.json", case / "calib-seeded.json")
            (output / "job.sh").write_text(
                "#!/bin/bash\nset -e\nsource /opt/ros/humble/setup.bash\n"
                "source /root/ros2_ws/install/setup.bash\n"
                "ros2 run direct_visual_lidar_calibration calibrate "
                f"/data/{output.name}/{record['case_id']} --background --auto_quit\n"
            )
            with (case / "koide.log").open("w") as log:
                subprocess.run(
                    [*docker, "start", "-a", container],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            state = json.loads(
                subprocess.check_output(
                    [*docker, "inspect", "--format", "{{json .State}}", container],
                    env=environment,
                    text=True,
                )
            )
            matrix = None
            error = None
            try:
                values = json.loads((case / "calib.json").read_text())["results"]["T_lidar_camera"]
                if (
                    len(values) != 7
                    or not np.isfinite(values).all()
                    or abs(np.linalg.norm(values[3:]) - 1) > 1e-6
                ):
                    raise ValueError("invalid transform")
                inverse = np.eye(4)
                inverse[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
                inverse[:3, 3] = values[:3]
                matrix = np.linalg.inv(inverse)
            except (KeyError, ValueError) as exc:
                error = str(exc)
            record["methods"]["koide"] = {
                "returncode": state["ExitCode"],
                "state": state,
                "parse_error": error,
                **pose_metrics(matrix, reference, state["ExitCode"] == 0),
            }
            save()
            print(record["case_id"], record["methods"], flush=True)
    finally:
        for record in records:
            for method in ["d2d", "koide"]:
                record["methods"].setdefault(
                    method, {"status": "not_completed", **pose_metrics(None, np.eye(4), False)}
                )
        save()
        subprocess.run(
            [*docker, "cp", str(original), f"{container}:/ros_entrypoint.sh"],
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    run(*(Path(v) for v in sys.argv[1:]))
