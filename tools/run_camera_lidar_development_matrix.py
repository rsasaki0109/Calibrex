"""Execute a frozen spent-data assisted-input matrix without outcome-based retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation
from tools.kitti_aligned_stereo_bundle import build
from tools.kitti_drive_fetch import central_directory, extract, fetch_range, head


class DrivePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    frame_count: int
    frames: list[int]
    archive_size: int
    archive_etag: str


class PerturbationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    rotation_deg: tuple[float, float, float]
    translation_m: tuple[float, float, float]


class DevelopmentMatrixPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: str
    selection_rule: str
    timeout_seconds: int
    expected_cases: int
    drives: list[DrivePlan]
    perturbations: list[PerturbationPlan]
    source_sha256: dict[str, str]


class MatrixResult(BaseModel):
    """All scheduled cases, including failed preparation/execution."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    plan_sha256: str
    driver_sha256: str
    records: list[dict[str, Any]]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_metrics(
    matrix: np.ndarray | None, reference: np.ndarray, completed: bool
) -> dict[str, Any]:
    """A failed or invalid execution never becomes a successful recovery."""
    valid = matrix is not None and matrix.shape == (4, 4) and np.isfinite(matrix).all()
    if valid:
        valid = bool(
            np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            and abs(np.linalg.det(matrix[:3, :3]) - 1) < 1e-6
        )
    if not valid:
        return {
            "valid_output": False,
            "rotation_deg": None,
            "translation_m": None,
            "recovered": False,
        }
    rotation = float(
        np.degrees(Rotation.from_matrix(matrix[:3, :3] @ reference[:3, :3].T).magnitude())
    )
    translation = float(np.linalg.norm(matrix[:3, 3] - reference[:3, 3]))
    return {
        "valid_output": True,
        "rotation_deg": rotation,
        "translation_m": translation,
        "recovered": bool(completed and rotation < 0.5 and translation < 0.20),
    }


def run(plan_path: Path, source: Path, output: Path, container: str) -> MatrixResult:
    """Run sequentially; restore the runtime entrypoint even after a failure."""
    plan = DevelopmentMatrixPlan.model_validate_json(plan_path.read_text()).model_dump()
    scheduled = sum(len(d["frames"]) for d in plan["drives"]) * len(plan["perturbations"])
    if scheduled != plan["expected_cases"]:
        raise ValueError("case inventory and denominator disagree")
    for path, expected in plan["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"frozen source changed: {path}")
    output.mkdir(parents=True, exist_ok=False)
    # This existing container mounts output.parent as /data.
    expected_mount = subprocess.check_output(
        [
            "docker",
            "--host=unix:///var/run/docker.sock",
            "inspect",
            "--format",
            "{{json .Mounts}}",
            container,
        ],
        text=True,
        env=docker_environment(),
    )
    mounts = json.loads(expected_mount)
    if not any(
        m["Destination"] == "/data" and Path(m["Source"]).resolve() == output.parent.resolve()
        for m in mounts
    ):
        raise ValueError("output must be directly below the existing /data bind mount")
    docker = ["docker", "--host=unix:///var/run/docker.sock"]
    environment = docker_environment()
    original = output / "original-entrypoint.sh"
    subprocess.run(
        [*docker, "cp", f"{container}:/ros_entrypoint.sh", str(original)],
        check=True,
        env=environment,
    )
    wrapper = output / "runtime-entrypoint.sh"
    wrapper.write_text(f"#!/bin/bash\nexec /bin/bash /data/{output.name}/job.sh\n")
    wrapper.chmod(0o755)
    records = []
    result = MatrixResult(
        plan_sha256=digest(plan_path), driver_sha256=digest(Path(__file__)), records=records
    )

    def save() -> None:
        result.records = records
        (output / "matrix.json").write_text(result.model_dump_json(indent=2) + "\n")
        (output / "matrix.schema.json").write_text(
            json.dumps(MatrixResult.model_json_schema(), indent=2) + "\n"
        )

    try:
        subprocess.run(
            [*docker, "cp", str(wrapper), f"{container}:/ros_entrypoint.sh"],
            check=True,
            env=environment,
        )
        for drive in plan["drives"]:
            key = drive["id"]
            date = key[:10]
            number = key[-4:]
            right_root = output / key
            right_root.mkdir()
            receipt = source / f"{key}_fetch.json"
            pin = json.loads(receipt.read_text())
            acquisition_error = None
            try:
                size, etag = head(pin["url"])
                if [size, etag] != [drive["archive_size"], drive["archive_etag"]]:
                    raise ValueError("archive identity changed")
                infos = central_directory(pin["url"], size, etag)
                prefix = f"{date}/{date}_drive_{number}_sync/image_03/"
                wanted = [
                    prefix + "timestamps.txt",
                    *[prefix + f"data/{i:010d}.png" for i in drive["frames"]],
                ]
                fetched = []
                for name in wanted:
                    info = next(i for i in infos if i.filename == name)
                    end = min(
                        size - 1,
                        info.header_offset
                        + 30
                        + len(info.filename.encode())
                        + len(info.extra)
                        + info.compress_size
                        + 4096,
                    )
                    data = extract(
                        fetch_range(pin["url"], info.header_offset, end, etag),
                        info.header_offset,
                        info,
                    )
                    path = right_root / Path(name).name
                    path.write_bytes(data)
                    fetched.append({"member": name, "sha256": digest(path), "bytes": len(data)})
                (right_root / "fetch.json").write_text(
                    json.dumps({"url": pin["url"], "etag": etag, "members": fetched}, indent=2)
                    + "\n"
                )
            except Exception as exc:
                acquisition_error = f"{type(exc).__name__}: {exc}"
            bag = source / "bags" / key / f"{key}_0.db3"
            metadata = json.loads((source / "bags" / key / "calibrex_conversion.json").read_text())
            reference = np.array(metadata["camera"]["t_cam_velo"])
            for frame in drive["frames"]:
                shared = right_root / f"frame-{frame}"
                preparation_error = acquisition_error
                if preparation_error is None:
                    try:
                        build(
                            bag,
                            source / "raw" / date / f"{date}_drive_{number}_sync",
                            source / "calibration" / date,
                            right_root / f"{frame:010d}.png",
                            right_root / "timestamps.txt",
                            frame,
                            shared,
                        )
                    except Exception as exc:
                        preparation_error = f"{type(exc).__name__}: {exc}"
                for perturbation in plan["perturbations"]:
                    case_id = f"{key}-{frame:010d}-{perturbation['id']}"
                    record = {
                        "case_id": case_id,
                        "drive": key,
                        "frame": frame,
                        "preparation_error": preparation_error,
                        "methods": {},
                    }
                    records.append(record)
                    initial = reference.copy()
                    initial[:3, :3] = (
                        Rotation.from_euler(
                            "xyz", perturbation["rotation_deg"], degrees=True
                        ).as_matrix()
                        @ reference[:3, :3]
                    )
                    initial[:3, 3] += perturbation["translation_m"]
                    record["initial"] = pose_metrics(initial, reference, True)
                    if preparation_error is not None:
                        for method in ["d2d", "koide"]:
                            record["methods"][method] = {
                                "status": "preparation_failed",
                                **pose_metrics(None, reference, False),
                            }
                        save()
                        continue
                    case = output / case_id
                    case.mkdir()
                    for name in [
                        "aligned.png",
                        "aligned.ply",
                        "depth.npy",
                        "frontend.json",
                        "calib.json",
                    ]:
                        shutil.copyfile(shared / name, case / name)
                    seed = {
                        "reference_T_camera_lidar": reference.tolist(),
                        "initial_T_camera_lidar": initial.tolist(),
                    }
                    (case / "seed.json").write_text(json.dumps(seed, indent=2) + "\n")
                    command = [
                        sys.executable,
                        "tools/run_aligned_stereo_d2d.py",
                        str(case),
                        str(case / "seed.json"),
                        str(case / "d2d.json"),
                    ]
                    cpus = sorted(os.sched_getaffinity(0))[:4]
                    command = ["taskset", "-c", ",".join(map(str, cpus)), *command]
                    native_env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "4"}
                    with (case / "d2d.log").open("w") as log:
                        try:
                            status = subprocess.run(
                                command,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                env=native_env,
                                timeout=plan["timeout_seconds"],
                            ).returncode
                        except subprocess.TimeoutExpired:
                            status = 124
                    native = None
                    native_status = "execution_failed"
                    if status == 0:
                        native = json.loads((case / "d2d.json").read_text())
                        native_status = native["result"]["status"]
                    record["methods"]["d2d"] = {
                        "returncode": status,
                        "status": native_status,
                        **pose_metrics(
                            None if native is None else np.array(native["output_T_camera_lidar"]),
                            reference,
                            status == 0 and native_status == "converged",
                        ),
                    }
                    calib = json.loads((case / "calib.json").read_text())
                    inverse = np.linalg.inv(initial)
                    calib["results"] = {
                        "init_T_lidar_camera": [
                            *inverse[:3, 3].tolist(),
                            *Rotation.from_matrix(inverse[:3, :3]).as_quat().tolist(),
                        ]
                    }
                    (case / "calib.json").write_text(json.dumps(calib, indent=2) + "\n")
                    shutil.copyfile(case / "calib.json", case / "calib-seeded.json")
                    (output / "job.sh").write_text(
                        "#!/bin/bash\nset -e\nsource /opt/ros/humble/setup.bash\n"
                        "source /root/ros2_ws/install/setup.bash\n"
                        "ros2 run direct_visual_lidar_calibration calibrate "
                        f"/data/{output.name}/{case_id} --background --auto_quit\n"
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
                    parse_error = None
                    try:
                        values = json.loads((case / "calib.json").read_text())["results"][
                            "T_lidar_camera"
                        ]
                        if (
                            len(values) != 7
                            or not np.isfinite(values).all()
                            or abs(np.linalg.norm(values[3:]) - 1) > 1e-6
                        ):
                            raise ValueError("invalid output transform")
                        inverse = np.eye(4)
                        inverse[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
                        inverse[:3, 3] = values[:3]
                        matrix = np.linalg.inv(inverse)
                    except (KeyError, ValueError) as exc:
                        parse_error = str(exc)
                    record["methods"]["koide"] = {
                        "returncode": state["ExitCode"],
                        "status": "completed" if state["ExitCode"] == 0 else "execution_failed",
                        "state": state,
                        "parse_error": parse_error,
                        **pose_metrics(matrix, reference, state["ExitCode"] == 0),
                    }
                    save()
                    print(
                        case_id,
                        {m: record["methods"][m]["recovered"] for m in ["d2d", "koide"]},
                        flush=True,
                    )
    finally:
        present = {r["case_id"] for r in records}
        for drive in plan["drives"]:
            for frame in drive["frames"]:
                for perturbation in plan["perturbations"]:
                    case_id = f"{drive['id']}-{frame:010d}-{perturbation['id']}"
                    if case_id not in present:
                        records.append(
                            {
                                "case_id": case_id,
                                "drive": drive["id"],
                                "frame": frame,
                                "preparation_error": "matrix interrupted before case",
                                "methods": {},
                            }
                        )
        for record in records:
            for method in ["d2d", "koide"]:
                if method not in record["methods"]:
                    record["methods"][method] = {
                        "status": "not_completed",
                        **pose_metrics(None, np.eye(4), False),
                    }
        save()
        subprocess.run(
            [*docker, "cp", str(original), f"{container}:/ros_entrypoint.sh"],
            check=True,
            env=environment,
        )
        save()
    return result


def docker_environment() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_TLS",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["plan", "source", "output"]:
        parser.add_argument(name, type=Path)
    parser.add_argument("--container", default="calibrex-koide-runtime-20261028")
    a = parser.parse_args()
    run(a.plan, a.source, a.output, a.container)


if __name__ == "__main__":
    main()
