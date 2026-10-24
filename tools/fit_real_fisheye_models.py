"""Fit MEI and Double Sphere stereo calibrations to real OpenCV board corners.

The supplied KB4 calibration initializes intrinsics and board poses. Each new model is fit
to measured 2D/3D corner observations, never to rendered images or KB4 rays.
Every fifth capture is held out; its pose is estimated using left corners only.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from calibrex.evaluation.fisheye_stereo import EquidistantCamera
from calibrex.evaluation.stereo_camera_models import DoubleSphereCamera, MeiCamera

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/fisheye-stereo-real"


def observations() -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Read original measured corners and board coordinates, without OpenCV."""
    output = []
    for name, dim in [("object", 3), ("left", 2), ("right", 2)]:
        root = ET.fromstring((ASSETS / "source" / (name + ".xml")).read_bytes())
        output.append(
            [
                np.fromstring(root.find(f"image_{i}").text, sep=" ").reshape(-1, dim)
                for i in range(34)
            ]
        )
    return output[0], output[1], output[2]


def camera(code: str, p: np.ndarray) -> Any:
    """Construct native cameras from fitted full-resolution parameters."""
    if code == "ds":
        return DoubleSphereCamera(*p[:6], 1280, 800)
    return MeiCamera(*p[:5], tuple(p[5:9]), 1280, 800)


def initialize_pose(xyz: np.ndarray, uv: np.ndarray, kb4: EquidistantCamera) -> np.ndarray:
    """Initialize a front-facing board pose from a bearing homography."""
    rays = kb4.unproject(uv[:, 0], uv[:, 1])
    norm = rays[:, :2] / rays[:, 2, None]
    rows = []
    for (x, y, _), (u, v) in zip(xyz, norm, strict=True):
        rows += [[-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
    _, _, vt = np.linalg.svd(rows)
    h = vt[-1].reshape(3, 3)
    if h[2, 2] < 0:
        h = -h
    scale = 2 / (np.linalg.norm(h[:, 0]) + np.linalg.norm(h[:, 1]))
    r1, r2, t = h[:, 0] * scale, h[:, 1] * scale, h[:, 2] * scale
    u, _, vt = np.linalg.svd(np.stack([r1, r2, np.cross(r1, r2)], axis=1))
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return np.r_[Rotation.from_matrix(rotation).as_rotvec(), t]


def project(code: str, p: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Project real board points with native camera geometry."""
    u, v, valid = camera(code, p).project(xyz)
    result = np.c_[u, v]
    result[~valid] = 1e5
    return np.nan_to_num(result, nan=1e5, posinf=1e5, neginf=-1e5)


def fit(code: str) -> dict[str, Any]:
    """Fit train captures and report untouched right-image reprojection diagnostics."""
    objects, left, right = observations()
    supplied = json.loads((ASSETS / "calibration.json").read_text())
    train = [i for i in range(34) if i % 5 != 0]
    hold = [i for i in range(34) if i % 5 == 0]
    init = []
    cameras = []
    for side in ["left", "right"]:
        k = supplied[side]
        fx, fy, cx, cy = k["intrinsics"]
        cameras.append(EquidistantCamera(fx, fy, cx, cy, tuple(k["distortion"]), 1280, 800))
        init.extend(
            [fx * 1.5, fy * 1.5, cx, cy, 0.5, 0.5]
            if code == "ds"
            else [fx * 1.8, fy * 1.8, cx, cy, 0.8, 0, 0, 0, 0]
        )
    n = 6 if code == "ds" else 9
    matrix = np.array(supplied["right_from_left"])
    init.extend(np.r_[Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]])
    poses = [initialize_pose(objects[i], left[i], cameras[0]) for i in range(34)]
    for i in train:
        init.extend(poses[i])
    per_view = len(objects[0]) * 4
    sparsity = lil_matrix((len(train) * per_view, len(init)), dtype=int)
    for j in range(len(train)):
        rows = slice(j * per_view, (j + 1) * per_view)
        sparsity[rows, 2 * n + 6 + j * 6 : 2 * n + 12 + j * 6] = 1
        sparsity[j * per_view : j * per_view + per_view // 2, :n] = 1
        sparsity[j * per_view + per_view // 2 : (j + 1) * per_view, n : 2 * n + 6] = 1

    def residual(p: np.ndarray) -> np.ndarray:
        stereo_r = Rotation.from_rotvec(p[2 * n : 2 * n + 3]).as_matrix()
        stereo_t = p[2 * n + 3 : 2 * n + 6]
        values = []
        for j, i in enumerate(train):
            pose = p[2 * n + 6 + j * 6 : 2 * n + 12 + j * 6]
            xyz = objects[i] @ Rotation.from_rotvec(pose[:3]).as_matrix().T + pose[3:]
            values += [
                (project(code, p[:n], xyz) - left[i]).ravel(),
                (project(code, p[n : 2 * n], xyz @ stereo_r.T + stereo_t) - right[i]).ravel(),
            ]
        return np.concatenate(values)

    low_one = (
        [100, 100, 0, 0, 0.001, 0.01] if code == "ds" else [100, 100, 0, 0, 0.001, -1, -1, -1, -1]
    )
    high_one = (
        [4000, 4000, 1280, 800, 3, 0.99] if code == "ds" else [4000, 4000, 1280, 800, 3, 1, 1, 1, 1]
    )
    lower = low_one * 2 + [-np.inf] * (len(init) - 2 * n)
    upper = high_one * 2 + [np.inf] * (len(init) - 2 * n)
    result = least_squares(
        residual,
        init,
        bounds=(lower, upper),
        jac_sparsity=sparsity,
        x_scale="jac",
        max_nfev=400,
        ftol=1e-5,
        xtol=1e-8,
        gtol=1e-6,
    )
    p = result.x
    tf = np.eye(4)
    tf[:3, :3] = Rotation.from_rotvec(p[2 * n : 2 * n + 3]).as_matrix()
    tf[:3, 3] = p[2 * n + 3 : 2 * n + 6]
    errors = []
    for j, i in enumerate(train):
        poses[i] = p[2 * n + 6 + j * 6 : 2 * n + 12 + j * 6].copy()
    for i in hold:

        def left_residual(pose: np.ndarray, i: int = i) -> np.ndarray:
            xyz = objects[i] @ Rotation.from_rotvec(pose[:3]).as_matrix().T + pose[3:]
            return (project(code, p[:n], xyz) - left[i]).ravel()

        pose = least_squares(left_residual, poses[i], max_nfev=100).x
        poses[i] = pose.copy()
        xyz = objects[i] @ Rotation.from_rotvec(pose[:3]).as_matrix().T + pose[3:]
        errors.extend(
            np.linalg.norm(
                project(code, p[n : 2 * n], xyz @ tf[:3, :3].T + tf[:3, 3]) - right[i], axis=1
            )
        )
    output = {
        "model": code,
        "parameters_left": p[:n].tolist(),
        "parameters_right": p[n : 2 * n].tolist(),
        "right_from_left": tf.tolist(),
        "image_size": [1280, 800],
        "board_poses_left": {str(i): pose.tolist() for i, pose in enumerate(poses)},
        "train_capture_ids": train,
        "holdout_capture_ids": hold,
        "train_reprojection_rmse_px": float(np.sqrt(np.mean(residual(p) ** 2) * 2)),
        "holdout_right_reprojection_rmse_px": float(np.sqrt(np.mean(np.array(errors) ** 2))),
        "solver_success": bool(result.success),
        "solver_message": result.message,
        "evaluations": result.nfev,
        "method": (
            "Joint stereo least squares on measured OpenCV 2D corners and board "
            "coordinates. KB4 initializes board poses only. Held-out poses use left "
            "corners only; right corners are untouched. "
        ),
        "limitation": (
            "Demonstration calibration on one physical rig; no independent metric "
            "depth truth or general lens accuracy claim. "
        ),
    }
    (ASSETS / f"calibration_{code}.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        code,
        output["train_reprojection_rmse_px"],
        output["holdout_right_reprojection_rmse_px"],
        result.nfev,
        result.message,
        flush=True,
    )
    return output


if __name__ == "__main__":
    for name in ["mei", "ds"]:
        fit(name)
