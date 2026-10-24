"""Build browser method demos from measured ACFR, ETHZ and comma2k19 observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from calibrex.data.ethz_hand_eye import read_ethz_robot_arm_hand_eye_motions
from calibrex.evaluation.holdout import split_indices
from calibrex.solvers.acfr_board_edges import extract_acfr_vlp_board_edges
from calibrex.solvers.horn_point_lidar_camera_solver import HornPointLidarCameraSolver
from calibrex.solvers.native_planar_board_solver import (
    read_acfr_vlp_horn_point_observations,
    read_acfr_vlp_plane_observations,
    read_acfr_vlp_point_plane_observations,
)
from calibrex.solvers.park_martin_hand_eye_solver import (
    HandEyeMotionPair,
    evaluate_hand_eye_motions,
)
from calibrex.solvers.planar_board_lidar_camera_solver import (
    PlanarBoardLidarCameraSolver,
    PlanarBoardSolverOptions,
)
from calibrex.solvers.planar_board_line_plane_solver import PlanarBoardLinePlaneSolver
from calibrex.solvers.point_plane_lidar_camera_solver import PointPlaneLidarCameraSolver

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/assets/evidence-demos/source"
OUTPUT = ROOT / "docs/app/samples"


def digest(path: Path) -> str:
    """Bind a source to its exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(label: str, value: float | str, unit: str = "") -> dict[str, Any]:
    """Create a displayed diagnostic without inferring a policy verdict."""
    return {"label": label, "value": value, "unit": unit}


def cloud(name: str, points: Any) -> dict[str, Any]:
    """Retain measured or explicitly transformed coordinates in metres."""
    return {"name": name, "points": np.round(np.asarray(points), 7).tolist()}


def board() -> dict[str, Any]:
    """Compare four native methods on the same real ACFR fit/holdout split."""
    path = SOURCE / "acfr_poses.csv"
    if digest(path) != "024bc6ed9009652761e9c0df49b106d325a10c88d41a80e9b36ea55fd567e110":
        raise ValueError("Unexpected ACFR input")
    plane_obs = read_acfr_vlp_plane_observations(path)
    point_obs = read_acfr_vlp_horn_point_observations(path)
    plane = PlanarBoardLidarCameraSolver().solve(
        plane_obs,
        options=PlanarBoardSolverOptions(normal_sign_policy="preserve"),
    )
    assert plane.transform_camera_lidar is not None
    rows = np.loadtxt(path, delimiter=",").reshape(-1, 19, 3)
    extracted = extract_acfr_vlp_board_edges(
        rows.reshape(-1, 3).tolist(), plane.transform_camera_lidar
    )
    methods = [
        ("plane", "Plane-only", plane),
        (
            "point-plane",
            "Point + plane",
            PointPlaneLidarCameraSolver().solve(read_acfr_vlp_point_plane_observations(path)),
        ),
        ("horn", "Horn points", HornPointLidarCameraSolver().solve(point_obs)),
        ("line-plane", "Line + plane", PlanarBoardLinePlaneSolver().solve(extracted.observations)),
    ]
    views = []
    for code, label, result in methods:
        tf = result.transform_camera_lidar
        if tf is None:
            raise ValueError(f"No transform for {label}")
        if (
            result.train_frame_ids != plane.train_frame_ids
            or result.holdout_frame_ids != plane.holdout_frame_ids
        ):
            raise ValueError("Board method splits differ")
        selected = [o for o in point_obs if o.frame_id in result.holdout_frame_ids]
        target = np.array([o.camera_point_m for o in selected])
        aligned = np.array([tf.transform_point(o.lidar_point_m) for o in selected])
        views.append(
            {
                "id": code,
                "label": label,
                "kind": "cloud",
                "title": label + " / held-out board centres",
                "clouds": [
                    cloud("Camera observations", target),
                    cloud("Transformed LiDAR", aligned),
                ],
                "metrics": [
                    metric(
                        "Shared holdout centre RMSE",
                        float(np.sqrt(np.mean(np.sum((target - aligned) ** 2, axis=1)))),
                        "m",
                    ),
                    metric("Fit / holdout", f"{len(result.train_frame_ids)} / {len(selected)}"),
                    metric("Solver", result.status),
                ],
                "note": (
                    "Same measured board centres, same split. Centre RMSE is a common "
                    "diagnostic; each method optimizes a different objective. No "
                    "independent extrinsic truth. "
                ),
                "result": result.as_dict(),
            }
        )
    # Targetless method evidence is kept on its own dataset, never ranked against ACFR.
    targetless = []
    for name, key in [
        ("pandey", "lidar_camera_mutual_information_score"),
        ("levinson", "levinson_edge_objective"),
    ]:
        p = ROOT / "docs/assets/png_luminance_decoding" / f"{name}-00-current/artifacts/result.yaml"
        data = yaml.safe_load(p.read_text())
        targetless.append(
            {
                "method": name,
                "metric": data["metrics"][key],
                "source": str(p.relative_to(ROOT / "docs")),
            }
        )
    return {
        "id": "comparison",
        "title": "One board dataset. Four methods.",
        "dataset": "ACFR VLP-16 / 40 measured board captures",
        "license": "Apache-2.0",
        "views": views,
        "targetless_evidence": targetless,
        "live_problem": {
            "source": [o.lidar_point_m for o in point_obs],
            "target": [o.camera_point_m for o in point_obs],
            "fit_ids": list(methods[2][2].train_frame_ids),
            "ids": [o.frame_id for o in point_obs],
            "initial_transform": methods[2][2].transform_camera_lidar.as_dict(),
        },
    }


def handeye() -> dict[str, Any]:
    """Evaluate eight real-pose motion methods on disjoint absolute-pose blocks."""
    from generate_ethz_hand_eye_benchmarks import _native_ax_xb_methods

    path = SOURCE / "ethz_robot.zip"
    if digest(path) != "2454578f731e656a940ddf51017b56e8d535c58d16a50629b85326005dedd6c3":
        raise ValueError("Unexpected ETHZ archive")
    dataset = read_ethz_robot_arm_hand_eye_motions(path)
    motions = [HandEyeMotionPair(m.pair_id, m.motion_a, m.motion_b) for m in dataset.motions]
    train_ids, hold_ids = split_indices(len(motions), 0.25, seed=0)
    train = [motions[i] for i in train_ids]
    hold = [motions[i] for i in hold_ids]
    views = []
    for code, label, _doi, solve in _native_ax_xb_methods():
        result = solve(train)
        x = result.transform_x
        if x is None:
            raise ValueError(f"No hand-eye transform: {code}")
        evaluation = evaluate_hand_eye_motions(hold, x)
        views.append(
            {
                "id": code,
                "label": label.replace("Calibrex ", ""),
                "kind": "cloud",
                "title": "AX = XB / held-out motion closure",
                "clouds": [
                    cloud("A * X origins", [m.motion_a.compose(x).translation_m for m in hold]),
                    cloud("X * B origins", [x.compose(m.motion_b).translation_m for m in hold]),
                ],
                "metrics": [
                    metric("Rotation closure RMSE", evaluation.rotation_closure_rmse_deg, "deg"),
                    metric("Translation closure RMSE", evaluation.translation_closure_rmse_m, "m"),
                    metric("Fit / holdout motions", f"{len(train)} / {len(hold)}"),
                ],
                "note": (
                    "Origins of the two sides of each held-out motion equation, in "
                    "the same frame. Closure measures consistency, not absolute "
                    "calibration accuracy. Relative-motion blocks do not reuse "
                    "absolute poses. "
                ),
                "transform": x.as_dict(),
                "evaluation": asdict(evaluation),
            }
        )
    return {
        "id": "handeye",
        "title": "Make two motion chains meet.",
        "dataset": "ETHZ ASL / measured robot-arm and target poses",
        "license": "ETHZ non-commercial dataset; upstream code BSD-3-Clause",
        "views": views,
        "fit_ids": [m.pair_id for m in train],
        "holdout_ids": [m.pair_id for m in hold],
    }


def comma(name: str) -> np.ndarray:
    """Read an original NumPy sample without allowing pickled objects."""
    return np.load(SOURCE / "comma" / (name + ".npy"), allow_pickle=False)


def enu(points: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Rotate Earth-fixed displacements into a local tangent plane."""
    x, y, z = origin
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.hypot(x, y) * (1 - 6.69437999014e-3))
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    basis = np.array([[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])
    return (points - origin) @ basis.T


def gps_ecef(data: np.ndarray) -> np.ndarray:
    """WGS84 receiver latitude/longitude/height to Earth-fixed coordinates."""
    lat, lon = np.deg2rad(data[:, 0]), np.deg2rad(data[:, 1])
    n = 6378137 / np.sqrt(1 - 6.69437999014e-3 * np.sin(lat) ** 2)
    h = data[:, 4]
    return np.c_[
        (n + h) * np.cos(lat) * np.cos(lon),
        (n + h) * np.cos(lat) * np.sin(lon),
        (n * (1 - 6.69437999014e-3) + h) * np.sin(lat),
    ]


def gnss() -> dict[str, Any]:
    """Show measured receiver tracks beside a fused trajectory, not truth."""
    positions = comma("global_pose_frame_positions")
    fused = enu(positions, positions[0])[::4]
    fused[:, 2] = 0
    clouds = [cloud("Fused camera pose (not truth)", fused)]
    for code, label in [("qcom", "QCOM GNSS"), ("ublox", "u-blox GNSS")]:
        measured = comma("processed_log_GNSS_live_gnss_" + code + "_value")
        points = enu(gps_ecef(measured), positions[0])
        points[:, 2] = 0  # altitude conventions may differ; plot only East/North.
        clouds.append(cloud(label, points))
    return {
        "id": "gnss",
        "title": "Three tracks. One real drive.",
        "dataset": "comma2k19 / measured receiver fixes and fused camera poses",
        "license": "MIT / comma.ai sample",
        "views": [
            {
                "id": "tracks",
                "label": "GNSS receivers",
                "kind": "cloud",
                "title": "East / North trajectory in metres",
                "clouds": clouds,
                "metrics": [
                    metric("QCOM fixes", len(clouds[1]["points"])),
                    metric("u-blox fixes", len(clouds[2]["points"])),
                    metric("Path extent", float(np.linalg.norm(fused[-1] - fused[0])), "m"),
                ],
                "note": (
                    "Actual fixes and fused poses from the same recording. Fusion "
                    "uses GNSS, so this is not independent truth. Height is excluded. "
                    "Inspect receiver differences; a lever arm cannot be inferred "
                    "from this plot alone. "
                ),
            }
        ],
        "lever_geometry": {
            "points": fused.tolist(),
            "body_to_enu": (
                enu(np.eye(3) + positions[0], positions[0]).T[None, :, :]
                @ Rotation.from_quat(
                    comma("global_pose_frame_orientations")[:, [1, 2, 3, 0]]
                ).as_matrix()[::4]
            ).tolist(),
        },
    }


def vehicle() -> dict[str, Any]:
    """Compare four measured wheel speeds with reported CAN and fused forward velocity."""
    times = comma("processed_log_CAN_wheel_speed_t")
    wheels = comma("processed_log_CAN_wheel_speed_value")
    pose_times = comma("global_pose_frame_times")
    q = comma(
        "global_pose_frame_orientations"
    )  # upstream Hamilton wxyz, FRD -> ECEF (utils/camera.py)
    body = Rotation.from_quat(q[:, [1, 2, 3, 0]]).inv().apply(comma("global_pose_frame_velocities"))
    select = np.flatnonzero((times >= pose_times[0]) & (times <= pose_times[-1]))[::10]
    t = times[select]
    reference = np.interp(t, pose_times, body[:, 0])
    views = []
    for i, label in enumerate(["Front left", "Front right", "Rear left", "Rear right"]):
        values = wheels[select, i]
        views.append(
            {
                "id": str(i),
                "label": label,
                "kind": "lines",
                "title": label + " wheel / fused camera forward speed",
                "chart": {
                    "x": (t - t[0]).tolist(),
                    "series": [
                        {"name": "Measured wheel speed", "values": values.tolist()},
                        {"name": "Fused camera forward", "values": reference.tolist()},
                    ],
                },
                "metrics": [
                    metric(
                        "Reported speed difference RMSE",
                        float(np.sqrt(np.mean((values - reference) ** 2))),
                        "m/s",
                    ),
                    metric("Measured wheel samples", len(times)),
                    metric("Channel", label),
                ],
                "note": (
                    "Actual wheel encoders and INS/GNSS/vision fused camera velocity. "
                    "Clock, wheel scale, turns and camera-to-vehicle alignment can "
                    "all affect the difference. This comparison does not identify "
                    "them separately. "
                ),
            }
        )
    views.append(
        {
            "id": "body",
            "label": "Body axes",
            "kind": "lines",
            "title": "Measured fused velocity / forward, right, down",
            "chart": {
                "x": (pose_times[::4] - pose_times[0]).tolist(),
                "series": [
                    {"name": name, "values": body[::4, i].tolist()}
                    for i, name in enumerate(["Forward", "Right", "Down"])
                ],
            },
            "metrics": [metric("Velocity frame", "camera FRD"), metric("Poses", len(body))],
            "note": (
                "Upstream fused ECEF velocity rotated using recorded Hamilton wxyz "
                "orientations into camera forward/right/down. This is not a claim "
                "that camera and vehicle frames are identical. "
            ),
        }
    )
    return {
        "id": "vehicle",
        "title": "Four wheels. Measured motion.",
        "dataset": "comma2k19 / CAN wheel encoders and fused camera-frame velocity",
        "license": "MIT / comma.ai sample",
        "views": views,
        "body_velocity": body[::4].tolist(),
    }


def radar() -> dict[str, Any]:
    """Show actual CAN-reported radar tracks; do not invent raw Doppler channels."""
    times = comma("processed_log_CAN_radar_t")
    values = comma("processed_log_CAN_radar_value")
    views = []
    for second in [5, 20, 40, 55]:
        mask = (times >= times[0] + second) & (times < times[0] + second + 0.1)
        rows = values[mask]
        rows = rows[np.isfinite(rows[:, :3]).all(axis=1) & (rows[:, 0] > 0)]
        points = np.c_[rows[:, 0], rows[:, 1], np.zeros(len(rows)), rows[:, 2]]
        views.append(
            {
                "id": str(second),
                "label": f"{second} s",
                "kind": "radar",
                "title": "Forward range / left distance · colour = reported relative speed",
                "clouds": [cloud("Measured radar tracks", points)],
                "metrics": [
                    metric("Reported tracks", len(rows)),
                    metric("Relative speed min", float(rows[:, 2].min()), "m/s"),
                    metric("Relative speed max", float(rows[:, 2].max()), "m/s"),
                ],
                "note": (
                    "Real radar targets decoded from CAN. Relative speed is the "
                    "upstream reported channel, not raw Doppler or a native "
                    "ego-velocity fit. Manual yaw rotates the reported target "
                    "coordinates only; no extrinsic accuracy is asserted. "
                ),
            }
        )
    return {
        "id": "radar",
        "title": "See the traffic your radar sees.",
        "dataset": "comma2k19 / measured CAN radar targets",
        "license": "MIT / comma.ai sample",
        "views": views,
    }


def build() -> None:
    """Write lazy demo assets and bind every original source and producer."""
    for code, make in [
        ("comparison", board),
        ("handeye", handeye),
        ("gnss", gnss),
        ("vehicle", vehicle),
        ("radar", radar),
    ]:
        data = make()
        data.update({"schema": "calibrex.measured_demo/v1", "capture_kind": "real"})
        path = OUTPUT / f"evidence_{code}.json"
        path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
        print(code, path.stat().st_size, len(data["views"]), flush=True)
    sources = [p for p in SOURCE.rglob("*") if p.is_file()]
    sources += [Path(__file__), ROOT / "tools/generate_ethz_hand_eye_benchmarks.py"]
    sources += list((ROOT / "src/calibrex/solvers").glob("*hand_eye*solver.py"))
    sources += [
        ROOT / "src/calibrex/solvers" / name
        for name in [
            "native_planar_board_solver.py",
            "planar_board_lidar_camera_solver.py",
            "point_plane_lidar_camera_solver.py",
            "horn_point_lidar_camera_solver.py",
            "planar_board_line_plane_solver.py",
            "acfr_board_edges.py",
        ]
    ]
    sources += [
        ROOT / "docs/assets/png_luminance_decoding" / f"{name}-00-current/artifacts/result.yaml"
        for name in ("pandey", "levinson")
    ]
    sources += [
        ROOT / "src/calibrex" / name
        for name in ["data/ethz_hand_eye.py", "evaluation/holdout.py", "core/geometry.py"]
    ]
    index = {
        "schema": "calibrex.measured_demo_index/v1",
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "assets": {
            p.name: digest(p)
            for p in OUTPUT.glob("evidence_*.json")
            if p.name != "evidence_demo_index.json"
        },
    }
    (OUTPUT / "evidence_demo_index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    build()
