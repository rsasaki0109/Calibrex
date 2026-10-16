"""Native process transport, failure ownership and complete synthetic TUM parity."""

import io
import pickle
import struct
import subprocess
import sys
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from calibrex.core.config import load_config
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.data.inspect import DatasetInspection
from calibrex.graph._rgbd_batches import _rgbd_binding
from calibrex.graph.joint_factors import (
    JointDepthPointToPlaneMeasurement,
    JointTrilinearDepthPointToPlaneMeasurement,
    JointTrilinearXYZPairMeasurement,
    JointTrilinearXYZPointToPlaneMeasurement,
    make_joint_centered_trilinear_depth_point_to_plane_factor,
    make_joint_depth_point_to_plane_factor,
    make_joint_prior_factor,
    make_joint_trilinear_xyz_pair_factor,
    make_joint_trilinear_xyz_point_to_plane_factor,
)
from calibrex.graph.joint_optimization import JointResidualBlock
from calibrex.solvers import _tum_window_processes as p
from calibrex.solvers import native_tum_joint_slac_solver as a
from calibrex.solvers._tum_window_transport import _dump_native, _factor_spec


def _roundtrip(value):
    stream = io.BytesIO()
    _dump_native(stream, value)
    return pickle.loads(stream.getvalue())


def _factors():
    pose = SE3((0.1, -0.0, 0.2), (0.13, -0.17, 0.11, 0.97))
    depth = JointDepthPointToPlaneMeasurement(
        "depth", "frame", (0.2, -0.3, 1.0), 1.2,
        (0.1, 0.2, 1.0), (0.3, -0.2, 0.7), pose, SE3.identity(), 0.7,
    )
    weights = (0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2, 0.05)
    ray = JointTrilinearDepthPointToPlaneMeasurement(
        "centered", "frame", depth.normalized_ray_sensor, depth.nominal_depth_m,
        weights, depth.plane_point_world_m, depth.plane_normal_world,
        pose, SE3.identity(), 0.7,
    )
    xyz = JointTrilinearXYZPointToPlaneMeasurement(
        "xyz", "frame", (0.2, -0.3, 1.2), weights,
        depth.plane_point_world_m, depth.plane_normal_world, pose, SE3.identity(), 0.7,
    )
    pair = JointTrilinearXYZPairMeasurement(
        "pair", "frame", (0.2, -0.3, 1.2), (0.21, -0.29, 1.19),
        weights, weights, depth.plane_normal_world, pose, pose, 0.7,
    )
    return (
        make_joint_depth_point_to_plane_factor(
            depth, pose_block="p", extrinsic_block="e", depth_calibration_block="d",
        ),
        make_joint_centered_trilinear_depth_point_to_plane_factor(
            ray, pose_block="p", extrinsic_block="e",
            depth_bias_block="b", centered_lattice_block="l",
        ),
        make_joint_trilinear_xyz_point_to_plane_factor(
            xyz, pose_block="p", extrinsic_block="e", xyz_lattice_block="x",
        ),
        make_joint_trilinear_xyz_pair_factor(
            pair, source_pose_block="p", target_pose_block="q", xyz_lattice_block="x",
        ),
        make_joint_prior_factor(
            factor_id="prior", observation_group="prior", block="p",
            target=(0.01, -0.0, -0.02, 0.03, 0.04, 0.05),
            sigma=(0.03, 0.07, 0.11, 0.13, 0.17, 0.19),
        ),
    )


@pytest.mark.parametrize("index", range(5))
def test_native_factor_transport_retains_residual_and_binding(index):
    factor = _factors()[index]
    restored = _roundtrip(factor)
    values = {
        "p": (0.01, -0.02, 0.03, 0.04, -0.05, 0.06),
        "q": (-0.01, 0.02, -0.03, -0.04, 0.05, -0.06),
        "e": (0.002, 0.003, -0.001, -0.04, 0.05, -0.02),
        "d": (0.01, -0.02), "b": (0.01,),
        "l": tuple(0.001 * i for i in range(8)),
        "x": tuple(0.001 * i for i in range(24)),
    }
    assert restored is not factor
    assert _factor_spec(restored) == _factor_spec(factor)
    assert _rgbd_binding(restored) is not None
    assert struct.pack(f"<{len(factor.residuals(values))}d", *factor.residuals(values)) == (
        struct.pack(f"<{len(restored.residuals(values))}d", *restored.residuals(values))
    )
    assert restored.sqrt_information == factor.sqrt_information


def test_transport_preserves_stored_quaternion_and_object_sharing():
    pose = SE3((-0.0, 0.1, -0.0), (0.13, -0.17, 0.11, 0.97))
    # A valid stored quaternion need not be invariant under another normalization.
    object.__setattr__(pose, "rotation_quat_xyzw", (0.1, 0.2, 0.3, 0.9273618495495702))
    result = _roundtrip((pose, pose))
    assert result[0] is result[1]
    assert struct.pack("<7d", *pose.translation_m, *pose.rotation_quat_xyzw) == struct.pack(
        "<7d", *result[0].translation_m, *result[0].rotation_quat_xyzw
    )


@pytest.mark.parametrize("mode", ["custom", "code", "ownership", "subclass"])
def test_transport_rejects_callbacks_without_native_ownership(mode):
    factor = _factors()[0]
    if mode == "custom":
        factor = JointResidualBlock("custom", "custom", ("p",), lambda _: (1.0,))
    elif mode == "code":
        factor.evaluator.__code__ = factor.evaluator.__code__.replace(co_name="replacement")
    elif mode == "ownership":
        factor = replace(factor, variable_names=("other",))
    else:
        class CustomFactor(JointResidualBlock):
            pass
        factor = CustomFactor(**factor.__dict__)
    with pytest.raises(ValueError, match=r"native|ownership"):
        _roundtrip(factor)


def _png(path: Path):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
            ">I", zlib.crc32(kind + payload)
        )
    width, height = 32, 24
    rows = b"".join(b"\0" + struct.pack(">32H", *(5000 + (x // 8) * 40 for x in range(width)))
                    for _ in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    ) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


@pytest.fixture
def tiny_tum(tmp_path):
    root = tmp_path / "tum"
    root.mkdir()
    (root / "depth").mkdir()
    for i in range(11):
        _png(root / f"depth/{i}.png")
    (root / "depth.txt").write_text(
        "".join(f"{i}.0 depth/{i}.png\n" for i in range(11)), encoding="utf-8"
    )
    (root / "groundtruth.txt").write_text(
        "".join(f"{i}.0 0 0 0 0 0 0 1\n" for i in range(11)), encoding="utf-8"
    )
    example = Path(__file__).resolve().parents[2] / (
        "examples/public_datasets/tum_rgbd_freiburg1_xyz/native_joint_slac_config.yaml"
    )
    config = yaml.safe_load(example.read_bytes())
    config["dataset"]["path"] = str(root)
    camera = next(v for v in config["sensors"].values() if v["type"] == "camera")
    camera["intrinsics"].update(fx=50.0, fy=50.0, cx=15.5, cy=11.5)
    config["solver"]["max_iterations"] = 1
    config["pipeline"]["factors"][a._FACTOR_NAME]["options"].update(
        frame_start_index=0, frame_stride=1, map_frame_count=1, query_frame_count=4,
        map_points_per_frame=768, query_points_per_frame=192,
        max_correspondences_per_frame=40, replication_start_indices=[0, 6],
        xyz_pair_max_correspondences_per_pair=40, xyz_lattice_local_rotation_updates=0,
        reassociation_max_outer_iterations=1,
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded = load_config(path)
    return loaded, FrameGraph.from_config(loaded), DatasetInspection("tum_rgbd", str(root), True)


@pytest.mark.parametrize("starts", [(0, 6), (6, 100), (100, 0, 6)])
def test_complete_synthetic_window_process_result_matches_serial(tiny_tum, monkeypatch, starts):
    config, frames, inspection = tiny_tum
    original = a._options
    monkeypatch.setattr(p.os, "cpu_count", lambda: 8)
    values = []
    for workers in (1, 2):
        monkeypatch.setattr(a, "_options", lambda config, n=workers: replace(
            original(config), window_workers=n, replication_start_indices=starts
        ))
        value = asdict(a.NativeTUMJointSlacSolver().solve(config, frames, inspection))
        assert value["available"]
        execution = value["provenance"].pop("native_tum_joint_slac_window_execution")
        assert execution["effective_workers"] == workers
        assert execution["backend"] == ("serial" if workers == 1 else "process")
        value["provenance"]["native_tum_joint_slac_options"].pop("window_workers")
        values.append(value)
    assert values[0] == values[1]


def test_child_failure_reports_window_and_stderr(tiny_tum, tmp_path, monkeypatch):
    config, _, inspection = tiny_tum
    job = p._WindowRequest(Path(inspection.path), [], a.TUMDepthIntrinsics(), config,
                           a.TUMJointSlacOptions(), 6)
    monkeypatch.setattr(p, "_worker_command", lambda *_: [
        sys.executable, "-c", "import sys; sys.stderr.write('fixture failure'); sys.exit(7)"
    ])
    children = p._WindowChildren(tmp_path)
    with pytest.raises(RuntimeError, match="window 6 worker exited 7: fixture failure"):
        children.run(job)
    assert not children.processes
    children.close()


def test_close_terminates_only_owned_children(tmp_path):
    children = p._WindowChildren(tmp_path)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32"
                               else 0)
    children.processes.add(process)
    children.close()
    assert process.poll() is not None
    assert children.closed
