import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation
from tools.cofii2p_input_boundary import (
    Request,
    compose_correction,
    map_input_pixels_to_source,
    pixel_map,
    prepare,
    rigid,
)


def make_request(root):
    image = root / "image.png"
    cloud = root / "cloud.npz"
    initial = root / "initial.json"
    camera = root / "camera.json"
    pixels = np.zeros((375, 1242, 3), dtype=np.uint8)
    pixels[:, :, 0] = 128
    assert cv2.imwrite(str(image), pixels)
    points = np.array([[1.0, 0, 5], [0, 2, 8], [2, 1, 10]])
    np.savez(
        cloud,
        point_lidar_m=points,
        intensity=np.array([[0.1], [0.2], [0.3]]),
        normal_lidar=np.tile([1.0, 0, 0], (3, 1)),
        point_id=np.array([9, 3, 7]),
    )
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    transform[:3, 3] = [0.2, -0.1, 0.3]
    initial.write_text(
        json.dumps(
            {
                "schema_version": "calibrex.explicit_initial_transform/v0.1",
                "transform_camera_lidar": transform.tolist(),
                "provenance": "declared test initializer",
            }
        )
    )
    camera.write_text(
        json.dumps(
            {
                "width": 1242,
                "height": 375,
                "intrinsic_matrix": [[720, 0, 620], [0, 720, 180], [0, 0, 1]],
            }
        )
    )
    request = {
        "schema_version": "calibrex.cofii2p_input/v0.1",
        "frame_id": "test",
        "source_crop_top_rows": 0,
    }
    for name, path in [
        ("image", image),
        ("cloud", cloud),
        ("initializer", initial),
        ("camera", camera),
    ]:
        request[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path = root / "request.json"
    path.write_text(json.dumps(request))
    return path, points, transform


def test_nonidentity_transform_normals_point_ids_and_reference_independence(tmp_path, monkeypatch):
    path, points, transform = make_request(tmp_path)
    reference = tmp_path / "unrelated_reference.json"
    reference.write_text('{"reference":1}')
    original_read = Path.read_text

    def guarded_read(p, *args, **kwargs):
        if p == reference:
            raise AssertionError("reference was read")
        return original_read(p, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    a = prepare(path, tmp_path / "a")
    reference.write_text('{"reference":999999}')
    b = prepare(path, tmp_path / "b")
    x, y = np.load(a.artifact.path), np.load(b.artifact.path)
    for k in x.files:
        np.testing.assert_array_equal(x[k], y[k])
    np.testing.assert_allclose(
        x["point_initial_camera_m"], points @ transform[:3, :3].T + transform[:3, 3]
    )
    np.testing.assert_allclose(x["normal_initial_camera"], np.tile([0, 1, 0], (3, 1)), atol=1e-12)
    np.testing.assert_array_equal(x["point_id"], [9, 3, 7])
    assert x["image_bgr_chw"].shape == (3, 160, 512)
    np.testing.assert_allclose(x["image_bgr_chw"][0], 128 / 255)
    np.testing.assert_array_equal(x["image_bgr_chw"][1:], 0)
    assert a.input_sha256 == b.input_sha256
    assert str(reference) not in a.input_sha256


def test_pixel_affine_roundtrip_and_principal_point_projection():
    affine, _, _ = pixel_map(1241, 375, 50)
    source = np.array([[620.0, 180.0], [100, 100], [1000, 300]])
    image = (np.column_stack([source, np.ones(3)]) @ affine.T)[:, :2]
    np.testing.assert_allclose(map_input_pixels_to_source(image, affine), source)
    k = np.array([[720.0, 0, 620], [0, 710, 180], [0, 0, 1.0]])
    point = np.array([0.2, -0.1, 5])
    p = k @ point
    p /= p[2]
    q = affine @ k @ point
    q /= q[2]
    np.testing.assert_allclose(affine @ p, q)


def test_composition_order_rejects_reflection_and_reference_fields(tmp_path):
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    initial = np.eye(4)
    initial[:3, 3] = [1, 2, 3]
    np.testing.assert_allclose(
        compose_correction(correction.tolist(), initial.tolist()), correction @ initial
    )
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(ValueError):
        rigid(reflection.tolist())
    path, _, _ = make_request(tmp_path)
    request = json.loads(path.read_text())
    request["reference_problem_path"] = "forbidden.json"
    with pytest.raises(ValidationError):
        Request.model_validate(request)


def test_digest_mismatch_fails_before_output(tmp_path):
    path, _, _ = make_request(tmp_path)
    (tmp_path / "image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        prepare(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
