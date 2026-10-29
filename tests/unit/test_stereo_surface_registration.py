"""Pose recovery with missing surfaces, partial overlap and unmatched distractors."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tools.stereo_surface_registration import register


@pytest.mark.parametrize("mode", ["complete", "hole", "occlusion", "combined", "foreground"])
def test_recovers_pose_despite_partial_surface_observation(mode):
    a, b = np.meshgrid(np.linspace(-0.8, 0.8, 25), np.linspace(-0.8, 0.8, 25))
    a, b = a.ravel(), b.ravel()
    target = np.concatenate(
        [
            np.column_stack((np.full(len(a), -0.65), a, b)),
            np.column_stack((a, np.full(len(a), 0.55), b)),
            np.column_stack((a, b, np.full(len(a), -0.45))),
        ]
    )
    camera = target.copy()
    if mode == "foreground":
        patch = (abs(a) < 0.25) & (abs(b) < 0.25)
        keep = np.r_[np.ones(2 * len(a), dtype=bool), ~patch]
        target = np.concatenate(
            (target[keep], np.column_stack((a[patch], b[patch], np.full(int(patch.sum()), -0.65))))
        )
    if mode in ["hole", "combined"]:
        target = target[np.tile(~((abs(a) < 0.25) & (abs(b) < 0.25)), 3)]
    if mode in ["occlusion", "combined"]:
        camera = camera[np.tile(b < 0.5, 3)]
        camera = np.concatenate((camera, np.random.default_rng(17).uniform(2, 3, (100, 3))))
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler("xyz", [1, -1, 0.5], degrees=True).as_matrix()
    truth[:3, 3] = [0.1, -0.05, 0.05]
    source = (camera - truth[:3, 3]) @ truth[:3, :3]
    fit = register(source, target, np.eye(4))
    assert fit["accepted"], fit["result"]["status"]
    estimate = np.array(fit["candidate_T_camera_lidar"])
    rot = Rotation.from_matrix(estimate[:3, :3] @ truth[:3, :3].T).magnitude()
    assert np.degrees(rot) < 0.05
    assert np.linalg.norm(estimate[:3, 3] - truth[:3, 3]) < 0.005
    assert fit["result"]["holdout_point_to_plane_rmse_m"] < 0.005
