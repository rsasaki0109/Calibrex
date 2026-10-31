import numpy as np
from tools.evaluate_surface_reliability import surface_gate


def test_near_plane_passes_far_plane_fails_uncertainty_gate():
    accepted, normals, valid = surface_gate(np.full((12, 12), 5.0))
    assert accepted.all() and valid.all()
    np.testing.assert_allclose(normals, np.broadcast_to([0, 0, 1], normals.shape), atol=1e-12)
    assert not surface_gate(np.full((12, 12), 20.0))[0].any()


def test_depth_discontinuity_and_missing_neighbors_are_rejected():
    d = np.full((12, 12), 5.0)
    d[:, 6:] = 20
    assert not surface_gate(d)[0][:, 3:7].any()
    d[:] = np.nan
    assert not surface_gate(d)[2].any()
