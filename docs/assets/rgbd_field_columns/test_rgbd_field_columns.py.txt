"""Host sum semantics, exact derivatives and bounded field-column scratch arrays."""

import numpy as np
import pytest
from tests.unit.test_rgbd_transform_reuse import batch_problem, inputs

from calibrex.graph import _rgbd_batches as batches
from calibrex.graph import _rgbd_field_columns as fields
from calibrex.graph import joint_optimization as graph


@pytest.mark.parametrize("controls", [1, 2, 8, 27])
@pytest.mark.parametrize("population", ["ordinary", "cancellation", "underflow"])
def test_ordered_probe_sums_match_all_host_float_bits(controls, population):
    rng = np.random.default_rng(20261017)
    if population == "cancellation":
        numbers = np.asarray([0.0, -0.0, 1.0, -1.0, 1e12, -1e12, 1e-300, -1e-300])
        weights, offsets = rng.choice(numbers, (128, controls)), rng.choice(numbers, (16, controls))
    else:
        scale = 1e-300 if population == "underflow" else 1.0
        weights = rng.uniform(-1.0, 1.0, (128, controls)) * scale
        offsets = rng.uniform(-1.0, 1.0, (16, controls)) * scale
    products = weights[None, :, :] * offsets[:, None, :]
    expected = np.asarray([
        sum(row) for row in products.reshape((-1, controls)).tolist()
    ]).reshape(products.shape[:2])
    assert fields._weighted_probes(weights, offsets).tobytes() == expected.tobytes()


def test_unknown_python_uses_ordinary_host_sum(monkeypatch):
    monkeypatch.setattr(fields, "_SUM_VERSION", None)
    weights = np.tile([1.0, 1.0, 1.0, -0.0], (128, 1))
    offsets = np.tile([1e12, 1.0, -1e12, -0.0], (16, 1))
    actual = fields._weighted_probes(weights, offsets)
    expected = np.full((16, 128), sum((weights[0] * offsets[0]).tolist()))
    assert actual.tobytes() == expected.tobytes()


def test_xyz_probe_deduplication_keeps_signed_zero():
    weights = np.tile([0.0, -0.0, 0.3, -0.2], (128, 1))
    offsets = np.asarray([
        [0.0, 0.0, 0.01, -0.03], [-0.0, 0.0, 0.01, -0.03],
        [0.0, -0.0, 0.01, -0.03], [0.0, 0.0, 0.01, -0.03],
    ])
    actual = fields._unique_weighted_probes(weights, offsets)
    expected = fields._weighted_probes(weights, offsets)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("kind", ["trilinear", "centered", "xyz", "pair"])
def test_probe_chunks_bound_scratch_and_keep_complete_scalar_jacobian(kind, monkeypatch):
    blocks, factors = batch_problem(kind, count=97, regularizers=True)
    values = inputs()
    layout = graph._variable_layout(blocks)
    options = graph.JointOptimizerOptions()
    monkeypatch.setattr(fields, "_PROBE_ROWS", 384)
    monkeypatch.setattr(fields, "_PRODUCT_ELEMENTS", 1536)
    rows = []
    products = []
    original_transform = fields._transform_points
    original_weighted = fields._weighted_probes

    def transformed(points, transform):
        rows.append(len(points))
        return original_transform(points, transform)

    def weighted(weights, offsets):
        products.append(len(weights) * len(offsets) * weights.shape[1])
        return original_weighted(weights, offsets)

    monkeypatch.setattr(fields, "_transform_points", transformed)
    monkeypatch.setattr(fields, "_weighted_probes", weighted)
    fast = graph._linearize(factors, values, blocks, layout, options)
    assert rows and max(rows) <= fields._PROBE_ROWS
    assert products and max(products) <= fields._PRODUCT_ELEMENTS
    monkeypatch.setattr(batches, "_MIN_POINTS", 1_000_000)
    scalar = graph._linearize(factors, values, blocks, layout, options)
    assert fast[0].tobytes() == scalar[0].tobytes()
    assert fast[1].tobytes() == scalar[1].tobytes()
