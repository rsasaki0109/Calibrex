"""Transport preserves the CPU objective; the optional runtime is injected."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.evaluation.camera_lidar_edge import EdgeCamera, EdgeObjective, PreparedFrame
from calibrex.solvers.browser_webgpu_edge_adapter import (
    BrowserWebGpuEdgeObjective,
    pack_edge_objective,
)


def frames() -> list[PreparedFrame]:
    rng = np.random.default_rng(42)
    return [
        PreparedFrame(
            time_s=float(i),
            gap_s=0.0,
            edge=rng.normal(size=(5 + i, 7 + i)).astype(np.float32),
            scale=i + 1,
            points=rng.normal(size=(4 + i, 3)).astype(np.float32),
            weights=rng.uniform(0.1, 2, size=4 + i).astype(np.float32),
            rings=32,
        )
        for i in range(2)
    ]


@pytest.mark.parametrize(
    "model,coefficients,kind,denominator",
    [
        ("none", (), 0, 0),
        ("radtan", (0.1, 0.2), 1, 0),
        ("radtan", tuple(range(8)), 1, 1),
        ("equidistant", (0.1, 0.2, 0.3, 0.4), 2, 0),
        ("equidistant", (), 0, 0),
    ],
)
def test_packing_keeps_all_points_images_and_weights(
    model: Any,
    coefficients: tuple[float, ...],
    kind: int,
    denominator: int,
) -> None:
    source = frames()
    start = np.eye(4)
    start[:3, :3] = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    objective = EdgeObjective(source, EdgeCamera(100, 110, 3, 2, model, coefficients), start)
    packed = pack_edge_objective(objective)
    samples = packed["samples"].reshape(-1, 12)
    assert samples.dtype == np.float32 and samples.flags.c_contiguous
    np.testing.assert_array_equal(samples[:, :3], objective._points)
    np.testing.assert_array_equal(samples[:, 3], objective._weights)
    integers = samples.view(np.uint32)
    np.testing.assert_array_equal(integers[:, 4], objective._offset)
    np.testing.assert_array_equal(integers[:, 5], objective._stride)
    np.testing.assert_array_equal(integers[:, 6], objective._v_limit + 1)
    np.testing.assert_array_equal(samples[:, 8], objective._shift)
    np.testing.assert_array_equal(samples[:, 9], objective._inverse_scale)
    np.testing.assert_array_equal(packed["images"], objective._flat)
    assert packed["totalWeight"] == objective._weights.sum(dtype=np.float64)
    assert packed["camera"][12:14].tolist() == [kind, denominator]


def test_integer_image_offsets_do_not_round_through_float32() -> None:
    objective = EdgeObjective(frames(), EdgeCamera(100, 110, 3, 2), np.eye(4))
    objective._offset[:] = 2**24 + 1
    packed = pack_edge_objective(objective)
    assert np.all(packed["samples"].view(np.uint32).reshape(-1, 12)[:, 4] == 2**24 + 1)


def test_malformed_radtan_coefficients_fail_explicitly() -> None:
    objective = EdgeObjective(frames(), EdgeCamera(100, 110, 3, 2, "radtan", (0.1,)), np.eye(4))
    with pytest.raises(ValueError, match="at least two"):
        pack_edge_objective(objective)


def test_development_fixtures_keep_all_declared_work_and_provenance(tmp_path: Path) -> None:
    import jsonschema
    from tools.prepare_browser_webgpu_edge_fixtures import prepare

    output = tmp_path / "fixtures"
    prepare(output)
    schema = json.loads(
        Path("tools/schemas/browser_webgpu_edge_probe.schema.json").read_text(
            encoding="utf-8",
        )
    )
    fixture_schema = schema["$defs"]["run"]["properties"]["fixture_provenance"]
    fixture_schema = {**fixture_schema, "$defs": schema["$defs"]}
    for path in output.glob("*.npz"):
        with np.load(path, allow_pickle=False) as fixture:
            provenance = json.loads(str(fixture["provenance_json"]))
            jsonschema.validate(provenance, fixture_schema)
            count = 48 if path.stem == "synthetic-48-frames" else 2
            points = 4000 if count == 48 else 37
            assert int(fixture["frames"]) == count
            assert sum(len(fixture[f"points_{i}"]) for i in range(count)) == count * points
            assert all(fixture[f"points_{i}"].dtype == np.float32 for i in range(count))
            assert provenance["accuracy_claim"] is False
    with pytest.raises(ValueError, match="already exists"):
        prepare(output)


def test_async_adapter_preserves_rotation_translation_order_and_lifetime() -> None:
    class Session:
        def __init__(self) -> None:
            self.closed = 0
            self.rows: Any = None

        async def scoreFromPython(self, rows: Any) -> np.ndarray:
            self.rows = rows.reshape(-1, 3, 4).copy()
            return np.array([0.125, -0.25])

        def destroy(self) -> None:
            self.closed += 1

    class Runtime:
        async def createFromPython(self, packed: Any) -> Session:
            assert len(packed["samples"]) == 9 * 12
            return session

    session = Session()

    async def run() -> None:
        start = np.eye(4)
        start[:3, 3] = [0.11, -0.22, 0.33]
        adapter = await BrowserWebGpuEdgeObjective.create(
            frames(),
            EdgeCamera(100, 110, 3, 2),
            start,
            runtime=Runtime(),
        )
        theta = np.array([[1, 2, 3, 0.01, -0.02, 0.03], [0, 0, 0, 1, 2, 3]])
        values = await adapter.score_many(theta)
        np.testing.assert_array_equal(values, [0.125, -0.25])
        np.testing.assert_array_equal(
            session.rows[0, :, :3],
            Rotation.from_rotvec(np.radians(theta[0, :3])).as_matrix().astype(np.float32),
        )
        np.testing.assert_array_equal(
            session.rows[:, :, 3],
            start[:3, 3].astype(np.float32) + theta[:, 3:].astype(np.float32),
        )
        assert (await adapter.score_many(np.empty((0, 6)))).shape == (0,)
        for bad in (np.zeros(6), np.zeros((2, 7)), np.full((1, 6), np.nan)):
            with pytest.raises(ValueError, match="finite N by 6"):
                await adapter.score_many(bad)
        adapter.close()
        adapter.close()
        assert session.closed == 1
        with pytest.raises(RuntimeError, match="closed"):
            await adapter.score_many(theta)

    asyncio.run(run())


@pytest.mark.parametrize("result", [np.zeros(1), np.array([np.nan, 1])])
def test_runtime_failures_are_never_reported_as_gpu_success(result: np.ndarray) -> None:
    class Session:
        async def scoreFromPython(self, rows: Any) -> np.ndarray:
            return result

    async def run() -> None:
        adapter = BrowserWebGpuEdgeObjective(
            EdgeObjective(frames(), EdgeCamera(100, 110, 3, 2), np.eye(4)),
            Session(),
        )
        with pytest.raises(RuntimeError, match="invalid edge scores"):
            await adapter.score_many(np.zeros((2, 6)))

    asyncio.run(run())
