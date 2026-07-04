from slac.core.config import load_config
from slac.core.frames import FrameGraph
from slac.data.inspect import inspect_dataset
from slac.solvers.open3d_slac_solver import Open3DSLACSolver


def test_open3d_slac_adapter_summarizes_manifest_inputs() -> None:
    config = load_config("examples/rgbd_open3d_slac/config.yaml")
    inspection = inspect_dataset(config.dataset)
    result = Open3DSLACSolver().solve(config, FrameGraph.from_config(config), inspection)
    assert result.backend == "open3d_slac"
    assert result.metrics["rgbd_fragment_count"].value == 3.0
    assert result.metrics["pose_graph_edges"].value == 2.0
    assert result.provenance["fragment_count"] == 3
    assert result.provenance["pose_graph_edges"] == 2
