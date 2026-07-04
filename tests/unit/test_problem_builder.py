from slac.core.config import load_config
from slac.core.frames import FrameGraph
from slac.data.inspect import inspect_dataset
from slac.graph.problem import build_problem


def test_problem_builder_compiles_minimal_config() -> None:
    config = load_config("examples/configs/minimal.yaml")
    problem = build_problem(config, FrameGraph.from_config(config), inspect_dataset(config.dataset))
    summary = problem.summary()
    assert summary["pipeline"] == "multi_sensor_slac"
    assert summary["factors"] >= 3
    assert "T_base_camera0" in {variable.name for variable in problem.variables}
    assert "surfel_map" in {variable.name for variable in problem.variables}
    assert problem.observability is not None
    assert problem.observability.gauge_count == 1


def test_problem_builder_uses_actual_root_frame_for_autonomous_config() -> None:
    config = load_config("examples/configs/autonomous_driving.yaml")
    problem = build_problem(config, FrameGraph.from_config(config), inspect_dataset(config.dataset))
    factor_variables = {variable for factor in problem.factors for variable in factor.variables}
    assert "T_base_link_camera0" in factor_variables
    assert "T_base_link_lidar0" in factor_variables
    assert "T_base_link_radar0" in factor_variables
    assert "T_base_camera0" not in factor_variables


def test_problem_builder_compiles_open3d_config() -> None:
    config = load_config("examples/rgbd_open3d_slac/config.yaml")
    problem = build_problem(config, FrameGraph.from_config(config), inspect_dataset(config.dataset))
    assert "open3d_control_grid" in {variable.name for variable in problem.variables}
    assert "open3d_slac" in {factor.name for factor in problem.factors}
