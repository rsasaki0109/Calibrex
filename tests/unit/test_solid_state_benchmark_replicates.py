from __future__ import annotations

from pathlib import Path

from tools.run_solid_state_benchmark_replicates import _variant_config


def test_replicate_config_injects_declared_continuous_time_options() -> None:
    base = {
        "pipeline": {
            "factors": {
                "lidar_rig_point_to_plane": {
                    "options": {"continuous_time_outlier_mad_scale": 3.5}
                }
            }
        }
    }

    config = _variant_config(
        base,
        output_dir=Path("outputs/probe"),
        holdout_start_fraction=0.7,
        sampling_seed=17,
        continuous_time_options={"continuous_time_outlier_mad_scale": 2.5},
    )
    options = config["pipeline"]["factors"]["lidar_rig_point_to_plane"]["options"]

    assert options["continuous_time_outlier_mad_scale"] == 2.5
    assert options["continuous_time_holdout_start_fraction"] == 0.7
    assert options["continuous_time_sampling_seed"] == 17
