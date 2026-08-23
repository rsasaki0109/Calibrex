# Solid-state LiDAR v0.4 train-only candidate selection

This is a candidate-selection note, not the final v0.4 benchmark. The
selection used public AgRob Modular-e data only and read train diagnostics;
temporal holdout metrics were not used to choose the setting.

## Frozen candidate

The current `adaptive_mad` profile uses
`continuous_time_outlier_mad_scale=3.5`. The v0.4 candidate fixes the same
profile at `2.5`, with the existing `minimum_threshold_m=0.02 m`, support
floor, adaptive voxel policy, solver budget, capture windows, and sampling
policy unchanged.

The diagnostics schema records the evidence used for this decision:

```text
slac.continuous_time_lidar_train_diagnostics/v0.1
selection_metric=train_rmse_m
holdout_used_for_selection=false
```

## AgRob train-only probes

Each row uses seed `0` and a different temporal train prefix. The baseline
values are the v0.3 `adaptive_mad` artifacts; candidate values were replayed
with the v0.4 setting. Lower train RMSE/P95 is preferred, while rank and
support are retained as constraints.

| Train prefix | Baseline train RMSE (m) | Candidate train RMSE (m) | Candidate P95 abs (m) | Candidate kept / rejected | Rank |
|---|---:|---:|---:|---:|---:|
| early (0.60) | 0.3458 | 0.3010 | 0.5861 | 494 / 47 | 6 |
| middle (0.70) | 0.3306 | 0.2836 | 0.5950 | 464 / 56 | 6 |
| late (0.80) | 0.2747 | 0.2064 | 0.4476 | 436 / 94 | 6 |

The candidate is therefore frozen before the v0.4 holdout matrix. Its
performance on TIERS and the GLIM identity control remains an evaluation
question; those datasets are not used to tune this setting.

The reproducible declaration is
`examples/public_datasets/solid_state_cross_dataset_benchmark_v04.yaml`.

The holdout matrix is now complete; see the
[v0.4 benchmark report](solid-state-cross-dataset-benchmark-v04.md) for the
independent evaluation result.
