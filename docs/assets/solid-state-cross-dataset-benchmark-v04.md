# Solid-state LiDAR public benchmark v0.4

This report records the checked public-data-only v0.4 run. It is a paired
temporal-holdout comparison of the frozen `continuous_time_outlier_mad_scale=2.5`
candidate against the uniform/no-rejection baseline. The candidate was fixed
from AgRob train-only diagnostics before this matrix was scored.

- Schema: `slac.solid_state_cross_dataset_benchmark/v0.2`
- Protocol: `bounded_continuous_time_uniform_vs_adaptive_mad25_solid_state_lidar`
- Spec SHA-256: `f0b9acb4d34171ad10787ffaba6dc0e4e85264b5dc14f6f76828f9255e86227f`
- Provenance sources: `142`

## Summary

- Scored datasets: **3/3**
- Scored paired replicates: **27/27**
- Adaptive wins: **25/27** (`0.926`)
- Uniform wins: **2/27**
- Mean holdout improvement: **57.20%**
- Median holdout improvement: **72.79%**
- Bootstrap 95% CI: **[45.69, 67.43]%**

Positive improvement means the adaptive candidate has lower final temporal-
holdout point-to-plane RMSE than `uniform_none` under the frozen protocol.

| Public pair/control | Replicates | Adaptive win rate | Mean improvement (95% CI) | v0.3 mean | Outcome |
|---|---:|---:|---:|---:|---|
| AgRob Modular-e Livox MID-70 ↔ RS-LiDAR | 9 | 0.778 | +16.85% (+5.90, +27.80) | −1.34% | adaptive 7/9; two retained counterexamples |
| TIERS LidarsCali VLP-16 ↔ Livox Horizon | 9 | 1.000 | +82.48% (+80.03, +85.17) | +78.76% | adaptive 9/9 |
| AIST GLIM identity control | 9 | 1.000 | +72.26% (+70.76, +73.60) | +65.67% | adaptive 9/9 |

Compared with v0.3, the aggregate mean improvement increases by **9.50
percentage points**, AgRob changes from mixed-negative to **7/9** adaptive
wins, and the other two public controls remain adaptive wins in every
replicate.

## Failure and interpretation gates

All 54 variant artifacts have rank 6 and sufficient holdout support. Nine of
the 54 variants carry the explicit `max_iterations` failure category; the
declaration keeps `require_converged: false`, so those rows remain visible and
scored rather than being silently removed. This is a performance caveat and a
follow-up budget-sensitivity item, not evidence of independently measured
absolute accuracy.

AgRob and TIERS use `trajectory_only` reference mode and GLIM is an identity
control. The public recordings do not provide independently surveyed
extrinsics or absolute clock truth. Therefore this result supports the frozen
paired temporal-holdout protocol; it is not a universal SOTA or absolute
calibration-accuracy claim.

## Reproduction and validation

The train-only selection evidence is in the
[candidate-selection note](solid-state-cross-dataset-benchmark-v04-train-selection.md)
and the frozen declaration is
`examples/public_datasets/solid_state_cross_dataset_benchmark_v04.yaml`.

```powershell
python tools/run_solid_state_cross_dataset_benchmark.py `
  outputs/solid_state_cross_dataset_benchmark_v04_late_holdout_seed_42.yaml `
  --output outputs/solid_state_cross_dataset_benchmark_v04.yaml `
  --markdown-output outputs/solid_state_cross_dataset_benchmark_v04.md `
  --html-output outputs/solid_state_cross_dataset_benchmark_v04.html

calibrex validate outputs/solid_state_cross_dataset_benchmark_v04.yaml `
  --kind solid-state-cross-dataset-benchmark
```

The full materialization commands and public-data limitations remain in the
[public benchmark runbook](../tutorials/solid_state_public_benchmark.md).
