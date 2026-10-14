# Public selective B-spline control comparison

Previous source: `392602713e9fc98b9a9defa86c3c29f106dec809`. Selective measurement source:
`75a41f22ee87a439f72610a842a48b26714924c6`. Original public GLIM Livox bag SHA-256:
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw data is not redistributed.

| Scope | Previous median | Selective median | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 77.870 s | 77.507 s | 1.005x |
| Signed control suite | 2.425 s | 0.090 s | 26.920x |
| Unchanged fits/covariance | 17.555 s | 17.058 s | 1.029x |

| Pair | Method | Whole seconds | Controls seconds |
| --- | --- | ---: | ---: |
| 0 | prior | 87.687028 | 2.754651 |
| 0 | current | 88.129986 | 0.090075 |
| 1 | current | 77.506864 | 0.074474 |
| 1 | prior | 76.849514 | 2.155256 |
| 2 | prior | 77.870133 | 2.424818 |
| 2 | current | 75.157340 | 0.091085 |

All three comparisons retain zero complete calibration, holdout and decision
difference. The audit checks full per-round covariance, fitted trajectories,
costs, convergence and source identity. Each worker keeps all 32 signed
perturbations and both IMU baseline scores. Final verdicts remain **INCONCLUSIVE**
and this clip provides no independent extrinsic truth.

The signed IMU controls previously recomputed plane, gyro and acceleration
residuals for each score. The selective version requests only its one IMU family,
skipping 34 unused LiDAR residual evaluations. The selected residuals keep
identical arithmetic. This is CPU work reduction and adds no GPU kernels.

Three alternating fresh-process pairs use one BLAS thread and separate initially
empty CuPy compilation caches. Driver/runtime caches may persist. Background
host load is unreserved; unchanged CPU fitting and whole-worker times also vary.
The whole-worker ratio is an observation, not a precise causal estimate or a
general guarantee. The control-suite ratio is not a whole-pipeline gain.
This cohort does not establish meaningful complete-pipeline acceleration.

All workers share the same six-second, four-round policy, 0.03 m voxels, three
local-map scans, 1 nm coordinate grid and returned-motion grids of 1 micrometre
translation and 0.1 microradian rotation-vector components. CPU remains the
default and grids remain disabled by default. No work or validation budget is
reduced. Whole timing includes imports, decoding, compilation, all odometry,
association, fits/covariance, noise updates, holdout, controls, hashes and saving.
The control observer adds no GPU synchronization and measures the suite once.

The index binds every complete worker, fitted array record, original log, exact
source archive/manifest, launcher, schema, test snapshot and publisher.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/selective_bspline_controls
```

See [implementation and reproduction](../../benchmarks/selective_bspline_controls.md).
