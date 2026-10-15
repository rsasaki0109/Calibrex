# Public GLIM LiDAR-IMU shared rotations

Three alternating fresh-process pairs compare direct gyro point deskew and
shared spline rotation terms with the preceding analytic-Jacobian baseline.
They use one BLAS thread, the original public GLIM Livox recording's first
12 seconds, four map rounds and the same 30-iteration fit caps. GPU is not
required. These measured ratios compare this cohort's two implementations;
ratios from separate cohorts should not be multiplied.

| Scope | Analytic-Jacobian baseline | Shared rotations | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 142.930 s | 100.615 s | 1.421x |
| Four fits, including covariance | 50.106 s | 33.072 s | 1.515x |

| Trial | Variant | Whole seconds | Fit seconds |
| --- | --- | ---: | ---: |
| 0 | prior | 155.435139 | 50.106017 |
| 0 | current | 100.615110 | 33.071726 |
| 1 | current | 103.203762 | 33.933937 |
| 1 | prior | 139.053514 | 49.604608 |
| 2 | prior | 142.929572 | 52.508813 |
| 2 | current | 94.802650 | 31.483618 |

## Method and source binding

- Prior: `6df5aa2526b1fb4d5bc8511b7e9a0265d1266abc`.
- Current measured source: `296746c8f58e2d740c119086cac62eab8d012a7e`.
- Original public `livox.bag` SHA-256: `f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`.
- PointCloud2 `/livox/points`, point-time `t`; IMU `/livox/imu`, acceleration in g.
- The 12-second clip yields roughly 11.5 reliable seconds and 118 controls;
  the full 61-second recording is not timed here. Supplied GT is not used.
- Whole time includes process startup, reading, initial estimates and gyro
  deskewed odometry, association, all fits and covariance, noise update,
  holdout and signed controls, result saving, hashing and verified imports.
  Fit time includes covariance. Parent source preparation is outside timing.

Every worker retains a schema-valid complete `result.json` and full per-round
calibration, trajectory and covariance in `measurement.json`. Both exact
source trees and their raw-byte manifests are archived. The original worker
logs, actual imported-module digests, executed comparison launcher, tool-only
schema and publisher are included. The publication index binds every file
except itself. Raw recording data is not redistributed.

## Quality and limits

All three result comparisons pass. Policy, block, control, separability and
convergence decisions agree; estimates, std, last-round changes, noise,
chi-square and correlations meet the fixed protocol tolerances. Across the
three pairs, the maximum rotation difference is 2.81e-10
deg, lever-arm difference 6.23e-12 m and clock-offset
difference 4.68e-13 s. The maximum whitened holdout RMS
difference is 2.24e-12. The independent archive audit additionally
checks full per-round covariance, trajectory and costs.

All six final results remain **INCONCLUSIVE**. The capture's estimated plane
map and its formal covariance provide no independent absolute extrinsic
truth. This is a performance/regression comparison for one clip, not an
accuracy or observability claim. Speed depends on point density, control
count, convergence and initial odometry. Exploratory profiles experienced
changing background load and their wall-time ratios are excluded. The
single formal cohort above supplies all reported speed ratios.

## Audit and reproduce

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/lidar_imu_shared_rotations
python -m tools.benchmark_bspline_lidar_imu \
  --baseline-commit 6df5aa2526b1fb4d5bc8511b7e9a0265d1266abc \
  --bag data/public/glim_versatile/livox.bag \
  --max-seconds 12 --max-rounds 4 --repeats 3 \
  --output outputs/glim-shared-rotations-comparison
```

The output directory must be new and runtime changes committed. The archive
audit uses CPU dependencies only. See the
[implementation and evaluation](../../benchmarks/lidar_imu_shared_rotations.md).
