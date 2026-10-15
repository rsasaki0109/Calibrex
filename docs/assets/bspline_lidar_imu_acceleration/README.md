# Public GLIM joint B-spline CPU acceleration

The complete native bag calibration runs 2.554x faster
on this public 12-second clip. Its four complete spline fits, including
marginal covariance, run 6.320x faster. Three alternating
fresh-process pairs use one BLAS thread and the same four map rounds and
30-iteration fit caps. GPU is not required.

| Scope | Prior median | Analytic median | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 325.065 s | 127.257 s | 2.554x |
| Four fits, including covariance | 251.629 s | 39.818 s | 6.320x |

| Trial | Variant | Whole seconds | Fit seconds |
| --- | --- | ---: | ---: |
| 0 | prior | 343.659968 | 259.109449 |
| 0 | current | 132.007329 | 50.824268 |
| 1 | current | 118.642378 | 39.817663 |
| 1 | prior | 325.065390 | 251.628897 |
| 2 | prior | 306.357344 | 231.575880 |
| 2 | current | 127.257188 | 39.379822 |

## Method and source binding

- Prior: `b89e50bbd9e48a05a2acb6010ac46e50ece9220f`.
- Current measured source: `4f3325740f9c821f1d198315f4534ae37482d181`.
- Original AIST GLIM `livox.bag` SHA-256: `f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`.
- PointCloud2 `/livox/points`, point-time `t`; IMU `/livox/imu`, acceleration in g.
- The first 12 seconds yield a roughly 11.5-second reliable odometry segment
  and 118 spline controls; the full 61-second recording is not timed here.
- Whole time runs from process launch through bag reading, angular-rate
  initialization, gyro-deskewed odometry, map association and all four fits,
  covariance, noise update, holdout and 32 signed controls, result saving,
  input hashing and verified imports. Source preparation in the parent is
  outside the timed worker. Fit time includes each fit's covariance.

Every worker has its complete schema-valid `result.json`, full per-round
covariance/trajectory/calibration in `measurement.json`, and original log.
Both exact frozen source trees are in tar archives; their raw-byte manifests
and each worker's actual module-import digests are retained. The executed
benchmark launcher and its tool-only schema are frozen here. `publication-index.json`
binds every published file except itself. The raw recording is not redistributed.

## Result agreement and limits

All three quality comparisons pass. Policy, block/control, separability and
convergence decisions agree; parameter estimates, std, last-round changes,
noise, control chi-square and correlations meet the fixed protocol tolerances.
For the first pair, rotation differs by 1.6e-09 deg,
lever arm by 1.43e-10 m and clock offset by
5.33e-13 s. Whitened holdout RMS differs by at most
1.63e-12. The audit additionally checks
full per-round covariance and trajectory arrays and costs.

All six workers report **INCONCLUSIVE**. This capture and its estimated plane
map do not provide independent absolute extrinsic truth. Formal covariance
holds the plane map fixed. This is a performance/regression comparison, not
an accuracy or observability claim, and does not predict all recordings' speed.

The exploratory profiles under ignored `outputs/` were used to identify the
rotation finite differences. Their initial-odometry timing changed substantially
between runs, so their total-time ratio is not used in these results. The
single formal cohort above is the source of every reported speed ratio.

## Audit and reproduce

```bash
python -m tools.audit_bspline_lidar_imu_benchmark
python -m tools.benchmark_bspline_lidar_imu \
  --baseline-commit b89e50bbd9e48a05a2acb6010ac46e50ece9220f \
  --bag data/public/glim_versatile/livox.bag \
  --max-seconds 12 --max-rounds 4 --repeats 3 \
  --output outputs/glim-joint-bspline-comparison
```

The comparison output must not exist and runtime source changes must be
committed. The archive audit needs the ordinary CPU dependencies only; it
does not open a bag or import CUDA. See the
[benchmark explanation](../../benchmarks/bspline_lidar_imu_acceleration.md).
