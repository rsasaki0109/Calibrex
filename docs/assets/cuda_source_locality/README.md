# Public CUDA source-locality comparison

Previous source: `9cf33f996aaaffaf40ab053fc6b3ad1d5f94f46d`. Measurement source:
`752e9a25502d2fe4b42f0fcbfaa2c257c13ac777`. Original public GLIM Livox bag SHA-256:
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw data is not redistributed.

| Scope | Previous CUDA median | Spatial-scheduling median | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 137.106 s | 135.282 s | 1.013x |
| Complete registration | 38.354 s | 36.116 s | 1.062x |
| Nested normal preparation | 14.217 s | 13.658 s | 1.041x |
| Unchanged fits/covariance | 28.471 s | 34.384 s | 0.828x |

| Pair | Method | Whole seconds | Registration seconds |
| --- | --- | ---: | ---: |
| 0 | prior | 107.690510 | 26.902308 |
| 0 | current | 215.891108 | 55.791071 |
| 1 | current | 112.908396 | 27.741631 |
| 1 | prior | 140.293149 | 38.353718 |
| 2 | prior | 137.106036 | 40.711358 |
| 2 | current | 135.282226 | 36.115793 |

All three comparisons retain zero complete calibration/holdout/decision
difference. The audit checks full per-round covariance, fitted trajectories,
costs, convergence and source identity. Final verdicts remain **INCONCLUSIVE**
without independent extrinsic truth.

The first spatial-scheduling worker records 896
dense source orderings over 17,281,155 source rows and
896 Morton-code kernel calls. Those orders
are reused across 4,777 fused iteration kernels.
CuPy's internal sorting launches are excluded from custom-kernel counters.
Every output and system term is written to its original row, retaining existing
FP64 arithmetic, reduction ordering and uncertainty repairs.

Registration wall time includes source ordering, normal preparation, iterative
queries/systems, reductions, small solves and native repairs. Normal preparation
is nested inside registration. Component ratios cannot be multiplied.
Whole timing includes imports, decoding, compilation, all odometry, association,
fits/covariance, noise updates, holdout, controls, hashes and saving.

Three alternating fresh-process pairs use one BLAS thread and separate initially
empty CuPy caches. Driver/runtime caches may persist. Host load is unreserved and
unchanged fits also vary. This chat launched no heavy tests concurrently, but
an independent CUDA sanitizer workload was observed on the shared host during
the first pair. This contended cohort cannot establish an isolated performance
effect. Its medians are retained observations, not a speedup claim or a general
speed guarantee.

All workers share the same six-second, four-round policy, 0.03 m voxels, three
local-map scans, 1 nm coordinate grid and returned-motion grids of 1 micrometre
translation and 0.1 microradian rotation-vector components. CPU remains the
default and grids remain disabled by default. No work or validation budget is
reduced. The index binds every worker, fitted array record, original log, exact
source archive/manifest, launcher, schema, test snapshot and publisher.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_source_locality
```

See [implementation and reproduction](../../benchmarks/cuda_source_locality.md).

A [monitored repeat](repeat-v2/README.md) retained all three exact quality
comparisons, but detected foreign GPU workloads in 32 of 52 samples.
It therefore also does not establish an isolated performance effect.
