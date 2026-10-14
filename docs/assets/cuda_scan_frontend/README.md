# Public GPU scan-front-end prototype comparison

Previous source: `e88844448ce9fde72ffa8983e9c5bef24df3ae3d`. Prototype measurement source:
`dda05b3ebaef7e27f28dcc05f7908fb1d0e4574c`. Original public GLIM Livox bag SHA-256:
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw data is not redistributed.

| Scope | Previous CUDA median | Prototype CUDA median | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 114.598 s | 87.859 s | 1.304x |
| Scan preprocessing | 13.573 s | 8.152 s | 1.665x |
| Spatial initialization | 13.348 s | 5.296 s | 2.521x |
| Normal preparation | 9.675 s | 8.419 s | 1.149x |
| Unchanged CPU fits/covariance | 19.914 s | 18.508 s | 1.076x |

| Pair | Method | Whole seconds | CPU fit seconds |
| --- | --- | ---: | ---: |
| 0 | prior | 134.740828 | 33.482295 |
| 0 | current | 91.858745 | 18.507995 |
| 1 | current | 87.859081 | 19.720539 |
| 1 | prior | 114.598297 | 19.914419 |
| 2 | prior | 80.103179 | 15.202112 |
| 2 | current | 71.571622 | 15.277348 |

All three comparisons have zero rotation, lever-arm, clock, holdout and decision
difference. The audit also checks full per-round fitted covariance, trajectories,
costs and convergence evidence. All final verdicts remain **INCONCLUSIVE** and
this clip provides no independent extrinsic truth.

The first prototype worker records 47,123,273 input
rows over 1,352 GPU preprocessing calls,
2,704 crop/centroid kernel calls and no native
preprocessing fallback. It builds 448 device trees using
1,344 construction kernel calls, and creates
231 native trees lazily for uncertainty repairs.

Three alternating fresh-process pairs use one BLAS thread and separate initially
empty CuPy compilation caches. Driver/runtime caches may persist. Host load is
unreserved. Whole times range substantially between attempts; unchanged CPU fits
also vary. These are observed medians, not a precise causal estimate or a
general speed guarantee. The prototype remains a draft because the complete
improvement is below the desired large acceleration.

The same six-second, four-round dense policy uses 0.03 m voxels, three local-map
scans, a 1 nm coordinate grid and returned-motion grids of 1 micrometre
translation and 0.1 microradian rotation-vector components. CPU remains the
default and all grids remain disabled by default. No return, neighbor or
calibration-iteration budget is reduced.

Whole timing includes imports, decoding, GPU setup/compilation, all odometry
passes, association, fitting/covariance, noise updates, holdout, signed controls,
hashes, saving and verified imports. Component observers add no synchronization.
Spatial initialization records constructor wall time; native repair trees created
later belong to registration/normal preparation. Some asynchronous tail work may
be charged to a following scope. Component ratios are not multiplied.

The index binds every complete result, original log, full fitted arrays, exact
source archives/manifests, launcher, schema, test snapshots and publisher.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_scan_frontend
```

See [implementation, limitations and reproduction](../../benchmarks/cuda_scan_frontend.md).
