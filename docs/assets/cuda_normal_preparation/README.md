# Complete public CUDA normal-preparation comparison

Previous source: `3f070af51921f60451c28b192ea815cc3d466025`. Current source:
`526dada8b41207a729e7f3c02da8ce4d8a04c92f`. The original public GLIM Livox bag SHA-256 is
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw data is not redistributed.

| Scope | Previous CUDA median | Fused-normal CUDA median | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 116.955 s | 90.017 s | 1.299x |
| Normal preparation within the worker | 32.733 s | 8.693 s | 3.766x |
| Four unchanged CPU fits, including covariance | 20.671 s | 17.056 s | 1.212x |

| Pair | Method | Whole seconds | Normal seconds |
| --- | --- | ---: | ---: |
| 0 | prior | 142.245773 | 42.603667 |
| 0 | current | 90.016579 | 8.692601 |
| 1 | current | 95.897507 | 9.259124 |
| 1 | prior | 116.954752 | 30.405365 |
| 2 | prior | 106.645805 | 32.732960 |
| 2 | current | 84.384773 | 7.027216 |

All three complete comparisons have zero rotation, lever-arm, clock, holdout and decision difference. The existing audit also checks every per-round covariance, trajectory, cost and convergence result.

The first current worker records 19,757,494 normal rows,
448 dedicated normal-kernel calls and
474 native normal repairs across 8
odometry passes. No spatial-return budget or quality tolerance was relaxed.

Three alternating fresh-process pairs use one BLAS thread and separate initially
empty CuPy compilation caches. Driver/runtime caches may persist and host load
is not reserved. The six-second, four-round workload retains the same dense
policy: 0.03 m voxels, three local-map scans, a 1 nm coordinate grid and returned
motion grids of 1 micrometre translation and 0.1 microradian rotation-vector
components. These grids remain disabled by default; CPU remains the default.

Whole timing includes imports, reading, GPU setup/compilation, all odometry,
association, fits/covariance, noise updates, holdout, signed controls, hashes,
saves and verified imports. The bound launcher observes outermost normal
preparation without inserting GPU synchronization. The preparation ratio is
part of the whole-worker ratio and is not multiplied by it. CPU fitting is
unchanged; its timing difference is not a GPU-fitting implementation.

All final real-data results remain **INCONCLUSIVE** and no independent extrinsic
truth is supplied. Component acceleration is substantial, while complete-run
acceleration is smaller. These observations apply to this explicit dense workload.

The publication index binds all complete results, original logs, full fitted
covariance/trajectory arrays, input and imported-source hashes, both exact source
archives/manifests, launcher, schema and publisher. Its CPU-only audit recomputes
both whole and preparation ratios and requires actual normal-kernel work.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_normal_preparation
```

See [implementation and reproduction](../../benchmarks/cuda_normal_preparation.md).
