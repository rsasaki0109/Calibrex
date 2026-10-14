# Rejected GPU gyro deskew prototype

Whole-worker medians improve by only 4.0%.
This does not establish the substantial whole-pipeline benefit sought by this
project. The prototype is rejected; active runtime code is restored to the
previous implementation. Its measured sources and GPU tests remain frozen here.

| Scope | Previous median | Prototype median | Observed ratio |
| --- | ---: | ---: | --- |
| Whole worker | 68.414 s | 65.699 s | 1.041x |
| Complete odometry | 35.495 s | 34.718 s | 1.022x |
| Nested complete registration | 18.066 s | 18.458 s | 0.979x |
| Nested normal preparation | 6.960 s | 6.951 s | 1.001x |
| Unchanged fit/covariance | 15.353 s | 13.943 s | 1.101x |

Three alternating previous/current pairs replay the same first six seconds of
the public GLIM Livox bag with four map rounds. All 12 complete fitted
trajectories/covariances/costs, holdout values and all 32 signed controls per pair
are exactly equal. Point, neighbor, iteration and validation budgets are equal.
Verdicts remain INCONCLUSIVE without independent extrinsic truth.

The prototype evaluates per-point gyro rotations in FP64 on the GPU and reuses
them across translation/registration rounds of a scan. Prefixes remain on the
device; quantization-boundary points are recomputed by the native implementation.
An explicit positive geometry grid is required; zero/default grids and custom
rotation callbacks preserve the native path. The measured candidate records
392 rotation calls over
9,408,000 rows, 784
translation calls and 606,953 native point repairs
per worker. This archive does not infer whole-pipeline speed from a microbenchmark.

Complete odometry includes gyro work, preprocessing, target and normal
preparation and registration. Registration and normals are nested scopes.
Standard `__call__` preprocessing excludes device gyro deskew; its timings are
retained but cannot compare total preprocessing across these two paths. The
unchanged fit timing difference illustrates residual measurement variation.
Component ratios must not be multiplied.

The observer detected no foreign or unknown CUDA compute processes in
43 ten-second observations. Sampling does not establish
continuous GPU exclusivity, observe all graphics activity or reserve the CPU.
Each worker runs in a fresh process with one BLAS/OMP/MKL thread and a separate
initially empty CuPy compilation cache. Whole timing includes imports, bag
decoding, compilation, all odometry, fits, covariance, controls and validation.
Process IDs and names are not published; raw bag data is not redistributed.

Baseline source: `9cf33f996aaaffaf40ab053fc6b3ad1d5f94f46d`. Measured prototype:
`6d5db689ade25af5b6f69de9f960f684449ce7eb`. Public input SHA-256:
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. The publication index binds all workers,
source archives/manifests, launcher/schema, tests, observer and publisher.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_gyro_deskew
```
