# Structured spline systems: complete public bag regression

One fixed-order fresh-process pair compares compiled runtime v0.2
(`fb162b33ffcc24f0400813e5f1e4d7a87c0e8717`) and v0.3 (`8801fa74f52560d80b077041e4de92d7a9dd7f6b`).
Both read the original public GLIM Livox bag's first six seconds, using CPU
odometry, the same default geometry, four map rounds and four fit workers.
Each worker starts with an empty private Numba cache. The complete worker
includes imports, decoding, all fits/covariance, noise updates, holdout,
32 signed controls, hashes, saves and verified frozen imports.

| Scope | Prior compiled | Structured compiled | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 54.809 s | 67.962 s | 0.806x |
| All four fits, including JIT and covariance | 13.826 s | 12.719 s | 1.087x |
| First fit | 5.884 s | 8.963 s | 0.657x |
| Subsequent three fits combined | 7.942 s | 3.756 s | 2.115x |

These exploratory timings do not establish whole-pipeline acceleration.
Host resources were unreserved. Unchanged odometry also varies from 24.242 s
to 36.494 s; first-use compilation is heavier in the new path. All raw times
and the negative whole-worker result are retained. No density, return,
iteration, validation or numerical-grid budget is changed.

Complete decisions and numerical evidence pass the existing bag gates.
The additional full-fit audit verifies all four paired trajectories, six
calibration blocks, covariance, costs and variance at `rtol=1e-8`, `atol=1e-9`.
Status, iterations, dimensions and parameter order agree exactly. The final
verdict remains INCONCLUSIVE; supplied GT is unused. This is regression
evidence, with no independent absolute calibration accuracy claim.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/numba_bspline_banded_bag
python docs/assets/numba_bspline_banded_bag/validate-full-fits.py.txt
```

Both audits require CPU dependencies, neither raw sensors, Numba nor CUDA.
The archive includes complete schema-valid results, every per-round fit,
source manifests/archives and exact executed helpers. The auxiliary report
binds its validator and frozen full-fit comparison helper.
The original bag hash is `f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`; raw arrays remain external.
See [implementation and limits](../../benchmarks/numba_bspline_planes.md).
