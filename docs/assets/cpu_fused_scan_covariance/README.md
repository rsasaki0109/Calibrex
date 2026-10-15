# Rejected fused CPU covariance experiment

| Scope | NumPy covariance | Fused Numba covariance | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker, including first compilation | 165.422 s | 177.744 s | 0.931x |
| Lazy normal preparation, including neighbourhood queries | 59.828 s | 51.136 s | 1.170x |
| Four unchanged fits, including covariance | 14.396 s | 18.932 s | 0.760x |

The candidate did not demonstrate a complete-worker improvement and is not
adopted. Runtime remains the native implementation from `d612581` (identical to
`7b367723` under `src/calibrex`). The observed ratio is not a stable slowdown
estimate: the unchanged fits also varied, and the host was unreserved.

Two fresh-process pairs alternate NumPy/Numba then Numba/NumPy on an unreserved
host. Both run eight CPU query workers and one BLAS thread, replaying the original
public GLIM Livox bag's first six seconds with four map rounds. Imports and the
first compilation belong to whole time; each compiled worker starts with a
separate empty Numba cache. The CPU fitter is unchanged. These observations do
not establish GPU gains or performance on default coarse scans.

Both variants execute identical frozen source bytes based on
`2d6e5caef559b4383bf4da44739bc9dc0146e619`. Input bag SHA-256:
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw scans and compiler caches are not redistributed.
The dense odometry settings are 0.03 m voxels, three local-map scans, a 1 nm
coordinate grid and returned-motion grids of 1 micrometre translation and
0.1 microradian rotation-vector components. Numerical grids remain off by default.

Every complete physical/evidence field, including the initial estimate, is
exactly equal after excluding timestamps, commands and independently validated
execution metadata. All four full fit records in each pair are exactly equal
after excluding wall time: covariance, trajectory, estimates, costs, dimensions,
status and iteration counts. All final verdicts remain **INCONCLUSIVE**; this
clip supplies no independent extrinsic truth.

Compiled work counts, the optional adapter source digest/version, explicit
backend choices and cold caches are validated independently before execution
metadata is excluded from the physical comparison. Complete worker time includes
startup, imports, bag decoding, all initial/gyro/feedback/final odometry passes,
association, four fits/covariance, noise updates, holdout, controls, hashes, saves
and source verification. Normal-preparation timing includes cache checks,
deduplication, SciPy neighbourhood queries, covariance and native PCA.

The publication index binds all full results, measurements/logs, source
capsules/manifests, launcher, schema, publisher and the candidate's correctness
run with bound test sources. Audit without a GPU, Numba compiler or bag:

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cpu_fused_scan_covariance
```

See [usage and reproduction](../../benchmarks/compiled_scan_covariance.md).
