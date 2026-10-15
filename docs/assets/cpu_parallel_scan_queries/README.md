# Complete CPU query-worker pilot

| Scope | One CPU query worker | Eight CPU query workers | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 313.401 s | 178.759 s | 1.753x |
| Four fits, including covariance | 18.110 s | 17.913 s | 1.011x |

One fresh-process pair replays the original public GLIM Livox bag's first six
seconds with four map rounds, one BLAS thread and CPU odometry throughout.
This is a preliminary fixed-order observation on an unreserved host. It does
not establish a stable speedup or compare CPU threading against GPU execution.
The fitter is unchanged; its observed timing difference is included in whole time.

Both workers execute identical frozen sources based on `b89db2537c5efafe668048cd61a407de13a377d2`.
The input bag SHA-256 is `f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw data is not redistributed.
Both use 0.03 m voxels, three local-map scans, a 1 nm coordinate grid and returned
motion grids of 1 micrometre translation and 0.1 microradian rotation-vector
components. Defaults remain one CPU query worker with these numerical grids off.

Every complete calibration field, including the stored initial estimate,
matches exactly after excluding only provenance timestamps/commands and the
declared query-worker counts. All four complete fit records, including full
covariance/trajectory arrays, costs, status and iterations, match exactly after
excluding fit wall times. Final judgments remain **INCONCLUSIVE**; this clip
supplies no independent extrinsic truth.

Whole timing includes process startup, imports, bag decoding, every initial,
gyro and feedback odometry pass, final odometry, association, fits/covariance,
noise updates, held-out evidence, signed controls, hashes, saves and source
verification. All worker measurements, full results, source capsules/manifests,
logs, launcher and schema are bound by the publication index.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cpu_parallel_scan_queries
```

See [usage and reproduction](../../benchmarks/parallel_scan_queries.md).
