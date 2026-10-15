# Rejected reusable CPU worker pilot

The prototype is **not adopted**. Complete-worker timing did not demonstrate
an improvement; the production runtime keeps the previous implementation.

| Scope | Previous eight-worker runtime | Reusable-worker prototype | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 249.395 s | 255.982 s | 0.974x |
| Four unchanged fits, including covariance | 31.456 s | 26.400 s | 1.192x |

Previous source: `7b367723690d867c446742d327a59433c12a705d`. Measured candidate:
`6f76feb3a019b1e0506ef6382581b0c781ff178a`. Both use eight CPU query workers, one BLAS thread,
the original public GLIM Livox bag's first six seconds and four map rounds.
This is one fixed-order fresh-process pair on an unreserved host, and does not
establish a systematic slowdown. Unchanged fitter timing varied substantially.

The bag SHA-256 is `f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. Raw data is not redistributed.
Both use 0.03 m voxels, three local-map scans, a 1 nm coordinate grid and returned
motion grids of 1 micrometre translation and 0.1 microradian rotation-vector
components. These numerical grids remain disabled by default.

Every complete calibration/evidence field and stored initial estimate agrees
exactly after excluding the declared source revisions, command provenance and
timestamps. Every complete fit field agrees exactly after excluding fit wall
times, including all covariance/trajectory arrays, costs, status and iterations.
Final verdicts remain INCONCLUSIVE: no independent extrinsic truth is supplied.

Whole timing includes startup, imports, decoding, all initial/gyro/feedback/final
odometry, association, fitting/covariance, evidence, hashes, saves and source
verification. Both full results, original worker measurements/logs, exact source
capsules/manifests, bound launcher/schema, publisher and prototype validation
sources/log are retained. The prototype passed all 64 selected tests, including
exact queries/PCA rows and worker cleanup after task failure.

To repeat the experiment, use the measured candidate revision above and the
command in [reproduction instructions](../../benchmarks/cpu_worker_reuse_rejected.md).
The current runtime retains the earlier implementation.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cpu_worker_reuse_rejected
```
