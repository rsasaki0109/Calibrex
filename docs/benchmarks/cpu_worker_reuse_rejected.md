# Rejected CPU worker-reuse pilot

A prototype reused a bounded thread pool within each CPU registration for
correspondence searches and normal neighborhoods, processed short query batches
on the caller and parallelized independent normal PCA rows. Each row kept its
original SciPy query arithmetic, neighbor order, covariance and LAPACK solve.
Workers were joined on normal completion and task failure.

The complete public-data comparison did not demonstrate an improvement, so the
runtime retains the earlier parallel-query implementation. No additional
thread-pool or parallel-PCA execution path is adopted.

| Scope | Previous eight-worker runtime | Reusable-worker prototype | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 249.395 s | 255.982 s | 0.974x |
| Four unchanged fits, including covariance | 31.456 s | 26.400 s | 1.192x |

Both workers used the same eight-worker budget. One fresh-process fixed-order
pair replayed the original public GLIM Livox bag's first six seconds with four
map rounds and one BLAS thread. Host load was unreserved; this observation does
not establish a systematic slowdown. Unchanged CPU fit timings also varied.
The complete-worker evidence did not justify adding the prototype's complexity.

Every complete calibration/evidence field and stored initial estimate agrees
exactly after excluding declared source revisions, command provenance and
timestamps. Every full per-round fit field, including covariance, trajectory,
costs, status and iterations, agrees exactly after excluding fit wall times.
Both results remain INCONCLUSIVE: the clip supplies no independent extrinsic
truth.

Both runs retain 0.03 m voxels, three local-map scans, a 1 nm coordinate grid and
returned-motion grids of 1 micrometre translation and 0.1 microradian
rotation-vector components. Those numerical grids remain disabled by default.
Whole timing includes startup, imports, bag decoding, every initial/gyro/
feedback/final odometry pass, association, fitting/covariance, evidence, hashes,
saves and source verification.

The [bound archive](../assets/cpu_worker_reuse_rejected/README.md) retains every
full result and fit, original measurement/log, both exact source capsules and
manifests, the executed launcher and schema, plus the prototype's validation
test sources and original test output. Its CPU-only audit checks actual import
bindings, worker budgets, complete physical fields and recomputed ratios. Raw
bag data is not redistributed.

## Reproduce the rejected candidate

Use the candidate revision recorded in the archive's protocol, which includes
the prototype. The current runtime has returned to the previous implementation.
The new `cpu-source` comparison keeps explicit worker budgets while evaluating
two source revisions; the existing `cpu-cpu` mode compares budgets on identical
frozen sources.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cpu-source \
  --baseline-commit 7b367723690d867c446742d327a59433c12a705d \
  --prior-query-workers 8 --current-query-workers 8 \
  --max-seconds 6 --max-rounds 4 --repeats 1 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-cpu-worker-reuse-rejected
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cpu_worker_reuse_rejected
```
