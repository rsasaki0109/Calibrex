# Rejected packed CUDA-neighbor prototype

The [latest three-pair comparison](../assets/cuda_packed_neighbors/repeat-v2/README.md)
detected no competing CUDA compute processes in 41 ten-second observations, but
**did not establish useful whole-pipeline acceleration**. Whole medians increase
from 62.250 s to 64.313 s (0.968x, 3.3% longer). Complete registration is nearly
unchanged, while the normal-preparation improvement is offset by other work.
The prototype is rejected; active runtime sources are restored to the previous
version. Its exact measured sources, kernels and device tests remain archived.

| Scope | Previous median | Prototype median | Observed ratio |
| --- | ---: | ---: | ---: |
| Whole worker | 62.250 s | 64.313 s | 0.968x |
| Complete registration | 17.290 s | 17.024 s | 1.016x |
| Nested normal preparation | 6.764 s | 5.822 s | 1.162x |
| Spatial tree construction | 3.897 s | 4.252 s | 0.916x |

All 12 full fitted trajectories, covariance matrices and costs, holdout values
and all 32 signed controls per pair are exactly equal. Normal preparation is
nested inside registration; component ratios must not be multiplied. Periodic
process sampling does not prove continuous GPU exclusivity or reserve CPU
resources. Fresh workers use separate initially empty compilation caches;
whole timing includes imports, decoding, all odometry, fits and validation.

The discarded prototype kept the existing Morton tree topology and packed leaf
coordinates once per target. The tree retained original coordinates and point IDs
for normal indexing, robust-system terms and native uncertainty repairs. Fused
registration queries read contiguous leaf coordinates while writing every
output and all 31 system terms to their original source rows.

Normal-neighbor queries maintain an exact max heap rather than shifting a sorted
candidate array for every hit. A final heap sort restores ascending distance and
original-index tie order. Arithmetic remains FP64 with FMA disabled. The normal
PCA, correspondence gates, repairs, pose solves and stopping rules are unchanged.
The extra coordinate buffer is counted in retained bytes and the payload cap.

The [complete public-data archive](../assets/cuda_packed_neighbors/README.md)
retains one previous/current six-second GLIM pair with four map rounds. All four
full fitted trajectories, covariance matrices and costs, holdout values and all
32 signed control records are exactly equal. Competing GPU workloads were
detected during that first comparison. The later repeated comparison above
provides the current assessment. Warm kernel experiments do not establish a
whole-worker speedup.

The previous source-row scheduling prototype is excluded from both versions.
Neither point/neighbor/iteration budgets nor validation sampling is reduced.
CPU remains the default. Real-data verdicts remain INCONCLUSIVE without
independent extrinsic truth.

Tests check exhaustive neighbor selection, sorted output, duplicate points,
extreme coordinate scales, partial and empty buffers, every fused output and
reduced-system bit, full registration, covariance, controls and temporal chaining.
The CPU archive audit recomputes ratios and rejects rebound false packing,
observer or registration records.

To rerun the discarded prototype, use a clean checkout of its measured commit
`7216d17c12203cd94152117cd1f0eda87360bcf8` for the benchmark command below.
The restored active runtime does not execute this prototype. The archive audit
can run from the current checkout without CUDA.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit 9cf33f996aaaffaf40ab053fc6b3ad1d5f94f46d \
  --max-seconds 6 --max-rounds 4 --repeats 3 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/cuda-packed-neighbors-comparison
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_packed_neighbors
```
