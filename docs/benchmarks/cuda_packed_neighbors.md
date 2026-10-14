# Packed target coordinates and exact GPU neighbor heaps

This prototype keeps the existing Morton tree topology and packs its leaf
coordinates once per target. The tree retains original coordinates and point IDs
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
detected during the comparison: **whole-pipeline acceleration is not established**.
This remains a draft pending a repeated comparison without detected GPU
competition. Warm kernel experiments do not establish a whole-worker speedup.

The previous source-row scheduling prototype is excluded from both versions.
Neither point/neighbor/iteration budgets nor validation sampling is reduced.
CPU remains the default. Real-data verdicts remain INCONCLUSIVE without
independent extrinsic truth.

Tests check exhaustive neighbor selection, sorted output, duplicate points,
extreme coordinate scales, partial and empty buffers, every fused output and
reduced-system bit, full registration, covariance, controls and temporal chaining.
The CPU archive audit recomputes ratios and rejects rebound false packing,
observer or registration records.

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
