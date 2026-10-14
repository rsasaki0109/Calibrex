# GPU scan-front-end prototype

The optional CUDA odometry path crops and voxelizes scans with dedicated FP64
kernels. Voxel candidates sort by signed coordinate keys and original row index.
Each centroid sums its points in original input order, retaining native
`bincount`/`add.at` arithmetic and lexicographic voxel order. Empty, nonfinite,
unsupported-key-range and oversized inputs retain native preprocessing.

The device spatial builder assigns isotropic 63-bit Morton codes, sorts them,
constructs a binary prefix tree and refits tight bounds with child-arrival
atomics. It is an independent implementation of the construction approach in
[Karras, HPG 2012](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees),
without incorporating external source code or adding a dependency.

Complete prefix subtrees of at most eight points become query leaves. Compaction
removes their descendants. Disjoint child boxes use an axis plane; overlapping
boxes retain conservative box traversal. A 96-entry stack covers the 63-bit
spatial code plus the bounded point-index suffix used for duplicate codes.
Original coordinates and original-index tie ordering are preserved. Complete
normal batches follow spatial order, while cached normals retain original row
indices. Native SciPy trees are constructed lazily for existing uncertainty
repairs. CPU computes coordinate extents and controls launches; this is not a
fully GPU-resident odometry pipeline.

CPU remains the default and core schemas are unchanged. The CUDA prototype adds
actual crop/centroid and spatial-construction counters. World-frame scan storage,
gyro modeling, pose solves/updates, spline fitting/covariance, association and
evidence evaluation continue to include CPU work. Preprocessed point arrays
still cross back to the host before later stages.

## Complete comparison

| Scope | Previous CUDA median | Prototype CUDA median | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 114.598 s | 87.859 s | 1.304x |
| Scan preprocessing | 13.573 s | 8.152 s | 1.665x |
| Spatial initialization | 13.348 s | 5.296 s | 2.521x |
| Normal preparation | 9.675 s | 8.419 s | 1.149x |
| Unchanged CPU fits/covariance | 19.914 s | 18.508 s | 1.076x |

Three alternating previous/current CUDA pairs replay the first six seconds of
the original public GLIM Livox bag with four map rounds. Each has zero complete
calibration/holdout/decision difference. Full covariance, trajectory, cost and
convergence records are retained and audited. Final real-data verdicts remain
**INCONCLUSIVE** because this clip has no independent extrinsic truth.

All workers use one BLAS thread and separate initially empty CuPy compilation
caches. Driver/runtime caches may persist; host background load is unreserved.
Whole times and unchanged CPU-fit times vary substantially. The observations
do not establish a precise causal effect or a general guarantee. The complete
improvement remains below the desired large acceleration, so this remains a
draft prototype.

Component timing observes constructor/preprocessing/normal-preparation wall
time without additional synchronization. Late native repair-tree construction
is part of registration/normal work rather than the initialization scope.
Asynchronous work can extend into the following scope. Component ratios are
parts of whole time and are not multiplied. CPU-fit timing differences do not
represent a GPU fitting implementation.

The [source-bound archive](../assets/cuda_scan_frontend/README.md) contains all
complete workers, full fitted arrays, original logs, executed sources, shaders
through their source modules, launcher, schema, tests and publication bindings.
CPU-only auditing requires actual GPU front-end work and validates every ratio.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit e88844448ce9fde72ffa8983e9c5bef24df3ae3d \
  --max-seconds 6 --max-rounds 4 --repeats 3 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/cuda-scan-frontend-comparison
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_scan_frontend
```

Use committed source and a fresh output directory. The coordinate and motion
grids are shared explicit workload settings and remain disabled by default.
Tests cover exact voxel bits and input addition order, crop/key boundaries,
native fallback, empty and uneven trees, exact subtree boxes, unique topology,
lazy repair snapshots, exhaustive neighbors, full registration and ICP evidence.
